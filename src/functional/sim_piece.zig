// Copyright 2024 TerseTS Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Implementation of the Sim-Piece algorithm from the paper
//! "Xenophon Kitsios, Panagiotis Liakos, Katia Papakonstantinopoulou, and Yannis Kotidis.
//! Sim-Piece: Highly Accurate Piecewise Linear Approximation through Similar Segment Merging.
//! Proc. VLDB Endow. 16, 8 2023.
//! https://doi.org/10.14778/3594512.3594521".

const std = @import("std");
const math = std.math;
const mem = std.mem;
const testing = std.testing;
const ArrayList = std.ArrayList;

const tersets = @import("../tersets.zig");
const Error = tersets.Error;
const DiscretePoint = tersets.DiscretePoint;

const SimPieceContainer = struct {
    timestamp: usize,
    intercept: f64,
    upper_bound_slope: f64,
    lower_bound_slope: f64,
};

/// Compress `uncompressed_values` within `error_bound` using "Sim-Piece" and write the
/// result to `compressed_values`. If an error occurs it is returned.
pub fn compressSimPiece(
    uncompressed_values: []const f64,
    compressed_values: *ArrayList(u8),
    allocator: mem.Allocator,
    error_bound: f32,
) Error!void {
    // Adjust the error bound to avoid exceeding it during decompression.
    const adjusted_error_bound = if (error_bound > 0)
        error_bound - tersets.ErrorBoundMargin
    else
        error_bound;

    var simpiece_container_list = ArrayList(SimPieceContainer).init(allocator);
    defer simpiece_container_list.deinit();

    var upper_bound_slope: f64 = math.f64_max;
    var lower_bound_slope: f64 = math.f64_min;

    // Initialize the `start_point` with the first uncompressed value.
    var start_point: DiscretePoint = .{ .time = 0, .value = uncompressed_values[0] };

    var quantized_start_value = quantize(uncompressed_values[0], adjusted_error_bound);

    // First point already part of `current_segment`, next point is at index one.
    var current_timestamp: usize = 1;
    while (current_timestamp < uncompressed_values.len) : (current_timestamp += 1) {
        const end_point: DiscretePoint = .{
            .time = current_timestamp,
            .value = uncompressed_values[current_timestamp],
        };

        const segment_size: usize = current_timestamp - start_point.time;
        const upper_limit: f64 = upper_bound_slope * segment_size + quantized_start_value;
        const lower_limit: f64 = lower_bound_slope * segment_size + quantized_start_value;

        if (segment_size > 2) {
            if ((upper_limit < (end_point.value - adjusted_error_bound)) or
                ((lower_limit > (end_point.value + adjusted_error_bound))))
            {
                simpiece_container_list.append(.{
                    .timestamp = start_point.time,
                    .intercept = quantized_start_value,
                    .upper_bound_slope = upper_bound_slope,
                    .lower_bound_slope = lower_bound_slope,
                });
                start_point = end_point;
                quantized_start_value = quantize(start_point.value, adjusted_error_bound);
                continue;
            }
        }

        const new_upper_bound_slope: f64 =
            (end_point.value + adjusted_error_bound - quantized_start_value) / segment_size;
        const new_lower_bound_slope: f64 =
            (end_point.value - adjusted_error_bound - quantized_start_value) / segment_size;
        if (segment_size == 2) {
            upper_bound_slope = new_upper_bound_slope;
            lower_bound_slope = new_lower_bound_slope;
        } else {
            if (end_point.value + adjusted_error_bound < upper_limit)
                upper_bound_slope = @max(new_upper_bound_slope, lower_limit);
            if (end_point.value - adjusted_error_bound > lower_limit)
                lower_bound_slope = @min(new_lower_bound_slope, upper_limit);
        }
    }
}

// fn compressSimPiecePhaseOne()

/// Quantizes the given `value` by the specified `error_bound`. This process ensures that
/// the quantized value remains within the error bound of the original value.
fn quantize(value: f64, error_bound: f32) f64 {
    return @floor(value / error_bound) * error_bound;
}
