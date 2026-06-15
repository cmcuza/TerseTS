// Copyright 2026 TerseTS Contributors
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

const std = @import("std");
const mem = std.mem;
const math = std.math;
const testing = std.testing;
const ArrayList = std.ArrayList;
const Allocator = mem.Allocator;

const tersets = @import("../../tersets.zig");
const configuration = @import("../../configuration.zig");
const Method = tersets.Method;
const Error = tersets.Error;

const shared_structs = @import("../../utilities/shared_structs.zig");
const shared_functions = @import("../../utilities/shared_functions.zig");

const DiscretePoint = shared_structs.DiscretePoint;

const calculateArea = shared_functions.calculateTriangleArea;

/// Compress `uncompressed_values` using "Largest Triangle Three Buckets" simplification algorithm by keeping
/// points which form the greatest triangle between an anchor point of the previous bucket and the average point
/// of the next bucket. The function writes the result to `compressed_values`. The `allocator` is used to
/// allocate memory for `method_configuration` parser. The `method_configuration` is expected to be of
/// `OutputThresholdNumber` type otherwise an `InvalidConfiguration` error is return.
/// If any other error occurs during the execution of the method, it is returned.
pub fn compress(
    allocator: Allocator,
    uncompressed_values: []const f64,
    compressed_values: *ArrayList(u8),
    method_configuration: []const u8,
) Error!void {
    const parsed_configuration = try configuration.parse(
        allocator,
        configuration.OutputThresholdNumber,
        method_configuration,
    );

    const threshold: u32 = parsed_configuration.output_threshold_number;

    // Return an error for empty value slice or for a threshold that is too low,
    // because at least 2 points are always necessary, as they anchor the line.
    if (uncompressed_values.len < 2) return Error.UnsupportedInput;
    if (threshold < 2) return Error.UnsupportedInput;

    // Series is already at or below the target size: keep every point
    // and just output (value, index) pairs normally
    if (uncompressed_values.len <= threshold) {
        try shared_functions.appendValue(allocator, f64, uncompressed_values[0], compressed_values);
        for (1..uncompressed_values.len) |i| {
            try shared_functions.appendValue(allocator, f64, uncompressed_values[i], compressed_values);
            try shared_functions.appendValue(allocator, usize, i, compressed_values);
        }
        return;
    }

    // The first and last point are in their own buckets
    const inner_bucket_count: u32 = threshold - 2;
    const points_per_bucket: u32 = @ceil((uncompressed_values.len - 2 / inner_bucket_count));

    var selected_point = DiscretePoint{ .index = 0, .value = uncompressed_values[0] };
    try shared_functions.appendValue(allocator, f64, uncompressed_values[0], compressed_values);

    for (0..inner_bucket_count) |bucket_idx| {
        // Calculate average of next bucket.
        // Our second to last bucket is potentially cut off due to the threshold not
        // dividing the number of points cleanly
        var avg = 0;
        var start = @floor((bucket_idx + 1) * points_per_bucket + 1);
        var end = @floor((bucket_idx + 2) * points_per_bucket + 1);
        end = math.clamp(end, 0, uncompressed_values.len);

        for (start..end) |point_idx| {
            avg = avg + uncompressed_values[point_idx];
        }
        avg = avg / (end - start);
        const avg_point = DiscretePoint{ .index = (start + end) / 2, .value = avg };

        // current bucket
        start = @floor(bucket_idx * points_per_bucket + 1);
        end = @floor((bucket_idx + 1) * points_per_bucket + 1);
        var max_area: f64 = calculateArea(selected_point, DiscretePoint{ .index = start, .value = uncompressed_values[start] }, avg_point);
        var best_point = DiscretePoint{ .index = start, .value = uncompressed_values[start] };

        for (start + 1..end) |point_idx| {
            const curr_area = calculateArea(
                selected_point,
                DiscretePoint{ .index = point_idx, .value = uncompressed_values[point_idx] },
                avg_point,
            );
            if (curr_area > max_area) {
                max_area = curr_area;
                best_point.index = point_idx;
                best_point.value = uncompressed_values[point_idx];
            }
        }

        // save the point with the highest rank and use that as the next starting point
        try shared_functions.appendValue(allocator, f64, best_point.value, compressed_values);
        try shared_functions.appendValue(allocator, usize, best_point.index, compressed_values);
        selected_point = best_point;
    }

    // Anchor the line to original end point
    try shared_functions.appendValue(allocator, f64, uncompressed_values[uncompressed_values.len - 1], compressed_values);
    try shared_functions.appendValue(allocator, usize, uncompressed_values.len, compressed_values);
}

/// Decompress `compressed_values` produced by "Largest Triangle Three Buckets" and write the
/// result to `decompressed_values`. If an error occurs it is returned.
pub fn decompress(allocator: Allocator, compressed_values: []const u8, decompressed_values: *ArrayList(f64)) Error!void {
    // The compressed representation is composed of two values after getting the first since all
    // segments are connected. Therefore, the condition checks that after the first value, the rest
    // of the values are in pairs (value, index) and that they are all of type 64-bit float.
    if ((compressed_values.len - 8) % 16 != 0) return Error.UnsupportedInput;

    const compressed_lines_and_index = mem.bytesAsSlice(f64, compressed_values);

    var index: usize = 0;

    // Extract the start point from the compressed representation.
    var start_point: DiscretePoint = .{ .index = 0, .value = compressed_lines_and_index[0] };
    try decompressed_values.append(allocator, start_point.value);

    // We need to create a segment for the linear function.
    var slope: f64 = undefined;
    var intercept: f64 = undefined;
    while (index < compressed_lines_and_index.len - 1) : (index += 2) {
        // index + 1 is the end value and index + 2 is the end time.
        const end_point: DiscretePoint = .{
            .index = @as(usize, @bitCast(compressed_lines_and_index[index + 2])),
            .value = compressed_lines_and_index[index + 1],
        };

        if (start_point.index + 1 < end_point.index) {
            // Create the linear approximation for the current segment.
            const duration: f64 = @floatFromInt(end_point.index - start_point.index);
            slope = (end_point.value - start_point.value) / duration;
            intercept = start_point.value - slope *
                @as(f64, @floatFromInt(start_point.index));

            var current_index: usize = start_point.index + 1;
            // Interpolate the values between the start and end points of the current segment.
            while (current_index < end_point.index) : (current_index += 1) {
                const y: f64 = slope * @as(f64, @floatFromInt(current_index)) + intercept;
                try decompressed_values.append(allocator, y);
            }
        }
        try decompressed_values.append(allocator, end_point.value);

        // The start point of the next segment is the end point of the current segment.
        start_point = end_point;
    }
}

pub fn extract(
    allocator: Allocator,
    compressed_values: []const u8,
    indices: *ArrayList(u64),
    coefficients: *ArrayList(f64),
) Error!void {
    //TODO
    _ = allocator;
    _ = compressed_values;
    _ = indices;
    _ = coefficients;
}

pub fn rebuild(
    allocator: Allocator,
    indices: []const u64,
    coefficients: []const f64,
    compressed_values: *ArrayList(u8),
) Error!void {
    //TODO
    _ = allocator;
    _ = indices;
    _ = coefficients;
    _ = compressed_values;
}

//TODO! Tests here
