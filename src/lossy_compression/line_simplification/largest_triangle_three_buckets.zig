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

pub fn decompress(allocator: Allocator, compressed_values: []const u8, decompressed_values: *ArrayList(f64)) Error!void {
    //TODO
    _ = allocator;
    _ = compressed_values;
    _ = decompressed_values;
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
