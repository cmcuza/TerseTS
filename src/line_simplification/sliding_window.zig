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

//! Implementation of "Sliding-Window" algorithm as described in the paper:
//! "E. Keogh, S. Chu, D. Hart and M. Pazzani.
//! An online algorithm for segmenting time series.
//! IEEE ICDM, pp. 289-296, 2001.
//! https://doi.org/10.1109/ICDM.2001.989531"
//! This specific implementation uses the "Root-Mean-Squared-Errors (RMSE)"
//! as the cost function. Future work may include other cost functions.

const std = @import("std");
const mem = std.mem;
const math = std.math;
const time = std.time;
const ArrayList = std.ArrayList;

const tersets = @import("../tersets.zig");
const Method = tersets.Method;
const Error = tersets.Error;

const shared_structs = @import("../utilities/shared_structs.zig");

const DiscretePoint = shared_structs.DiscretePoint;
const LinearFunction = shared_structs.LinearFunction;

const tester = @import("../tester.zig");

const testing = std.testing;

/// Compresses `uncompressed_values` using the "Sliding Window" simplification algorithm.
/// This algorithm iteratively merges points to minimize the RMSE, ensuring that the resulting
/// compressed sequence stays within the specified `error_bound`. The function writes the
/// simplified sequence to the `compressed_values`. If an error occurs it is returned.
pub fn compress(
    uncompressed_values: []const f64,
    compressed_values: *ArrayList(u8),
    error_bound: f32,
) Error!void {
    var seg_start: usize = 0;

    // Return error if the error bound is negative.
    if (error_bound < 0) return Error.UnsupportedErrorBound;

    // Iterate through the input values to segment them.
    while (seg_start < uncompressed_values.len - 1) {
        // We can skip the next point as it has 0 error.
        var seg_end = seg_start + 2;

        // Expand the segment as long as the RMSE is within the error bound.
        while ((seg_end < uncompressed_values.len) and
            (computeRMSE(uncompressed_values, seg_start, seg_end) < error_bound)) : (seg_end += 1)
        {}

        // Store the segment's start value, end index, and end value in the compressed output.
        try appendValue(f64, uncompressed_values[seg_start], compressed_values);
        try appendValue(usize, seg_end - 1, compressed_values);
        try appendValue(f64, uncompressed_values[seg_end - 1], compressed_values);

        // Move to the next segment.
        seg_start = seg_end;
    }

    // Check if the last point was not enconded by checking if seg_start is at that point.
    // If the last point was not encoded, we need to insert it as a single-point segment to
    // maintain the compressed representation as a multiple of 3 values (start_value, end_time,
    // end_value). This is important because the decompression logic expects the compressed
    // data to be a sequence of 3-tuples, allowing safe and correct decoding. Adding one extra
    // value for the last point (if needed) has negligible impact on the compression ratio,
    // since it only occurs once per time series (if at all), regardless of the input size.
    if (seg_start == uncompressed_values.len - 1) {
        try appendValue(f64, uncompressed_values[seg_start], compressed_values);
        try appendValue(usize, seg_start, compressed_values);
        try appendValue(f64, uncompressed_values[seg_start], compressed_values);
    }
}

/// Decompress `compressed_values` produced by "Sliding Window". The function writes the result to
/// `decompressed_values`. If an error occurs it is returned.
pub fn decompress(compressed_values: []const u8, decompressed_values: *ArrayList(f64)) Error!void {
    // The compressed representation is composed of three values: (start_value, end_time, end_value)
    // all of type 64-bit float.
    if (compressed_values.len % 24 != 0) return Error.UnsupportedInput;

    const compressed_lines_and_index = mem.bytesAsSlice(f64, compressed_values);

    var first_timestamp: usize = 0;
    var index: usize = 0;

    while (index < compressed_lines_and_index.len) : (index += 3) {
        const start_point = .{ .time = first_timestamp, .value = compressed_lines_and_index[index] };
        const end_point = .{
            .time = @as(usize, @bitCast(compressed_lines_and_index[index + 1])),
            .value = compressed_lines_and_index[index + 2],
        };

        // Check if the segment has only two points. If so, we can directly append their values.
        if (start_point.time + 1 < end_point.time) {
            const duration: f64 = @floatFromInt(end_point.time - start_point.time);

            const slope = (end_point.value - start_point.value) / duration;
            const intercept = start_point.value - slope *
                @as(f64, @floatFromInt(start_point.time));

            try decompressed_values.append(start_point.value);
            var current_timestamp: usize = start_point.time + 1;

            // Interpolate the values between the start and end points of the current segment.
            while (current_timestamp < end_point.time) : (current_timestamp += 1) {
                const y: f64 = slope * @as(f64, @floatFromInt(current_timestamp)) + intercept;
                try decompressed_values.append(y);
            }
            try decompressed_values.append(end_point.value);
            first_timestamp = current_timestamp + 1;
        } else {
            // If the start and end points are the distance 1,
            // append the start point and end points directly.
            try decompressed_values.append(start_point.value);
            // Check wheter the point is the same. If so, then we are at the end of the time series.
            // Thus, do not insert the end point. Otherwise, insert the end point.
            if (start_point.time != end_point.time)
                try decompressed_values.append(end_point.value);
            first_timestamp += 2;
        }
    }
}

/// Computes the Root-Mean-Squared-Errors (RMSE) for a segment of the `uncompressed_values`.
/// This function calculates the error between the actual values and the predicted values
/// based on a linear regression model fitted to the segment defined by `seg_start` and `seg_end`.
fn computeRMSE(uncompressed_values: []const f64, seg_start: usize, seg_end: usize) f64 {
    const seg_len: f64 = @floatFromInt(seg_end - seg_start + 1);
    if (seg_len <= 1) return 0.0; // If the segment has one or no points, return zero error.

    const slope: f64 = (uncompressed_values[seg_end] - uncompressed_values[seg_start]) / (seg_len - 1);
    const intercept: f64 = uncompressed_values[seg_start] - slope * @as(f64, @floatFromInt(seg_start));

    // Calculate the RMSE between actual and predicted values.
    var sse: f64 = 0;
    var i = seg_start;
    while (i <= seg_end) : (i += 1) {
        const pred = slope * @as(f64, @floatFromInt(i)) + intercept; // Predicted value.
        const diff = uncompressed_values[i] - pred; // Difference between actual and predicted.
        sse += diff * diff; // Accumulate squared differences.
    }

    // Return the RMSE.
    return math.sqrt(sse / seg_len);
}

/// Computes the maximum absolute (Chebyshev, L-infinity) error between the actual values and the
/// linear interpolation over a segment of the input array. This function fits a straight
/// line between the values at `seg_start` and `seg_end` in `uncompressed_values`, then
/// calculates the maximum absolute difference between the actual values and the predicted
/// values (from the fitted line) for all indices in the segment `[seg_start, seg_end]`.
fn computeMaxAbsoluteError(uncompressed_values: []const f64, seg_start: usize, seg_end: usize) f64 {
    const seg_len: f64 = @floatFromInt(seg_end - seg_start + 1);
    if (seg_len <= 2) return 0.0; // If the segment has less than 3 points, return zero error.

    const slope: f64 = (uncompressed_values[seg_end] - uncompressed_values[seg_start]) / (seg_len - 1);
    const intercept: f64 = uncompressed_values[seg_start] - slope * @as(f64, @floatFromInt(seg_start));

    // Calculate the maximum absolute error of the segment.
    var linf: f64 = 0;
    var i = seg_start;
    while (i <= seg_end) : (i += 1) {
        const pred = slope * @as(f64, @floatFromInt(i)) + intercept; // Predicted value.
        const diff = @abs(uncompressed_values[i] - pred); // Difference between actual and predicted.
        linf = @max(diff, linf);
    }

    // Return max abs.
    return linf;
}

/// Append `value` of `type` determined at compile time to `compressed_values`.
fn appendValue(comptime T: type, value: T, compressed_values: *ArrayList(u8)) !void {
    // Compile-time type check
    switch (@TypeOf(value)) {
        f64, usize => {
            const value_as_bytes: [8]u8 = @bitCast(value);
            try compressed_values.appendSlice(value_as_bytes[0..]);
        },
        else => @compileError("Unsupported type for append value function"),
    }
}

/// Test if the RMSE of the linear regression line that fits the points in the segment in `values`
/// is within the `error_bound`.
pub fn testRMSEisWithinErrorBound(
    values: []const f64,
    error_bound: f32,
) !void {
    // At least two points are needed to form a line.
    if (values.len < 2) return;

    const rmse = computeRMSE(values, 0, values.len - 1);
    try testing.expect(rmse <= error_bound);
}

test "sliding-window can compress and decompress with zero error bound" {
    const allocator = testing.allocator;

    // Output buffer.
    var uncompressed_values = ArrayList(f64).init(allocator);
    defer uncompressed_values.deinit();
    var compressed_values = ArrayList(u8).init(allocator);
    defer compressed_values.deinit();
    var decompressed_values = ArrayList(f64).init(allocator);
    defer decompressed_values.deinit();
    const error_bound: f32 = 0.0;

    const max_lines: usize = @intFromFloat(@round(tester.generateBoundedRandomValue(f64, 4, 25, undefined)));
    // Sliding-Window cannot handle very large value due to numerical issues with `math.order()`.
    for (0..max_lines) |_| {
        try tester.generateBoundedRandomValues(&uncompressed_values, -1e16, 1e16, undefined);
    }
    // Call the compress and decompress functions.
    try compress(uncompressed_values.items, &compressed_values, error_bound);
    try decompress(compressed_values.items, &decompressed_values);

    // Check if the decompressed values have the same lenght as the compressed ones.
    try testing.expectEqual(uncompressed_values.items.len, decompressed_values.items.len);
}

test "sliding-window random lines and random error bound compress and decompress" {
    const allocator = testing.allocator;

    var uncompressed_values = ArrayList(f64).init(allocator);
    defer uncompressed_values.deinit();
    var compressed_values = ArrayList(u8).init(allocator);
    defer compressed_values.deinit();
    var decompressed_values = ArrayList(f64).init(allocator);
    defer decompressed_values.deinit();

    const error_bound: f32 = tester.generateBoundedRandomValue(f32, 0, 1, undefined);

    const max_lines: usize = @intFromFloat(@round(tester.generateBoundedRandomValue(f64, 4, 25, undefined)));
    for (0..max_lines) |_| {
        // Generate a random linear function and add it to the uncompressed values.
        try tester.generateRandomLinearFunction(&uncompressed_values, undefined);
    }

    try compress(
        uncompressed_values.items,
        &compressed_values,
        error_bound,
    );

    try decompress(compressed_values.items, &decompressed_values);

    // Check if the decompressed values have the same lenght as the compressed ones.
    try testing.expectEqual(uncompressed_values.items.len, decompressed_values.items.len);

    // In theory, the linear interpolation of all segments formed by the slices of preserved points, should have a RMSE
    // within the error bound otherwise there a mistake. Since the error bound and the poitns are unknown, we need to
    // used the compressed representation to access each of the points preserved and their index `current_point_index`.
    // Then, the RMSE of the linear regression of the segment formed by the slices from
    // `previous_point_index`..`current_point_index` should be less than `error_bound`.
    const compressed_representation = mem.bytesAsSlice(f64, compressed_values.items);

    var index: usize = 0;
    var previous_point_index: usize = 0;
    while (index < compressed_representation.len - 1) : (index += 3) {
        const current_point_index = @min(
            @as(usize, @bitCast(compressed_representation[index + 1])),
            uncompressed_values.items.len - 1,
        );

        // Check if the point is within the error bound.
        try testRMSEisWithinErrorBound(
            uncompressed_values.items[previous_point_index .. current_point_index + 1],
            error_bound,
        );
        previous_point_index = current_point_index + 1;
    }
}
