// Copyright 2025 TerseTS Contributors
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
const configuration = @import("../configuration.zig");
const Method = tersets.Method;
const Error = tersets.Error;

const shared_structs = @import("../utilities/shared_structs.zig");
const shared_functions = @import("../utilities/shared_functions.zig");

const DiscretePoint = shared_structs.DiscretePoint;
const LinearFunction = shared_structs.LinearFunction;

const tester = @import("../tester.zig");

const testing = std.testing;

/// Compresses `uncompressed_values` using the "Sliding Window" simplification algorithm.
/// This algorithm iteratively merges points to minimize the RMSE, ensuring that the resulting
/// compressed sequence stays within the specified `error_bound`. The function writes the
/// simplified sequence to the `compressed_values`. The `allocator` is used to allocate memory
/// for the `method_configuration` parser. The `method_configuration` is expected to be of
/// `AggregateError` type otherwise an `InvalidConfiguration` error is return. If any other
/// error occurs during the execution of the method, it is returned.
pub fn compress(
    allocator: mem.Allocator,
    uncompressed_values: []const f64,
    compressed_values: *ArrayList(u8),
    method_configuration: []const u8,
) Error!void {
    const parsed_configuration = try configuration.parse(
        allocator,
        configuration.AggregateError,
        method_configuration,
    );

    const error_bound: f32 = parsed_configuration.aggregate_error_bound;

    var seg_start: usize = 0;
    // Iterate through the input values to segment them.
    while (seg_start < uncompressed_values.len - 1) {
        // We can skip the next point as it has 0 error.
        var seg_end = seg_start + 2;

        // Expand the segment as long as the RMSE is within the error bound.
        while ((seg_end < uncompressed_values.len) and
            (try computeRMSE(uncompressed_values, seg_start, seg_end) <= error_bound)) : (seg_end += 1)
        {}

        // Store the segment's start value, end index, and end value in the compressed output.
        try shared_functions.appendValue(f64, uncompressed_values[seg_start], compressed_values);
        try shared_functions.appendValue(f64, uncompressed_values[seg_end - 1], compressed_values);
        try shared_functions.appendValue(usize, seg_end - 1, compressed_values);

        // Move to the next segment.
        seg_start = seg_end;
    }

    // Check if the last point was not enconded by checking if seg_start is at that point.
    // If the last point was not encoded, we need to insert it as a single-point segment to
    // maintain the compressed representation as a multiple of 3 values (start_value, end_value,
    // end_time). This is important because the decompression logic expects the compressed
    // data to be a sequence of 3-tuples, allowing safe and correct decoding. Adding one extra
    // value for the last point (if needed) has negligible impact on the compression ratio,
    // since it only occurs once per time series (if at all), regardless of the input size.
    if (seg_start == uncompressed_values.len - 1) {
        try shared_functions.appendValue(f64, uncompressed_values[seg_start], compressed_values);
        try shared_functions.appendValue(f64, uncompressed_values[seg_start], compressed_values);
        try shared_functions.appendValue(usize, seg_start, compressed_values);
    }
}

/// Decompress `compressed_values` produced by "Sliding Window". The function writes the result to
/// `decompressed_values`. If an error occurs it is returned.
pub fn decompress(compressed_values: []const u8, decompressed_values: *ArrayList(f64)) Error!void {
    // The compressed representation is composed of three values: (start_value, end_value, end_time)
    // all of type 64-bit float, except end_time which is usize.
    if (compressed_values.len % 24 != 0) return Error.UnsupportedInput;

    const compressed_lines_and_index = mem.bytesAsSlice(f64, compressed_values);

    var first_timestamp: usize = 0;
    var index: usize = 0;

    while (index < compressed_lines_and_index.len) : (index += 3) {
        const start_point = .{ .time = first_timestamp, .value = compressed_lines_and_index[index] };
        const end_point = .{
            .time = @as(usize, @bitCast(compressed_lines_and_index[index + 2])),
            .value = compressed_lines_and_index[index + 1],
        };

        // Check if the segment has more than two points.
        if (start_point.time + 1 < end_point.time) {
            const duration: f64 = @floatFromInt(end_point.time - start_point.time);

            const slope = (end_point.value - start_point.value) / duration;
            const intercept = start_point.value;

            try decompressed_values.append(start_point.value);
            var current_timestamp: usize = start_point.time + 1;

            // Interpolate the values between the start and end points of the current segment.
            while (current_timestamp < end_point.time) : (current_timestamp += 1) {
                const scaled_time = @as(f64, @floatFromInt(current_timestamp - start_point.time));
                const y: f64 = slope * scaled_time + intercept;
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
            if (start_point.time != end_point.time) {
                try decompressed_values.append(end_point.value);
                first_timestamp += 2;
            } else {
                // If the start and end points are the same, we are at the end of the time series.
                // Thus, do not insert the end point.
                first_timestamp += 1;
            }
        }
    }
}

/// Computes the Root-Mean-Squared-Errors (RMSE) for a segment of the `uncompressed_values`.
/// This function calculates the error between the actual values and the predicted values
/// based on a linear regression model fitted to the segment defined by `seg_start` and `seg_end`.
fn computeRMSE(uncompressed_values: []const f64, seg_start: usize, seg_end: usize) Error!f64 {
    const seg_len: f64 = @floatFromInt(seg_end - seg_start + 1);
    if (seg_len <= 1) return 0.0; // If the segment has one or no points, return zero error.

    // Check if the elements of the segment are within the valid range.
    if (uncompressed_values[seg_start] > tester.max_test_value or
        !math.isFinite(uncompressed_values[seg_start]) or
        uncompressed_values[seg_end] > tester.max_test_value or
        !math.isFinite(uncompressed_values[seg_end]))
    {
        return Error.UnsupportedInput;
    }
    // Calculate the slope and intercept of the line connecting the start and end points.
    const slope: f64 = (uncompressed_values[seg_end] - uncompressed_values[seg_start]) / (seg_len - 1);
    const intercept = uncompressed_values[seg_start];
    var sse: f64 = 0;
    var i: usize = seg_start;
    while (i <= seg_end) : (i += 1) {
        if (uncompressed_values[i] > tester.max_test_value or
            !math.isFinite(uncompressed_values[i])) return Error.UnsupportedInput;

        const scaled_time = @as(f64, @floatFromInt(i - seg_start)); // small numbers: 0,1,2,...
        const pred = intercept + slope * scaled_time;
        const diff = uncompressed_values[i] - pred;
        sse += diff * diff;
    }
    return math.sqrt(sse / seg_len);
}

test "sliding-window can compress and decompress bounded values with zero error bound" {
    const allocator = testing.allocator;
    const error_bound: f32 = 0.0;

    var uncompressed_values = ArrayList(f64).init(allocator);
    defer uncompressed_values.deinit();

    try tester.generateBoundedRandomValues(
        &uncompressed_values,
        -1e15,
        1e15,
        undefined,
    );

    const configuration_json = try std.fmt.allocPrint(
        allocator,
        "{{\"aggregate_error_type\": \"rmse\", \"aggregate_error_bound\": {d}}}",
        .{error_bound},
    );
    defer allocator.free(configuration_json);

    const compressed_values = try tersets.compress(
        allocator,
        uncompressed_values.items,
        tersets.Method.SlidingWindow,
        configuration_json,
    );
    defer compressed_values.deinit();

    const decompressed_values = try tersets.decompress(allocator, compressed_values.items);
    defer decompressed_values.deinit();

    try testing.expect(shared_functions.isWithinErrorBound(
        uncompressed_values.items,
        decompressed_values.items,
        error_bound,
    ));
}

test "sliding-window cannot compress and decompress nan values" {
    const allocator = testing.allocator;
    const uncompressed_values = [3]f64{ 343.0, math.nan(f64), 520.0 };
    var compressed_values = std.ArrayList(u8).init(allocator);
    compressed_values.deinit();

    const method_configuration =
        \\ { "aggregate_error_type": "rmse", "aggregate_error_bound": 0.1 }
    ;

    compress(
        allocator,
        uncompressed_values[0..],
        &compressed_values,
        method_configuration,
    ) catch |err| {
        try testing.expectEqual(Error.UnsupportedInput, err);
        return;
    };

    try testing.expectFmt(
        "",
        "The Sliding-Window method cannot compress nan values",
        .{},
    );
}

test "sliding-window cannot compress and decompress unbounded values" {
    const allocator = testing.allocator;
    const uncompressed_values = [3]f64{ 343.0, 1e20, 520.0 };
    var compressed_values = std.ArrayList(u8).init(allocator);
    compressed_values.deinit();

    const method_configuration =
        \\ { "aggregate_error_type": "rmse", "aggregate_error_bound": 0.1 }
    ;

    compress(
        allocator,
        uncompressed_values[0..],
        &compressed_values,
        method_configuration,
    ) catch |err| {
        try testing.expectEqual(Error.UnsupportedInput, err);
        return;
    };

    try testing.expectFmt(
        "",
        "The Sliding-Window method cannot compress unbounded values",
        .{},
    );
}

test "sliding-window compress and decompress random lines and random error bound" {
    const allocator = testing.allocator;
    const random = tester.getDefaultRandomGenerator();

    var uncompressed_values = ArrayList(f64).init(allocator);
    defer uncompressed_values.deinit();
    var compressed_values = ArrayList(u8).init(allocator);
    defer compressed_values.deinit();
    var decompressed_values = ArrayList(f64).init(allocator);
    defer decompressed_values.deinit();

    const error_bound: f32 = tester.generateBoundedRandomValue(f32, 0.01, 1e6, undefined);

    try tester.generateRandomLinearFunctions(&uncompressed_values, random);

    const method_configuration = try std.fmt.allocPrint(
        allocator,
        "{{\"aggregate_error_type\": \"rmse\", \"aggregate_error_bound\": {d}}}",
        .{error_bound},
    );
    defer allocator.free(method_configuration);

    try compress(
        allocator,
        uncompressed_values.items,
        &compressed_values,
        method_configuration,
    );

    try decompress(compressed_values.items, &decompressed_values);

    // Check if the decompressed values have the same lenght as the compressed ones.
    try testing.expectEqual(uncompressed_values.items.len, decompressed_values.items.len);

    // In theory, the linear interpolation of all segments formed by the slices of preserved points,
    // should have a RMSE within the error bound otherwise there a mistake. Since the error bound
    // and the poitns are unknown, we need to used the compressed representation to access each of
    // the points preserved and their index `current_point_index`. Then, the RMSE of the linear
    // regression of the segment formed by the slices from `previous_point_index` to
    // `current_point_index` should be less than `error_bound`.
    const compressed_representation = mem.bytesAsSlice(f64, compressed_values.items);

    var index: usize = 0;
    var previous_point_index: usize = 0;
    while (index < compressed_representation.len - 1) : (index += 3) {
        const current_point_index = @min(
            @as(usize, @bitCast(compressed_representation[index + 2])),
            uncompressed_values.items.len - 1,
        );

        // Check if the point is within the error bound.
        try shared_functions.testRMSEisWithinErrorBound(
            uncompressed_values.items[previous_point_index .. current_point_index + 1],
            error_bound,
        );
        previous_point_index = current_point_index + 1;
    }
}

test "check sliding window configuration parsing" {
    // Tests the configuration parsing and functionality of the `compress` function.
    // The test verifies that the provided configuration is correctly interpreted and
    // that the `configuration.AggregateError` is expected in the function.
    const allocator = testing.allocator;

    const uncompressed_values = &[4]f64{ 19.0, 48.0, 28.0, 3.0 };

    var compressed_values = ArrayList(u8).init(allocator);
    defer compressed_values.deinit();

    const method_configuration =
        \\ {"aggregate_error_type": "rmse", "aggregate_error_bound": 0.3}
    ;

    // The configuration is properly defined. No error expected.
    try compress(
        allocator,
        uncompressed_values,
        &compressed_values,
        method_configuration,
    );
}
