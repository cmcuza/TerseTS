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

//! Implementation of the Largest Triangle Three Buckets (LTTB) downsampling algorithm
//! from the thesis "Steinarsson, S. Downsampling Time Series for Visual Representation.
//! M.Sc. thesis, University of Iceland, 2013.
//! https://hdl.handle.net/1946/15343".

const std = @import("std");
const mem = std.mem;
const math = std.math;
const testing = std.testing;
const ArrayList = std.ArrayList;
const Allocator = mem.Allocator;

const extractors = @import("../../utilities/extractors.zig");
const rebuilders = @import("../../utilities/rebuilders.zig");

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

    // Return an error  for a threshold that is too low, because at least
    // 2 points are always necessary, as they anchor the line.
    if (threshold < 2) return Error.UnsupportedInput;

    // Series is already at or below the target size: keep every point
    // and just output (value, index) pairs normally.
    if (uncompressed_values.len <= threshold) {
        try shared_functions.appendValue(allocator, f64, uncompressed_values[0], compressed_values);
        for (1..uncompressed_values.len) |i| {
            try shared_functions.appendValue(allocator, f64, uncompressed_values[i], compressed_values);
            try shared_functions.appendValue(allocator, usize, i, compressed_values);
        }
        return;
    }

    // The first and last points are always kept, so the remaining threshold - 2
    // output points are chosen from the len - 2 interior points (indices 1..len-1).
    const inner_threshold: usize = threshold - 2;
    const inner_point_count = uncompressed_values.len - 2;

    var selected_point = DiscretePoint{ .index = 0, .value = uncompressed_values[0] };
    try shared_functions.appendValue(allocator, f64, uncompressed_values[0], compressed_values);

    for (0..inner_threshold) |bucket_idx| {
        // Split the interior points proportionally so the buckets always cover exactly
        // [1, len-1) with no overshoot, regardless of how evenly the points divide.
        const start = 1 + bucket_idx * inner_point_count / inner_threshold;
        const end = 1 + (bucket_idx + 1) * inner_point_count / inner_threshold;

        // Average of the next bucket. For the final bucket this range collapses onto the
        // last point, which is exactly the anchor we want to aim the triangle at.
        const next_start = end;
        const next_end = @min(
            1 + (bucket_idx + 2) * inner_point_count / inner_threshold,
            uncompressed_values.len,
        );
        var avg: f64 = 0;
        for (next_start..next_end) |point_idx| {
            avg += uncompressed_values[point_idx];
        }
        avg /= @as(f64, @floatFromInt(next_end - next_start));
        const avg_point = DiscretePoint{ .index = (next_start + next_end) / 2, .value = avg };

        // Pick the interior point in the current bucket that forms the largest triangle
        // with the previously selected point and the next bucket's average.
        var best_point = DiscretePoint{ .index = start, .value = uncompressed_values[start] };
        var max_area = calculateArea(selected_point, best_point, avg_point);
        for (start + 1..end) |point_idx| {
            const candidate = DiscretePoint{ .index = point_idx, .value = uncompressed_values[point_idx] };
            const curr_area = calculateArea(selected_point, candidate, avg_point);
            if (curr_area > max_area) {
                max_area = curr_area;
                best_point = candidate;
            }
        }

        // Save the winning point and use it as the anchor for the next bucket.
        try shared_functions.appendValue(allocator, f64, best_point.value, compressed_values);
        try shared_functions.appendValue(allocator, usize, best_point.index, compressed_values);
        selected_point = best_point;
    }

    // Anchor the line to the original end point.
    try shared_functions.appendValue(allocator, f64, uncompressed_values[uncompressed_values.len - 1], compressed_values);
    try shared_functions.appendValue(allocator, usize, uncompressed_values.len - 1, compressed_values);
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

/// Extracts `indices` and `coefficients` from "Largest Triangle Three Buckets" `compressed_values`.
/// The binary representation follows the same pattern as "Visvalingam-Whyatt" and "SwingFilter",
/// so this function calls `extractCoefficientIndexTuplesWithStartCoefficient`.
/// All structural and corruption checks are performed by the delegated function.
/// Any loss of index information can lead to failures during later decompression.
/// The `allocator` handles the memory of the output arrays. Allocation errors are propagated.
pub fn extract(
    allocator: Allocator,
    compressed_values: []const u8,
    indices: *ArrayList(u64),
    coefficients: *ArrayList(f64),
) Error!void {
    // Delegate to CoefficientIndexTuplesWithStartCoefficient extractor.
    // "Largest Triangle Three Buckets" uses the same representation as SwingFilter.
    try extractors.extractCoefficientIndexTuplesWithStartCoefficient(
        allocator,
        compressed_values,
        indices,
        coefficients,
    );
}

/// Rebuilds "Largest Triangle Three Buckets" `compressed_values` from the provided `indices` and `coefficients`.
/// The representation matches "Visvalingam-Whyatt" and "SwingFilter",
/// so the function delegates to `rebuildCoefficientIndexTuplesWithStartCoefficient`.
/// All format validation and corruption checks are performed by that routine.
/// Any loss or misalignment of indices may cause failures when decoding the rebuilt representation.
/// The `allocator` handles the memory of the output arrays. Allocation errors are propagated.
pub fn rebuild(
    allocator: Allocator,
    indices: []const u64,
    coefficients: []const f64,
    compressed_values: *ArrayList(u8),
) Error!void {
    // Delegate to CoefficientIndexTuplesWithStartCoefficient rebuilder.
    // "Largest Triangle Three Buckets" uses the same representation as SwingFilter.
    try rebuilders.rebuildCoefficientIndexTuplesWithStartCoefficient(
        allocator,
        indices,
        coefficients,
        compressed_values,
    );
}

test "lttb keeps all values when threshold is at least the input length" {
    const allocator = testing.allocator;

    // With a threshold greater than or equal to the number of points, every point is kept and the
    // round-trip is lossless.
    const uncompressed_values: []const f64 = &[_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };

    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);
    var decompressed_values = ArrayList(f64).empty;
    defer decompressed_values.deinit(allocator);

    const method_configuration =
        \\ {"output_threshold_number": 16}
    ;

    try compress(allocator, uncompressed_values, &compressed_values, method_configuration);
    try decompress(allocator, compressed_values.items, &decompressed_values);

    try testing.expectEqual(uncompressed_values.len, decompressed_values.items.len);
    for (uncompressed_values, decompressed_values.items) |expected, actual| {
        try testing.expectEqual(expected, actual);
    }
}

test "lttb downsamples with known result" {
    const allocator = testing.allocator;

    // Nine points are reduced to a threshold of four.
    const uncompressed_values: []const f64 = &[_]f64{ 1.0, 2.0, 1.5, 3.0, 4.0, 3.5, 5.0, 6.0, 7.0 };
    const threshold: usize = 4;

    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);

    const method_configuration =
        \\ {"output_threshold_number": 4}
    ;

    try compress(allocator, uncompressed_values, &compressed_values, method_configuration);

    // The compressed representation stores the first value (8 bytes) followed by a (value, index)
    // pair (16 bytes) for every other kept point, so exactly `threshold` points must be kept.
    const kept_points = (compressed_values.items.len - 8) / 16 + 1;
    try testing.expectEqual(threshold, kept_points);

    // Reinterpret the byte stream as f64s: slot 0 is the first kept value, then the kept
    // (value, index) pairs follow, so the kept values live at slots 0, 1, 3, 5.
    const kept = mem.bytesAsSlice(f64, compressed_values.items);
    try testing.expectEqual(uncompressed_values[0], kept[0]);
    try testing.expectEqual(uncompressed_values[2], kept[1]);
    try testing.expectEqual(uncompressed_values[5], kept[3]);
    try testing.expectEqual(uncompressed_values[8], kept[5]);
}

test "lttb returns error for threshold below 2" {
    const allocator = testing.allocator;

    const uncompressed_values: []const f64 = &[_]f64{ 1.0, 2.0, 3.0 };

    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);

    const method_configuration =
        \\ {"output_threshold_number": 1}
    ;

    try testing.expectError(
        Error.UnsupportedInput,
        compress(allocator, uncompressed_values, &compressed_values, method_configuration),
    );
}

test "lttb with threshold 2 keeps only first and last point" {
    const allocator = testing.allocator;

    // inner_threshold = 0, the loop never executes, only endpoints are kept.
    const uncompressed_values: []const f64 = &[_]f64{ 1.0, 3.0, 5.0, 7.0, 9.0 };

    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);

    const method_configuration =
        \\ {"output_threshold_number": 2}
    ;

    try compress(allocator, uncompressed_values, &compressed_values, method_configuration);

    // Calculate the amount of points stored, verify that there are 2 (first and last point).
    // 8 bytes for start value, 8 for end value, 8 for end index.
    const kept_points = (compressed_values.items.len - 8) / 16 + 1;
    try testing.expectEqual(@as(usize, 2), kept_points);

    // Should keep the start and end value of uncompressed_values.
    const kept = mem.bytesAsSlice(f64, compressed_values.items);
    try testing.expectEqual(uncompressed_values[0], kept[0]);
    try testing.expectEqual(uncompressed_values[uncompressed_values.len - 1], kept[1]);
}

test "lttb with threshold equal to len-1 preserves round-trip" {
    const allocator = testing.allocator;

    // 5 points, threshold 4: inner_threshold = 2, inner_point_count = 3.
    // Two buckets cover three interior points (indices 1,2,3).
    const uncompressed_values: []const f64 = &[_]f64{ 1.0, 2.0, 1.5, 3.0, 5.0 };

    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);
    var decompressed_values = ArrayList(f64).empty;
    defer decompressed_values.deinit(allocator);

    const method_configuration =
        \\ {"output_threshold_number": 4}
    ;

    try compress(allocator, uncompressed_values, &compressed_values, method_configuration);

    const kept_points = (compressed_values.items.len - 8) / 16 + 1;
    try testing.expectEqual(@as(usize, 4), kept_points);

    // Endpoints are always first and last.
    const kept = mem.bytesAsSlice(f64, compressed_values.items);
    try testing.expectEqual(uncompressed_values[0], kept[0]);
    try testing.expectEqual(uncompressed_values[4], kept[5]);

    try decompress(allocator, compressed_values.items, &decompressed_values);
    try testing.expectEqual(uncompressed_values.len, decompressed_values.items.len);
}

test "lttb handles uneven bucket distribution" {
    const allocator = testing.allocator;

    // 12 points, threshold 5: inner_threshold = 3, inner_point_count = 10.
    // Buckets: floor(0*10/3)=0, floor(1*10/3)=3, floor(2*10/3)=6, floor(3*10/3)=10.
    // Bucket sizes: 3, 3, 4 — uneven.
    const uncompressed_values: []const f64 = &[_]f64{
        0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 5.0,
    };

    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);
    var decompressed_values = ArrayList(f64).empty;
    defer decompressed_values.deinit(allocator);

    const method_configuration =
        \\ {"output_threshold_number": 5}
    ;

    try compress(allocator, uncompressed_values, &compressed_values, method_configuration);

    const kept_points = (compressed_values.items.len - 8) / 16 + 1;
    try testing.expectEqual(@as(usize, 5), kept_points);

    try decompress(allocator, compressed_values.items, &decompressed_values);
    try testing.expectEqual(uncompressed_values.len, decompressed_values.items.len);
}

test "lttb handles collinear points" {
    const allocator = testing.allocator;

    // All points lie strictly on y = 2x + 1. Every triangle has area zero.
    // The algorithm should still select a point from each bucket.
    const uncompressed_values: []const f64 = &[_]f64{
        1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0,
    };

    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);
    var decompressed_values = ArrayList(f64).empty;
    defer decompressed_values.deinit(allocator);

    const method_configuration =
        \\ {"output_threshold_number": 4}
    ;

    try compress(allocator, uncompressed_values, &compressed_values, method_configuration);

    const kept_points = (compressed_values.items.len - 8) / 16 + 1;
    try testing.expectEqual(@as(usize, 4), kept_points);

    try decompress(allocator, compressed_values.items, &decompressed_values);
    try testing.expectEqual(uncompressed_values.len, decompressed_values.items.len);
}

test "lttb round-trip with threshold 3 preserves length" {
    const allocator = testing.allocator;

    // Minimal downsampling: inner_threshold = 1, one interior point selected.
    const uncompressed_values: []const f64 = &[_]f64{ 0.0, 10.0, 5.0, 20.0, 15.0 };

    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);
    var decompressed_values = ArrayList(f64).empty;
    defer decompressed_values.deinit(allocator);

    const method_configuration =
        \\ {"output_threshold_number": 3}
    ;

    try compress(allocator, uncompressed_values, &compressed_values, method_configuration);

    const kept_points = (compressed_values.items.len - 8) / 16 + 1;
    try testing.expectEqual(@as(usize, 3), kept_points);

    try decompress(allocator, compressed_values.items, &decompressed_values);
    try testing.expectEqual(uncompressed_values.len, decompressed_values.items.len);
}

test "lttb compress and decompress preserve length on larger data" {
    const allocator = testing.allocator;

    var prng = std.Random.DefaultPrng.init(0);
    const random = prng.random();

    var uncompressed_values = ArrayList(f64).empty;
    defer uncompressed_values.deinit(allocator);
    for (0..100) |_| {
        try uncompressed_values.append(allocator, random.float(f64) * 100.0);
    }

    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);
    var decompressed_values = ArrayList(f64).empty;
    defer decompressed_values.deinit(allocator);

    const method_configuration =
        \\ {"output_threshold_number": 20}
    ;

    try compress(allocator, uncompressed_values.items, &compressed_values, method_configuration);
    try decompress(allocator, compressed_values.items, &decompressed_values);

    try testing.expectEqual(uncompressed_values.items.len, decompressed_values.items.len);
}

test "lttb returns an error for an invalid configuration" {
    const allocator = testing.allocator;

    // LTTB expects an OutputThresholdNumber configuration; any other shape is invalid.
    const uncompressed_values: []const f64 = &[_]f64{ 1.0, 2.0, 3.0, 4.0 };

    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);

    const method_configuration =
        \\ {"abs_error_bound": 0.1}
    ;

    try testing.expectError(
        Error.InvalidConfiguration,
        compress(allocator, uncompressed_values, &compressed_values, method_configuration),
    );
}

test "check lttb configuration parsing" {
    // Verifies that a well-formed OutputThresholdNumber configuration is accepted by compress.
    const allocator = testing.allocator;

    const uncompressed_values: []const f64 = &[_]f64{ 19.0, 48.0, 28.0, 3.0 };

    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);

    const method_configuration =
        \\ {"output_threshold_number": 4}
    ;

    try compress(allocator, uncompressed_values, &compressed_values, method_configuration);
}
