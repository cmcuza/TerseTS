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

//! Implementation of "Bottom-Up" algorithm as described in the paper:
//! "E. Keogh, S. Chu, D. Hart and M. Pazzani.
//! An online algorithm for segmenting time series.
//! IEEE ICDM, pp. 289-296, 2001.
//! https://doi.org/10.1109/ICDM.2001.989531"
//! This specific implementation uses the "Root-Mean-Squared-Errors (RMSE)"
//! as the cost function. Future work may include other cost functions.

const std = @import("std");
const mem = std.mem;
const math = std.math;
const ArrayList = std.ArrayList;
const Allocator = mem.Allocator;

const tersets = @import("../../tersets.zig");
const configuration = @import("../../configuration.zig");
const Error = tersets.Error;

const IndexedPriorityQueue = @import("../../utilities/indexed_priority_queue.zig").IndexedPriorityQueue;

const shared_functions = @import("../../utilities/shared_functions.zig");

const tester = @import("../../tester.zig");
const extractors = @import("../../utilities/extractors.zig");
const rebuilders = @import("../../utilities/rebuilders.zig");

const testing = std.testing;

/// Compresses `uncompressed_values` using the "Bottom-Up" simplification algorithm.
/// This algorithm iteratively merges points to minimize the sum of squared errors,
/// ensuring that the resulting compressed sequence stays within the specified `error_bound`.
/// The function writes the simplified sequence to the `compressed_values`. The `allocator`
/// is used to allocate memory for the IndexedPriorityQueue used in the implementation and
/// the `method_configuration` parser. The `method_configuration` is expected to be of
/// `AggregateError` type otherwise an `InvalidConfiguration` error is return. If any other
/// error occurs during the execution of the method, it is returned.
pub fn compress(
    allocator: Allocator,
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

    // If we have 2 or fewer points, we store them without compression.
    if (uncompressed_values.len <= 2) {
        try shared_functions.appendValue(allocator, f64, uncompressed_values[0], compressed_values);
        try shared_functions.appendValue(allocator, f64, uncompressed_values[1], compressed_values);
        try shared_functions.appendValue(allocator, usize, 1, compressed_values);
        return;
    }

    const segment_count = (uncompressed_values.len + 1) / 2;

    // Initialize an indexed priority queue to store the cost of merging each segment with the
    // segment to its right. Segment IDs are dense and stable for the whole compression run.
    var heap = try IndexedPriorityQueue(f64, compareMergeCost).init(allocator, segment_count);
    defer heap.deinit();

    const costs = try allocator.alloc(f64, segment_count);
    defer allocator.free(costs);
    const left_segments = try allocator.alloc(usize, segment_count);
    defer allocator.free(left_segments);
    const right_segments = try allocator.alloc(usize, segment_count);
    defer allocator.free(right_segments);
    const segment_starts = try allocator.alloc(usize, segment_count);
    defer allocator.free(segment_starts);
    const segment_ends = try allocator.alloc(usize, segment_count);
    defer allocator.free(segment_ends);

    // Compute all pairwise merging costs of the segments.
    try buildInitialPairwiseSegmentCost(
        &heap,
        costs,
        left_segments,
        right_segments,
        segment_starts,
        segment_ends,
        uncompressed_values,
    );

    while (heap.count() > 2) {
        // Peek into the segment with the lowest cost.
        const min_segment = try heap.peek();
        const min_segment_index = min_segment.index;

        // Check if the cost is within the error bound.
        if (min_segment.priority > error_bound) {
            break;
        }

        // Now is safe to pop the segment from the heap.
        _ = try heap.pop();

        const right_segment_index = right_segments[min_segment_index];

        // Update the start of the right segment to include the start of the merged segment.
        segment_starts[right_segment_index] = segment_starts[min_segment_index];

        // If the current minimum segment is not the first segment, update the left segment.
        if (segment_starts[min_segment_index] != 0) {
            const left_segment_index = left_segments[min_segment_index];

            // Update the neighboring segment links to bypass the removed segment.
            left_segments[right_segment_index] = left_segment_index;
            right_segments[left_segment_index] = right_segment_index;

            // Compute the merge cost of merging the left and right segments.
            costs[left_segment_index] = try mergeCostFromState(
                uncompressed_values,
                segment_starts,
                segment_ends,
                left_segment_index,
                right_segment_index,
            );

            // Update the left segment in the heap.
            try heap.update(left_segment_index, costs[left_segment_index]);
        }

        // If the right segment is not the last segment, update its merge cost with its right neighbor.
        if (segment_ends[right_segment_index] != uncompressed_values.len - 1) {
            // Compute the merge cost of merging the right segment with its right neighbor.
            costs[right_segment_index] = try mergeCostFromState(
                uncompressed_values,
                segment_starts,
                segment_ends,
                right_segment_index,
                right_segments[right_segment_index],
            );
        }

        // Update the right segment in the heap.
        try heap.update(right_segment_index, costs[right_segment_index]);
    }

    // Sort remaining points by original index to preserve order.
    var remaining_segments = ArrayList(SegmentMergeCost).empty;
    defer remaining_segments.deinit(allocator);
    try remaining_segments.ensureTotalCapacity(allocator, heap.count());

    for (0..segment_count) |segment_index| {
        if (!heap.contains(segment_index)) continue;
        try remaining_segments.append(allocator, .{
            .index = segment_index,
            .cost = costs[segment_index],
            .left_seg = left_segments[segment_index],
            .right_seg = right_segments[segment_index],
            .seg_start = segment_starts[segment_index],
            .seg_end = segment_ends[segment_index],
        });
    }
    mem.sort(SegmentMergeCost, remaining_segments.items, {}, SegmentMergeCost.firstThan);

    // Output compressed series: (seg_start, index, end_value) pairs.
    for (remaining_segments.items) |segment| {
        const seg_start = segment.seg_start;
        const seg_end = segment.seg_end;
        // Append the start value, the end index, and the end value to the compressed representation.
        try shared_functions.appendValue(allocator, f64, uncompressed_values[seg_start], compressed_values);
        try shared_functions.appendValue(allocator, f64, uncompressed_values[seg_end], compressed_values);
        try shared_functions.appendValue(allocator, usize, seg_end, compressed_values);
    }

    return;
}

/// Decompress `compressed_values` produced by "Bottom-Up". The function writes the result to
/// `decompressed_values`. If an error occurs it is returned.
pub fn decompress(
    allocator: Allocator,
    compressed_values: []const u8,
    decompressed_values: *ArrayList(f64),
) Error!void {
    // The compressed representation is composed of three values: (start_value, end_value, end_time)
    // all of type 64-bit float, except end_time which is usize.
    if (compressed_values.len % 24 != 0) return Error.UnsupportedInput;

    const compressed_lines_and_index = mem.bytesAsSlice(f64, compressed_values);

    var first_index: usize = 0;
    var index: usize = 0;

    while (index < compressed_lines_and_index.len) : (index += 3) {
        const start_point = .{ .index = first_index, .value = compressed_lines_and_index[index] };
        const end_point = .{
            .index = @as(usize, @bitCast(compressed_lines_and_index[index + 2])),
            .value = compressed_lines_and_index[index + 1],
        };

        // Check if the segment has only two points. If so, we can directly append their values.
        if (start_point.index + 1 < end_point.index) {
            const duration: f64 = @floatFromInt(end_point.index - start_point.index);

            const slope = (end_point.value - start_point.value) / duration;
            const intercept = start_point.value - slope *
                @as(f64, @floatFromInt(start_point.index));

            try decompressed_values.append(allocator, start_point.value);
            var current_index: usize = start_point.index + 1;

            // Interpolate the values between the start and end points of the current segment.
            while (current_index < end_point.index) : (current_index += 1) {
                const y: f64 = slope * @as(f64, @floatFromInt(current_index)) + intercept;
                try decompressed_values.append(allocator, y);
            }
            try decompressed_values.append(allocator, end_point.value);
            first_index = current_index + 1;
        } else {
            // If the start and end points are one point apart,
            // append the start point and end points directly.
            try decompressed_values.append(allocator, start_point.value);

            // Append the end point only if it is different from the start point.
            // This is to avoid duplicates in the decompressed values.
            if (start_point.index != end_point.index) {
                try decompressed_values.append(allocator, end_point.value);
            }

            first_index += 2;
        }
    }
}

/// Extracts `indices` and `coefficients` from BottomUp's `compressed_values`. BottomUp uses the
/// same three-value repeating representation as SlideFilter, so this function forwards to
/// `extractSlide`. All corruption checks and structural validation occur in that routine. Any loss
/// of information on indices can lead to failures when decoding. The `allocator` handles the memory
/// of the output arrays. Allocation errors are propagated.
pub fn extract(
    allocator: Allocator,
    compressed_values: []const u8,
    indices: *ArrayList(u64),
    coefficients: *ArrayList(f64),
) Error!void {
    // Delegate to DoubleCoefficientIndexTriples extractor.
    // BottomUp uses the same representation as SlideFilter.
    try extractors.extractDoubleCoefficientIndexTriples(
        allocator,
        compressed_values,
        indices,
        coefficients,
    );
}

/// Rebuilds BottomUp's `compressed_values` from the provided `indices` and `coefficients`.
/// Because the format matches SlideFilter exactly, this wrapper forwards to `rebuildSlide`.
/// All format and corruption checks are performed internally. Incorrect or inconsistent
/// indices may produce corrupted output that cannot be decompressed. The `allocator`
/// handles the memory allocations of the output arrays. Allocation errors are propagated.
pub fn rebuild(
    allocator: Allocator,
    indices: []const u64,
    coefficients: []const f64,
    compressed_values: *ArrayList(u8),
) Error!void {
    // Delegate to DoubleCoefficientIndexTriples extractor.
    // BottomUp uses the same representation as SlideFilter.
    try rebuilders.rebuildDoubleCoefficientIndexTriples(
        allocator,
        indices,
        coefficients,
        compressed_values,
    );
}

/// `SegmentMergeCost` represents a segment and its associated `cost` with the next segment,
/// which is calculated based on RMSE. The `cost` is used to determine the order in which segments
/// are merged during the simplification process. The `merge_index` is the index of the segment, the
/// `left_seg` and `right_seg`represent the IDs of the segments being merged. The `seg_start` and
/// `seg_end` represent the start and end indices of the segment in the original data.
const SegmentMergeCost = struct {
    // The segment ID is used to identify the segment in the heap.
    index: usize,
    // The cost of merging this segment with the next one.
    cost: f64,
    // The ID of the left segment in the merge.
    left_seg: usize,
    // The ID of the right segment in the merge.
    right_seg: usize,
    // The start index of the segment in the original data.
    seg_start: usize,
    // The end index of the segment in the original data.
    seg_end: usize,

    /// Order by the `index` field (ascending). This is used to sort segments by their original
    /// position in the series to preserve the order after simplification.
    fn firstThan(_: void, seg_1: SegmentMergeCost, seg_2: SegmentMergeCost) bool {
        return seg_1.index < seg_2.index;
    }
};

/// Comparison function for the `IndexedPriorityQueue`. It compares merge costs between two segments.
fn compareMergeCost(cost_1: f64, cost_2: f64) math.Order {
    return math.order(cost_1, cost_2);
}

/// Build initial pair-wise segments costs (two points per segment if possible).
fn buildInitialPairwiseSegmentCost(
    heap: *IndexedPriorityQueue(f64, compareMergeCost),
    costs: []f64,
    left_segments: []usize,
    right_segments: []usize,
    segment_starts: []usize,
    segment_ends: []usize,
    uncompressed_values: []const f64,
) Error!void {
    // We need to create a segment for every two points in the uncompressed values.
    // The first segment is always the first two points.
    // The second segment is the next two points, and so on.
    for (0..costs.len) |segment_index| {
        const seg_start = segment_index * 2;
        const seg_end: usize = if (seg_start + 1 < uncompressed_values.len) seg_start + 1 else seg_start;

        left_segments[segment_index] = if (segment_index == 0) 0 else segment_index - 1;
        right_segments[segment_index] = segment_index + 1;
        segment_starts[segment_index] = seg_start;
        segment_ends[segment_index] = seg_end;
        costs[segment_index] = math.inf(f64);
    }

    for (0..costs.len) |segment_index| {
        if (segment_index + 1 < costs.len) {
            costs[segment_index] = try mergeCostFromState(
                uncompressed_values,
                segment_starts,
                segment_ends,
                segment_index,
                segment_index + 1,
            );
        }
        try heap.add(segment_index, costs[segment_index]);
    }
}

/// Incremental cost of merging the neighbouring segments identified by dense segment IDs.
fn mergeCostFromState(
    uncompressed_values: []const f64,
    segment_starts: []const usize,
    segment_ends: []const usize,
    segment_one: usize,
    segment_two: usize,
) !f64 {
    const merged_start = @min(segment_starts[segment_one], segment_starts[segment_two]);
    const merged_end = @max(segment_ends[segment_one], segment_ends[segment_two]);
    return try shared_functions.computeRMSE(uncompressed_values, merged_start, merged_end);
}

test "bottom-up can compress and decompress with zero error bound" {
    const allocator = testing.allocator;
    const error_bound: f32 = 0.0;

    var uncompressed_values = ArrayList(f64).empty;
    defer uncompressed_values.deinit(allocator);

    try tester.generateBoundedRandomValues(
        allocator,
        &uncompressed_values,
        -1e15,
        1e15,
        null,
    );

    const method_configuration = try std.fmt.allocPrint(
        allocator,
        "{{\"aggregate_error_type\": \"rmse\", \"aggregate_error_bound\": {d}}}",
        .{error_bound},
    );
    defer allocator.free(method_configuration);

    var compressed_values = try tersets.compress(
        allocator,
        uncompressed_values.items,
        tersets.Method.BottomUp,
        method_configuration,
    );
    defer compressed_values.deinit(allocator);

    var decompressed_values = try tersets.decompress(allocator, compressed_values.items);
    defer decompressed_values.deinit(allocator);

    try testing.expect(shared_functions.isWithinErrorBound(
        uncompressed_values.items,
        decompressed_values.items,
        error_bound,
    ));
}

test "bottom-up cannot compress and decompress nan values" {
    const allocator = testing.allocator;
    const uncompressed_values = [3]f64{ 343.0, math.nan(f64), 520.0 };
    var compressed_values = ArrayList(u8).empty;
    compressed_values.deinit(allocator);

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
        "The Bottom-Up method cannot compress nan values",
        .{},
    );
}

test "bottom-up cannot compress and decompress unbounded values" {
    const allocator = testing.allocator;
    const uncompressed_values = [3]f64{ 343.0, 1e20, 520.0 };
    var compressed_values = ArrayList(u8).empty;
    compressed_values.deinit(allocator);

    const method_configuration =
        \\ { "aggregate_error_type": "rmse", "aggregate_error_bound": 0.1 }
    ;

    compress(allocator, uncompressed_values[0..], &compressed_values, method_configuration) catch |err| {
        try testing.expectEqual(Error.UnsupportedInput, err);
        return;
    };

    try testing.expectFmt(
        "",
        "The Bottom-Up method cannot compress unbounded values",
        .{},
    );
}

test "bottom-up random lines and random error bound compress and decompress" {
    const allocator = testing.allocator;
    const random = tester.getDefaultRandomGenerator();

    var uncompressed_values = ArrayList(f64).empty;
    defer uncompressed_values.deinit(allocator);
    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);
    var decompressed_values = ArrayList(f64).empty;
    defer decompressed_values.deinit(allocator);

    const error_bound: f32 = tester.generateBoundedRandomValue(f32, 0.01, 1e6, null);

    try tester.generateRandomLinearFunctions(allocator, &uncompressed_values, random);

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

    try decompress(allocator, compressed_values.items, &decompressed_values);

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

test "check bottom-up configuration parsing" {
    // Tests the configuration parsing and functionality of the `compress` function.
    // The test verifies that the provided configuration is correctly interpreted and
    // that the `configuration.AggregateError` is expected in the function.
    const allocator = testing.allocator;

    const uncompressed_values = &[4]f64{ 19.0, 48.0, 28.0, 3.0 };

    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);

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
