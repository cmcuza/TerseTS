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
const time = std.time;
const ArrayList = std.ArrayList;
const Allocator = mem.Allocator;

const tersets = @import("../../tersets.zig");
const configuration = @import("../../configuration.zig");
const Method = tersets.Method;
const Error = tersets.Error;

const HashedPriorityQueue = @import(
    "../../utilities/hashed_priority_queue.zig",
).HashedPriorityQueue;

const shared_structs = @import("../../utilities/shared_structs.zig");
const shared_functions = @import("../../utilities/shared_functions.zig");

const DiscretePoint = shared_structs.DiscretePoint;
const LinearFunction = shared_structs.LinearFunction;

const tester = @import("../../tester.zig");
const extractors = @import("../../utilities/extractors.zig");
const rebuilders = @import("../../utilities/rebuilders.zig");

const testing = std.testing;

/// Compresses `uncompressed_values` using the "Bottom-Up" simplification algorithm.
/// This algorithm iteratively merges points to minimize the sum of squared errors,
/// ensuring that the resulting compressed sequence stays within the specified `error_bound`.
/// The function writes the simplified sequence to the `compressed_values`. The `allocator`
/// is used to allocate memory for the HashedPriorityQueue used in the implementation and
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

    // Initialize a hashed priority queue to store the effective area of triangles formed by every
    // sequence of three consecutive points. The priority is determined by the area.
    var heap = try HashedPriorityQueue(
        SegmentMergeCost,
        void,
        compareSegmentMergeCost,
        SegmentMergeCostHashContext,
    ).init(allocator, {});
    defer heap.deinit();

    // Compute all pairwise merging cost of the segments.
    try buildInitialPairwiseSegmentCost(&heap, uncompressed_values);

    // Placeholder for the segment cost to be used in the loop to search for neighboring segments.
    // During the merging process, we keep track of neighboring segments: the "left segment" is
    // the segment immediately before the current one, and the "right segment" is the segment
    // immediately after. These relationships are maintained using the `left_seg` and `right_seg`
    // fields in the `SegmentMergeCost` struct.
    var placeholder_segment_cost: SegmentMergeCost = .{
        .index = undefined,
        .cost = undefined,
        .left_seg = undefined,
        .right_seg = undefined,
        .seg_start = undefined,
        .seg_end = undefined,
    };

    while (heap.len > 2) {
        // Peek into the segment with the lowest cost.
        const min_segment: SegmentMergeCost = try heap.peek();

        // Check if the cost is within the error bound.
        if (min_segment.cost > error_bound) {
            break;
        }

        // Now is safe to pop the point from the heap.
        _ = try heap.pop();

        // Update the index of the placeholder segment to the right segment of the current minimum segment.
        placeholder_segment_cost.index = min_segment.right_seg;

        // Retrieve the right segment from the heap using its index.
        var right_seg: SegmentMergeCost = try heap.get(try heap.getIndex(placeholder_segment_cost));

        // Update the start of the right segment to include the start of the merged segment.
        right_seg.seg_start = min_segment.seg_start;

        // If the current minimum segment is not the first segment, update the left segment.
        if (min_segment.seg_start != 0) {
            // Update the index of the placeholder segment to the left segment of the current minimum segment.
            placeholder_segment_cost.index = min_segment.left_seg;

            // Retrieve the left segment from the heap using its index.
            var left_seg: SegmentMergeCost = try heap.get(try heap.getIndex(placeholder_segment_cost));

            // Update the right segment's left neighbor to the left segment's index.
            right_seg.left_seg = left_seg.index;

            // Update the left segment's right neighbor to the right segment's index.
            left_seg.right_seg = right_seg.index;

            // Compute the merge cost of merging the left and right segments.
            const merge_cost = try mergeCost(uncompressed_values, left_seg, right_seg);

            // Update the cost of the left segment with the computed merge cost.
            left_seg.cost = merge_cost;

            // Update the left segment in the heap.
            try heap.update(left_seg, left_seg);
        }

        // If the right segment is not the last segment, update its merge cost with its right neighbor.
        if (right_seg.seg_end != uncompressed_values.len - 1) {
            // Update the index of the placeholder segment to the right neighbor of the right segment.
            placeholder_segment_cost.index = right_seg.right_seg;

            // Retrieve the right neighbor of the right segment from the heap.
            const right_to_right_seg = try heap.get(try heap.getIndex(placeholder_segment_cost));

            // Compute the merge cost of merging the right segment with its right neighbor.
            const merge_cost = try mergeCost(uncompressed_values, right_seg, right_to_right_seg);

            // Update the cost of the right segment with the computed merge cost.
            right_seg.cost = merge_cost;
        }

        // Update the right segment in the heap.
        try heap.update(right_seg, right_seg);
    }

    // Sort remaining points by original index to preserve order.
    mem.sort(SegmentMergeCost, heap.items[0..heap.len], {}, SegmentMergeCost.firstThan);

    // Output compressed series: (seg_start, index, end_value) pairs.
    for (0..heap.len) |index| {
        const seg_start = heap.items[index].seg_start;
        const seg_end = heap.items[index].seg_end;
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

/// `SegmentMergeCostHashContext` provides context for hashing and comparing `SegmentMergeCost` items for use
/// in `HashMap`. It defines how `SegmentMergeCost` are hashed and compared for equality.
const SegmentMergeCostHashContext = struct {
    /// Hashes the `index: usize` by bitcasting it to `u64`.
    pub fn hash(_: SegmentMergeCostHashContext, seg_merge_cost: SegmentMergeCost) u64 {
        return @as(u64, @intCast(seg_merge_cost.index));
    }
    /// Compares two `index` for equality.
    pub fn eql(
        _: SegmentMergeCostHashContext,
        seg_merge_error_one: SegmentMergeCost,
        seg_merge_error_two: SegmentMergeCost,
    ) bool {
        return seg_merge_error_one.index == seg_merge_error_two.index;
    }
};

/// Comparison function for the `HashedPriorityQueue`. It compares merge cost between two segments.
fn compareSegmentMergeCost(_: void, seg_1: SegmentMergeCost, seg_2: SegmentMergeCost) math.Order {
    if (seg_1.cost == seg_2.cost)
        return math.Order.eq;
    return math.order(seg_1.cost, seg_2.cost);
}

/// Build initial pair-wise segments costs (two points per segment if possible).
fn buildInitialPairwiseSegmentCost(
    heap: *HashedPriorityQueue(
        SegmentMergeCost,
        void,
        compareSegmentMergeCost,
        SegmentMergeCostHashContext,
    ),
    uncompressed_values: []const f64,
) Error!void {
    var seg_id: usize = 1;
    var seg_start: usize = 2;

    // We need to create a segment for every two points in the uncompressed values.
    // The first segment is always the first two points.
    // The second segment is the next two points, and so on.
    var previous_seg = SegmentMergeCost{
        .index = 0,
        .cost = math.inf(f64),
        .left_seg = 0,
        .right_seg = 1,
        .seg_start = 0,
        .seg_end = 1,
    };

    while (seg_start < uncompressed_values.len) : (seg_start += 2) {
        const seg_end: usize = if (seg_start + 1 < uncompressed_values.len) seg_start + 1 else seg_start;

        const current_seg = SegmentMergeCost{
            .index = seg_id,
            .cost = math.inf(f64),
            .left_seg = seg_id - 1,
            .right_seg = seg_id + 1,
            .seg_start = seg_start,
            .seg_end = seg_end,
        };

        // Compute the merge cost of merging the previous and current segments.
        const merge_cost = try mergeCost(uncompressed_values, previous_seg, current_seg);
        previous_seg.cost = merge_cost;
        try heap.add(previous_seg);

        seg_id += 1;
        previous_seg = current_seg;
    }

    // Inserts the previous segment, which is likely the last segment in the sequence.
    // In the bottom-up line simplification algorithm, the last segment cannot be merged
    // with any segment to its right, as there are no further segments. Therefore, it is
    // added with an infinite cost to indicate that no further merging is possible.
    try heap.add(previous_seg);

    // If the last segment is not a pair, we need to add it to the heap.
    if (seg_start < uncompressed_values.len) {
        var last_seg = SegmentMergeCost{
            .index = seg_id,
            .cost = math.inf(f64),
            .left_seg = seg_id - 1,
            .right_seg = seg_id + 1,
            .seg_start = seg_start,
            .seg_end = seg_start,
        };
        const merge_cost = try mergeCost(uncompressed_values, previous_seg, last_seg);
        last_seg.cost = merge_cost;
        try heap.add(last_seg);
    }
}

/// Incremental cost of merging the neighbouring segments `seg_one` and `seg_two`.
fn mergeCost(uncompressed_values: []const f64, seg_one: SegmentMergeCost, seg_two: SegmentMergeCost) !f64 {
    const merged_start = @min(seg_one.seg_start, seg_two.seg_start);
    const merged_end = @max(seg_one.seg_end, seg_two.seg_end);
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
