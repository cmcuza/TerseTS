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

//! Implementation of "Bottom-Up" algorithm as described in the paper:
//! "E. Keogh, S. Chu, D. Hart and M. Pazzani.
//! An online algorithm for segmenting time series.
//! IEEE ICDM, pp. 289-296, 2001.
//! https://doi.org/10.1109/ICDM.2001.989531"
//! This specific implementation uses the "Root-Sum-of-Squared-Errors"
//! as the cost function. Future work may include other cost functions.

const std = @import("std");
const mem = std.mem;
const math = std.math;
const time = std.time;
const ArrayList = std.ArrayList;

const tersets = @import("../tersets.zig");
const Method = tersets.Method;
const Error = tersets.Error;

const HashedPriorityQueue = @import(
    "../utilities/hashed_priority_queue.zig",
).HashedPriorityQueue;

const shared = @import("../utilities/shared_structs.zig");

const DiscretePoint = shared.DiscretePoint;
const LinearFunction = shared.LinearFunction;
const Segment = shared.Segment;

const tester = @import("../tester.zig");

/// Compresses `uncompressed_values` using the "Bottom-Up" simplification algorithm.
/// This algorithm iteratively merges points to minimize the sum of squared errors,
/// ensuring that the resulting compressed sequence stays within the specified `error_bound`.
/// The function writes the simplified sequence to the `compressed_values`. The `allocator`
/// is used to allocate memory for the HashedPriorityQueue used in the implementation.
/// If an error occurs it is returned.
pub fn compress(
    uncompressed_values: []const f64,
    compressed_values: *ArrayList(u8),
    allocator: mem.Allocator,
    error_bound: f32,
) Error!void {

    // If we have 2 or fewer points, store them without compression.
    if (uncompressed_values.len <= 2) {
        try appendValue(f64, uncompressed_values[0], compressed_values);
        try appendValue(usize, 1, compressed_values);
        try appendValue(f64, uncompressed_values[1], compressed_values);
        return;
    }

    if (error_bound < 0) {
        return Error.UnsupportedErrorBound;
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

    try buildInitialPairwiseSegmentCost(&heap, uncompressed_values);

    // Placeholder for the segment cost to be used in the loop to search for neighboring segments.
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

        placeholder_segment_cost = min_segment.left_seg;
        const left_seg: SegmentMergeCost = try heap.get(try heap.getIndex(placeholder_segment_cost));

        placeholder_segment_cost.index = min_segment.right_point;
        const right_seg: SegmentMergeCost = try heap.get(try heap.getIndex(placeholder_segment_cost));

        right_seg.seg_start = min_segment.seg_start;
        right_seg.left_seg = left_seg.index;
        left_seg.right_seg = right_seg.index;

        if (right_seg.seg_start != 0) {
            // Compute the merge cost of merging the previous and current segments.
            const merge_cost = mergeCost(uncompressed_values, left_seg, right_seg);
            const new_left_seg = left_seg;
            new_left_seg.cost = merge_cost;
            try heap.update(left_seg, new_left_seg);
        }

        if (right_seg.seg_end != uncompressed_values.len - 1) {
            // Compute the merge cost of merging the previous and current segments.
            placeholder_segment_cost = right_seg.right_seg;
            const right_to_right_seg = try heap.get(try heap.getIndex(placeholder_segment_cost));
            const merge_cost = mergeCost(uncompressed_values, right_seg, right_to_right_seg);
            const new_right_seg = right_seg;
            new_right_seg.cost = merge_cost;
            try heap.update(right_seg, new_right_seg);
        }
    }
}

/// `SegmentMergeCost` represents a segment and its associated `cost` with the next segment,
/// which is calculated based RSSE. The `cost` is used to determine the order in which segments
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
) !void {
    var seg_id: usize = 1;
    var seg_start: usize = 2;

    // We need to create a segment for every two points in the uncompressed values.
    // The first segment is always the first two points.
    // The second segment is the next two points, and so on.
    var previous_seg = SegmentMergeCost{
        .index = 0,
        .cost = std.math.inf(f64),
        .left_seg = 0,
        .right_seg = 1,
        .seg_start = 0,
        .seg_end = 1,
    };

    while (seg_start < uncompressed_values.len) : (seg_start += 2) {
        const seg_end: usize = if (seg_start + 1 < uncompressed_values.len) seg_start + 1 else seg_start;

        const current_seg = SegmentMergeCost{
            .index = seg_id,
            .cost = std.math.inf(f64),
            .left_seg = seg_id - 1,
            .right_seg = seg_id + 1,
            .seg_start = seg_start,
            .seg_end = seg_end,
        };

        // Compute the merge cost of merging the previous and current segments.
        const merge_cost = mergeCost(uncompressed_values, previous_seg, current_seg);
        previous_seg.cost = merge_cost;
        try heap.add(previous_seg);

        seg_id += 1;
        previous_seg = current_seg;
    }

    // If the last segment is not a pair, we need to add it to the heap.
    if (seg_start < uncompressed_values.len) {
        var last_seg = SegmentMergeCost{
            .index = seg_id,
            .cost = std.math.inf(f64),
            .left_seg = seg_id - 1,
            .right_seg = seg_id + 1,
            .seg_start = seg_start,
            .seg_end = seg_start,
        };
        const merge_cost = mergeCost(uncompressed_values, previous_seg, last_seg);
        last_seg.cost = merge_cost;
        try heap.add(last_seg);
    }
}

/// Incremental cost of merging the neighbouring segments `seg_one` and `seg_two`.
fn mergeCost(uncompressed_values: []const f64, seg_one: SegmentMergeCost, seg_two: SegmentMergeCost) f64 {
    const merged_start = @min(seg_one.seg_start, seg_two.seg_start);
    const merged_end = @max(seg_one.seg_end, seg_two.seg_end);
    return computeRSSE(uncompressed_values, merged_start, merged_end);
}

fn computeRSSE(uncompressed_values: []const f64, seg_start: usize, seg_end: usize) f64 {
    const seg_len: f64 = @floatFromInt(seg_end - seg_start + 1);
    if (seg_len <= 1) return 0.0;

    var sum_x: f64 = 0;
    var sum_y: f64 = 0;
    var sum_x2: f64 = 0;
    var sum_xy: f64 = 0;

    var i: usize = seg_start;
    while (i <= seg_end) : (i += 1) {
        const x: f64 = @floatFromInt(i);
        const y: f64 = uncompressed_values[i];
        sum_x += x;
        sum_y += y;
        sum_x2 += x * x;
        sum_xy += x * y;
    }

    const var_x: f64 = sum_x2 - (sum_x * sum_x / seg_len);
    const cov_xy = sum_xy - (sum_x * sum_y / seg_len);
    const slope = cov_xy / var_x;
    const mean_x = sum_x / seg_len;
    const mean_y = sum_y / seg_len;
    const intercept = mean_y - slope * mean_x;

    var sse: f64 = 0;
    i = seg_start;
    while (i <= seg_end) : (i += 1) {
        const pred = slope * @as(f64, @floatFromInt(i)) + intercept;
        const diff = uncompressed_values[i] - pred;
        sse += diff * diff;
    }
    return std.math.sqrt(sse);
}

/// Append `value` of `type` determined at compile time to `compressed_values`.
fn appendValue(comptime T: type, value: T, compressed_values: *std.ArrayList(u8)) !void {
    // Compile-time type check
    switch (@TypeOf(value)) {
        f64, usize => {
            const value_as_bytes: [8]u8 = @bitCast(value);
            try compressed_values.appendSlice(value_as_bytes[0..]);
        },
        else => @compileError("Unsupported type for append value function"),
    }
}
