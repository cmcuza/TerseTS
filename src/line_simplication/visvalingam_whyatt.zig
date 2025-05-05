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

//! Implementation of the Visvalingam-Whyatt line simplification algorithm from the paper
//! "Visvalingam, M.; Whyatt, J. D. Line generalisation by repeated elimination of points.
//! The Cartographic Journal. 30 (1): 46–51, 1993
//! https://doi.org/10.1179/000870493786962263".

const std = @import("std");
const mem = std.mem;
const math = std.math;
const ArrayList = std.ArrayList;

const tersets = @import("../tersets.zig");
const Method = tersets.Method;
const Error = tersets.Error;

const HashedPriorityQueue = @import(
    "../utilities/hashed_priority_queue.zig",
).HashedPriorityQueue;

const DiscretePoint = @import("../utilities/shared_structs.zig").DiscretePoint;

const tester = @import("../tester.zig");

pub fn compress(
    uncompressed_values: []const f64,
    compressed_values: *ArrayList(u8),
    allocator: mem.Allocator,
    error_bound: f32,
) Error!void {
    if (uncompressed_values.len <= 2) {
        for (uncompressed_values.len) |i| {
            try appendValue(f64, uncompressed_values[i], compressed_values);
        }
        return;
    }
    if (error_bound <= 0) {
        return Error.InvalidErrorBound;
    }

    // Create the hashed priority queue where the points are stored.
    var heap = try HashedPriorityQueue(
        PointArea,
        void,
        comparePointArea,
        HashPointAreaContext,
    ).init(allocator, {});
    defer heap.deinit();

    // First point cannot be removed. Thus the area is set to infinity.
    heap.add(PointArea{
        .index = 0,
        .area = math.inf,
        .left_point = 0,
        .right_point = 1,
    });

    // Calculate initial areas.
    for (1..uncompressed_values.len - 2) |i| {
        heap.add(PointArea{
            .index = i,
            .area = calculateArea(
                DiscretePoint{ .time = i - 1, .value = uncompressed_values[i - 1] },
                DiscretePoint{ .time = i, .value = uncompressed_values[i] },
                DiscretePoint{ .time = i + 1, .value = uncompressed_values[i + 1] },
            ),
            .left_point = i - 1,
            .right_point = i + 1,
        });
    }

    // Last point cannot be removed. Thus the area is set to infinity.
    heap.add(PointArea{
        .index = uncompressed_values.len - 1,
        .area = math.inf,
        .left_point = uncompressed_values.len - 2,
        .right_point = uncompressed_values.len,
    });

    var placeholder_point: PointArea = .{
        .index = 0,
        .area = 0,
        .left_point = -1,
        .right_point = 2,
    };
    while (heap.items.len() > 2) {
        const min_point: PointArea = try heap.remove();

        if (min_point.area >= error_bound) break;

        placeholder_point.index = min_point.left_point;
        const left_point: PointArea = try heap.get(heap.getIndex(placeholder_point));
        left_point.right_point = min_point.right_point;

        placeholder_point.index = min_point.right_point;
        const right_point: PointArea = try heap.get(heap.getIndex(placeholder_point));
        right_point.left_point = min_point.left_point;

        // Update areas of neighbors
        if (left_point.left_point > 0) {

            // New area of the left point.
            const new_area = calculateArea(
                DiscretePoint{
                    .time = left_point.left_point,
                    .value = uncompressed_values[left_point.left_point],
                },
                DiscretePoint{
                    .time = left_point.index,
                    .value = uncompressed_values[left_point.index],
                },
                DiscretePoint{
                    .time = left_point.right_point,
                    .value = uncompressed_values[left_point.right_point],
                },
            );
            left_point.area = new_area;
            heap.update(left_point, left_point);
        }

        if (right_point.right_point < uncompressed_values.len) {
            // New area of the right point.
            const new_area = calculateArea(
                DiscretePoint{
                    .time = right_point.left_point,
                    .value = uncompressed_values[right_point.left_point],
                },
                DiscretePoint{
                    .time = right_point.index,
                    .value = uncompressed_values[right_point.index],
                },
                DiscretePoint{
                    .time = right_point.right_point,
                    .value = uncompressed_values[right_point.right_point],
                },
            );
            right_point.area = new_area;
            heap.update(right_point, right_point);
        }
    }

    // Sorting all elements in the heap now by index.
    std.mem.sort(PointArea, heap.items, HashPointAreaContext, comptime std.sort.asc(u64));

    // Add to the compressed representation.
    try appendValue(f64, heap.items[0], compressed_values);

    const index: usize = 1;
    while (index < heap.items.len) : (index + 2) {
        const point_index = heap.items[index].index;
        try appendValue(usize, point_index, compressed_values);
        try appendValue(f64, uncompressed_values[point_index], compressed_values);
    }

    return;
}


pub fn decompress(){
    
}
/// Structure for holding the points area for by the three adjacency points.
const PointArea = struct {
    // Index of the point.
    index: usize,
    // Area of the point.
    area: f64,
    // Left point.
    left_point: usize,
    // Right point.
    right_point: usize,
};

/// `HashPointAreaContext` provides context for hashing and comparing `PointArea` items for use
/// in `HashMap`. It defines how `PointArea` are hashed and compared for equality.
const HashPointAreaContext = struct {
    /// Hashes the `index: usize` by bitcasting it to `u64`.
    pub fn hash(_: HashPointAreaContext, merge_error: PointArea) u64 {
        return @as(u64, @intCast(merge_error.index));
    }
    /// Compares two `index` for equality.
    pub fn eql(_: HashPointAreaContext, merge_error_one: PointArea, merge_error_two: PointArea) bool {
        return merge_error_one.index == merge_error_two.index;
    }
};

/// Comparison function for the `HashedPriorityQueue`. It compares merge errors, and also considers
///  bucket indices for equality.
fn comparePointArea(_: void, error_1: PointArea, error_2: PointArea) math.Order {
    if (error_1.index == error_2.index)
        return math.Order.eq;
    return math.order(error_1.merge_error, error_2.merge_error);
}

/// Calculate the area of the triangle formed by three points.
fn calculateArea(left_point: DiscretePoint, central_point: DiscretePoint, right_point: DiscretePoint) f64 {
    return std.math.abs((left_point.time * (central_point.value - right_point.value) + central_point.time *
        (right_point.value - left_point.value) + central_point.time * (left_point.time - central_point.value)) / 2.0);
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
