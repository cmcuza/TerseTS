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
const time = std.time;
const ArrayList = std.ArrayList;

const tersets = @import("../tersets.zig");
const Method = tersets.Method;
const Error = tersets.Error;

const HashedPriorityQueue = @import(
    "../utilities/hashed_priority_queue.zig",
).HashedPriorityQueue;

const DiscretePoint = @import("../utilities/shared_structs.zig").DiscretePoint;
const LinearFunction = @import("../utilities/shared_structs.zig").LinearFunction;
const Segment = @import("../utilities/shared_structs.zig").Segment;

const tester = @import("../tester.zig");

/// Compress `uncompressed_values` using "Visvalingam-Whyatt" simplification algorithm by keeping
/// points whose effective area is greater than the `error_bound`. The function writes the result
/// to `compressed_values`. The `allocator` is used to allocate memory for the HashedPriorityQueue
/// used in the implementation. If an error occurs it is returned.
pub fn compress(
    uncompressed_values: []const f64,
    compressed_values: *ArrayList(u8),
    allocator: mem.Allocator,
    error_bound: f32,
) Error!void {
    // If we have 2 or fewer points, copy them directly.
    if (uncompressed_values.len <= 2) {
        for (0..uncompressed_values.len) |i| {
            try appendValue(f64, uncompressed_values[i], compressed_values);
        }
        return;
    }
    if (error_bound < 0) {
        return Error.IncorrectInput;
    }

    // Initialize hashed priority queue with area as priority.
    var heap = try HashedPriorityQueue(
        PointArea,
        void,
        comparePointArea,
        HashPointAreaContext,
    ).init(allocator, {});
    defer heap.deinit();

    // First point cannot be removed. Thus the area is set to infinity.
    try heap.add(PointArea{
        .index = 0,
        .area = math.inf(f64),
        .left_point = 0,
        .right_point = 1,
    });

    // Calculate initial areas.
    for (1..uncompressed_values.len - 1) |i| {
        try heap.add(PointArea{
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
    try heap.add(PointArea{
        .index = uncompressed_values.len - 1,
        .area = math.inf(f64),
        .left_point = uncompressed_values.len - 2,
        .right_point = uncompressed_values.len,
    });

    // Placeholder for the point area to be used in the loop to search neighborg points.
    var placeholder_point: PointArea = .{
        .index = 0,
        .area = 0,
        .left_point = 0,
        .right_point = 0,
    };

    // Main simplification loop
    while (heap.items.len > 2) {
        // Get the point with the smallest area.
        const min_point: PointArea = try heap.peek();

        // The area is greater than the error bound.
        if (min_point.area >= error_bound) {
            break;
        }

        // Now is safe to remove the point with the smallest area.
        _ = try heap.pop();

        // Adjust neighbors of removed point
        placeholder_point.index = min_point.left_point;
        var left_point: PointArea = try heap.get(try heap.getIndex(placeholder_point));
        left_point.right_point = min_point.right_point;

        placeholder_point.index = min_point.right_point;
        var right_point: PointArea = try heap.get(try heap.getIndex(placeholder_point));
        right_point.left_point = min_point.left_point;

        // Update areas of left neighbor.
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
        }
        // Update the left point in the heap. Even if the area is not changed, it is necessary to update the
        // left point to update the pointer to its right point.
        try heap.update(left_point, left_point);

        // Update area of right neighbor.
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
        }
        // Update the right point in the heap. Even if the area is not changed, it is necessary to update the
        // right point to update the pointer to its left point.
        try heap.update(right_point, right_point);
    }

    // Sort remaining points by original index to preserve order.
    std.mem.sort(PointArea, heap.items[0..heap.len], {}, PointArea.firstThan);

    // Output compressed series: first point, then (index, value) pairs.
    try appendValue(f64, uncompressed_values[0], compressed_values);
    for (1..heap.len) |index| {
        const point_index = heap.items[index].index;
        try appendValue(usize, point_index, compressed_values);
        try appendValue(f64, uncompressed_values[point_index], compressed_values);
    }

    return;
}

/// Decompress `compressed_values` produced by "Visvalingam-Whyatt" and write the
/// result to `decompressed_values`. If an error occurs it is returned.
pub fn decompress(compressed_values: []const u8, decompressed_values: *ArrayList(f64)) Error!void {
    // The compressed representation is composed of two values after getting the first since all
    // segments are connected. Therefore, the condition checks that after the first value, the rest
    // of the values are in pairs (index, value) and that they are all of type 64-bit float.
    if ((compressed_values.len - 8) % 16 != 0) return Error.IncorrectInput;

    const compressed_lines_and_index = mem.bytesAsSlice(f64, compressed_values);

    var index: usize = 0;

    // Extract the start point from the compressed representation.
    var start_point: DiscretePoint = .{ .time = 0, .value = compressed_lines_and_index[0] };
    try decompressed_values.append(start_point.value);

    // We need to create a segment for the linear function.
    var slope: f64 = undefined;
    var intercept: f64 = undefined;
    while (index < compressed_lines_and_index.len - 1) : (index += 2) {
        // index + 1 is the end value and index + 2 is the end time.
        const end_point: DiscretePoint = .{
            .time = @as(usize, @bitCast(compressed_lines_and_index[index + 1])),
            .value = compressed_lines_and_index[index + 2],
        };

        if (start_point.time < end_point.time) {
            // Create the linear approximation for the current segment.
            if (end_point.time != start_point.time) {
                const duration: f64 = @floatFromInt(end_point.time - start_point.time);
                slope = (end_point.value - start_point.value) / duration;
                intercept = start_point.value - slope *
                    @as(f64, @floatFromInt(start_point.time));
            } else {
                slope = 0.0;
                intercept = start_point.value;
            }
            var current_timestamp: usize = start_point.time + 1;
            // Interpolate the values between the start and end points of the current segment.
            while (current_timestamp < end_point.time) : (current_timestamp += 1) {
                const y: f64 = slope * @as(f64, @floatFromInt(current_timestamp)) + intercept;
                try decompressed_values.append(y);
            }
            try decompressed_values.append(end_point.value);
        }

        // The start point of the next segment is the end point of the current segment.
        start_point = end_point;
    }
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

    /// order by the `index` field (ascending).
    fn firstThan(_: void, point_1: PointArea, point_2: PointArea) bool {
        return point_1.index < point_2.index;
    }
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
fn comparePointArea(_: void, point_1: PointArea, point_2: PointArea) math.Order {
    if (point_1.area == point_2.area)
        return math.Order.eq;
    return math.order(point_1.area, point_2.area);
}

/// Return the absolute area of the triangle defined by three points.
fn calculateArea(left_point: DiscretePoint, central_point: DiscretePoint, right_point: DiscretePoint) f64 {
    const x1: f64 = @floatFromInt(left_point.time);
    const y1: f64 = left_point.value;
    const x2: f64 = @floatFromInt(central_point.time);
    const y2: f64 = central_point.value;
    const x3: f64 = @floatFromInt(right_point.time);
    const y3: f64 = right_point.value;

    return @abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0);
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

/// Creates a linear function that passes throught the two points of the `segment` and returns it
/// in `linear_function`.
fn createLinearFunction(start_point: DiscretePoint, end_point: DiscretePoint, linear_function: *LinearFunction) void {
    if (end_point.time != start_point.time) {
        const duration: f80 = @floatFromInt(end_point.time - start_point.time);
        linear_function.slope = (end_point.value - start_point.value) / duration;
        linear_function.intercept = start_point.value - linear_function.slope *
            @as(f80, @floatFromInt(start_point.time));
    } else {
        linear_function.slope = 0.0;
        linear_function.intercept = start_point.value;
    }
}

/// Test if the area of all the triangles defined by three points is within the error bound.
pub fn testAreaWithinErrorBound(
    values: []const f64,
    error_bound: f32,
) !void {
    if (values.len < 3) return;

    // Calculate initial areas.
    for (1..values.len - 1) |i| {
        const area = calculateArea(
            DiscretePoint{ .time = i - 1, .value = values[i - 1] },
            DiscretePoint{ .time = i, .value = values[i] },
            DiscretePoint{ .time = i + 1, .value = values[i + 1] },
        );
        try std.testing.expect(area <= error_bound);
    }
}

test "vw compress and decompress with zero error bound" {
    // Initialize a random number generator.
    const seed: u64 = @bitCast(time.milliTimestamp());
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();

    const allocator = std.testing.allocator;

    // Output buffer.
    var uncompressed_values = ArrayList(f64).init(allocator);
    defer uncompressed_values.deinit();
    var compressed_values = ArrayList(u8).init(allocator);
    defer compressed_values.deinit();
    var decompressed_values = ArrayList(f64).init(allocator);
    defer decompressed_values.deinit();
    const error_bound: f32 = 0.0;

    try tester.generateBoundedRandomValues(&uncompressed_values, 0, 1000000, random);

    // Call the compress function.
    try compress(uncompressed_values.items, &compressed_values, allocator, error_bound);
    try decompress(compressed_values.items, &decompressed_values);

    for (0..uncompressed_values.items.len) |i| {
        try std.testing.expectEqual(uncompressed_values.items[i], decompressed_values.items[i]);
    }
}

test "vw compress and compress with known result" {
    const allocator = std.testing.allocator;

    // Input data.
    const uncompressed_values: []const f64 = &[_]f64{ 1.0, 1.5, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0 };
    const error_bound: f32 = 2.5;

    // Output buffer.
    var compressed_values = ArrayList(u8).init(allocator);
    defer compressed_values.deinit();
    var decompressed_values = ArrayList(f64).init(allocator);
    defer decompressed_values.deinit();

    // Call the compress function.
    try compress(uncompressed_values, &compressed_values, allocator, error_bound);
    try decompress(compressed_values.items, &decompressed_values);

    // Check if the decompressed values have the same lenght as the compressed ones.
    try std.testing.expectEqual(uncompressed_values.len, decompressed_values.items.len);

    // In theory, all triangles formed by the slices of removed points should be within the error otherwise the
    // point cannot be removed. In this case, the error is 2.5,  and the area of the triangles is always less than
    // this value except when removing the element in position 5. Therefore, the area of the triangles formed by the
    // slices of the removed points from 0..5 should be less than 2.5.
    try testAreaWithinErrorBound(uncompressed_values[0..6], error_bound);
}

test "vw compress and compress with random data" {
    // Initialize a random number generator.
    const seed: u64 = @bitCast(time.milliTimestamp());
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();

    const allocator = std.testing.allocator;

    // Output buffer.
    var uncompressed_values = ArrayList(f64).init(allocator);
    defer uncompressed_values.deinit();
    var compressed_values = ArrayList(u8).init(allocator);
    defer compressed_values.deinit();
    var decompressed_values = ArrayList(f64).init(allocator);
    defer decompressed_values.deinit();
    const error_bound: f32 = random.float(f32);

    try tester.generateBoundedRandomValues(&uncompressed_values, 0, 1, random);

    // Call the compress function.
    try compress(uncompressed_values.items, &compressed_values, allocator, error_bound);
    try decompress(compressed_values.items, &decompressed_values);

    // Check if the decompressed values have the same lenght as the compressed ones.
    try std.testing.expectEqual(uncompressed_values.items.len, decompressed_values.items.len);

    // In theory, all triangles formed by the slices of removed points should be within the error otherwise the
    // point cannot be removed. In this case, the error bound is unknown as well as which points are finally
    // preserved in the compressed representation. Therefore, we need to used the compressed representation to access
    // each of the points preserved and their index `current_point_index`. Then, the area of the triangles formed by the
    // slices of the removed points from `previous_point_index`..`current_point_index` should be less than `error_bound`.
    const compressed_representation = mem.bytesAsSlice(f64, compressed_values.items);

    std.debug.print("len={}\n", .{compressed_representation.len});
    var index: usize = 0;
    var previous_point_index: usize = 0;
    while (index < compressed_representation.len - 1) : (index += 2) {
        const current_point_index = @as(usize, @bitCast(compressed_representation[index + 1]));

        // Check if the point is within the error bound.
        try testAreaWithinErrorBound(uncompressed_values.items[previous_point_index .. current_point_index + 1], error_bound);
        previous_point_index = current_point_index;
    }
}
