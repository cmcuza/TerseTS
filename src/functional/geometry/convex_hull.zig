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

//! This file contains the Graham algorithm to incrementally create or maintain a convex hull
//! as well as the necessary structures, enums, and auxiliary functions. The algorithm is described
//! in: "Mark De Berg, and Marc van Kreveld. Computational geometry: algorithms and applications.
//! Springer Science & Business Media, 2000. https://doi.org//10.1007/978-3-540-77974-2.
//! Convex Hulls are necessary in the implementation of the compression algorithms [1] and [2].
//! [1] https://doi.org/10.14778/1687627.1687645 (Slide Filter).
//! [2] https://doi.org/10.1109/TSP.2006.875394 (Optimal PLA).

const std = @import("std");
const mem = std.mem;
const testing = std.testing;

const tersets = @import("../../tersets.zig");
const Error = tersets.Error;

/// Enum for the angle's `Turn` of three consecutive points. The angle can represent a `right` or
/// `left` turn. If there is no turn, then the points are `colinear`.
const Turn = enum(i8) { right, left, colinear };

/// Point structure to represent a point by `time` and `value`.
const Point = struct { time: usize, value: f64 };

/// Set of points used to store the upper and lower hull of the convex hull. `points` are memory
/// slide that it's dynamically increased as `len` exceeds `max_len`. A good estimation of
/// `max_len` can improve both execution time and memory consumption.
pub const PointSet = struct {
    points: []Point,
    len: usize,
    max_len: usize,
    allocator: *const mem.Allocator,

    // Initialize the container with a given `allocator` and max number of points `num_points`.
    // The max number of points it is not fixed. The max number of points will increase as more
    // elements are added.
    pub fn init(allocator: *const mem.Allocator, num_points: usize) !PointSet {
        return PointSet{
            .points = try allocator.alloc(Point, num_points),
            .len = 0,
            .max_len = num_points,
            .allocator = allocator,
        };
    }

    // Deinitialize the container and free the allocated memory.
    pub fn deinit(self: *PointSet) void {
        self.allocator.free(self.points);
    }

    // Add `point` to the `PointSet`.
    pub fn add(self: *PointSet, point: Point) !void {
        if (self.len >= self.max_len) {
            // Resize the set of `points`.
            self.max_len = self.max_len * 2;
            self.points = try self.allocator.realloc(self.points, self.max_len);
        }
        self.points[self.len] = point;
        self.len += 1;
    }

    // Remove the last point from the set.
    pub fn pop(self: *PointSet) !void {
        if (self.len == 0) return Error.EmptySet;
        self.len -= 1;
    }
};

/// Graham algorithm to add a new point to `upper_hull` and `lower_hull`. The algorithm ensures
/// that the upper and lower hull form a Convex Hull.
pub fn addPointToConvexHull(upper_hull: *PointSet, lower_hull: *PointSet, point: Point) !void {
    if (upper_hull.len < 2) {
        // The first two points can be add directly.
        try upper_hull.add(point);
    } else {
        // Update upper hull.
        var top: usize = upper_hull.len - 1;
        while ((top > 0) and (getTurn(
            upper_hull.points[top - 1],
            upper_hull.points[top],
            point,
        ) != Turn.right)) : (top -= 1) {
            try upper_hull.pop();
        }
        try upper_hull.add(point);
    }

    if (lower_hull.len < 2) {
        // The first two points can be add directly.
        try lower_hull.add(point);
    } else {
        // Update lower hull.
        var top: usize = lower_hull.len - 1;
        while ((top > 0) and (getTurn(
            lower_hull.points[top - 1],
            lower_hull.points[top],
            point,
        ) != Turn.left)) : (top -= 1) {
            try lower_hull.pop();
        }
        try lower_hull.add(point);
    }
}

/// Return the type of turn created by the `first_point`, `middle_point and the `last_point`.
fn getTurn(first_point: Point, middle_point: Point, last_point: Point) Turn {
    const distance_last_middle: f64 = @floatFromInt(last_point.time - middle_point.time);
    const distance_middle_first: f64 = @floatFromInt(middle_point.time - first_point.time);

    const value = (middle_point.value - first_point.value) * distance_last_middle -
        (last_point.value - middle_point.value) * distance_middle_first;

    if (value == 0) return Turn.colinear;

    return if (value > 0) Turn.right else Turn.left;
}

test "incremental convex hull with known result" {
    const allocator = testing.allocator;

    var upper_hull = try PointSet.init(&allocator, 2);
    defer upper_hull.deinit();
    var lower_hull = try PointSet.init(&allocator, 2);
    defer lower_hull.deinit();

    var point: Point = Point{ .time = 0, .value = 3 };
    try addPointToConvexHull(&upper_hull, &lower_hull, point);
    point.time = 1;
    point.value = 2;
    try addPointToConvexHull(&upper_hull, &lower_hull, point);
    point.time = 2;
    point.value = 3.5;
    try addPointToConvexHull(&upper_hull, &lower_hull, point);
    point.time = 3;
    point.value = 5;
    try addPointToConvexHull(&upper_hull, &lower_hull, point);
    point.time = 4;
    point.value = 3;
    try addPointToConvexHull(&upper_hull, &lower_hull, point);
    point.time = 5;
    point.value = 4;
    try addPointToConvexHull(&upper_hull, &lower_hull, point);
    point.time = 6;
    point.value = 4;
    try addPointToConvexHull(&upper_hull, &lower_hull, point);
    point.time = 7;
    point.value = 3;
    try addPointToConvexHull(&upper_hull, &lower_hull, point);
    point.time = 8;
    point.value = 4.5;
    try addPointToConvexHull(&upper_hull, &lower_hull, point);
    point.time = 9;
    point.value = 3.5;
    try addPointToConvexHull(&upper_hull, &lower_hull, point);
    point.time = 10;
    point.value = 2.5;
    try addPointToConvexHull(&upper_hull, &lower_hull, point);
    point.time = 11;
    point.value = 2.5;
    try addPointToConvexHull(&upper_hull, &lower_hull, point);
    point.time = 12;
    point.value = 3.5;
    try addPointToConvexHull(&upper_hull, &lower_hull, point);
    point.time = 13;
    point.value = 2.5;
    try addPointToConvexHull(&upper_hull, &lower_hull, point);
    point.time = 14;
    point.value = 2.5;
    try addPointToConvexHull(&upper_hull, &lower_hull, point);
    point.time = 15;
    point.value = 2.5;
    try addPointToConvexHull(&upper_hull, &lower_hull, point);
    point.time = 16;
    point.value = 3;
    try addPointToConvexHull(&upper_hull, &lower_hull, point);
    point.time = 17;
    point.value = 3;
    try addPointToConvexHull(&upper_hull, &lower_hull, point);
    point.time = 18;
    point.value = 3;
    try addPointToConvexHull(&upper_hull, &lower_hull, point);
    point.time = 19;
    point.value = 3;
    try addPointToConvexHull(&upper_hull, &lower_hull, point);
    point.time = 20;
    point.value = 2.8;
    try addPointToConvexHull(&upper_hull, &lower_hull, point);

    try testing.expectEqual(5, upper_hull.len);
    try testing.expectEqual(4, lower_hull.len);

    // Expected Upper Hull.
    try testing.expectEqual(0, upper_hull.points[0].time);
    try testing.expectEqual(3, upper_hull.points[1].time);
    try testing.expectEqual(8, upper_hull.points[2].time);
    try testing.expectEqual(19, upper_hull.points[3].time);
    try testing.expectEqual(20, upper_hull.points[4].time);
    // Expected Lower Hull.
    try testing.expectEqual(0, lower_hull.points[0].time);
    try testing.expectEqual(1, lower_hull.points[1].time);
    try testing.expectEqual(15, lower_hull.points[2].time);
    try testing.expectEqual(20, lower_hull.points[3].time);
}

test "incremental convex hull random elements" {
    const num_points: usize = 1000;
    const allocator = testing.allocator;
    var rnd = std.rand.DefaultPrng.init(0);
    var upper_hull = try PointSet.init(&allocator, num_points);
    defer upper_hull.deinit();
    var lower_hull = try PointSet.init(&allocator, num_points);
    defer lower_hull.deinit();

    var point: Point = Point{ .time = 0, .value = rnd.random().float(f64) };
    for (1..num_points) |i| {
        try addPointToConvexHull(&upper_hull, &lower_hull, point);
        point.time = i;
        point.value = rnd.random().float(f64);
    }

    // All points in the Upper Hull should turn to the right.
    for (1..upper_hull.len - 1) |i| {
        const turn = getTurn(upper_hull.points[i - 1], upper_hull.points[i], upper_hull.points[i + 1]);
        try testing.expectEqual(turn, Turn.right);
    }
    // All points in the Lower Hull should turn to the left.
    for (1..lower_hull.len - 1) |i| {
        const turn = getTurn(lower_hull.points[i - 1], lower_hull.points[i], lower_hull.points[i + 1]);
        try testing.expectEqual(turn, Turn.left);
    }
}
