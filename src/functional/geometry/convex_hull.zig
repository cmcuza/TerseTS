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

//! Implementation of Graham's scan which can maintain a convex hull from the paper
//! "Graham, R.L. An Efficient Algorithm for Determining the Convex Hull of a Finite Planar Set.
//! Information Processing Letters. 1 (4): 132–133. 1972.
//! https://doi.org/10.1016/0020-0190(72)90045-2.
//! Convex Hulls are used by the Slide Filter and Optimal PLA algorithms implemented in
//! `src/functional/swing_slide_filter.zig` and `src/functional/optimal_pla.zig`.

const std = @import("std");
const mem = std.mem;
const ArrayList = std.ArrayList;
const testing = std.testing;

const tersets = @import("../../tersets.zig");
const DiscretePoint = tersets.DiscretePoint;

/// Enum for the angle's `Turn` of three consecutive points A, B, and C. Essentially, it describes
/// whether the path from A to B to C makes a `left` turn, a `right` turn, or continues in a
/// straight line also called collinear.
const Turn = enum(i8) { right, left, collinear };

/// Convex Hull formed by an upper and lower hull. The hulls are formed by the input data with
/// discrete time axis thus, the ConvexHull is always represented by `DiscretePoint`.
pub const ConvexHull = struct {
    lower_hull: ArrayList(DiscretePoint),
    upper_hull: ArrayList(DiscretePoint),

    // Initialize the container with a given `allocator`.
    pub fn init(allocator: mem.Allocator) !ConvexHull {
        return ConvexHull{
            .lower_hull = ArrayList(DiscretePoint).init(allocator),
            .upper_hull = ArrayList(DiscretePoint).init(allocator),
        };
    }

    // Deinitialize the container and free the allocated memory.
    pub fn deinit(self: *ConvexHull) void {
        self.lower_hull.deinit();
        self.upper_hull.deinit();
    }

    /// Add a new `point` to the convex hull.
    pub fn addPoint(self: *ConvexHull, point: DiscretePoint) !void {
        try addPointToHull(&self.upper_hull, Turn.right, point);
        try addPointToHull(&self.lower_hull, Turn.left, point);
    }

    /// Auxiliary function to add a new `point` to a given `hull` of the convex hull. The function uses
    /// the given `turn` to correctly add the new point.
    fn addPointToHull(hull: *ArrayList(DiscretePoint), turn: Turn, point: DiscretePoint) !void {
        if (hull.items.len < 2) {
            // The first two points can be add directly.
            try hull.append(point);
        } else {
            var top: usize = hull.items.len - 1;
            // Remove the last point as long as the `turn` is not the provided.
            while ((top > 0) and (computeTurn(
                hull.items[top - 1],
                hull.items[top],
                point,
            ) != turn)) : (top -= 1) {
                _ = hull.pop();
            }
            try hull.append(point);
        }
    }
};

/// Compute turn created by the path from `first_point` to `middle_point` to `last_point`. If this
/// function is part of the structure and does not use the `self` parameter, the compiler returns
/// an error. However, since it is used for testing purposes it cannot be private.
fn computeTurn(
    first_point: DiscretePoint,
    middle_point: DiscretePoint,
    last_point: DiscretePoint,
) Turn {
    const distance_last_middle: f64 = @floatFromInt(last_point.time - middle_point.time);
    const distance_middle_first: f64 = @floatFromInt(middle_point.time - first_point.time);

    const cross_product = (middle_point.value - first_point.value) * distance_last_middle -
        (last_point.value - middle_point.value) * distance_middle_first;

    return if (cross_product == 0)
        Turn.collinear
    else if (cross_product > 0)
        Turn.right
    else
        Turn.left;
}

test "incremental convex hull with known result" {
    const allocator = testing.allocator;

    var convex_hull = try ConvexHull.init(allocator);
    defer convex_hull.deinit();

    try convex_hull.addPoint(.{ .time = 0, .value = 3 });
    try convex_hull.addPoint(.{ .time = 1, .value = 2 });
    try convex_hull.addPoint(.{ .time = 2, .value = 3.5 });
    try convex_hull.addPoint(.{ .time = 3, .value = 5 });
    try convex_hull.addPoint(.{ .time = 4, .value = 3 });
    try convex_hull.addPoint(.{ .time = 5, .value = 4 });
    try convex_hull.addPoint(.{ .time = 6, .value = 4 });
    try convex_hull.addPoint(.{ .time = 7, .value = 3 });
    try convex_hull.addPoint(.{ .time = 8, .value = 4.5 });
    try convex_hull.addPoint(.{ .time = 9, .value = 3.5 });
    try convex_hull.addPoint(.{ .time = 10, .value = 2.5 });
    try convex_hull.addPoint(.{ .time = 11, .value = 2.5 });
    try convex_hull.addPoint(.{ .time = 12, .value = 3.5 });
    try convex_hull.addPoint(.{ .time = 13, .value = 2.5 });
    try convex_hull.addPoint(.{ .time = 14, .value = 2.5 });
    try convex_hull.addPoint(.{ .time = 15, .value = 2.5 });
    try convex_hull.addPoint(.{ .time = 16, .value = 3 });
    try convex_hull.addPoint(.{ .time = 17, .value = 3 });
    try convex_hull.addPoint(.{ .time = 18, .value = 3 });
    try convex_hull.addPoint(.{ .time = 19, .value = 3 });
    try convex_hull.addPoint(.{ .time = 20, .value = 2.8 });

    try testing.expectEqual(5, convex_hull.upper_hull.items.len);
    try testing.expectEqual(4, convex_hull.lower_hull.items.len);

    // Expected Upper Hull.
    try testing.expectEqual(0, convex_hull.upper_hull.items[0].time);
    try testing.expectEqual(3, convex_hull.upper_hull.items[1].time);
    try testing.expectEqual(8, convex_hull.upper_hull.items[2].time);
    try testing.expectEqual(19, convex_hull.upper_hull.items[3].time);
    try testing.expectEqual(20, convex_hull.upper_hull.items[4].time);

    // Expected Lower Hull.
    try testing.expectEqual(0, convex_hull.lower_hull.items[0].time);
    try testing.expectEqual(1, convex_hull.lower_hull.items[1].time);
    try testing.expectEqual(15, convex_hull.lower_hull.items[2].time);
    try testing.expectEqual(20, convex_hull.lower_hull.items[3].time);
}

test "incremental convex hull random elements" {
    const num_points: usize = 1000;
    const allocator = testing.allocator;
    var rnd = std.rand.DefaultPrng.init(0);

    var convex_hull = try ConvexHull.init(allocator);
    defer convex_hull.deinit();

    for (0..num_points) |i| {
        try convex_hull.addPoint(.{ .time = i, .value = rnd.random().float(f64) });
    }

    // All points in the Upper Hull should turn to the right.
    for (1..convex_hull.upper_hull.items.len - 1) |i| {
        const turn = computeTurn(
            convex_hull.upper_hull.items[i - 1],
            convex_hull.upper_hull.items[i],
            convex_hull.upper_hull.items[i + 1],
        );
        try testing.expectEqual(turn, Turn.right);
    }

    // All points in the Lower Hull should turn to the left.
    for (1..convex_hull.lower_hull.items.len - 1) |i| {
        const turn = computeTurn(
            convex_hull.lower_hull.items[i - 1],
            convex_hull.lower_hull.items[i],
            convex_hull.lower_hull.items[i + 1],
        );
        try testing.expectEqual(turn, Turn.left);
    }
}
