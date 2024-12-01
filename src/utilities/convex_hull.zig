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

const tersets = @import("../tersets.zig");
const Error = tersets.Error;

const shared = @import("shared_structs.zig");
const DiscretePoint = shared.DiscretePoint;
const ContinousPoint = shared.ContinousPoint;
const Segment = shared.Segment;
const LinearFunction = shared.LinearFunction;

/// Enum for the angle's `Turn` of three consecutive points A, B, and C. Essentially, it describes
/// whether the path from A to B to C makes a `left` turn, a `right` turn, or continues in a
/// straight line also called collinear.
const Turn = enum(i8) { right, left, collinear };

/// Enum to determine the hull in the convex hull.
const HullType = enum(i8) { upperHull, lowerHull };

/// Convex Hull formed by an upper and lower hull. The hulls are formed by the input data with
/// discrete time axis thus, the ConvexHull is always represented by `DiscretePoint`.
pub const ConvexHull = struct {
    lower_hull: ArrayList(DiscretePoint),
    upper_hull: ArrayList(DiscretePoint),
    allocator: mem.Allocator,

    // Initialize the container with a given `allocator`.
    pub fn init(allocator: mem.Allocator) !ConvexHull {
        return ConvexHull{
            .lower_hull = ArrayList(DiscretePoint).init(allocator),
            .upper_hull = ArrayList(DiscretePoint).init(allocator),
            .allocator = allocator,
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

    /// Invalidates all element pointers in the upper and lower hull. The capacity is preserved.
    pub fn clean(self: *ConvexHull) void {
        self.upper_hull.clearRetainingCapacity();
        self.lower_hull.clearRetainingCapacity();
    }

    /// Returns the points in the `upper_hull` except the last one.
    pub fn getUpperHullExceptLast(self: *ConvexHull) []const DiscretePoint {
        const upper_hull_len = self.upper_hull.items.len;
        if (upper_hull_len <= 1) {
            // Return an empty array if there's only one or no items.
            return &[_]DiscretePoint{};
        }
        return self.upper_hull.items[0 .. upper_hull_len - 1];
    }

    /// Returns the points in the `lower_hull` except the last one.
    pub fn getLowerHullExceptLast(self: *ConvexHull) []const DiscretePoint {
        const lower_hull_len = self.lower_hull.items.len;
        if (lower_hull_len <= 1) {
            // Return an empty array if there's only one or no items.
            return &[_]DiscretePoint{};
        }
        return self.lower_hull.items[0 .. lower_hull_len - 1];
    }

    /// Computes the `LinearFunction` with minimum square error based on the Minimum-Area Bounding
    /// Rectangle (MABR) of a Convex Hull. The MABR is computed using the Rotating Calipers
    /// algorithm described in the paper:
    /// Shamos, Michael (1978). "Computational Geometry" (PDF). Yale University. pp. 76–81.
    /// This implementation is based mostly on the description found in:
    /// https://en.wikipedia.org/wiki/Rotating_calipers.
    pub fn computeMABRLinearFunction(self: *ConvexHull) Error!LinearFunction {
        const convex_hull_len: usize = self.len();

        // Check if the Convex Hull has only one point. If so, the linear function is a horizontal
        // line thus `slope=0` and `intercept` is the value of the first and only point.
        if (convex_hull_len == 1) {
            const first_point: DiscretePoint = self.at(0);
            return LinearFunction{ .intercept = first_point.value, .slope = 0.0 };
        }
        // Check if the Convex Hull has only two points. If so, the linear function passes through
        // the two points in the Convex Hull.
        if (convex_hull_len == 2) {
            const first_point = self.at(0);
            const second_point = self.at(1);
            const delta_time: f80 = @floatFromInt(second_point.time - first_point.time);
            const slope = (second_point.value - first_point.value) / delta_time;
            const intercept_value: f80 = first_point.value - slope * @as(f80, @floatFromInt(first_point.time));
            return LinearFunction{
                .slope = slope,
                .intercept = intercept_value,
            };
        }

        // Initialize variables to track the minimum area and corresponding parameters.
        var min_area: f64 = std.math.floatMax(f64); // Set initial min_area to the maximum possible.
        var min_slope: f64 = 0.0; // Slope of the line corresponding to min_area.
        var min_intercept: f64 = 0.0; // Intercept point corresponding to min_area.

        // Iterate over each edge of the convex hull.
        for (0..convex_hull_len) |i| {
            // Get the current edge (segment) of the convex hull.
            const first_point: DiscretePoint = self.at(i);
            const second_point: DiscretePoint = self.at(i + 1);
            const segment = Segment{ .start_point = first_point, .end_point = second_point };

            // Compute the angle between the current edge and the x-axis.
            const angle = angleToXAxis(segment);

            // Initialize min/max values for rotated points.
            var minX: f64 = std.math.floatMax(f64); // Minimum x-coordinate after rotation.
            var maxX: f64 = std.math.floatMin(f64); // Maximum x-coordinate after rotation.
            var minY: f64 = std.math.floatMax(f64); // Minimum y-coordinate after rotation.
            var maxY: f64 = std.math.floatMin(f64); // Maximum y-coordinate after rotation.

            // Rotate all points and update min/max coordinates.
            for (0..convex_hull_len) |j| {
                const point: DiscretePoint = self.at(j);

                // Rotate point by -angle to align the edge with the x-axis
                const rotated_point: ContinousPoint = rotateToXAxis(DiscretePoint, point, -angle);

                // Update min/max x and y values
                minX = @min(minX, rotated_point.time);
                maxX = @max(maxX, rotated_point.time);
                minY = @min(minY, rotated_point.value);
                maxY = @max(maxY, rotated_point.value);
            }

            // Compute the area of the bounding rectangle in the rotated coordinate system.
            const width = maxX - minX; // Width of the bounding rectangle.
            const height = maxY - minY; // Height of the bounding rectangle.
            const area = width * height; // Area of the bounding rectangle.

            // Update minimum area and corresponding parameters if a smaller area is found.
            if (area < min_area) {
                min_area = area;

                // Compute the slope of the edge (tan(angle)).
                min_slope = std.math.tan(angle);

                // Compute the center of the bounding rectangle in rotated coordinates.
                const centerX = (minX + maxX) / 2.0;
                const centerY = (minY + maxY) / 2.0;
                const center_rotated = ContinousPoint{ .time = centerX, .value = centerY };

                // Rotate the center point back to the original coordinate system.
                const center_point = rotateToXAxis(ContinousPoint, center_rotated, angle);

                // Compute the intercept value using the center point.
                const intercept_value = center_point.value - min_slope * center_point.time;
                min_intercept = intercept_value;
            }
        }

        return LinearFunction{ .slope = min_slope, .intercept = min_intercept };
    }

    /// Merges another `other: ConvexHull` into the current `self: ConvexHull` in place. This
    /// operation modifies the `self` object by appending points from the `other` convex hull.
    /// If appending to the hulls fails due to allocation issues, the error is returned.
    pub fn merge(self: *ConvexHull, other: *ConvexHull) !void {
        // Special case: if `other` has only one point, directly add it to self's hulls.
        if (other.len() == 1) {
            try self.addPoint(other.at(0));
            return;
        }

        // Special case: if `self` has only one point, transfer points from `other` directly.
        if (self.len() == 1) {
            const other_points = try other.getAllPointsSorted();
            defer other_points.deinit(); // Ensure the allocated list is freed.
            for (other_points.items) |point| {
                try self.addPoint(point); // Add each point to `self`.
            }
            return;
        }

        // Find tangent between the upper hulls.
        const upper_tangent = try findTangent(
            &self.upper_hull,
            &other.upper_hull,
            .upperHull,
        );

        // Find tangent between the lower hulls.
        const lower_tangent = try findTangent(
            &self.lower_hull,
            &other.lower_hull,
            .lowerHull,
        );

        // Add points to `self`'s upper hull starting from the right-most point
        // of the `other`s upper hull's tangent.
        for (upper_tangent.hull_two_idx..other.upper_hull.items.len) |i| {
            try self.upper_hull.append(other.upper_hull.items[i]);
        }

        // Add points to `self`'s lower hull starting from the right-most point
        // of the `other`s lower hull's tangent.
        for (lower_tangent.hull_two_idx..other.lower_hull.items.len) |i| {
            try self.lower_hull.append(other.lower_hull.items[i]);
        }
    }

    /// Helper function to get all points as an `ArrayList(DiscretePoint)` from the ConvexHull.
    /// If the `ConvexHull` is empty, an empty ArrayList is returned.
    fn getAllPoints(self: *ConvexHull) !ArrayList(DiscretePoint) {
        // Since the first and final element are repeated, we substract two from the sum of the
        // lower and upper hull length.
        var hull: ArrayList(DiscretePoint) = try ArrayList(DiscretePoint).initCapacity(
            self.allocator,
            self.lower_hull.items.len + self.upper_hull.items.len - 2,
        );

        for (self.lower_hull.items) |item| {
            try hull.append(item);
        }
        for (1..self.upper_hull.items.len - 1) |i| {
            try hull.append(self.upper_hull.items[i]);
        }
        return hull;
    }

    /// Helper function to get all points sorted as an `ArrayList(DiscretePoint)`. The function
    /// merges the lower and upper hulls, which are already sorted, in O(N) time. If the
    /// `ConvexHull` is empty, an empty ArrayList is returned.
    pub fn getAllPointsSorted(self: *ConvexHull) !ArrayList(DiscretePoint) {
        // Initialize the result list with enough capacity for both hulls.
        var all_points = try ArrayList(DiscretePoint).initCapacity(
            self.allocator,
            self.lower_hull.items.len + self.upper_hull.items.len - 2,
        );

        var lower_idx: usize = 0;
        var upper_idx: usize = 1;

        // Merge lower_hull and upper_hull into all_points. Since the first and final element are
        // repeated, we start `upper_idx=1` and only loop until `self.upper_hull.items.len - 1`.
        while (lower_idx < self.lower_hull.items.len and upper_idx < self.upper_hull.items.len - 1) {
            const lower_point = self.lower_hull.items[lower_idx];
            const upper_point = self.upper_hull.items[upper_idx];

            if (lower_point.time <= upper_point.time) {
                // Add the point from the lower hull.
                try all_points.append(lower_point);
                lower_idx += 1;
            } else {
                // Add the point from the upper hull.
                try all_points.append(upper_point);
                upper_idx += 1;
            }
        }

        // Append remaining points from the lower hull, if any.
        while (lower_idx < self.lower_hull.items.len) {
            try all_points.append(self.lower_hull.items[lower_idx]);
            lower_idx += 1;
        }

        // Append remaining points from the upper hull, if any. Since the first and final element are
        // repeated, we start `upper_idx=1` and only loop until `self.upper_hull.items.len - 1`.
        while (upper_idx < self.upper_hull.items.len - 1) {
            try all_points.append(self.upper_hull.items[upper_idx]);
            upper_idx += 1;
        }

        return all_points;
    }

    /// Auxiliary function to add a new `point` to a given `hull` of the convex hull. The function
    /// uses the given `turn` to correctly add the new point.
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

    /// Returns the length of the `ConvexHull`.
    fn len(self: *ConvexHull) usize {
        const lower_hull_len = self.lower_hull.items.len;
        const upper_hull_len = self.upper_hull.items.len;
        const convex_hull_len = lower_hull_len + upper_hull_len;

        // There are no elements in the Convex Hull.
        if (convex_hull_len == 0) {
            return 0;
        }

        // There is only one element but since it is repeated `convex_hull_len==2`.
        if (convex_hull_len == 2) {
            return 1;
        }

        // Since the first and final element are repeated, we substract two from the sum of the
        // lower and upper hull length.
        return convex_hull_len - 2;
    }

    /// Returns the item at the given `index` counterclockwise concatenating the lower and upper
    /// hull and considering the repeated first and last element.
    fn at(self: *ConvexHull, index: usize) DiscretePoint {
        const lower_hull_len = self.lower_hull.items.len;
        const convex_hull_len = self.len();
        var new_index = @mod(index, convex_hull_len);
        if (new_index < lower_hull_len) {
            return self.lower_hull.items[new_index];
        } else {
            new_index = new_index - lower_hull_len + 1;
            return self.upper_hull.items[new_index];
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

/// Compute the angle between a segment and the x-axis.
fn angleToXAxis(segment: Segment) f64 {
    const deltaX: f64 = @as(f64, @floatFromInt(segment.end_point.time)) - @as(f64, @floatFromInt(segment.start_point.time));
    const deltaY: f64 = segment.end_point.value - segment.start_point.value;
    return std.math.atan2(deltaY, deltaX);
}

/// Rotate a point around the origin by a given angle.
fn rotateToXAxis(comptime Point: type, point: Point, angle: f64) ContinousPoint {
    if (Point == DiscretePoint) {
        const newX: f64 = @as(f64, @floatFromInt(point.time)) *
            std.math.cos(angle) - point.value * std.math.sin(angle);
        const newY: f64 = @as(f64, @floatFromInt(point.time)) *
            std.math.sin(angle) + point.value * std.math.cos(angle);
        return ContinousPoint{ .time = newX, .value = newY };
    }
    const newX: f64 = point.time * std.math.cos(angle) - point.value * std.math.sin(angle);
    const newY: f64 = point.time * std.math.sin(angle) + point.value * std.math.cos(angle);
    return ContinousPoint{ .time = newX, .value = newY };
}

/// Finds the tangent between convex hulls `hull_one` and `hull_two` for either the upper or lower
/// hull determined by `hull_type`. This function ensures the resulting tangent line lies entirely
/// outside both convex hulls and does not cross any points within the hulls. The function returns
/// a struct containing the indices of the tangent points in `hull_one` and `hull_two`. It returns
/// `IndexError` if either hull is empty.
fn findTangent(
    hull_one: *ArrayList(DiscretePoint),
    hull_two: *ArrayList(DiscretePoint),
    hull_type: HullType,
) !struct {
    hull_one_idx: usize,
    hull_two_idx: usize,
} {
    // Ensure the input hulls are not empty.
    if (hull_one.items.len == 0 or hull_two.items.len == 0) {
        return Error.EmptyInput;
    }

    // Start with the rightmost point of hull_one and the leftmost point of hull_two.
    var hull_one_idx: usize = hull_one.items.len - 1; // Rightmost point of hull_one.
    var hull_two_idx: usize = 0; // Leftmost point of hull_two.

    // Iterate until a valid tangent is found.
    while (true) {
        // Check the turn direction at hull_two.
        const hull_two_turn = computeTurn(
            hull_one.items[hull_one_idx], // Current point in hull_one.
            hull_two.items[hull_two_idx], // Current point in hull_two.
            hull_two.items[(hull_two_idx + 1) % hull_two.items.len], // Next point in hull_two.
        );

        // Adjust hull_two_idx based on the turn and hull type.
        const hull_two_turn_violation = switch (hull_type) {
            // If the turn is left, the current tangent dips into hull_two.
            .upperHull => hull_two_turn == Turn.left,
            // If the turn is right, the current tangent dips into hull_two.
            .lowerHull => hull_two_turn == Turn.right,
        };

        if (hull_two_turn_violation) {
            // Move to the next point in hull_two (clockwise adjustment).
            hull_two_idx = (hull_two_idx + 1) % hull_two.items.len;
            continue; // Re-evaluate the tangent.
        }

        // Check the turn direction at hull_one.
        const hull_one_turn = computeTurn(
            hull_one.items[(hull_one_idx - 1 + hull_one.items.len) % hull_one.items.len], // Previous point in hull_one.
            hull_one.items[hull_one_idx], // Current point in hull_one.
            hull_two.items[hull_two_idx], // Current point in hull_two.
        );

        // Adjust hull_one_idx based on the turn and hull type.
        const hull_one_turn_violation = switch (hull_type) {
            // If the turn is left, the current tangent dips into hull_one.
            .upperHull => hull_one_turn == Turn.left,
            // If the turn is right, the current tangent dips into hull_one.
            .lowerHull => hull_one_turn == Turn.right,
        };

        if (hull_one_turn_violation) {
            // Move to the previous point in hull_one (counterclockwise adjustment).
            hull_one_idx = (hull_one_idx - 1 + hull_one.items.len) % hull_one.items.len;
            _ = hull_one.pop(); // Remove the point that violates convexity.
            continue; // Re-evaluate the tangent.
        }

        // If neither index needs adjustment, we have found a valid tangent.
        break;
    }

    // Return the indices of the tangent points.
    return .{ .hull_one_idx = hull_one_idx, .hull_two_idx = hull_two_idx };
}

test "Create incrementally convex hull with known result" {
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

test "Create incrementally a convex hull with random elements" {
    const num_points: usize = 1000;
    const allocator = testing.allocator;
    var rnd = std.rand.DefaultPrng.init(0);

    var convex_hull = try ConvexHull.init(allocator);
    defer convex_hull.deinit();

    for (0..num_points) |i| {
        try convex_hull.addPoint(.{ .time = i, .value = rnd.random().float(f64) });
    }

    try testConvexHullProperty(&convex_hull);
}

test "Compute MABR LinearFunction for known Convex Hull one" {
    const allocator = std.testing.allocator;

    // Define a set of points forming a simple rectangle.
    const points = [_]DiscretePoint{
        .{ .time = 0, .value = 0.0 },
        .{ .time = 1, .value = 3.0 },
        .{ .time = 2, .value = 2.0 },
        .{ .time = 3, .value = 5.0 },
        .{ .time = 4, .value = 4.0 },
    };

    // Initialize the convex hull and add points to it.
    var convex_hull = try ConvexHull.init(allocator);
    defer convex_hull.deinit();

    for (points) |point| {
        try convex_hull.addPoint(point);
    }

    // Calculate the MABR Linear Function.
    const mabr_linear_function = try convex_hull.computeMABRLinearFunction();

    // Known Linear Function with `slope=1` and `intercept=1`.
    try testing.expect(@abs(mabr_linear_function.slope - 1) <= 0.0001);
    try testing.expect(@abs(mabr_linear_function.intercept - 1) <= 0.0001);
}

test "Compute MABR LinearFunction for known Convex Hull two" {
    const allocator = std.testing.allocator;

    // Define a set of points forming a simple rectangle.
    const points = [_]DiscretePoint{
        .{ .time = 0, .value = 0.0 },
        .{ .time = 1, .value = -2.0 },
        .{ .time = 2, .value = 3.0 },
        .{ .time = 3, .value = -1.0 },
        .{ .time = 4, .value = 2.0 },
    };
    // Initialize the convex hull and add points to it.
    var convex_hull = try ConvexHull.init(allocator);
    defer convex_hull.deinit();

    for (points) |point| {
        try convex_hull.addPoint(point);
    }

    // Calculate MABR Linear Function.
    const mabr_linear_function = try convex_hull.computeMABRLinearFunction();

    // Known Linear Function with `slope=1.5` and `intercept=2.75`.
    try testing.expect(@abs(mabr_linear_function.slope - 1.5) <= 0.0001);
    try testing.expect(@abs(mabr_linear_function.intercept + 2.75) <= 0.0001);
}

test "Compute MABR LinearFunction for random Convex Hull" {
    const num_points: usize = 1000;
    const allocator = testing.allocator;
    var rnd = std.rand.DefaultPrng.init(0);

    var convex_hull = try ConvexHull.init(allocator);
    defer convex_hull.deinit();

    for (0..num_points) |i| {
        try convex_hull.addPoint(.{ .time = i, .value = rnd.random().float(f64) });
    }

    // Calculate MABR Linear Function. The exact value of the `intercept` is unknown but it must be
    // less than 1.0 given that all points lie strictly within [0,1]. This is because the linear function is the height bisector and goes through the midpoint of the rectangle side which has as lower and upper bounds 0 and 1.
    const mabr_linear_function = try convex_hull.computeMABRLinearFunction();

    try testing.expect(mabr_linear_function.intercept <= 1.0);
}

test "Merge convex hulls with known result" {
    const allocator = testing.allocator;

    var convex_hull_one = try ConvexHull.init(allocator);
    defer convex_hull_one.deinit();

    var convex_hull_two = try ConvexHull.init(allocator);
    defer convex_hull_two.deinit();

    try convex_hull_one.addPoint(.{ .time = 0, .value = 3 });
    try convex_hull_one.addPoint(.{ .time = 1, .value = 2 });
    try convex_hull_one.addPoint(.{ .time = 2, .value = 3.5 });
    try convex_hull_one.addPoint(.{ .time = 3, .value = 5 });
    try convex_hull_one.addPoint(.{ .time = 4, .value = 3 });
    try convex_hull_one.addPoint(.{ .time = 5, .value = 4 });
    try convex_hull_one.addPoint(.{ .time = 6, .value = 4 });
    try convex_hull_one.addPoint(.{ .time = 7, .value = 3 });
    try convex_hull_one.addPoint(.{ .time = 8, .value = 4.5 });
    try convex_hull_one.addPoint(.{ .time = 9, .value = 3.5 });
    try convex_hull_one.addPoint(.{ .time = 10, .value = 2.5 });
    try convex_hull_one.addPoint(.{ .time = 11, .value = 2.5 });
    try convex_hull_one.addPoint(.{ .time = 12, .value = 3.5 });
    try convex_hull_one.addPoint(.{ .time = 13, .value = 2.5 });
    try convex_hull_one.addPoint(.{ .time = 14, .value = 2.5 });
    try convex_hull_one.addPoint(.{ .time = 15, .value = 2.5 });
    try convex_hull_one.addPoint(.{ .time = 16, .value = 3 });
    try convex_hull_one.addPoint(.{ .time = 17, .value = 3 });
    try convex_hull_one.addPoint(.{ .time = 18, .value = 3 });
    try convex_hull_one.addPoint(.{ .time = 19, .value = 3 });
    try convex_hull_one.addPoint(.{ .time = 20, .value = 2.8 });

    try convex_hull_two.addPoint(.{ .time = 21, .value = 1 });
    try convex_hull_two.addPoint(.{ .time = 22, .value = 2.5 });
    try convex_hull_two.addPoint(.{ .time = 23, .value = 6 });
    try convex_hull_two.addPoint(.{ .time = 24, .value = 2 });
    try convex_hull_two.addPoint(.{ .time = 25, .value = 6 });
    try convex_hull_two.addPoint(.{ .time = 26, .value = 9 });
    try convex_hull_two.addPoint(.{ .time = 27, .value = 3 });
    try convex_hull_two.addPoint(.{ .time = 28, .value = 4.5 });
    try convex_hull_two.addPoint(.{ .time = 29, .value = 10 });
    try convex_hull_two.addPoint(.{ .time = 30, .value = 1.5 });

    try convex_hull_one.merge(&convex_hull_two);

    // Expected Upper Hull.
    try testing.expectEqual(0, convex_hull_one.upper_hull.items[0].time);
    try testing.expectEqual(3, convex_hull_one.upper_hull.items[1].time);
    try testing.expectEqual(29, convex_hull_one.upper_hull.items[2].time);
    try testing.expectEqual(30, convex_hull_one.upper_hull.items[3].time);

    // Expected Lower Hull.
    try testing.expectEqual(0, convex_hull_one.lower_hull.items[0].time);
    try testing.expectEqual(1, convex_hull_one.lower_hull.items[1].time);
    try testing.expectEqual(21, convex_hull_one.lower_hull.items[2].time);
    try testing.expectEqual(30, convex_hull_one.lower_hull.items[3].time);
}

test "Merge convex hulls with random elements" {
    const num_points: usize = 100;
    const allocator = testing.allocator;
    var rnd = std.rand.DefaultPrng.init(0);

    var convex_hull_one = try ConvexHull.init(allocator);
    defer convex_hull_one.deinit();

    for (0..num_points) |i| {
        try convex_hull_one.addPoint(.{ .time = i, .value = rnd.random().float(f64) });
    }

    var convex_hull_two = try ConvexHull.init(allocator);
    defer convex_hull_two.deinit();

    for (num_points..2 * num_points) |i| {
        try convex_hull_two.addPoint(.{ .time = i, .value = rnd.random().float(f64) });
    }

    // Merge convex_hull_two into convex_hull_one.
    try convex_hull_one.merge(&convex_hull_two);

    try testConvexHullProperty(&convex_hull_one);
}

test "Merge single element's convex hull with other convex hull" {
    const num_points: usize = 100;
    const allocator = testing.allocator;
    var rnd = std.rand.DefaultPrng.init(0);

    var convex_hull_one = try ConvexHull.init(allocator);
    defer convex_hull_one.deinit();

    try convex_hull_one.addPoint(.{ .time = 0, .value = rnd.random().float(f64) });

    var convex_hull_two = try ConvexHull.init(allocator);
    defer convex_hull_two.deinit();

    for (1..num_points) |i| {
        try convex_hull_two.addPoint(.{ .time = i, .value = rnd.random().float(f64) });
    }

    try convex_hull_one.merge(&convex_hull_two);

    try testConvexHullProperty(&convex_hull_one);
}

test "Merge convex hull with single element's convex hull" {
    const num_points: usize = 100;
    const allocator = testing.allocator;
    var rnd = std.rand.DefaultPrng.init(0);

    var convex_hull_one = try ConvexHull.init(allocator);
    defer convex_hull_one.deinit();

    for (0..num_points) |i| {
        try convex_hull_one.addPoint(.{ .time = i, .value = rnd.random().float(f64) });
    }

    var convex_hull_two = try ConvexHull.init(allocator);
    defer convex_hull_two.deinit();

    try convex_hull_two.addPoint(.{ .time = num_points, .value = rnd.random().float(f64) });

    try convex_hull_one.merge(&convex_hull_two);

    try testConvexHullProperty(&convex_hull_one);
}

/// Validates that the Convex Hull `convex_hull` satisfies the required properties:
/// 1. In the `upper_hull`, every sequence of three consecutive points must turn to the right.
/// 2. In the `lower_hull`, every sequence of three consecutive points must turn to the left.
/// If any of these properties are violated, an error is returned.
fn testConvexHullProperty(convex_hull: *ConvexHull) !void {
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
