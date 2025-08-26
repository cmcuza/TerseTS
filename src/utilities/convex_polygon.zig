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

//! This module implements convex polygon clipping in the (slope, intercept) parameter space
//! for linear functions, supporting incremental intersection of half-plane constraints from the paper:
//! "O'Rourke, Joseph. An online algorithm for fitting straight lines between data ranges."
//! Communications of the ACM 24.9 (1981): 574-578.
//! https://dl.acm.org/doi/pdf/10.1145/358746.458758.
//! The implementation is partially based on the code released at
//! https://github.com/and-gue/NeaTS (accessed on 15-08-25).

const std = @import("std");
const mem = std.mem;
const ArrayList = std.ArrayList;
const testing = std.testing;
const shared_structs = @import("../utilities/shared_structs.zig");
const LinearFunction = shared_structs.LinearFunction;
const ContinousPoint = shared_structs.ContinousPoint;

const shared_functions = @import("../utilities/shared_functions.zig");
const tester = @import("../tester.zig");

const XAxisDomain = struct { start: f64, end: f64 };

/// Result of clipping a convex polygon with a new half-plane (lower or upper bound).
/// If `NoClip`, the polygon is fully inside the half-plane; no changes are needed.
/// If `Reject`, the polygon lies entirely outside the half-plane; intersection is empty.
/// If `Clip`, the polygon intersects the half-plane. In this case, the structure contains
/// `upper_index_offset` and `lower_index_offset` indicating how far into the upper and lower
/// chains the existing polygon must be trimmed; and `new_upper_segment` and `new_lower_segment`
/// to replace the border segments created at the intersection with the clipping line.
const ClipOutcome = union(enum) {
    NoClip,
    Reject,
    Clip: struct {
        upper_index_offset: usize, // offset into upper bound chain.
        lower_index_offset: usize, // offset into lower bound chain.
        new_upper_segment: BorderLine, // replacement segment on upper bound.
        new_lower_segment: BorderLine, // replacement segment on lower bound.
    },
};

/// `BorderLine` represents Line in the parameter-space of the slope and intercept of a LinearFunction.
/// The structure contains the `definition` of the LinearFunction as well as the `x_axis_domain` in
/// which the line is defined. This structure is used to contain the borders of the ConvexPolygon.
pub const BorderLine = struct {
    x_axis_domain: XAxisDomain, // Start and end of the x-axis domain of the border line.
    definition: LinearFunction, // Parameters defining the border line.

    /// Creates a new BorderLine with `definition=(slope, intercept)` and `x_axis_domain=(start, end)`.
    pub fn init(slope: f64, intercept: f64, start: f64, end: f64) BorderLine {
        return BorderLine{
            .x_axis_domain = .{ .start = start, .end = end },
            .definition = .{ .slope = slope, .intercept = intercept },
        };
    }

    /// Builds a BorderLine from two ContinousPoints in the slope-intercept parameter-space. If
    /// `continous_point_one` and `continous_point_two` have the same time axis, the function
    /// returns a degenerate horizontal segment over that interval.
    pub fn buildBorderLineFromPoints(
        continous_point_one: ContinousPoint,
        continous_point_two: ContinousPoint,
    ) BorderLine {
        if (shared_functions.isApproximatelyEqual(
            continous_point_one.time,
            continous_point_two.time,
        )) {
            // Vertical in (m,b): represent as zero-slope b line over [m, m].
            return .{
                .x_axis_domain = .{
                    .start = continous_point_one.time,
                    .end = continous_point_two.time,
                },
                .definition = .{
                    .slope = 0,
                    .intercept = continous_point_one.value,
                },
            };
        }
        const slope = (continous_point_two.value - continous_point_one.value) /
            (continous_point_two.time - continous_point_one.time);
        // In the slope-intercept space the equation of the function is inverted.
        const intercept = continous_point_one.value - slope * continous_point_one.time;

        return .{ .definition = .{
            .slope = slope,
            .intercept = intercept,
        }, .x_axis_domain = .{
            .start = continous_point_one.time,
            .end = continous_point_two.time,
        } };
    }

    /// Computes a ContinousPoint which contains the time value equal to the `start` of the BorderLine
    /// and the value is equal to the evaluated start in the `definition` of the `self` BorderLine.
    pub fn evaluateAtStart(self: *const BorderLine) ContinousPoint {
        const start = self.x_axis_domain.start;
        return .{
            .time = start,
            .value = self.definition.slope * start + self.definition.intercept,
        };
    }

    /// Computes a ContinousPoint which contains the time value equal to the `end` of the BorderLine
    /// and the value is equal to the evaluated `end` in the `definition` of the `self` BorderLine.
    pub fn evaluateAtEnd(self: *const BorderLine) ContinousPoint {
        const end = self.x_axis_domain.end;
        return .{
            .time = end,
            .value = self.definition.slope * end + self.definition.intercept,
        };
    }

    /// Evaluates the `self` BorderLine at a given `x_axis_value` using the its `definition`.
    /// The function returns the y_axis_value as a f64.
    pub fn evaluateAt(self: *const BorderLine, x_axis_value: f64) f64 {
        return self.definition.slope * x_axis_value + self.definition.intercept;
    }

    /// Returns true or false depending of the `self` BorderLine is strictly less than the
    /// `other_border_line` BorderLine. The comparison is done only in the y_axis of the
    /// evaluated `x_axis_domain`. Since dealing with floating points, we consider rounding
    /// error when comparing by using approximate equalities.
    pub fn isLessThan(self: *const BorderLine, other_border_line: BorderLine) bool {
        const start = self.x_axis_domain.start;
        const end = self.x_axis_domain.end;

        const self_evaluate_start = self.evaluateAt(start);
        const self_evaluate_end = self.evaluateAt(end);
        const other_evaluate_start = other_border_line.evaluateAt(start);
        const other_evaluate_end = other_border_line.evaluateAt(end);

        return (other_evaluate_start > self_evaluate_start) and
            !shared_functions.isApproximatelyEqual(other_evaluate_start, self_evaluate_start) and
            (other_evaluate_end > self_evaluate_end) and
            !shared_functions.isApproximatelyEqual(other_evaluate_end, self_evaluate_end);
    }

    /// Computes the intersection between `self` and `other_border_line`. Depending on the value
    /// of `is_upper`, the returned segment corresponds to the portion of `self` from its start up
    /// to the intersection point, or the portion of `self` from the intersection point to its end.
    /// The intersection is computed in the (slope, intercept) parameter space. If the lines are
    /// parallel (equal slopes), the function asserts that they are coincident (equal intercepts)
    /// and returns a degenerate horizontal segment spanning the current x-axis domain of `self`.
    pub fn computeIntersection(
        self: *const BorderLine,
        other_border_line: BorderLine,
        is_upper: bool,
    ) BorderLine {
        const self_slope = self.definition.slope;
        const self_intercept = self.definition.intercept;
        const other_slope = other_border_line.definition.slope;
        const other_intercept = other_border_line.definition.intercept;

        const start_point = self.evaluateAtStart();
        const end_point = self.evaluateAtEnd();

        // Parallel lines: return a horizontal segment over our current interval.
        if (shared_functions.isApproximatelyEqual(self_slope, other_slope)) {
            std.debug.assert(shared_functions.isApproximatelyEqual(self_intercept, other_intercept));
            return BorderLine{ .definition = .{
                .slope = 0.0,
                .intercept = start_point.value,
            }, .x_axis_domain = .{
                .start = start_point.time,
                .end = end_point.time,
            } };
        }

        // Intersection abscissa in x: seg_s*m + seg_q = ln_s*m + ln_q.
        const x_intercept = (self_intercept - other_intercept) / (other_slope - self_slope);
        const y_intercept = self.evaluateAt(x_intercept);

        if (is_upper) {
            if (shared_functions.isApproximatelyEqual(x_intercept, start_point.time)) {
                return BorderLine{ .definition = .{
                    .slope = 0,
                    .intercept = y_intercept,
                }, .x_axis_domain = .{
                    .start = x_intercept,
                    .end = start_point.time,
                } };
            }
            return BorderLine{
                .definition = self.definition,
                .x_axis_domain = .{
                    .start = start_point.time,
                    .end = x_intercept,
                },
            };
        } else {
            if (shared_functions.isApproximatelyEqual(x_intercept, end_point.time)) {
                return BorderLine{
                    .definition = .{
                        .slope = 0,
                        .intercept = y_intercept,
                    },
                    .x_axis_domain = .{
                        .start = x_intercept,
                        .end = end_point.time,
                    },
                };
            }
            return BorderLine{
                .definition = self.definition,
                .x_axis_domain = .{
                    .start = x_intercept,
                    .end = end_point.time,
                },
            };
        }
    }
};

/// `ConvexPolygon` represents a convex polygon in the (slope, intercept) parameter space.
/// It maintains the feasible region for linear model parameters under bounded error constraints.
/// The polygon is defined by two chains of line segments (`upper_bound_lines` and
/// `lower_bound_lines`) and supports incremental updates via intersection with new half-plane
/// constraints. A key function updates the polygon which performs clipping operations to refine
/// the feasible region in the polygon.
pub const ConvexPolygon = struct {
    // Vertices defining the upper segments of the Polygon.
    upper_bound_lines: ArrayList(BorderLine),
    // Vertices defining the lower segments of the Polygon.
    lower_bound_lines: ArrayList(BorderLine),
    // Index of the first valid segment in the upper bound vertex list.
    upper_bound_start: usize,
    // Index of the first valid segment in the lower bound vertex list.
    lower_bound_start: usize,
    // Allocator for memory management.
    allocator: mem.Allocator,

    // Initializes an empty convex polygon with the given `allocator`.
    pub fn init(allocator: mem.Allocator) ConvexPolygon {
        return ConvexPolygon{
            .upper_bound_lines = ArrayList(BorderLine).init(allocator),
            .lower_bound_lines = ArrayList(BorderLine).init(allocator),
            .upper_bound_start = 0,
            .lower_bound_start = 0,
            .allocator = allocator,
        };
    }

    // Releases all allocated memory used by the ConvexPolygon `self`.
    pub fn deinit(self: *ConvexPolygon) void {
        self.upper_bound_lines.deinit();
        self.lower_bound_lines.deinit();
    }

    /// Updates the ConvexPolygon `self` with `new_upper_bound` and `new_lower_bound`, updating the
    /// internal upper and lower bound lines in-place. Returns `true` if the polygon remains
    /// feasible after the update, `false` if the new constraints would empty the region. The function
    /// handles three cases. First, if the polygon is empty, the function adds the two incoming lines.
    /// If one border line on each chain, build the first four border lines polygon. General case, clip
    /// by lower, then clip by upper (O’Rourke online step). If an error occurs, it is returned.
    pub fn update(
        self: *ConvexPolygon,
        new_upper_bound: BorderLine,
        new_lower_bound: BorderLine,
    ) !bool {
        // Case 1: Empty polygon, seed with the two lines.
        if (self.isEmpty()) {
            try self.upper_bound_lines.append(new_upper_bound);
            try self.lower_bound_lines.append(new_lower_bound);
            self.upper_bound_start = 0;
            self.lower_bound_start = 0;
            return true;
        }

        // Case 2: expand the borders into the first proper polygon.
        if (self.upper_bound_lines.items.len == 1 and self.lower_bound_lines.items.len == 1) {
            const seed_upper = self.upper_bound_lines.items[0];
            const seed_lower = self.lower_bound_lines.items[0];

            // Clip the seed segments by (lower, then upper) to get the two “spine” segments.
            const upper_start =
                seed_upper.computeIntersection(new_lower_bound, false)
                    .computeIntersection(new_upper_bound, true);

            const lower_start =
                seed_lower.computeIntersection(new_lower_bound, false)
                    .computeIntersection(new_upper_bound, true);

            // Build the bridge edges that close the polygon (left and right vertical-ish edges in (x,y)).
            const upper_end = BorderLine.buildBorderLineFromPoints(
                upper_start.evaluateAtEnd(), // right endpoint of upper spine.
                lower_start.evaluateAtEnd(), // right endpoint of lower spine.
            );
            const lower_end = BorderLine.buildBorderLineFromPoints(
                upper_start.evaluateAtStart(), // left endpoint of upper spine.
                lower_start.evaluateAtStart(), // left endpoint of lower spine.
            );

            // Replace the seed with the four segments defining the first polygon.
            self.upper_bound_lines.clearRetainingCapacity();
            self.lower_bound_lines.clearRetainingCapacity();

            try self.upper_bound_lines.append(upper_start);
            try self.upper_bound_lines.append(upper_end);

            try self.lower_bound_lines.append(lower_start);
            try self.lower_bound_lines.append(lower_end);

            self.upper_bound_start = 0;
            self.lower_bound_start = 0;

            return true;
        }

        // Case 3: General step, clip by LOWER, then clip by UPPER.
        // 3a) Clip with lower bound.
        switch (self.clipWithLowerBound(new_lower_bound)) {
            .Reject => return false, // would empty the region
            .NoClip => {}, // nothing to do before upper clip
            .Clip => |clip_info| {
                // Advance the upper-chain cursor and replace that segment.
                self.upper_bound_start += clip_info.upper_index_offset;
                self.upper_bound_lines.items[self.upper_bound_start] = clip_info.new_upper_segment;

                // On the lower chain: truncate up to the cut, then append the cut segment
                // followed by the “bridge” that connects left endpoints of the two cut segments.
                const bridge_lower = BorderLine.buildBorderLineFromPoints(
                    clip_info.new_upper_segment.evaluateAtStart(),
                    clip_info.new_lower_segment.evaluateAtStart(),
                );
                try self.lower_bound_lines.resize(self.lower_bound_start + clip_info.lower_index_offset);
                try self.lower_bound_lines.append(clip_info.new_lower_segment);
                try self.lower_bound_lines.append(bridge_lower);
            },
        }

        // 3b) Clip with upper bound.
        switch (self.clipWithUpperBound(new_upper_bound)) {
            .Reject => return false, // would empty the region
            .NoClip => return true, // done; lower clip (if any) already applied
            .Clip => |clip_info| {
                // On the upper chain: truncate up to the cut, then append the cut segment
                // followed by the “bridge” that connects right endpoints of the two cut segments.
                const bridge_upper = BorderLine.buildBorderLineFromPoints(
                    clip_info.new_upper_segment.evaluateAtEnd(),
                    clip_info.new_lower_segment.evaluateAtEnd(),
                );
                try self.upper_bound_lines.resize(self.upper_bound_start + clip_info.upper_index_offset);
                try self.upper_bound_lines.append(clip_info.new_upper_segment);
                try self.upper_bound_lines.append(bridge_upper);

                // Advance the lower-chain cursor and replace that segment.
                self.lower_bound_start += clip_info.lower_index_offset;
                self.lower_bound_lines.items[self.lower_bound_start] = clip_info.new_lower_segment;
            },
        }

        return true;
    }

    // Clears the contents of the `ConvexPolygon` by resetting the lower and upper bound start
    // indices to zero, and clearing the `lower_bound_lines` and `upper_bound_lines` while retaining
    // their allocated capacity. This prepares the polygon for reuse without reallocating memory.
    pub fn clear(self: *ConvexPolygon) void {
        self.lower_bound_start = 0;
        self.upper_bound_start = 0;
        self.lower_bound_lines.clearRetainingCapacity();
        self.upper_bound_lines.clearRetainingCapacity();
    }

    /// Returns a representative feasible linear function for the `self` ConvexPolygon.
    /// We use the midpoint of the two extreme corners in (slope, intercept) space.
    /// The polygon needs to have at least one element to contain a feasible solution.
    /// If only one upper/lower border lines, returns a LinearFunction with slope zero;
    pub fn computeFeasibleSolution(self: *const ConvexPolygon) LinearFunction {
        std.debug.assert(!self.isEmpty());

        const upper_left = self.upperLeft();
        const lower_right = self.lowerRight();

        var slope: f64 = 0.0;
        var intercept: f64 = 0.0;

        if ((self.lower_bound_lines.items.len == 1) and (self.upper_bound_lines.items.len == 1)) {
            intercept = (upper_left.value + lower_right.value) / 2.0;
        } else {
            slope = (upper_left.time + lower_right.time) / 2.0;
            intercept = (upper_left.value + lower_right.value) / 2.0;
        }

        return LinearFunction{
            .slope = slope,
            .intercept = intercept,
        };
    }

    // Checks whether the polygon has any border line.
    fn isEmpty(self: *const ConvexPolygon) bool {
        return (self.upper_bound_lines.items.len == 0) and (self.lower_bound_lines.items.len == 0);
    }

    /// Clips the convex polygon against a new half-plane boundary `bound_line`.
    /// If `is_upper = true`, the clipping is performed against the *upper* hull; otherwise against
    /// the lower bound line. The function searches for where the new boundary intersects the
    /// polygon’s current upper and lower chains, computes the intersection segments, and
    /// returns a `ClipOutcome`. In case of intersection, the returned `Clip` variant contains
    /// absolute indices into the chain arrays and the replacement border segments.
    fn clip(self: *const ConvexPolygon, bound_line: BorderLine, is_upper: bool) ClipOutcome {
        // 1) Locate intersection positions on both chains.
        const upper_chain_slice = self.upper_bound_lines.items[self.upper_bound_start..];
        const upper_rel_index = searchIntersection(
            upper_chain_slice,
            bound_line,
            false,
        );
        const upper_abs_index = self.upper_bound_start + upper_rel_index;

        const lower_chain_slice = self.lower_bound_lines.items[self.lower_bound_start..];
        const lower_rel_index_from_right = searchIntersection(
            lower_chain_slice,
            bound_line,
            true,
        );
        const lower_abs_index = self.lower_bound_start + lower_rel_index_from_right;

        // 2) Compute clipped segments at those positions.
        const new_upper_segment = self.upper_bound_lines.items[upper_abs_index]
            .computeIntersection(
            bound_line,
            is_upper,
        );
        const new_lower_segment = self.lower_bound_lines.items[lower_abs_index]
            .computeIntersection(
            bound_line,
            is_upper,
        );

        // 3) Package result (indices are absolute into their arrays).
        return .{ .Clip = .{
            .upper_index_offset = upper_rel_index,
            .lower_index_offset = lower_rel_index_from_right,
            .new_upper_segment = new_upper_segment,
            .new_lower_segment = new_lower_segment,
        } };
    }

    /// Attempts to clip the convex polygon with a new upper boundary line `new_upper_bound_line`.
    /// The candidate boundary is first narrowed to the polygon’s current valid x-axis domain
    /// (between the leftmost point of the upper chain and the rightmost point of the lower chain).
    /// Depending on relative positions, if `.NoClip` is returned if the new upper bound lies
    /// completely above the polygon. If `.Clip`, is returned if the new upper bound intersects the
    /// polygon in a valid way; in this case the actual split is delegated to the function clip.
    /// If `.Reject`, is returned if the new upper bound invalidates the polygon (removes all area).
    fn clipWithUpperBound(
        self: *const ConvexPolygon,
        new_upper_bound_line: BorderLine,
    ) ClipOutcome {
        // Guard: polygon must be initialized.
        if (self.isEmpty()) return .Reject;

        // Lower chain’s rightmost endpoint in (x, y).
        const lower_start_line = self.lower_bound_lines.items[self.lower_bound_start];
        const lower_right_point = lower_start_line.evaluateAtEnd();

        // Upper chain’s leftmost endpoint in (x, y).
        const upper_start_line = self.upper_bound_lines.items[self.upper_bound_start];
        const upper_left_point = upper_start_line.evaluateAtStart();

        // Narrow the candidate upper line to the polygon’s current x-axis domain [x_left, x_right].
        var narrowed_line = new_upper_bound_line;
        narrowed_line.x_axis_domain.start = upper_left_point.time;
        narrowed_line.x_axis_domain.end = lower_right_point.time;

        // Evaluate the narrowed line at both ends.
        const upper_left_eval = narrowed_line.evaluateAtStart().value;
        const upper_right_eval = narrowed_line.evaluateAtEnd().value;

        // 1) NoClip: line lies entirely above polygon’s lower-right corner, no clipping required.
        if (upper_right_eval >= lower_right_point.value) {
            return .NoClip;
        }

        // 2) Clip: line is above the polygon’s upper-left corner and below the lower-right corner.
        if ((upper_left_eval >= upper_left_point.value) and
            (upper_right_eval < lower_right_point.value))
        {
            // Perform actual split on both upper & lower chains with this upper bound.
            return self.clip(narrowed_line, true); // true = clip with upper bound.
        }

        // 3) Otherwise: boundary misses in a way that invalidates the polygon.
        return .Reject;
    }

    /// Attempts to clip the convex polygon with a new lower boundary line `new_lower_bound_line`.
    /// The candidate boundary is first narrowed to the polygon’s current valid x-axis domain
    /// (between the leftmost point of the upper chain and the rightmost point of the lower chain).
    /// Depending on relative positions, if `.NoClip` is returned if the new upper bound lies
    /// completely above the polygon. If `.Clip`, the new lower bound intersects the polygon in a
    /// valid way; the actual split is delegated to the function clip. If `.Reject`, the new lower
    /// bound misses in a way that would remove all area.
    fn clipWithLowerBound(
        self: *const ConvexPolygon,
        new_lower_bound_line: BorderLine,
    ) ClipOutcome {
        // Guard: polygon must be initialized.
        if (self.isEmpty()) return .Reject;

        // Lower chain’s rightmost endpoint in (x, y).
        const lower_start_line = self.lower_bound_lines.items[self.lower_bound_start];
        const lower_right_point = lower_start_line.evaluateAtEnd();

        // Upper chain’s leftmost endpoint in (x, y).
        const upper_start_line = self.upper_bound_lines.items[self.upper_bound_start];
        const upper_left_point = upper_start_line.evaluateAtStart();

        // Narrow the candidate lower line to the polygon’s current x-axis domain [x_left, x_right].
        var narrowed_line = new_lower_bound_line;
        narrowed_line.x_axis_domain.start = upper_left_point.time;
        narrowed_line.x_axis_domain.end = lower_right_point.time;

        // Evaluate the narrowed line at both ends of the domain.
        const left_eval = narrowed_line.evaluateAtStart().value;
        const right_eval = narrowed_line.evaluateAtEnd().value;

        // 1) NoClip: the new lower bound is at/below the polygon's upper-left corner.
        if (left_eval <= upper_left_point.value) {
            return .NoClip;
        }

        // 2) Clip: it starts above the upper-left corner but ends at/below the lower-right corner.
        if ((left_eval > upper_left_point.value) and (right_eval <= lower_right_point.value)) {
            return self.clip(narrowed_line, false); // false = clip with lower bound.
        }

        // 3) Otherwise: the bound misses in a way that invalidates the polygon.
        return .Reject;
    }

    /// Returns the upper-left corner point of the polygon (start of current upper bound line).
    fn upperLeft(self: *const ConvexPolygon) ContinousPoint {
        const upper_line = self.upper_bound_lines.items[self.upper_bound_start];
        return upper_line.evaluateAtStart();
    }

    /// Returns the lower-right corner point of the polygon (end of current lower bound line).
    fn lowerRight(self: *const ConvexPolygon) ContinousPoint {
        const lower_line = self.lower_bound_lines.items[self.lower_bound_start];
        return lower_line.evaluateAtEnd();
    }
};

/// Performs a binary search on a monotonic `border_line` segments to find the first segment
/// that is not strictly below the given `target` segment, according to the segment ordering.
/// The ordering is defined such that `a < b` if `b` lies strictly above `a` at both endpoints,
/// as determined by `isLessThan`. If `reversed` is false, searches from left to right; if true,
/// searches from right to left, effectively reversing the `border_line`'s direction for the search.
/// Returns the index of the first segment that is not strictly below `target`. If all segments are
/// strictly below `target`, returns the last index.
fn searchIntersection(
    border_line: []const BorderLine,
    target: BorderLine,
    reversed: bool,
) usize {
    // The algorithm assumes non-empty chains.
    std.debug.assert(border_line.len > 0);

    var lower: usize = 0;
    var higher: usize = border_line.len;
    const n: usize = border_line.len - 1;

    while (lower < higher) {
        const middle = lower + (higher - lower) / 2;
        const idx = if (!reversed) middle else n - middle; // map "from-right" index to forward index.
        if (border_line[idx].isLessThan(target)) {
            lower = middle + 1; // Go right.
        } else {
            higher = middle; // Keep mid (go left).
        }
    }

    const clamped_lower_bound = @min(n, lower);
    if (!reversed)
        return clamped_lower_bound;

    return n - clamped_lower_bound;
}

/// Helper function to adds a new point constraint to the convex polygon, representing the feasible
/// region for a linear function passing within `eps` of the point (`x_axis`, `y_axis`). Constructs
/// two half-plane boundaries (upper and lower) at the given x position, offset by `eps`, and
/// updates the polygon by intersecting with these constraints. Returns `true` if the polygon
/// remains non-empty after the update, `false` otherwise.
fn addPoint(poly: *ConvexPolygon, x_axis: usize, y_axis: f64, eps: f64) !bool {
    const slope = -@as(f64, @floatFromInt(x_axis)); // (-x_k)
    const upper_intercept = y_axis + eps;
    const lower_intercept = y_axis - eps;

    const neg_inf = -std.math.inf(f64);
    const pos_inf = std.math.inf(f64);

    const upper = BorderLine.init(slope, upper_intercept, neg_inf, pos_inf);
    const lower = BorderLine.init(slope, lower_intercept, neg_inf, pos_inf);

    return try poly.update(upper, lower);
}

test "convex polygon can update random linear sequences with slope break" {
    const allocator = testing.allocator;
    const random = tester.getDefaultRandomGenerator();

    // Create polygon.
    var poly = ConvexPolygon.init(allocator);
    defer poly.deinit();

    const epsilon = 0.8;

    const m1: f64 = random.float(f64) * 10;
    const b1: f64 = random.float(f64);
    // Generate points for first line (should intersect).
    for (0..20) |i| {
        const y = m1 * @as(f64, @floatFromInt(i)) + b1 + random.float(f64) * 0.1;
        const ok = try addPoint(&poly, i, y, epsilon);
        try std.testing.expect(ok);
    }

    // Now second line: slope -1, intercept 10
    const m2: f64 = -random.float(f64) * 10;
    const b2: f64 = -random.float(f64);

    var rejected = false;
    for (20..40) |i| {
        const y = m2 * @as(f64, @floatFromInt(i)) + b2 + random.float(f64) * 0.1;
        const ok = try addPoint(&poly, i, y, epsilon);
        if (!rejected) {
            rejected = true;
            try std.testing.expect(!ok);
            poly.clear();
        } else {
            try std.testing.expect(ok);
        }
    }

    try std.testing.expect(rejected); // we expect eventual rejection
}
