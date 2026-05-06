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

//! Geometric primitives for polygon clipping and feasibility tests used by
//! Mixed-Type PLA.

const std = @import("std");
const math = std.math;
const mem = std.mem;
const Allocator = mem.Allocator;
const ArrayList = std.ArrayList;
const testing = std.testing;

const shared_structs = @import("shared_structs.zig");
const shared_functions = @import("shared_functions.zig");

pub const ContinousPoint = shared_structs.ContinousPoint;

pub const LinearFunction = shared_structs.LinearFunction;

pub const ParameterSpacePoint = shared_structs.ParameterSpacePoint;

/// Marks whether a boundary chain is the upper or lower boundary of the feasible
/// parameter-space polygon; used to select geometric test directions.
pub const ChainType = enum { upper, lower };

/// Direction of a half-plane constraint induced by `(t, m +/- delta)`: keep points
/// above (`point_to_above`) or below (`point_to_below`) the boundary line.
pub const HalfplaneDirection = enum { point_to_above, point_to_below };

/// Result of intersecting a half-plane with a boundary chain: `contain_all` keeps
/// everything, `contain_some` trims to a smaller non-empty chain, `contain_none`
/// excludes all, and `non_exist` means the chain was already empty.
pub const ContainmentResult = enum { non_exist, contain_none, contain_some, contain_all };

/// Three-way point classification relative to a boundary: strictly inside
/// (`include`), strictly outside (`exclude`), or within tolerance (`touch`).
pub const PointRelation = enum { exclude, touch, include };

/// Selects which extremal polygon vertex to retrieve: right-most (upper chain)
/// or left-most (lower chain).
pub const EndmostSide = enum { right_most, left_most };

pub const sign_value_diff: f64 = 0.0000001;
pub const sign_time_diff: f64 = 0.0000001;
pub const max_uni: i64 = 222222222;

/// Dynamic value/time tolerances computed in `compress()` and threaded through
/// algorithm structs so geometric tests scale with data amplitude.
pub const Tolerances = struct {
    val: f64,
    time: f64,
};

/// Default tolerances for contexts that bypass dynamic initialization (e.g. tests).
const default_tols =
    Tolerances{ .val = sign_value_diff, .time = sign_time_diff };

/// Per-chain limits derived from algorithm parameters (`delta`, `eps`).
const ChainMeters = struct {
    size: i64,
    thr: f64,

    /// Compute chain meters from `delta` and `eps`: size is `ceil(4/eps)` when
    /// `eps > 0`, else `max_uni`; threshold is `eps * delta`.
    fn init(delta: f64, eps: f64) ChainMeters {
        const size: i64 = if (eps > 0)
            @intFromFloat(@ceil(4.0 / eps))
        else
            max_uni;
        return .{ .size = size, .thr = eps * delta };
    }
};

/// Half-plane constraint in `(k, b)` space defined by separating line `sep` and
/// feasible-side `direction`; consumed by chain/polygon intersection routines.
pub const Halfplane = struct {
    sep: LinearFunction,
    direction: HalfplaneDirection,

    /// Classify parameter-space point `p` relative to this half-plane.
    pub fn isInner(self: Halfplane, p: ParameterSpacePoint, tol: f64) PointRelation {
        return isInnerParamPoint(self.sep, p, self.direction, tol);
    }
};

/// Chain vertex in the convex polygon: wraps a `ParameterSpacePoint` plus a
/// `color` marker for intersection-generated vertices.
const Edge = struct {
    p: ParameterSpacePoint,
    color: bool,
};

/// Vertical error tube at one timestamp expanded by `delta`: `upper=(t,m+delta)`
/// and `lower=(t,m-delta)`. This is the core input unit for polygon clipping.
pub const ErrorTubeSegment = struct {
    upper: ContinousPoint,
    lower: ContinousPoint,

    /// Build a `ErrorTubeSegment` from one point and error bound `delta`.
    pub fn fromPointAndDelta(
        point: ContinousPoint,
        delta: f64,
    ) ErrorTubeSegment {
        return .{
            .upper = .{ .index = point.index, .value = point.value + delta },
            .lower = .{ .index = point.index, .value = point.value - delta },
        };
    }

    /// Create a zero-initialized `ErrorTubeSegment`.
    pub fn empty() ErrorTubeSegment {
        return .{
            .upper = .{ .index = 0.0, .value = 0.0 },
            .lower = .{ .index = 0.0, .value = 0.0 },
        };
    }

    /// Return segment midpoint value, used as a degenerate fallback.
    pub fn midpointValue(self: ErrorTubeSegment) f64 {
        return (self.upper.value + self.lower.value) / 2.0;
    }

    /// Return whether upper/lower timestamps coincide within `tols_time`.
    pub fn isVertical(self: ErrorTubeSegment, tolerances: Tolerances) bool {
        return @abs(self.upper.index - self.lower.index) < tolerances.time;
    }

    /// Intersect extreme line `exl` with this tube and write result to `dp`.
    /// For vertical tubes evaluate at tube time; otherwise intersect against the
    /// line through `(upper, lower)`.
    pub fn hittingLine(
        self: ErrorTubeSegment,
        dp: *ContinousPoint,
        exl: LinearFunction,
        tolerances: Tolerances,
    ) void {
        if (self.isVertical(tolerances)) {
            dp.* = .{
                .index = self.upper.index,
                .value = evaluateLinear(exl, self.upper.index),
            };
        } else {
            const lw = linearFromTwoPoints(self.upper, self.lower);
            if (@abs(lw.slope - exl.slope) > 1e-10) {
                const t = (exl.intercept - lw.intercept) /
                    (lw.slope - exl.slope);
                // Evaluate the extreme line at time t
                const m = exl.slope * t + exl.intercept;
                dp.* = .{ .index = t, .value = m };
            } else {
                dp.* = .{
                    .index = (self.upper.index + self.lower.index) / 2.0,
                    .value = (self.upper.value + self.lower.value) / 2.0,
                };
            }
        }
    }
};

/// Doubly-ended chain of polygon `Edge` vertices (upper or lower boundary) with a dedicated
/// `extreme_vertex`. `padBoundaryChain` and `cutBoundaryChain` perform incremental half-plane clipping
/// as new samples arrive.
pub const BoundaryChain = struct {
    chain_type: ChainType,
    edges: ArrayList(Edge),
    extreme_vertex: ParameterSpacePoint,
    has_extreme_vertex: bool,
    tolerances: Tolerances,
    allocator: Allocator,

    /// Create an empty `BoundaryChain` of the given type with no extreme vertex.
    pub fn create(allocator: Allocator, chain_type: ChainType, tolerances: Tolerances) BoundaryChain {
        return .{
            .chain_type = chain_type,
            .edges = ArrayList(Edge).empty,
            .extreme_vertex = .{ .x_axis = 0.0, .y_axis = 0.0 },
            .has_extreme_vertex = false,
            .tolerances = tolerances,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *BoundaryChain) void {
        self.edges.deinit(self.allocator);
    }

    /// Initialize chain with one or two start vertices plus extreme point. If `middle_vertex`
    /// is null only `start_vertex` is added.
    pub fn initializeChain(
        self: *BoundaryChain,
        start_vertex: ParameterSpacePoint,
        middle_vertex: ?ParameterSpacePoint,
        extreme_vertex: ParameterSpacePoint,
    ) Allocator.Error!void {
        self.edges.clearRetainingCapacity();
        try self.edges
            .append(self.allocator, .{
            .p = start_vertex,
            .color = false,
        });
        if (middle_vertex) |middle| {
            try self.edges
                .append(self.allocator, .{
                .p = middle,
                .color = false,
            });
        }
        self.extreme_vertex = extreme_vertex;
        self.has_extreme_vertex = true;
    }

    /// Overwrite the current extreme vertex.
    pub fn setExtremeVertex(self: *BoundaryChain, vertex: ParameterSpacePoint) void {
        self.extreme_vertex = vertex;
    }

    /// Return current extreme vertex.
    pub fn getExtremeVertex(self: BoundaryChain) ParameterSpacePoint {
        return self.extreme_vertex;
    }

    /// Return approximate chain size metric used to bound complexity.
    pub fn approximateStateSize(self: BoundaryChain) usize {
        const edge_count = self.edges.items.len;
        if (self.has_extreme_vertex) {
            return @intFromFloat(
                @as(f64, @floatFromInt(edge_count)) * 1.5 + 1.0 + 0.5,
            );
        } else {
            return @intFromFloat(
                @as(f64, @floatFromInt(edge_count)) * 1.5,
            );
        }
    }

    /// Insert a colored edge at the front; used for `loadVisibleRegionConstraints` intersection vertices that
    /// must precede existing edges.
    pub fn pushFrontIntersectionVertex(
        self: *BoundaryChain,
        front_vertex: ParameterSpacePoint,
    ) Allocator.Error!void {
        try self.edges.insert(self.allocator, 0, .{
            .p = front_vertex,
            .color = true,
        });
    }

    /// Intersect this chain with half-plane `h` by scanning from end-most  (`reverse=false`) or
    /// front (`reverse=true`), trimming excluded vertices and inserting/marking boundary
    ///  intersections. Returns containment status.
    pub fn padBoundaryChain(
        self: *BoundaryChain,
        half_plane: Halfplane,
        reverse: bool,
    ) Allocator.Error!ContainmentResult {
        if (!self.has_extreme_vertex) return .non_exist;

        if (!reverse) {
            // Normal case: search from extreme vertex inward.
            var back_vertex = self.extreme_vertex;
            const inner = half_plane.isInner(back_vertex, self.tolerances.val);

            if (inner != .exclude) return .contain_all;

            while (self.edges.items.len > 0) {
                const front_vertex = self.edges.items[
                    self.edges.items.len - 1
                ].p;
                const front_inner = half_plane.isInner(
                    front_vertex,
                    self.tolerances.val,
                );

                if (front_inner == .include) {
                    const current_edge = linearFromTwoPoints(
                        .{ .index = front_vertex.x_axis, .value = front_vertex.y_axis },
                        .{ .index = back_vertex.x_axis, .value = back_vertex.y_axis },
                    );
                    if (intersectTwoLines(current_edge, half_plane.sep)) |intersection_vertex| {
                        try self.edges.append(self.allocator, .{
                            .p = intersection_vertex,
                            .color = false,
                        });
                    }
                    return .contain_some;
                } else if (front_inner == .touch) {
                    self.edges.items[
                        self.edges.items.len - 1
                    ].color = true;
                    return .contain_some;
                } else {
                    // Update both the tracker and the actual chain boundary.
                    back_vertex = front_vertex;
                    self.extreme_vertex = front_vertex;
                    _ = self.edges.pop();
                }
            }

            self.has_extreme_vertex = false;
            return .contain_none;
        } else {
            // Reverse case: search from front inward.
            var front_vertex: ParameterSpacePoint = undefined;
            if (self.edges.items.len == 0) {
                front_vertex = self.extreme_vertex;
            } else {
                front_vertex = self.edges.items[0].p;
            }

            const inner = half_plane.isInner(
                front_vertex,
                self.tolerances.val,
            );
            if (inner != .exclude) return .contain_all;

            while (self.edges.items.len > 0) {
                var back_vertex: ParameterSpacePoint = undefined;
                if (self.edges.items.len > 1) {
                    back_vertex = self.edges.items[1].p;
                } else {
                    back_vertex = self.extreme_vertex;
                }

                const back_inner = half_plane.isInner(
                    back_vertex,
                    self.tolerances.val,
                );
                if (back_inner == .include) {
                    const cur_edge = linearFromTwoPoints(
                        .{ .index = front_vertex.x_axis, .value = front_vertex.y_axis },
                        .{ .index = back_vertex.x_axis, .value = back_vertex.y_axis },
                    );
                    if (intersectTwoLines(cur_edge, half_plane.sep)) |intersection_vertex| {
                        self.edges.items[0].p = intersection_vertex;
                    }
                    return .contain_some;
                } else if (back_inner == .touch) {
                    _ = self.edges.orderedRemove(0);
                    return .contain_some;
                } else {
                    _ = self.edges.orderedRemove(0);
                    front_vertex = back_vertex;
                }
            }

            self.has_extreme_vertex = false;
            return .contain_none;
        }
    }

    /// Trim excluded vertices from the opposite end of `padBoundaryChain` and return the
    /// first surviving/boundary vertex, or `null` if none survives. `reverse`
    /// selects direction (used by `loadVisibleRegionConstraints`).
    pub fn cutBoundaryChain(
        self: *BoundaryChain,
        half_plane: Halfplane,
        reverse: bool,
    ) ?ParameterSpacePoint {
        if (!self.has_extreme_vertex) return null;

        if (!reverse) {
            // Normal case: search from front inward.
            var front_vertex: ParameterSpacePoint = undefined;
            if (self.edges.items.len > 0) {
                front_vertex = self.edges.items[0].p;
            } else {
                front_vertex = self.extreme_vertex;
            }

            const inner = half_plane.isInner(front_vertex, self.tolerances.val);
            if (inner != .exclude) return null;

            while (self.edges.items.len > 0) {
                var back_vertex: ParameterSpacePoint = undefined;
                if (self.edges.items.len > 1) {
                    back_vertex = self.edges.items[1].p;
                } else {
                    back_vertex = self.extreme_vertex;
                }

                const back_inner = half_plane.isInner(back_vertex, self.tolerances.val);
                if (back_inner == .exclude) {
                    _ = self.edges.orderedRemove(0);
                    front_vertex = back_vertex;
                } else if (back_inner == .touch) {
                    _ = self.edges.orderedRemove(0);
                    return back_vertex;
                } else {
                    // include: compute intersection.
                    const cur_edge = linearFromTwoPoints(
                        .{ .index = front_vertex.x_axis, .value = front_vertex.y_axis },
                        .{ .index = back_vertex.x_axis, .value = back_vertex.y_axis },
                    );
                    if (intersectTwoLines(cur_edge, half_plane.sep)) |intersection_vertex| {
                        self.edges.items[0].p = intersection_vertex;
                        return intersection_vertex;
                    }
                    return front_vertex;
                }
            }

            self.has_extreme_vertex = false;
            return null;
        } else {
            // Reverse case: search from extreme vertex inward.
            var back_vertex = self.extreme_vertex;
            const inner = half_plane.isInner(back_vertex, self.tolerances.val);

            if (inner != .exclude) return null;

            while (self.edges.items.len > 0) {
                const front_vertex = self.edges.items[
                    self.edges.items.len - 1
                ].p;
                const front_inner = half_plane.isInner(front_vertex, self.tolerances.val);

                if (front_inner == .include) {
                    const cur_edge = linearFromTwoPoints(
                        .{ .index = front_vertex.x_axis, .value = front_vertex.y_axis },
                        .{ .index = back_vertex.x_axis, .value = back_vertex.y_axis },
                    );
                    if (intersectTwoLines(cur_edge, half_plane.sep)) |intersection_vertex| {
                        // Mutate both local tracker and actual chain state.
                        back_vertex = intersection_vertex;
                        self.extreme_vertex = intersection_vertex;
                    }
                    return back_vertex;
                } else if (front_inner == .touch) {
                    // Mutate both local tracker and actual chain state.
                    back_vertex = front_vertex;
                    self.extreme_vertex = front_vertex;
                    _ = self.edges.pop();
                    return back_vertex;
                } else {
                    // Mutate both local tracker and actual chain state.
                    back_vertex = front_vertex;
                    self.extreme_vertex = front_vertex;
                    _ = self.edges.pop();
                }
            }

            self.has_extreme_vertex = false;
            return null;
        }
    }

    /// Deep-copy this chain (including all edges).
    pub fn cloneBoundaryChain(self: BoundaryChain, allocator: Allocator) Allocator.Error!BoundaryChain {
        var new_chain = BoundaryChain{
            .chain_type = self.chain_type,
            .edges = ArrayList(Edge).empty,
            .extreme_vertex = self.extreme_vertex,
            .has_extreme_vertex = self.has_extreme_vertex,
            .tolerances = self.tolerances,
            .allocator = allocator,
        };
        try new_chain.edges.appendSlice(allocator, self.edges.items);
        return new_chain;
    }
};

/// Convex feasible polygon in `(k, b)` parameter space, represented by upper and lower
/// `BoundaryChain` boundaries. Stores admissible `(slope, intercept)` pairs shrunk
/// incrementally by half-plane intersections from incoming data samples.
pub const ExtendedPolygon = struct {
    chain_meters: ChainMeters,
    upper_edges: BoundaryChain,
    lower_edges: BoundaryChain,
    instantiated: bool,
    tolerances: Tolerances,

    /// Create an uninstantiated polygon with error bound `delta`, size tolerance `eps`,
    /// and dynamic `tolerances`. Neither chain holds vertices until `initializePolygon`
    /// or `reInitializePolygon` is called.
    pub fn create(
        allocator: Allocator,
        delta: f64,
        eps: f64,
        tols: Tolerances,
    ) ExtendedPolygon {
        return .{
            .chain_meters = ChainMeters
                .init(delta, eps),
            .upper_edges = BoundaryChain
                .create(allocator, .upper, tols),
            .lower_edges = BoundaryChain
                .create(allocator, .lower, tols),
            .instantiated = false,
            .tolerances = tols,
        };
    }

    pub fn deinit(self: *ExtendedPolygon) void {
        self.upper_edges.deinit();
        self.lower_edges.deinit();
    }

    /// Initialize the polygon from `first_point` and `second_point`, each expanded by `delta`.
    /// In dual `(k, b)` space each tube boundary maps to a line; their four pairwise intersections
    /// `left_middle`, `right_middle`, `middle_top`, `middle_bottom` form the initial
    /// quadrilateral. Distinct timestamps guarantee non-parallel lines, so intersections always exist.
    pub fn initializePolygon(
        self: *ExtendedPolygon,
        first_point: ContinousPoint,
        second_point: ContinousPoint,
        delta: f64,
    ) Allocator.Error!void {
        self.instantiated = true;

        const first_upper_line =
            LinearFunction{
                .slope = -first_point.index,
                .intercept = first_point.value + delta,
            };
        const first_lower_line =
            LinearFunction{
                .slope = -first_point.index,
                .intercept = first_point.value - delta,
            };
        const second_upper_line =
            LinearFunction{
                .slope = -second_point.index,
                .intercept = second_point.value + delta,
            };
        const second_lower_line =
            LinearFunction{
                .slope = -second_point.index,
                .intercept = second_point.value - delta,
            };

        // Intersections succeed for distinct timestamps because slopes differ.
        const left_middle =
            intersectTwoLines(first_upper_line, second_lower_line) orelse unreachable;
        const right_middle =
            intersectTwoLines(first_lower_line, second_upper_line) orelse unreachable;
        const middle_top =
            intersectTwoLines(first_upper_line, second_upper_line) orelse unreachable;
        const middle_bottom =
            intersectTwoLines(first_lower_line, second_lower_line) orelse unreachable;

        try self.upper_edges
            .initializeChain(
            left_middle,
            middle_top,
            right_middle,
        );
        try self.lower_edges
            .initializeChain(
            right_middle,
            middle_bottom,
            left_middle,
        );
    }

    /// Intersect `first_line` and `second_line` in parameter space. When the lines are
    /// parallel — which occurs when `limiting_segment` and `current_segment` share a
    /// timestamp, making their dual lines have identical slopes -`intersectTwoLines` returns
    /// `null`. Defaulting to `(0, 0)` in that case would inject a zero constraint and instantly
    /// destroy the feasible polygon for extreme-valued data. Instead, the intersection is capped
    /// at `fallback_x_axis` (±`max_coordinate`) and `second_line` is evaluated there, preserving
    /// the open tube of the feasible region.
    pub fn computeSafeIntersection(
        first_line: LinearFunction,
        second_line: LinearFunction,
        fallback_x_axis: f64,
    ) ParameterSpacePoint {
        if (intersectTwoLines(first_line, second_line)) |p| {
            return p;
        }
        // Parallel lines: no finite intersection; cap at the infinity sentinel.
        return .{
            .x_axis = fallback_x_axis,
            .y_axis = evaluateLinear(second_line, fallback_x_axis),
        };
    }

    /// Reinitialize the polygon for a new fitting round from `limiting_segment` (the boundary
    /// carried over from the previous round) and `current_segment` (the first sample of the
    /// new round). Each tube boundary at timestamp `t` dualises to a line with slope
    /// `time_base - t`. The four polygon corners are the pairwise intersections of the
    /// upper/lower half-planes of `limiting_segment` and `current_segment`.
    /// Three degenerate cases require special handling. If
    /// `limiting_segment.upper.index ≈ current_segment.upper.index`, the dual lines on the left
    /// side are parallel and sentinels push `left_top` and `left_bottom` to −inf. If
    /// `limiting_segment.lower.index ≈ current_segment.upper.index`, the dual lines on the right
    /// side are parallel and sentinels push `right_top` and `right_bottom` to +inf.
    /// Crossing corners are also handled explicitly: when
    /// `limiting_segment.upper.index > limiting_segment.lower.index` or the reverse, the bounding
    /// lines may cross and the polygon degenerates to a triangle; a fully collapsed polygon
    /// (`is_polygon_empty`) is resolved via `closed_direction`.
    pub fn reInitializePolygon(
        self: *ExtendedPolygon,
        limiting_segment: ErrorTubeSegment,
        current_segment: ErrorTubeSegment,
        time_base: f64,
        closed_direction: ChainType,
    ) Allocator.Error!void {
        self.instantiated = true;

        // Slope `time_base - t` anchors each constraint line at the new round's time origin.
        const current_upper_halfplane = Halfplane{
            .sep = .{
                .slope = time_base - current_segment.upper.index,
                .intercept = current_segment.upper.value,
            },
            .direction = .point_to_below,
        };
        const current_lower_halfplane = Halfplane{
            .sep = .{
                .slope = time_base - current_segment.lower.index,
                .intercept = current_segment.lower.value,
            },
            .direction = .point_to_above,
        };
        const limiting_upper_halfplane = Halfplane{
            .sep = .{
                .slope = time_base - limiting_segment.upper.index,
                .intercept = limiting_segment.upper.value,
            },
            .direction = .point_to_below,
        };
        const limiting_lower_halfplane = Halfplane{
            .sep = .{
                .slope = time_base - limiting_segment.lower.index,
                .intercept = limiting_segment.lower.value,
            },
            .direction = .point_to_above,
        };

        var left_top = ParameterSpacePoint{ .x_axis = 0.0, .y_axis = 0.0 };
        var left_bottom = ParameterSpacePoint{ .x_axis = 0.0, .y_axis = 0.0 };
        var right_top = ParameterSpacePoint{ .x_axis = 0.0, .y_axis = 0.0 };
        var right_bottom = ParameterSpacePoint{ .x_axis = 0.0, .y_axis = 0.0 };

        // `max_coordinate` acts as ±∞ in parameter space. Using a finite sentinel rather than
        // `math.inf` avoids IEEE edge cases in downstream chain operations.
        const max_coordinate: f64 = 100000000.0;
        if (@abs(limiting_segment.upper.index - current_segment.upper.index) < 1e-10) {
            // `limiting_segment.upper` and `current_segment.upper` share a timestamp; their dual
            // lines are parallel. Left corners are pushed to the -inf sentinel to keep the polygon
            // open on the left. Right corners are computed normally from `limiting_lower`.
            left_top = .{
                .x_axis = -max_coordinate,
                .y_axis = current_segment.upper.value,
            };
            left_bottom = .{
                .x_axis = -10.0 * max_coordinate,
                .y_axis = current_segment.lower.value,
            };
            right_top =
                computeSafeIntersection(
                    limiting_lower_halfplane.sep,
                    current_upper_halfplane.sep,
                    max_coordinate,
                );
            right_bottom =
                computeSafeIntersection(
                    limiting_lower_halfplane.sep,
                    current_lower_halfplane.sep,
                    max_coordinate,
                );
        } else if (@abs(limiting_segment.lower.index - current_segment.upper.index) < 1e-10) {
            // `limiting_segment.lower` and `current_segment.upper` share a timestamp; their dual
            // lines are parallel. Right corners are pushed to the +inf sentinel. Left corners are
            // computed normally from `limiting_upper`.
            left_top =
                computeSafeIntersection(
                    limiting_upper_halfplane.sep,
                    current_upper_halfplane.sep,
                    -max_coordinate,
                );
            left_bottom =
                computeSafeIntersection(
                    limiting_upper_halfplane.sep,
                    current_lower_halfplane.sep,
                    -max_coordinate,
                );
            right_top = .{ .x_axis = 10.0 * max_coordinate, .y_axis = current_segment.upper.value };
            right_bottom = .{ .x_axis = max_coordinate, .y_axis = current_segment.lower.value };
        } else {
            // General case: no coincident timestamps, so all four corners have finite intersections.
            left_top =
                computeSafeIntersection(
                    limiting_upper_halfplane.sep,
                    current_upper_halfplane.sep,
                    -max_coordinate,
                );
            left_bottom =
                computeSafeIntersection(
                    limiting_upper_halfplane.sep,
                    current_lower_halfplane.sep,
                    -max_coordinate,
                );
            right_top =
                computeSafeIntersection(
                    limiting_lower_halfplane.sep,
                    current_upper_halfplane.sep,
                    max_coordinate,
                );
            right_bottom =
                computeSafeIntersection(
                    limiting_lower_halfplane.sep,
                    current_lower_halfplane.sep,
                    max_coordinate,
                );

            var is_polygon_empty = false;

            if (limiting_segment.upper.index > limiting_segment.lower.index) {
                // Upper boundary timestamp is later than lower boundary timestamp.
                // The two bounding lines may cross, turning the quadrilateral into a triangle.
                if (left_top.x_axis >= right_top.x_axis and left_bottom.x_axis <= right_bottom.x_axis) {
                    // Top corners crossed; recompute `right_top` as the self-intersection of
                    // the two limiting half-planes and build a degenerate triangle on top.
                    right_top =
                        computeSafeIntersection(
                            limiting_upper_halfplane.sep,
                            limiting_lower_halfplane.sep,
                            max_coordinate,
                        );
                    try self.upper_edges
                        .initializeChain(left_bottom, null, right_top);
                    try self.lower_edges
                        .initializeChain(right_top, right_bottom, left_bottom);
                    return;
                } else if (left_bottom.x_axis > right_bottom.x_axis) {
                    // Both corner pairs crossed: polygon fully collapsed.
                    is_polygon_empty = true;
                }
            } else if (limiting_segment.upper.index < limiting_segment.lower.index) {
                // Lower boundary timestamp is later; symmetric degenerate case on the bottom.
                if (left_bottom.x_axis >= right_bottom.x_axis and left_top.x_axis <= right_top.x_axis) {
                    // Bottom corners crossed; recompute `left_bottom` as the self-intersection
                    // of the two limiting half-planes and build a degenerate triangle on bottom.
                    left_bottom =
                        computeSafeIntersection(
                            limiting_upper_halfplane.sep,
                            limiting_lower_halfplane.sep,
                            -max_coordinate,
                        );
                    try self.upper_edges
                        .initializeChain(left_bottom, left_top, right_top);
                    try self.lower_edges
                        .initializeChain(right_top, null, left_bottom);
                    return;
                } else if (left_top.x_axis > right_top.x_axis) {
                    // Both corner pairs crossed: polygon fully collapsed.
                    is_polygon_empty = true;
                }
            }

            if (is_polygon_empty) {
                // Polygon collapsed entirely. Which degenerate side survives is determined by
                // `closed_direction`: the chain boundary that terminated the previous round.
                // `.lower` closed → collapse right side onto left; `.upper` closed → vice versa.
                if (closed_direction == .lower) {
                    right_top = left_top;
                    right_bottom = left_bottom;
                } else {
                    left_top = right_top;
                    left_bottom = right_bottom;
                }
            }
        }

        // Standard quadrilateral: upper chain [left_bottom, left_top, right_top],
        // lower chain [right_top, right_bottom, left_bottom].
        try self.upper_edges
            .initializeChain(left_bottom, left_top, right_top);
        try self.lower_edges
            .initializeChain(right_top, right_bottom, left_bottom);
    }

    /// Return approximate polygon size metric (sum of chain sizes), or 0 if
    /// uninstantiated.
    pub fn approximateSize(self: ExtendedPolygon) usize {
        if (self.instantiated) {
            return self.upper_edges.approximateStateSize() +
                self.lower_edges.approximateStateSize();
        }
        return 0;
    }

    /// Return endmost vertex from upper (`right_most`) or lower (`left_most`) chain.
    pub fn getEndmostVertex(
        self: ExtendedPolygon,
        side: EndmostSide,
    ) ParameterSpacePoint {
        return switch (side) {
            .right_most => self.upper_edges.getExtremeVertex(),
            .left_most => self.lower_edges.getExtremeVertex(),
        };
    }

    /// Overwrite the endmost vertices on both chains with `upper_point` and `lower_point`.
    pub fn setEndmostVertices(
        self: *ExtendedPolygon,
        upper_point: ParameterSpacePoint,
        lower_point: ParameterSpacePoint,
    ) void {
        self.upper_edges.setExtremeVertex(upper_point);
        self.lower_edges.setExtremeVertex(lower_point);
    }

    pub fn isInstantiated(self: ExtendedPolygon) bool {
        return self.instantiated;
    }

    pub fn setUninstantiated(self: *ExtendedPolygon) void {
        self.instantiated = false;
    }

    /// Select the best current solution line from the feasible polygon. If uninstantiated,
    /// returns a horizontal line at `current_boundary_value`. Otherwise picks the endmost
    /// vertex with the smaller absolute slope, minimising approximation excursion.
    pub fn selectSolution(
        self: ExtendedPolygon,
        shift: f64,
        curb: f64,
    ) LinearFunction {
        if (!self.instantiated) {
            return .{ .slope = 0.0, .intercept = curb };
        }
        const up = self.upper_edges.getExtremeVertex();
        const lp = self.lower_edges.getExtremeVertex();
        if (@abs(up.x_axis) > @abs(lp.x_axis)) {
            return linearFromParamPoint(lp, shift);
        } else {
            return linearFromParamPoint(up, shift);
        }
    }

    /// Intersect polygon with half-plane `h`: apply `padBoundaryChain` to one chain and
    /// `cutBoundaryChain` to the other, then update endmost from cut result when present.
    /// Returns containment from the `padBoundaryChain` side.
    pub fn intersect(
        self: *ExtendedPolygon,
        half_plane: Halfplane,
    ) Allocator.Error!ContainmentResult {
        if (half_plane.direction == .point_to_below) {
            const relationship =
                try self.upper_edges.padBoundaryChain(half_plane, false);
            const end = self.lower_edges
                .cutBoundaryChain(half_plane, false);
            if (end) |e| self.upper_edges
                .setExtremeVertex(e);
            return relationship;
        } else {
            const relationship = try self.lower_edges
                .padBoundaryChain(half_plane, false);
            const end = self.upper_edges
                .cutBoundaryChain(half_plane, false);
            if (end) |e| self.lower_edges
                .setExtremeVertex(e);
            return relationship;
        }
    }

    /// Load visible region constraints during restart: reverse-direction variant of `intersect`
    /// that prepends new intersection vertices via `pushFrontIntersectionVertex`.
    pub fn loadVisibleRegionConstraints(
        self: *ExtendedPolygon,
        half_plane: Halfplane,
    ) Allocator.Error!ContainmentResult {
        if (half_plane.direction == .point_to_below) {
            const relationship =
                try self.upper_edges.padBoundaryChain(half_plane, true);
            const new_start = self.lower_edges
                .cutBoundaryChain(half_plane, true);
            if (new_start) |start| try self.upper_edges
                .pushFrontIntersectionVertex(start);
            return relationship;
        } else {
            const relationship = try self.lower_edges
                .padBoundaryChain(half_plane, true);
            const new_start = self.upper_edges
                .cutBoundaryChain(half_plane, true);
            if (new_start) |start| try self.lower_edges
                .pushFrontIntersectionVertex(start);
            return relationship;
        }
    }

    /// Deep-copy this polygon, including both boundary chains.
    pub fn cloneExtendedPolygon(self: ExtendedPolygon, allocator: Allocator) Allocator.Error!ExtendedPolygon {
        return .{
            .chain_meters = self.chain_meters,
            .upper_edges = try self.upper_edges.cloneBoundaryChain(allocator),
            .lower_edges = try self.lower_edges.cloneBoundaryChain(allocator),
            .instantiated = self.instantiated,
            .tolerances = self.tolerances,
        };
    }
};

/// Incremental boundary chain that tracks the extremal boundary of the feasible region across
/// multiple data samples. Unlike `BoundaryChain`, which performs polygon clipping, `VisibleRegionChain`
/// maintains a list of `ContinousPoint` `vertices` and a `reference_line`. As new data points arrive
/// via `updateVisibleRegion`, dominated vertices are popped to restore the boundary invariant.
pub const VisibleRegionChain = struct {
    chain_type: ChainType,
    chain_meters: ChainMeters,
    vertices: ArrayList(ContinousPoint),
    reference_line: LinearFunction,
    tolerances: Tolerances,
    allocator: Allocator,

    /// Create a new `VisibleRegionChain` of the given `chain_type` with error bound
    /// `delta`, size tolerance `eps`, and dynamic `tolerances`.
    pub fn create(
        allocator: Allocator,
        chain_type: ChainType,
        delta: f64,
        eps: f64,
        tolerances: Tolerances,
    ) VisibleRegionChain {
        return .{
            .chain_type = chain_type,
            .chain_meters = ChainMeters.init(delta, eps),
            .vertices = ArrayList(ContinousPoint).empty,
            .reference_line = .{ .slope = 0.0, .intercept = 0.0 },
            .tolerances = tolerances,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *VisibleRegionChain) void {
        self.vertices.deinit(self.allocator);
    }

    /// Reset the chain from `seed_dual_point` and `seed_data_point`. Clears all
    /// existing vertices, rebuilds `reference_line` from `seed_dual_point` and
    /// `shift_time`, then calls `updateVisibleRegion` to seed the first vertex.
    pub fn resetFromSeedPoint(
        self: *VisibleRegionChain,
        seed_dual_point: ParameterSpacePoint,
        seed_data_point: ContinousPoint,
        shift_time: f64,
    ) void {
        self.vertices.clearRetainingCapacity();
        self.reference_line = linearFromParamPoint(seed_dual_point, shift_time);
        self.updateVisibleRegion(seed_data_point);
    }
    /// Add `new_point` to the chain, maintaining the visible region invariant.
    /// `adjusted_point` is `new_point` shifted by `chain_limits.threshold`:
    /// down for upper chains, up for lower chains. Dominated vertices are then
    /// popped from the back until the boundary invariant is restored. When the chain is
    /// reduced to fewer than two vertices, `adjusted_point` is checked against
    /// `adjusted_reference_line`; it is appended, replaces the front vertex,
    /// or triggers `unreachable` if it falls outside the feasible region.
    pub fn updateVisibleRegion(self: *VisibleRegionChain, new_point: ContinousPoint) void {
        var adjusted_point: ContinousPoint = undefined;
        if (self.chain_type == .upper) {
            adjusted_point = .{
                .index = new_point.index,
                .value = new_point.value - self.chain_meters.thr,
            };
        } else {
            adjusted_point = .{
                .index = new_point.index,
                .value = new_point.value + self.chain_meters.thr,
            };
        }

        if (self.vertices.items.len == 0) {
            self.vertices.append(self.allocator, adjusted_point) catch unreachable;
            return;
        }

        // Pop dominated vertices. `edge_line` is the visible region edge between `prev_vertex` and
        // `last_vertex`; `boundary_direction` is `.point_to_above` for upper chains so that
        // points below the edge are dominated and removed.
        while (self.vertices.items.len >= 2) {
            const last_vertex = self.vertices.items[self.vertices.items.len - 1];
            const prev_vertex = self.vertices.items[self.vertices.items.len - 2];

            const edge_line = linearFromTwoPoints(prev_vertex, last_vertex);
            const boundary_direction: HalfplaneDirection = if (self.chain_type == .upper)
                .point_to_above
            else
                .point_to_below;

            const vertex_relation = isInnerParamPoint(
                edge_line,
                .{ .x_axis = adjusted_point.index, .y_axis = adjusted_point.value },
                boundary_direction,
                self.tolerances.val,
            );

            if (vertex_relation == .include) {
                self.vertices.append(self.allocator, adjusted_point) catch unreachable;
                return;
            } else {
                _ = self.vertices.pop();
            }
        }

        // Fewer than two vertices remain: check `adjusted_point` against
        // `adjusted_reference_line` (the seed line shifted by `chain_limits.threshold`).
        var ln_copy = self.reference_line;
        if (self.chain_type == .upper) {
            ln_copy.intercept -= self.chain_meters.thr;
        } else {
            ln_copy.intercept += self.chain_meters.thr;
        }

        // `reference_direction` mirrors `boundary_direction`: upper chains test above,
        // lower chains test below.
        const point_dir: HalfplaneDirection = if (self.chain_type == .upper)
            .point_to_above
        else
            .point_to_below;

        const rel =
            isInnerPoint(ln_copy, adjusted_point, point_dir, self.tolerances.val);
        if (rel == .include) {
            self.vertices.append(self.allocator, adjusted_point) catch unreachable;
        } else if (rel == .touch) {
            self.vertices.append(self.allocator, adjusted_point) catch unreachable;
            if (self.vertices.items.len > 0) {
                _ = self.vertices.orderedRemove(0);
            }
        } else {
            unreachable;
        }
    }
    /// Add `new_point` to the chain. Delegates to `updateVisibleRegion` and
    /// returns `true` unconditionally.
    pub fn addPoint(self: *VisibleRegionChain, new_point: ContinousPoint) bool {
        self.updateVisibleRegion(new_point);
        return true;
    }

    /// Remove and return the last vertex from the chain, or `null` if the
    /// chain is empty.
    pub fn popBack(self: *VisibleRegionChain) ?ContinousPoint {
        if (self.vertices.items.len == 0) return null;
        return self.vertices.pop();
    }

    /// Return the first vertex of the chain, or `null` if the chain is empty.
    pub fn front(self: VisibleRegionChain) ?ContinousPoint {
        if (self.vertices.items.len == 0) return null;
        return self.vertices.items[0];
    }

    /// Return the `reference_line` (the extremal line seeded at `resetFromSeedPoint`).
    pub fn getExtremeLine(self: VisibleRegionChain) LinearFunction {
        return self.reference_line;
    }

    /// Return an approximate size metric for this chain.
    // fn ApproximateStateSize(self: VisibleRegionChain) usize {
    //     if (self.vertices.items.len == 0) return 1;
    //     return @intFromFloat(
    //         @as(f64, @floatFromInt(self.vertices.items.len)) * 1.5 + 1.0,
    //     );
    // }

    /// Clear all vertices from the chain.
    pub fn clear(self: *VisibleRegionChain) void {
        self.vertices.clearRetainingCapacity();
    }

    /// Return the `HalfplaneDirection` for this chain's type: upper chains
    /// produce `.point_to_below` constraints; lower chains produce `.point_to_above`.
    pub fn pointToDirection(self: VisibleRegionChain) HalfplaneDirection {
        return if (self.chain_type == .upper)
            .point_to_below
        else
            .point_to_above;
    }

    /// Deep-copy this `VisibleRegionChain`, including all `vertices`.
    pub fn cloneVisibleRegionChain(
        self: VisibleRegionChain,
        allocator: Allocator,
    ) Allocator.Error!VisibleRegionChain {
        var new_chain = VisibleRegionChain{
            .chain_type = self.chain_type,
            .chain_meters = self.chain_meters,
            .vertices = ArrayList(ContinousPoint).empty,
            .reference_line = self.reference_line,
            .tolerances = self.tolerances,
            .allocator = allocator,
        };
        try new_chain.vertices.appendSlice(allocator, self.vertices.items);
        return new_chain;
    }
};

/// Classify `val` as `.include`, `.exclude`, or `.touch` using tolerance `tol`.
fn judgeValueByTwoSides(val: f64, tol: f64) PointRelation {
    if (val > tol) return .include;
    if (val < -tol) return .exclude;
    return .touch;
}

/// Evaluate linear function `f` at `x` as `f.slope * x + f.intercept`.
pub fn evaluateLinear(f: LinearFunction, x: f64) f64 {
    return f.slope * x + f.intercept;
}

/// Build the unique line through `p1` and `p2`; if timestamps are near-identical
/// (`1e-10`), return a horizontal line through `p1.value`.
pub fn linearFromTwoPoints(p1: ContinousPoint, p2: ContinousPoint) LinearFunction {
    if (@abs(p1.index - p2.index) < 1e-10) {
        return .{ .slope = 0.0, .intercept = p1.value };
    }
    const slope = (p1.value - p2.value) / (p1.index - p2.index);
    const intercept = p1.value - slope * p1.index;
    return .{ .slope = slope, .intercept = intercept };
}
/// Build a line through point `p` with given `slope`.
fn linearFromPointAndSlope(p: ContinousPoint, slope: f64) LinearFunction {
    return .{ .slope = slope, .intercept = p.value - slope * p.index };
}

/// Convert parameter-space point `(x, y)` plus `shift` into primal line
/// `(slope=x, intercept=y - x * shift).
pub fn linearFromParamPoint(
    pp: ParameterSpacePoint,
    shift: f64,
) LinearFunction {
    return .{
        .slope = pp.x_axis,
        .intercept = pp.y_axis - pp.x_axis * shift,
    };
}

/// Compute the parameter-space intersection of two lines; returns `null` for
///  near-parallel lines (`|slope diff| < 1e-10`).
pub fn intersectTwoLines(
    l1: LinearFunction,
    l2: LinearFunction,
) ?ParameterSpacePoint {
    if (@abs(l1.slope - l2.slope) < 1e-10) return null;

    const x = (l2.intercept - l1.intercept) / (l1.slope - l2.slope);
    // Evaluate y from the first line instead of cross-multiplying
    const y = l1.slope * x + l1.intercept;

    return .{
        .x_axis = x,
        .y_axis = y,
    };
}

/// Classify a point against half-plane (`line`, `direction`) as inside,
/// outside, or touching using tolerance `tol`.
fn isInnerPoint(
    line: LinearFunction,
    p: ContinousPoint,
    direction: HalfplaneDirection,
    tol: f64,
) PointRelation {
    var diff = evaluateLinear(line, p.index);
    if (direction == .point_to_above) {
        diff = p.value - diff;
    } else {
        diff = diff - p.value;
    }
    return judgeValueByTwoSides(diff, tol);
}

/// Parameter-space version of `isInnerPoint` using `p.x_axis` for evaluation
/// and `p.y_axis` for comparison.
fn isInnerParamPoint(
    line: LinearFunction,
    p: ParameterSpacePoint,
    direction: HalfplaneDirection,
    tol: f64,
) PointRelation {
    var diff = evaluateLinear(line, p.x_axis);
    if (direction == .point_to_above) {
        diff = p.y_axis - diff;
    } else {
        diff = diff - p.y_axis;
    }
    return judgeValueByTwoSides(diff, tol);
}

test "ExtendedPolygon maintains feasibility for collinear points" {
    const allocator = std.testing.allocator;
    const tolerances = Tolerances{ .val = sign_value_diff, .time = sign_time_diff };
    const delta = 0.5;

    // Create polygon.
    var poly = ExtendedPolygon.create(allocator, delta, 0.0, tolerances);
    defer poly.deinit();

    // Initialize with two points on the line y = 2x
    const p1 = ContinousPoint{ .index = 1.0, .value = 2.0 };
    const p2 = ContinousPoint{ .index = 2.0, .value = 4.0 };
    try poly.initializePolygon(p1, p2, delta);
    try std.testing.expect(poly.isInstantiated());

    // Add a 3rd point on the same line (y = 2x)
    const p3 = ContinousPoint{ .index = 3.0, .value = 6.0 };

    // Test upper constraint (Slope must be -t, intercept must include +delta).
    const upper_hp = Halfplane{
        .sep = .{ .slope = -p3.index, .intercept = p3.value + delta },
        .direction = .point_to_below,
    };
    const upper_result = try poly.intersect(upper_hp);
    try std.testing.expect(upper_result != .contain_none and upper_result != .non_exist);

    // Test lower constraint (Slope must be -t, intercept must include -delta).
    const lower_hp = Halfplane{
        .sep = .{ .slope = -p3.index, .intercept = p3.value - delta },
        .direction = .point_to_above,
    };
    const lower_result = try poly.intersect(lower_hp);
    try std.testing.expect(lower_result != .contain_none and lower_result != .non_exist);
}

test "ExtendedPolygon rejects points after a severe slope break" {
    const allocator = std.testing.allocator;
    const tolerances = Tolerances{ .val = sign_value_diff, .time = sign_time_diff };
    const delta = 0.5;

    // Create polygon.
    var poly = ExtendedPolygon.create(allocator, delta, 0.0, tolerances);
    defer poly.deinit();

    // Initialize with points from a positive slope (y = x)
    const p1 = ContinousPoint{ .index = 0.0, .value = 0.0 };
    const p2 = ContinousPoint{ .index = 1.0, .value = 1.0 };
    try poly.initializePolygon(p1, p2, delta);

    var rejected = false;

    // Feed points from a completely different line (y = -5x + 20).
    for (2..10) |i| {
        const x: f64 = @floatFromInt(i);
        const y: f64 = -5.0 * x + 20.0;

        const upper_hp = Halfplane{
            .sep = .{ .slope = -x, .intercept = y + delta },
            .direction = .point_to_below,
        };
        const upper_result = try poly.intersect(upper_hp);

        const lower_hp = Halfplane{
            .sep = .{ .slope = -x, .intercept = y - delta },
            .direction = .point_to_above,
        };
        const lower_result = try poly.intersect(lower_hp);

        // If either boundary causes the polygon to empty, the segment terminates.
        if (upper_result == .contain_none or lower_result == .contain_none) {
            rejected = true;
            break;
        }
    }

    try std.testing.expect(rejected);
}
