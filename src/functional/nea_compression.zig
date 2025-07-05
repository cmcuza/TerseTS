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

//! Implementation of "NeaTS" (Nonlinear error-bounded approximation for Time Series) compression from the paper:
//! "Andrea Guerra, Giorgio Vinciguerra, Antonio Boffa and Paolo Ferragina.
//! Learned Compression of Nonlinear Time Series With Random Access.
//! https://doi.org/10.48550/arXiv.2412.16266.
//! The implementation is partially based on the authors implementation at
//! https://github.com/and-gue/NeaTS (accessed on 15-05-25).

const std = @import("std");
const mem = std.mem;
const ArrayList = std.ArrayList;
const testing = std.testing;

const tersets = @import("../tersets.zig");
const tester = @import("../tester.zig");
const Error = tersets.Error;
const Method = tersets.Method;

const shared_structs = @import("../utilities/shared_structs.zig");
const DiscretePoint = shared_structs.DiscretePoint;

/// Point structure for polygon vertices (can handle negative coordinates)
const PolygonPoint = struct {
    x: f64,
    y: f64,
};

/// Parameter structure for segment creation
const Parameters = struct {
    param1: f64,
    param2: f64,
};

/// Edge information for shortest path reconstruction
const EdgeInfo = struct {
    prev_node: usize,
    segment: Segment,
};

/// Function types supported by NeaTS
const FunctionType = enum(u8) {
    Linear = 0,
    Quadratic = 1,
    Exponential = 2,
    Power = 3,
    Sqrt = 4,
};

/// Active segment tracking (corresponds to Jf,ε in Algorithm 1)
const ActiveSegment = struct {
    start_idx: usize,
    end_idx: usize,
    function_type: FunctionType,
    epsilon: f64,
    polygon: ConvexPolygon,
    best_params: ?Parameters,
    allocator: mem.Allocator,

    fn init(start_idx: usize, function_type: FunctionType, epsilon: f64, allocator: mem.Allocator) ActiveSegment {
        return ActiveSegment{
            .start_idx = start_idx,
            .end_idx = start_idx + 1, // Initially covers just the first point
            .function_type = function_type,
            .epsilon = epsilon,
            .polygon = ConvexPolygon.init(allocator),
            .best_params = null,
            .allocator = allocator,
        };
    }

    fn deinit(self: *ActiveSegment) void {
        self.polygon.deinit();
    }

    fn extendTo(self: *ActiveSegment, data: []const f64, new_end: usize) !bool {
        // Try to extend segment to new_end using O'Rourke's algorithm
        const k = new_end - 1; // Current data point index
        const x_k = @as(f64, @floatFromInt(k - self.start_idx)); // 0-based relative coordinate
        const y_k = data[k];

        std.debug.print("  Trying to extend segment [{}, {}) to include point {} (x={:.1}, y={:.1})\n", .{ self.start_idx, self.end_idx, k, x_k, y_k });

        // Transform constraints based on function type (Table I from paper)
        const transformed_constraint: ?TransformedConstraint = switch (self.function_type) {
            .Linear => TransformedConstraint{
                .m_coeff = -x_k,
                .lower = y_k - self.epsilon,
                .upper = y_k + self.epsilon,
            },
            .Quadratic => blk: {
                if (x_k <= 0.0) break :blk null; // skip x=0 for quadratic
                // For quadratic f(x) = θ₁x² + θ₂ from the paper
                // Constraint: y_k - ε ≤ θ₁x_k² + θ₂ ≤ y_k + ε
                // Rearrange to: (y_k - ε) ≤ (-x_k²)θ₁ + θ₂ ≤ (y_k + ε)
                const m_coeff = -(x_k * x_k); // coefficient of θ₁
                const lower = y_k - self.epsilon;
                const upper = y_k + self.epsilon;
                break :blk TransformedConstraint{ .m_coeff = m_coeff, .lower = lower, .upper = upper };
            },
            .Exponential => if (y_k - self.epsilon > 0 and y_k + self.epsilon > 0) TransformedConstraint{
                .m_coeff = -x_k,
                .lower = @log(y_k - self.epsilon),
                .upper = @log(y_k + self.epsilon),
            } else null,
            .Power => if (x_k > 0 and y_k - self.epsilon > 0 and y_k + self.epsilon > 0) TransformedConstraint{
                .m_coeff = -@log(x_k),
                .lower = @log(y_k - self.epsilon),
                .upper = @log(y_k + self.epsilon),
            } else null,
            .Sqrt => TransformedConstraint{
                .m_coeff = -@sqrt(@max(0, x_k)),
                .lower = y_k - self.epsilon,
                .upper = y_k + self.epsilon,
            },
        };

        if (transformed_constraint) |tc| {
            std.debug.print("    Constraint: {:.3} <= {:.3}*m + b <= {:.3}\n", .{ tc.lower, tc.m_coeff, tc.upper });
            // Add half-plane constraints to polygon
            try self.polygon.addConstraint(tc.m_coeff, tc.lower, true); // Lower bound
            try self.polygon.addConstraint(tc.m_coeff, tc.upper, false); // Upper bound

            if (!self.polygon.isEmpty()) {
                std.debug.print("    Extension SUCCESS to {}\n", .{new_end});
                self.end_idx = new_end;
                if (self.polygon.getFeasiblePoint()) |point| {
                    self.best_params = transformParameters(point, self.function_type);
                }
                return true;
            } else {
                std.debug.print("    Extension FAILED - polygon empty\n", .{});
            }
        }
        return false;
    }

    fn isValidAt(self: *const ActiveSegment, pos: usize) bool {
        return pos >= self.start_idx and pos < self.end_idx and self.best_params != null;
    }

    fn getCostForRange(self: *const ActiveSegment, start: usize, end: usize, data: []const f64) usize {
        if (self.best_params == null) return std.math.maxInt(usize);

        const params = self.best_params.?;

        // Calculate actual residuals for this function fit
        var max_residual: f64 = 0;
        for (start..end) |i| {
            const x = @as(f64, @floatFromInt(i - self.start_idx));
            const predicted = switch (self.function_type) {
                .Linear => params.param1 * x + params.param2,
                .Quadratic => params.param1 * x * x + params.param2,
                .Exponential => if (params.param2 > 0) params.param2 * @exp(params.param1 * x) else 0,
                .Power => if (x > 0 and params.param2 > 0) params.param2 * std.math.pow(f64, x, params.param1) else 0,
                .Sqrt => params.param1 * @sqrt(@max(0, x)) + params.param2,
            };
            const actual = data[i]; // You'll need to pass data as a parameter
            const residual = @abs(predicted - actual);
            max_residual = @max(max_residual, residual);
        }

        // Calculate bits needed for actual residuals
        const correction_bits = if (max_residual <= 0.5)
            0
        else
            @as(usize, @intFromFloat(@ceil(@log2(2.0 * max_residual + 1.0))));

        const num_points = end - start;
        const correction_cost = (num_points * correction_bits + 7) / 8;
        return 17 + correction_cost;
    }

    fn getSegmentForRange(self: *const ActiveSegment, start: usize, end: usize) Segment {
        const params = self.best_params orelse Parameters{ .param1 = 0, .param2 = 0 };
        return Segment{
            .start_idx = start,
            .end_idx = end,
            .function_type = self.function_type,
            .param1 = params.param1,
            .param2 = params.param2,
        };
    }
};

/// Update active segments for current position k (Algorithm 1 logic)
fn updateActiveSegments(
    active_segments: *ArrayList(ActiveSegment),
    data: []const f64,
    k: usize,
    function_types: []const FunctionType,
    error_bounds: []const f64,
    allocator: mem.Allocator,
) !void {
    // Extend existing segments
    var i: usize = 0;
    while (i < active_segments.items.len) {
        // print the length of active segments
        std.debug.print("Active segments length: {}\n", .{active_segments.items.len});
        std.debug.print("Processing active segment at index {}\n", .{i});
        // Try to extend the segment at position i to k + 1
        // // check if k + 1 is within bounds
        // if (k + 1 >= data.len) {
        //     std.debug.print("Skipping segment extension: k+1 out of bounds\n", .{});
        //     i += 1;
        //     continue;
        // }
        const can_extend = active_segments.items[i].extendTo(data, k + 1) catch false;
        std.debug.print("Extending segment at index {}: can_extend={}\n", .{ i, can_extend });
        if (!can_extend) {
            // Segment can't be extended, remove it
            std.debug.print("Removing segment at index {}\n", .{i});
            active_segments.items[i].deinit();
            _ = active_segments.swapRemove(i);
        } else {
            i += 1;
        }
    }

    // Start new segments at position k
    for (function_types) |func_type| {
        for (error_bounds) |epsilon| {
            if (k + 1 < data.len) { // Ensure we have at least one more point
                std.debug.print("Creating new segment at position {} with function type {s} and epsilon {:.3}\n", .{ k, @tagName(func_type), epsilon });
                var new_segment = ActiveSegment.init(k, func_type, epsilon, allocator);
                // Try to extend to the next point to initialize
                // check if k+1 is within bounds
                if (k + 2 >= data.len) {
                    std.debug.print("Skipping new segment creation: k+2 out of bounds\n", .{});
                    new_segment.deinit();
                    continue;
                }
                const can_init = new_segment.extendTo(data, k + 2) catch false;
                if (can_init) {
                    std.debug.print("New segment initialized successfully: [{}, {}) type={s}\n", .{ new_segment.start_idx, new_segment.end_idx, @tagName(new_segment.function_type) });
                    try active_segments.append(new_segment);
                } else {
                    new_segment.deinit();
                }
            }
        }
    }
}

/// Transformed constraint structure (unified type for all function types)
const TransformedConstraint = struct {
    m_coeff: f64,
    lower: f64,
    upper: f64,
};

/// Main NeaTS compression function implementing Algorithm 1 from the paper
pub fn compress(
    data: []const f64,
    out: *ArrayList(u8),
    allocator: mem.Allocator,
    error_bound: f32,
) Error!void {
    if (data.len < 2) return Error.UnsupportedInput;
    const n = data.len;

    // --- 1) Prepare function/error‐bound table ---
    const FTs = [_]FunctionType{ .Linear, .Quadratic, .Exponential, .Power, .Sqrt };
    const Eps = [_]f64{@as(f64, error_bound)};
    const FE = comptime FTs.len * Eps.len;

    // A J–entry holds the last [i,j) we found for (f,ε)
    const JEntry = struct {
        start_idx: usize,
        end_idx: usize,
        segment: Segment,
    };
    var jtable = try allocator.alloc(JEntry, FE);
    defer allocator.free(jtable);
    // initialize all entries so j<=0 => we'll recompute at k=0
    for (jtable) |*e| e.* = JEntry{ .start_idx = 0, .end_idx = 0, .segment = Segment{ .start_idx = 0, .end_idx = 0, .function_type = .Linear, .param1 = 0, .param2 = 0 } };

    // --- 2) DP arrays ---
    var dist = try allocator.alloc(usize, n + 1);
    var prev = try allocator.alloc(?EdgeInfo, n + 1);
    defer allocator.free(dist);
    defer allocator.free(prev);

    for (dist) |*d| d.* = std.math.maxInt(usize);
    for (prev) |*p| p.* = null;
    dist[0] = 0;

    // --- 3) Main loop (Algorithm 1, ll. 7–26) ---
    for (0..n) |k| {
        //  3a) For each (f,ε), if j≤k, recompute via MAKEAPPROXIMATION
        var idx: usize = 0;
        for (FTs) |function_type| {
            for (Eps) |eps| {
                var je = &jtable[idx];
                if (je.end_idx <= k) {
                    // MAKEAPPROXIMATION: findLongestSegment returns [?Segment]
                    if (try findLongestSegment(data, k, function_type, eps, allocator)) |seg| {
                        je.start_idx = seg.start_idx; // should be k
                        je.end_idx = seg.end_idx;
                        je.segment = seg;
                    } else {
                        // no segment → mark as empty
                        je.start_idx = k + 1;
                        je.end_idx = k + 1;
                    }
                }
                idx += 1;
            }
        }

        //  3b) Relax edges (prefix + suffix) for every live segment at k
        idx = 0;
        for (FTs) |function_type| {
            _ = function_type; // silence unused variable warning
            for (Eps) |eps| {
                _ = eps; // silence unused variable warning
                const je = &jtable[idx];
                // only if k lies in [i,j)
                if (je.start_idx <= k and k < je.end_idx) {
                    // prefix (i→k)
                    const c1 = je.segment.getCost();
                    if (dist[je.start_idx] != std.math.maxInt(usize)) {
                        const nc = dist[je.start_idx] + c1;
                        if (nc < dist[k]) {
                            dist[k] = nc;
                            prev[k] = EdgeInfo{ .prev_node = je.start_idx, .segment = je.segment };
                        }
                    }
                    // suffix (k→j)
                    const c2 = je.segment.getCostForRange(k, je.end_idx, data);
                    if (dist[k] != std.math.maxInt(usize)) {
                        const nc = dist[k] + c2;
                        if (nc < dist[je.end_idx]) {
                            dist[je.end_idx] = nc;
                            prev[je.end_idx] = EdgeInfo{ .prev_node = k, .segment = je.segment };
                        }
                    }
                }
                idx += 1;
            }
        }
    }

    // --- 4) Backtrack & serialize (same as before) ---
    if (dist[n] == std.math.maxInt(usize)) return Error.UnsupportedInput;
    var segs = ArrayList(Segment).init(allocator);
    defer segs.deinit();

    var cur = n;
    while (cur > 0) {
        if (prev[cur]) |ei| {
            try segs.append(ei.segment);
            cur = ei.prev_node;
        } else {
            return Error.UnsupportedInput;
        }
    }
    std.mem.reverse(Segment, segs.items);

    try out.append(@intCast(segs.items.len));
    for (segs.items) |s| {
        try out.append(@intFromEnum(s.function_type));
        try appendF64(s.param1, out);
        try appendF64(s.param2, out);
        try appendU32(@intCast(s.start_idx), out);
        try appendU32(@intCast(s.end_idx), out);
    }
}

/// Decompress function (lossy)
pub fn decompress(
    compressed_values: []const u8,
    decompressed_values: *ArrayList(f64),
) Error!void {
    if (compressed_values.len == 0) return Error.UnsupportedInput;

    var offset: usize = 0;

    // Read number of segments
    if (offset >= compressed_values.len) return Error.UnsupportedInput;
    const num_segments = compressed_values[offset];
    offset += 1;

    std.debug.print("Decompress: {} segments\n", .{num_segments});

    // Read each segment
    for (0..num_segments) |_| {
        if (offset + 25 > compressed_values.len) return Error.UnsupportedInput;

        const function_type: FunctionType = @enumFromInt(compressed_values[offset]);
        offset += 1;

        const param1 = readF64(compressed_values[offset..]);
        offset += 8;
        const param2 = readF64(compressed_values[offset..]);
        offset += 8;
        const start_idx = readU32(compressed_values[offset..]);
        offset += 4;
        const end_idx = readU32(compressed_values[offset..]);
        offset += 4;

        const segment = Segment{
            .start_idx = start_idx,
            .end_idx = end_idx,
            .function_type = function_type,
            .param1 = param1,
            .param2 = param2,
        };

        std.debug.print("  Segment: {s} [{}, {}) params=({:.3}, {:.3})\n", .{ @tagName(function_type), start_idx, end_idx, param1, param2 });

        // Generate values for this segment
        for (start_idx..end_idx) |i| {
            // print current i
            std.debug.print("    Generating value for index {}\n", .{i});
            const x = @as(f64, @floatFromInt(i - segment.start_idx));
            const value = segment.evaluate(x);
            try decompressed_values.append(value);
            std.debug.print("    x={:.1} -> {:.3}\n", .{ x, value });
        }
    }
}

/// Find the longest segment starting at `start_idx` that can be ε-approximated by the given function type
/// Following O'Rourke's algorithm as described in the paper
fn findLongestSegment(
    data: []const f64,
    start_idx: usize,
    function_type: FunctionType,
    epsilon: f64,
    allocator: mem.Allocator,
) !?Segment {
    if (start_idx >= data.len) return null;
    if (start_idx + 1 >= data.len) return null; // Need at least 2 points

    std.debug.print("    findLongestSegment: {s} from {}\n", .{ @tagName(function_type), start_idx });

    var polygon = ConvexPolygon.init(allocator);
    defer polygon.deinit();

    var longest_valid_end: ?usize = null;
    var best_params: ?Parameters = null;

    // Start with the first point to establish the initial polygon
    const first_y = data[start_idx];
    const first_constraint: ?TransformedConstraint = switch (function_type) {
        .Linear => TransformedConstraint{
            .m_coeff = 0.0, // x_k = 0, so constraint is just on b
            .lower = first_y - epsilon,
            .upper = first_y + epsilon,
        },
        .Quadratic => null, // Skip first point for quadratic (x=0 not useful for x² term)
        .Exponential => if (first_y - epsilon > 0 and first_y + epsilon > 0) TransformedConstraint{
            .m_coeff = 0.0, // x_k = 0
            .lower = @log(first_y - epsilon),
            .upper = @log(first_y + epsilon),
        } else null,
        .Power => null, // Skip first point for power (need x > 0)
        .Sqrt => TransformedConstraint{
            .m_coeff = 0.0, // sqrt(0) = 0
            .lower = first_y - epsilon,
            .upper = first_y + epsilon,
        },
    };

    // Add the first constraint if valid
    if (first_constraint) |fc| {
        try polygon.addConstraint(fc.m_coeff, fc.lower, true);
        try polygon.addConstraint(fc.m_coeff, fc.upper, false);
    }

    // Try to extend the segment as far as possible (O'Rourke's algorithm)
    for (start_idx + 1..data.len + 1) |end_idx| {
        const k = end_idx - 1;
        const x_k = @as(f64, @floatFromInt(k - start_idx)); // Relative x coordinate (0-based)
        const y_k = data[k];

        std.debug.print("      Point {}: x_k={:.1}, y_k={:.1}\n", .{ k, x_k, y_k });

        // Transform constraints based on function type (Table I from paper)
        const transformed_constraint: ?TransformedConstraint = switch (function_type) {
            .Linear => TransformedConstraint{
                .m_coeff = -x_k, // t_k = x_k
                .lower = y_k - epsilon,
                .upper = y_k + epsilon,
            },
            .Quadratic => blk: {
                if (x_k <= 0.0) break :blk null;
                // For quadratic f(x) = θ₁x² + θ₂ from the paper
                // Constraint: y_k - ε ≤ θ₁x_k² + θ₂ ≤ y_k + ε
                // Rearrange to: (y_k - ε) ≤ (-x_k²)θ₁ + θ₂ ≤ (y_k + ε)
                const m_coeff = -(x_k * x_k); // coefficient of θ₁
                const lower = y_k - epsilon;
                const upper = y_k + epsilon;
                break :blk TransformedConstraint{ .m_coeff = m_coeff, .lower = lower, .upper = upper };
            },
            .Exponential => if (y_k - epsilon > 0 and y_k + epsilon > 0) TransformedConstraint{
                .m_coeff = -x_k,
                .lower = @log(y_k - epsilon),
                .upper = @log(y_k + epsilon),
            } else null,
            .Power => if (x_k > 0 and y_k - epsilon > 0 and y_k + epsilon > 0) TransformedConstraint{
                .m_coeff = -@log(x_k),
                .lower = @log(y_k - epsilon),
                .upper = @log(y_k + epsilon),
            } else null,
            .Sqrt => TransformedConstraint{
                .m_coeff = -@sqrt(@max(0, x_k)),
                .lower = y_k - epsilon,
                .upper = y_k + epsilon,
            },
        };

        if (transformed_constraint) |tc| {
            std.debug.print("      Constraint: m_coeff={:.3}, lower={:.3}, upper={:.3}\n", .{ tc.m_coeff, tc.lower, tc.upper });

            // Clip the polygon by the new half-planes
            try polygon.addConstraint(tc.m_coeff, tc.lower, true); // lower bound
            try polygon.addConstraint(tc.m_coeff, tc.upper, false); // upper bound

            if (!polygon.isEmpty()) {
                longest_valid_end = end_idx;
                if (polygon.getFeasiblePoint()) |pt| {
                    best_params = transformParameters(pt, function_type);
                    std.debug.print("      Feasible point: m={:.3}, b={:.3}\n", .{ pt.x, pt.y });
                }
            } else {
                std.debug.print("      Polygon became empty\n", .{});
                break;
            }
        } else {
            std.debug.print("      Invalid constraints\n", .{});
            if (function_type == .Quadratic or function_type == .Power) {
                // For quadratic and power, continue to next point
                continue;
            } else {
                break;
            }
        }
    }

    // If we found a valid furthest end, build the segment
    if (longest_valid_end) |end_idx| {
        if (best_params) |params| {
            std.debug.print("    Final segment: [{}, {}) params=({:.3}, {:.3})\n", .{ start_idx, end_idx, params.param1, params.param2 });
            return Segment{
                .start_idx = start_idx,
                .end_idx = end_idx,
                .function_type = function_type,
                .param1 = params.param1,
                .param2 = params.param2,
            };
        }
    }

    std.debug.print("    No valid segment found\n", .{});
    return null;
}

/// A convex polygon representing feasible parameter region (following O'Rourke's algorithm)
const ConvexPolygon = struct {
    vertices: ArrayList(PolygonPoint),
    allocator: mem.Allocator,

    fn init(allocator: mem.Allocator) ConvexPolygon {
        return ConvexPolygon{
            .vertices = ArrayList(PolygonPoint).init(allocator),
            .allocator = allocator,
        };
    }

    fn deinit(self: *ConvexPolygon) void {
        self.vertices.deinit();
    }

    fn isEmpty(self: *const ConvexPolygon) bool {
        return self.vertices.items.len == 0;
    }

    /// Add half-plane constraint: b >= slope * m + intercept (following O'Rourke's notation)
    fn addConstraint(self: *ConvexPolygon, slope: f64, intercept: f64, is_lower: bool) !void {
        if (self.vertices.items.len == 0) {
            // Initialize with a large feasible region (following the paper)
            try self.vertices.append(PolygonPoint{ .x = -1000.0, .y = -1000.0 });
            try self.vertices.append(PolygonPoint{ .x = 1000.0, .y = -1000.0 });
            try self.vertices.append(PolygonPoint{ .x = 1000.0, .y = 1000.0 });
            try self.vertices.append(PolygonPoint{ .x = -1000.0, .y = 1000.0 });
        }

        // Sutherland-Hodgman clipping algorithm for half-planes
        var new_vertices = ArrayList(PolygonPoint).init(self.allocator);
        defer {
            self.vertices.deinit();
            self.vertices = new_vertices;
        }

        if (self.vertices.items.len == 0) return;

        // Process each edge of the polygon
        for (0..self.vertices.items.len) |i| {
            const curr = self.vertices.items[i];
            const prev = self.vertices.items[if (i == 0) self.vertices.items.len - 1 else i - 1];

            const curr_inside = if (is_lower)
                curr.y >= slope * curr.x + intercept
            else
                curr.y <= slope * curr.x + intercept;

            const prev_inside = if (is_lower)
                prev.y >= slope * prev.x + intercept
            else
                prev.y <= slope * prev.x + intercept;

            if (curr_inside) {
                if (!prev_inside) {
                    // Entering the region - add intersection
                    if (intersectLine(prev, curr, slope, intercept)) |intersection| {
                        try new_vertices.append(intersection);
                    }
                }
                try new_vertices.append(curr);
            } else if (prev_inside) {
                // Leaving the region - add intersection
                if (intersectLine(prev, curr, slope, intercept)) |intersection| {
                    try new_vertices.append(intersection);
                }
            }
        }
    }

    /// Get a feasible point from the polygon (if any exists)
    fn getFeasiblePoint(self: *const ConvexPolygon) ?PolygonPoint {
        if (self.vertices.items.len == 0) return null;

        // Return centroid as a reasonable feasible point
        var sum_x: f64 = 0;
        var sum_y: f64 = 0;
        for (self.vertices.items) |vertex| {
            sum_x += vertex.x;
            sum_y += vertex.y;
        }

        return PolygonPoint{
            .x = sum_x / @as(f64, @floatFromInt(self.vertices.items.len)),
            .y = sum_y / @as(f64, @floatFromInt(self.vertices.items.len)),
        };
    }
};

/// Find intersection of line segment with half-plane boundary
fn intersectLine(p1: PolygonPoint, p2: PolygonPoint, slope: f64, intercept: f64) ?PolygonPoint {
    const dx = p2.x - p1.x;
    const dy = p2.y - p1.y;

    if (@abs(dx) < 1e-10) {
        // Vertical line case
        const x = p1.x;
        const y = slope * x + intercept;
        if (y >= @min(p1.y, p2.y) and y <= @max(p1.y, p2.y)) {
            return PolygonPoint{ .x = x, .y = y };
        }
        return null;
    }

    const line_slope = dy / dx;
    const line_intercept = p1.y - line_slope * p1.x;

    if (@abs(line_slope - slope) < 1e-10) return null; // Parallel lines

    const x = (intercept - line_intercept) / (line_slope - slope);
    const y = slope * x + intercept;

    // Check if intersection is within the line segment
    const t = (x - p1.x) / dx;
    if (t >= 0.0 and t <= 1.0) {
        return PolygonPoint{ .x = x, .y = y };
    }

    return null;
}

/// A fitted segment with function parameters
const Segment = struct {
    start_idx: usize,
    end_idx: usize, // exclusive
    function_type: FunctionType,
    param1: f64,
    param2: f64,

    fn getCost(self: *const Segment) usize {
        // Cost is just the number of parameters (2 parameters = 16 bytes)
        _ = self; // suppress unused parameter warning
        return 17; // 16 bytes for parameters + 1 byte for function type
    }

    fn getCostForRange(self: *const Segment, start: usize, end: usize, data: []const f64) usize {
        // Calculate actual residuals for this function fit
        var max_residual: f64 = 0;
        for (start..end) |i| {
            const x = @as(f64, @floatFromInt(i - self.start_idx));
            const predicted = self.evaluateAt(x, data);
            const actual = data[i];
            const residual = @abs(predicted - actual);
            max_residual = @max(max_residual, residual);
        }

        // Calculate bits needed for actual residuals
        const correction_bits = if (max_residual <= 0.5)
            0
        else
            @as(usize, @intFromFloat(@ceil(@log2(2.0 * max_residual + 1.0))));

        const num_points = end - start;
        const correction_cost = (num_points * correction_bits + 7) / 8;
        return 17 + correction_cost;
    }

    fn evaluate(self: *const Segment, x: f64) f64 {
        return switch (self.function_type) {
            .Linear => self.param1 * x + self.param2,
            .Quadratic => {
                // For quadratic, we need the first point's y-value as constant term
                // This is a limitation - we should store it, but for now we'll approximate
                const c = self.param2; // This isn't quite right, but will work for simple cases
                return self.param1 * x * x + c;
            },
            .Exponential => if (self.param2 > 0) self.param2 * @exp(self.param1 * x) else 0,
            .Power => if (x > 0 and self.param2 > 0) self.param2 * std.math.pow(f64, x, self.param1) else 0,
            .Sqrt => self.param1 * @sqrt(@max(0, x)) + self.param2,
        };
    }

    fn evaluateAt(self: *const Segment, x: f64, data: []const f64) f64 {
        _ = data; // suppress unused parameter warning for most cases
        return switch (self.function_type) {
            .Linear => self.param1 * x + self.param2,
            .Quadratic => self.param1 * x * x + self.param2, // f(x) = θ₁x² + θ₂
            .Exponential => if (self.param2 > 0) self.param2 * @exp(self.param1 * x) else 0,
            .Power => if (x > 0 and self.param2 > 0) self.param2 * std.math.pow(f64, x, self.param1) else 0,
            .Sqrt => self.param1 * @sqrt(@max(0, x)) + self.param2,
        };
    }
};

/// Transform parameters back from constraint space (Table I from paper)
fn transformParameters(point: PolygonPoint, function_type: FunctionType) Parameters {
    return switch (function_type) {
        .Linear => Parameters{ .param1 = point.x, .param2 = point.y }, // f(x) = θ₁x + θ₂
        .Quadratic => Parameters{ .param1 = point.x, .param2 = point.y }, // f(x) = θ₁x² + θ₂
        .Exponential => Parameters{ .param1 = point.x, .param2 = @exp(point.y) }, // f(x) = θ₂e^(θ₁x)
        .Power => Parameters{ .param1 = point.x, .param2 = @exp(point.y) }, // f(x) = θ₂x^θ₁
        .Sqrt => Parameters{ .param1 = point.x, .param2 = point.y }, // f(x) = θ₁√x + θ₂
    };
}

/// Updated segment evaluation with proper parameter handling
fn evaluateSegment(segment: *const Segment, x: f64) f64 {
    return switch (segment.function_type) {
        .Linear => segment.param1 * x + segment.param2,
        .Quadratic => {
            // Use the form from paper: f(x) = θ₁x² + θ₂
            return segment.param1 * x * x + segment.param2;
        },
        .Exponential => if (segment.param2 > 0) segment.param2 * @exp(segment.param1 * x) else 0,
        .Power => if (x > 0 and segment.param2 > 0) segment.param2 * std.math.pow(f64, x, segment.param1) else 0,
        .Sqrt => segment.param1 * @sqrt(@max(0, x)) + segment.param2,
    };
}

// Helper functions for serialization
fn appendF64(value: f64, list: *ArrayList(u8)) !void {
    const bytes: [8]u8 = @bitCast(value);
    try list.appendSlice(&bytes);
}

fn readF64(bytes: []const u8) f64 {
    return @bitCast(bytes[0..8].*);
}

fn appendU32(value: u32, list: *ArrayList(u8)) !void {
    const bytes: [4]u8 = @bitCast(value);
    try list.appendSlice(&bytes);
}

fn readU32(bytes: []const u8) u32 {
    return @bitCast(bytes[0..4].*);
}

test "neats handles linear data" {
    const allocator = testing.allocator;
    const error_bound: f32 = 1.0;

    std.debug.print("\n=== Testing linear data ===\n", .{});

    // Linear data: y = 2x + 3 for x = 0,1,2,3,4,5,6,7,8,9
    const uncompressed_values = [_]f64{ 3, 5, 7, 9, 11, 13, 15, 17, 19, 21 };

    std.debug.print("Original data: ", .{});
    for (uncompressed_values) |val| {
        std.debug.print("{:.1} ", .{val});
    }
    std.debug.print("\n", .{});

    var compressed_values = ArrayList(u8).init(allocator);
    defer compressed_values.deinit();

    try compress(&uncompressed_values, &compressed_values, allocator, error_bound);

    var decompressed_values = ArrayList(f64).init(allocator);
    defer decompressed_values.deinit();

    try decompress(compressed_values.items, &decompressed_values);

    std.debug.print("Decompressed: ", .{});
    for (decompressed_values.items) |val| {
        std.debug.print("{:.1} ", .{val});
    }
    std.debug.print("\n", .{});

    try testing.expectEqual(uncompressed_values.len, decompressed_values.items.len);

    for (uncompressed_values, decompressed_values.items) |original, decompressed| {
        const abs_error = @abs(original - decompressed);
        std.debug.print("Error: {:.1} - {:.1} = {:.3}\n", .{ original, decompressed, abs_error });
        try testing.expect(abs_error <= error_bound);
    }
}

test "neats handles quadratic data" {
    const allocator = testing.allocator;
    const error_bound: f32 = 1.0;

    std.debug.print("\n=== Testing quadratic data ===\n", .{});

    // Quadratic data: y = x²+1 for x = 0,1,2,3,4,5,6,7,8,9,10
    const uncompressed_values = [_]f64{ 1, 2, 5, 10, 17, 26, 37, 50, 65, 82 };

    std.debug.print("Original data: ", .{});
    for (uncompressed_values) |val| {
        std.debug.print("{:.1} ", .{val});
    }
    std.debug.print("\n", .{});

    var compressed_values = ArrayList(u8).init(allocator);
    defer compressed_values.deinit();

    try compress(&uncompressed_values, &compressed_values, allocator, error_bound);

    var decompressed_values = ArrayList(f64).init(allocator);
    defer decompressed_values.deinit();

    try decompress(compressed_values.items, &decompressed_values);

    std.debug.print("Decompressed: ", .{});
    for (decompressed_values.items) |val| {
        std.debug.print("{:.1} ", .{val});
    }
    std.debug.print("\n", .{});

    try testing.expectEqual(uncompressed_values.len, decompressed_values.items.len);

    for (uncompressed_values, decompressed_values.items) |original, decompressed| {
        const abs_error = @abs(original - decompressed);
        std.debug.print("Error: {:.1} - {:.1} = {:.3}\n", .{ original, decompressed, abs_error });
        try testing.expect(abs_error <= error_bound);
    }
}
