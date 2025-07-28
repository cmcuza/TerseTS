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

pub fn compress(
    data: []const f64,
    compressed_values: *ArrayList(u8),
    allocator: mem.Allocator,
    error_bound: f32,
) Error!void {
    if (data.len < 2) return Error.UnsupportedInput;

    // Check if preprocessing shift is needed
    const shift_amount = calculateShiftAmount(data, error_bound);

    // Only create mutable copy if shift is needed
    const working_data = if (shift_amount == 0.0)
        data
    else blk: {
        const mutable_data = try allocator.alloc(f64, data.len);
        @memcpy(mutable_data, data);
        applyShift(mutable_data, shift_amount);
        break :blk mutable_data;
    };
    defer if (shift_amount != 0.0) allocator.free(working_data);

    // Store preprocessing info - always store shift amount, 0.0 means no shift applied
    try appendF32(shift_amount, compressed_values);

    // Function types and epsilon values
    const function_types = [_]FunctionType{ .Linear, .Quadratic, .Exponential, .Power, .Sqrt };
    const epsilon_values = [_]f32{error_bound};

    // Partition using DP
    var optimal_segments = try partitionTimeSeries(working_data, &function_types, &epsilon_values, allocator);
    defer optimal_segments.deinit();

    // Serialize segments - only store end_idx, start_idx can be inferred
    try appendU32(@intCast(optimal_segments.items.len), compressed_values);
    for (optimal_segments.items) |segment| {
        try compressed_values.append(@intFromEnum(segment.function_type));
        try appendF64(segment.parameters.param1, compressed_values);
        try appendF64(segment.parameters.param2, compressed_values);
        if (segment.parameters.shift_value) |shift| {
            try compressed_values.append(1);
            try appendF64(shift, compressed_values);
        } else {
            try compressed_values.append(0);
        }
        // Only store end_idx - start_idx will be inferred during decompression
        try appendU32(@intCast(segment.end_idx), compressed_values);
    }
}

/// Decompression function with improved parameter display
pub fn decompress(
    compressed_values: []const u8,
    decompressed_values: *ArrayList(f64),
) Error!void {
    if (compressed_values.len == 0) return Error.UnsupportedInput;

    var offset: usize = 0;

    // Read preprocessing info - always read shift amount, 0.0 means no shift was applied
    if (offset + 4 > compressed_values.len) return Error.UnsupportedInput;
    const shift_amount = readF32(compressed_values[offset..]);
    offset += 4;

    // Read segments
    if (offset + 4 > compressed_values.len) return Error.UnsupportedInput;
    const num_segments = readU32(compressed_values[offset..]);
    offset += 4;

    var segments = ArrayList(Segment).init(std.heap.page_allocator);
    defer segments.deinit();

    var current_start_idx: u32 = 0; // Track start index for sequential segments

    for (1..num_segments + 1) |_| { // 1-based loop for display
        // Length check, only end_idx stored
        if (offset + 21 > compressed_values.len) return Error.UnsupportedInput;

        const function_type: FunctionType = @enumFromInt(compressed_values[offset]);
        offset += 1;

        const param1 = readF64(compressed_values[offset..]);
        offset += 8;
        const param2 = readF64(compressed_values[offset..]);
        offset += 8;

        var shift_value: ?f64 = null;
        if (compressed_values[offset] == 1) {
            offset += 1;
            shift_value = readF64(compressed_values[offset..]);
            offset += 8;
        } else {
            offset += 1;
        }

        // Only read end_idx - start_idx is inferred from previous segment
        const end_idx = readU32(compressed_values[offset..]);
        offset += 4;

        const segment = Segment{
            .start_idx = current_start_idx, // Use inferred start index
            .end_idx = end_idx,
            .function_type = function_type,
            .parameters = FunctionParameters{
                .param1 = param1,
                .param2 = param2,
                .shift_value = shift_value,
            },
        };

        try segments.append(segment);

        // Update start index for next segment
        current_start_idx = end_idx;
    }

    // Validate and sort segments
    std.sort.heap(Segment, segments.items, {}, segmentLessThan);

    // Validate proper partitioning
    if (segments.items.len == 0 or segments.items[0].start_idx != 0) {
        return Error.UnsupportedInput;
    }

    for (0..segments.items.len - 1) |i| {
        if (segments.items[i].end_idx != segments.items[i + 1].start_idx) {
            return Error.UnsupportedInput;
        }
    }

    const first_type = segments.items[0].function_type;

    // Generate values - use 1-based indexing
    for (segments.items) |segment| {
        for (segment.start_idx..segment.end_idx) |i| {
            const value = segment.evaluate(@as(f64, @floatFromInt(i + 1))); // Use 1-based indexing for evaluation
            try decompressed_values.append(value);
        }
    }

    // Postprocess
    postprocessData(decompressed_values.items, shift_amount, first_type);
}

/// Function types from Table I of the paper
const FunctionType = enum(u8) {
    Linear = 1, // θ1x + θ2
    Quadratic = 2, // θ1x² + θ2
    Exponential = 3, // θ2e^(θ1x)
    Sqrt = 4, // θ1√x + θ2
    Power = 5, // θ2x^θ1

    pub fn toString(self: FunctionType) []const u8 {
        return switch (self) {
            .Linear => "Linear",
            .Quadratic => "Quadratic",
            .Exponential => "Exponential",
            .Sqrt => "Sqrt",
            .Power => "Power",
        };
    }
};

/// Point in parameter space (m, b coordinates)
const ParameterPoint = struct {
    m: f64, // corresponds to θ1 after transformation
    b: f64, // corresponds to θ2 after transformation
};

/// Half-plane constraint: b >= slope * m + intercept or b <= slope * m + intercept
const HalfPlane = struct {
    slope: f64,
    intercept: f64,
    is_lower: bool, // true for >=, false for <=
};

/// Improved convex polygon with enhanced parameter finding
const ConvexPolygon = struct {
    vertices: ArrayList(ParameterPoint),
    allocator: mem.Allocator,

    fn init(allocator: mem.Allocator) ConvexPolygon {
        return ConvexPolygon{
            .vertices = ArrayList(ParameterPoint).init(allocator),
            .allocator = allocator,
        };
    }

    fn deinit(self: *ConvexPolygon) void {
        self.vertices.deinit();
    }

    fn isEmpty(self: *const ConvexPolygon) bool {
        return self.vertices.items.len == 0;
    }

    /// Initialize with large feasible region
    fn initializeLargeFeasibleRegion(self: *ConvexPolygon) !void {
        const bound = 1000.0;
        try self.vertices.append(ParameterPoint{ .m = -bound, .b = -bound });
        try self.vertices.append(ParameterPoint{ .m = bound, .b = -bound });
        try self.vertices.append(ParameterPoint{ .m = bound, .b = bound });
        try self.vertices.append(ParameterPoint{ .m = -bound, .b = bound });
    }

    /// Add constraint
    fn addConstraint(self: *ConvexPolygon, constraint: HalfPlane) !void {
        if (self.vertices.items.len == 0) {
            try self.initializeLargeFeasibleRegion();
        }

        var new_vertices = ArrayList(ParameterPoint).init(self.allocator);
        defer {
            self.vertices.deinit();
            self.vertices = new_vertices;
        }

        if (self.vertices.items.len == 0) return;

        // Sutherland-Hodgman clipping algorithm
        for (0..self.vertices.items.len) |i| {
            const curr = self.vertices.items[i];
            const prev_idx = if (i == 0) self.vertices.items.len - 1 else i - 1;
            const prev = self.vertices.items[prev_idx];

            const curr_inside = self.isPointInside(curr, constraint);
            const prev_inside = self.isPointInside(prev, constraint);

            if (curr_inside) {
                if (!prev_inside) {
                    // Entering - add intersection
                    if (self.intersectLine(prev, curr, constraint)) |intersection| {
                        try new_vertices.append(intersection);
                    }
                }
                try new_vertices.append(curr);
            } else if (prev_inside) {
                // Leaving - add intersection
                if (self.intersectLine(prev, curr, constraint)) |intersection| {
                    try new_vertices.append(intersection);
                }
            }
        }
    }

    fn isPointInside(self: *const ConvexPolygon, point: ParameterPoint, constraint: HalfPlane) bool {
        _ = self;
        const value = point.b - constraint.slope * point.m - constraint.intercept;
        return if (constraint.is_lower) value >= -1e-10 else value <= 1e-10;
    }

    fn intersectLine(self: *const ConvexPolygon, p1: ParameterPoint, p2: ParameterPoint, constraint: HalfPlane) ?ParameterPoint {
        _ = self;
        const dx = p2.m - p1.m;
        const dy = p2.b - p1.b;

        if (@abs(dx) < 1e-10) {
            // Vertical line
            const m = p1.m;
            const b = constraint.slope * m + constraint.intercept;
            if (b >= @min(p1.b, p2.b) and b <= @max(p1.b, p2.b)) {
                return ParameterPoint{ .m = m, .b = b };
            }
            return null;
        }

        const line_slope = dy / dx;
        const line_intercept = p1.b - line_slope * p1.m;

        if (@abs(line_slope - constraint.slope) < 1e-10) return null; // Parallel

        const m = (constraint.intercept - line_intercept) / (line_slope - constraint.slope);
        const b = constraint.slope * m + constraint.intercept;

        // Check if intersection is within line segment
        const t = (m - p1.m) / dx;
        if (t >= 0.0 and t <= 1.0) {
            return ParameterPoint{ .m = m, .b = b };
        }

        return null;
    }

    fn getFeasiblePoint(self: *const ConvexPolygon) ?ParameterPoint {
        if (self.vertices.items.len == 0) return null;

        var sum_m: f64 = 0;
        var sum_b: f64 = 0;
        for (self.vertices.items) |vertex| {
            sum_m += vertex.m;
            sum_b += vertex.b;
        }

        const centroid_m = sum_m / @as(f64, @floatFromInt(self.vertices.items.len));
        const centroid_b = sum_b / @as(f64, @floatFromInt(self.vertices.items.len));

        return ParameterPoint{ .m = centroid_m, .b = centroid_b };
    }
};

/// Function parameters after transformation back from parameter space
const FunctionParameters = struct {
    param1: f64, // θ1
    param2: f64, // θ2
    shift_value: ?f64 = null, // for functions that need domain shifting
};

/// Transform parameters from (m,b) space back to function parameters
fn transformParameters(point: ParameterPoint, function_type: FunctionType, shift_value: ?f64) FunctionParameters {
    return switch (function_type) {
        .Linear => FunctionParameters{ .param1 = point.m, .param2 = point.b },
        .Quadratic => FunctionParameters{ .param1 = point.m, .param2 = point.b },
        .Exponential => FunctionParameters{ .param1 = point.m, .param2 = @exp(point.b) },
        .Power => FunctionParameters{ .param1 = point.m, .param2 = @exp(point.b) },
        .Sqrt => FunctionParameters{ .param1 = point.m, .param2 = point.b, .shift_value = shift_value },
    };
}

/// Transform constraints to parameter space according to Table I
fn getConstraints(x: f64, y: f64, epsilon: f32, function_type: FunctionType) ?struct { lower: HalfPlane, upper: HalfPlane } {
    const eps = @as(f64, epsilon); // Convert f32 to f64 for calculations

    return switch (function_type) {
        .Linear => .{
            .lower = HalfPlane{ .slope = -x, .intercept = y - eps, .is_lower = true },
            .upper = HalfPlane{ .slope = -x, .intercept = y + eps, .is_lower = false },
        },
        .Quadratic => blk: {
            if (x <= 0.0) break :blk null; // Never happens (x ≥ 1)
            break :blk .{
                .lower = HalfPlane{ .slope = -(x * x), .intercept = y - eps, .is_lower = true },
                .upper = HalfPlane{ .slope = -(x * x), .intercept = y + eps, .is_lower = false },
            };
        },
        .Exponential => blk: {
            if (y - eps <= 0 or y + eps <= 0) break :blk null; // Never happens (preprocessing)
            break :blk .{
                .lower = HalfPlane{ .slope = -x, .intercept = @log(y - eps), .is_lower = true },
                .upper = HalfPlane{ .slope = -x, .intercept = @log(y + eps), .is_lower = false },
            };
        },
        .Power => blk: {
            if (x <= 0 or y - eps <= 0 or y + eps <= 0) break :blk null; // Never happens
            break :blk .{
                .lower = HalfPlane{ .slope = -@log(x), .intercept = @log(y - eps), .is_lower = true },
                .upper = HalfPlane{ .slope = -@log(x), .intercept = @log(y + eps), .is_lower = false },
            };
        },
        .Sqrt => blk: {
            if (x < 0) break :blk null; // x must be non-negative for sqrt
            break :blk .{
                .lower = HalfPlane{ .slope = -@sqrt(x), .intercept = y - eps, .is_lower = true },
                .upper = HalfPlane{ .slope = -@sqrt(x), .intercept = y + eps, .is_lower = false },
            };
        },
    };
}

/// Segment with function parameters and bounds
const Segment = struct {
    start_idx: usize,
    end_idx: usize, // exclusive
    function_type: FunctionType,
    parameters: FunctionParameters,

    fn evaluate(self: *const Segment, x: f64) f64 {
        // Use absolute position instead of relative position to match C++ impl
        return switch (self.function_type) {
            .Linear => self.parameters.param1 * x + self.parameters.param2,
            .Quadratic => self.parameters.param1 * x * x + self.parameters.param2,
            .Exponential => self.parameters.param2 * @exp(self.parameters.param1 * x),
            .Power => blk: {
                if (x <= 0) break :blk self.parameters.param2;
                break :blk self.parameters.param2 * std.math.pow(f64, x, self.parameters.param1);
            },
            .Sqrt => blk: {
                if (x < 0) break :blk self.parameters.param2;
                break :blk self.parameters.param1 * @sqrt(x) + self.parameters.param2;
            },
        };
    }

    fn getCost(self: *const Segment, data: []const f64, epsilon: f32) usize {
        // Verify segment meets error bound (validation)
        var max_error: f64 = 0;
        for (self.start_idx..self.end_idx) |i| {
            const predicted = self.evaluate(@as(f64, @floatFromInt(i + 1))); // 1-based indexing
            const abs_error = @abs(predicted - data[i]);
            max_error = @max(max_error, abs_error);
        }

        const eps = @as(f64, epsilon);

        // If segment doesn't meet error bound, return very high cost
        if (max_error > eps) {
            return std.math.maxInt(usize);
        }

        // Cost: two f64 parameters + function type bits + end_idx (32 bits)
        const model_cost = 64 + 64 + 3 + 32; // = 163 bits

        return model_cost;
    }
};

/// NEW: Validate that segment satisfies error bounds
fn validateSegment(segment: *const Segment, data: []const f64, epsilon: f32) bool {
    var max_error: f64 = 0;
    for (segment.start_idx..segment.end_idx) |i| {
        const predicted = segment.evaluate(@as(f64, @floatFromInt(i + 1)));
        const actual = data[i];
        const abs_error = @abs(predicted - actual);
        max_error = @max(max_error, abs_error);
    }
    const eps = @as(f64, epsilon);
    const is_valid = max_error <= eps;

    return is_valid;
}

/// Find longest segment using O'Rourke's algorithm
fn findLongestSegment(
    data: []const f64,
    start_idx: usize,
    function_type: FunctionType,
    epsilon: f32,
    shift_value: ?f64,
    allocator: mem.Allocator,
) !?Segment {
    if (start_idx >= data.len or start_idx + 1 >= data.len) {
        return null;
    }

    var polygon = ConvexPolygon.init(allocator);
    defer polygon.deinit();

    var longest_valid_end: ?usize = null;
    var best_params: ?FunctionParameters = null;
    var points_processed: usize = 0;

    // O'Rourke's algorithm: process data points left-to-right
    for (start_idx..data.len) |end_idx| {
        const k = end_idx;
        const x_k = @as(f64, @floatFromInt(k + 1)); // 1-based indexing
        const y_k = data[k];
        points_processed += 1;

        // Get constraints for this data point
        const constraints = getConstraints(x_k, y_k, epsilon, function_type) orelse {
            break;
        };

        // Add constraints to polygon
        try polygon.addConstraint(constraints.lower);
        try polygon.addConstraint(constraints.upper);

        if (!polygon.isEmpty()) {
            longest_valid_end = end_idx + 1; // +1 because end_idx is exclusive
            if (polygon.getFeasiblePoint()) |point| {
                best_params = transformParameters(point, function_type, shift_value);
            }
        } else {
            break; // Polygon became empty
        }
    }

    if (longest_valid_end) |end_idx| {
        if (best_params) |params| {
            const segment = Segment{
                .start_idx = start_idx,
                .end_idx = end_idx,
                .function_type = function_type,
                .parameters = params,
            };

            // Validate the segment meets error bounds
            if (!validateSegment(&segment, data, epsilon)) {
                return null;
            }
            return segment;
        }
    }

    return null;
}

/// Preprocessing functions for exponential/power functions
/// Calculate shift amount needed without modifying data
fn calculateShiftAmount(data: []const f64, epsilon: f32) f32 {
    var min_val = std.math.inf(f64);
    for (data) |val| {
        min_val = @min(min_val, val);
    }

    const eps = @as(f64, epsilon);
    if (min_val - eps <= 0) {
        return @as(f32, @floatCast(eps + 1.0 - min_val));
    }

    return 0.0;
}

/// Apply shift to mutable data
fn applyShift(data: []f64, shift_amount: f32) void {
    const shift_f64 = @as(f64, shift_amount);
    for (data) |*val| {
        val.* += shift_f64;
    }
}

fn postprocessData(data: []f64, shift_amount: f32, first_segment_type: FunctionType) void {
    if (shift_amount == 0.0) return;

    const shift_f64 = @as(f64, shift_amount);
    for (data, 1..) |*val, idx| { // Start from 1 for consistency
        if (idx == 1 and first_segment_type == .Power) continue;
        val.* -= shift_f64;
    }
}

/// Dynamic programming partitioning algorithm - now uses f32 error bounds
fn partitionTimeSeries(
    data: []const f64,
    function_types: []const FunctionType,
    epsilon_values: []const f32,
    allocator: mem.Allocator,
) !ArrayList(Segment) {
    const n = data.len;

    // DP arrays
    var dist = try allocator.alloc(usize, n + 1);
    var segments = try allocator.alloc(?Segment, n + 1);
    defer allocator.free(dist);
    defer allocator.free(segments);

    // Initialize
    for (dist) |*d| d.* = std.math.maxInt(usize);
    for (segments) |*s| s.* = null;
    dist[0] = 0;

    // DP main loop
    for (0..n) |start_pos| {
        if (dist[start_pos] == std.math.maxInt(usize)) continue;

        var found_any_segment = false;

        // Try each function type and epsilon
        for (function_types) |function_type| {
            for (epsilon_values) |epsilon| {
                if (try findLongestSegment(data, start_pos, function_type, epsilon, null, allocator)) |segment| {
                    const end_pos = segment.end_idx;
                    const segment_cost = segment.getCost(data, epsilon);
                    const new_cost = dist[start_pos] + segment_cost;

                    if (new_cost < dist[end_pos]) {
                        dist[end_pos] = new_cost;
                        segments[end_pos] = segment;
                        found_any_segment = true;
                    }
                }
            }
        }
        // Fallback: if no valid segment was found, create a single-point segment
        if (!found_any_segment and start_pos < n) {
            const fallback_segment = Segment{
                .start_idx = start_pos,
                .end_idx = start_pos + 1,
                .function_type = .Linear,
                .parameters = FunctionParameters{
                    .param1 = 0.0,
                    .param2 = data[start_pos],
                    .shift_value = null,
                },
            };

            const cost = fallback_segment.getCost(data, epsilon_values[0]);
            const new_cost = dist[start_pos] + cost;

            if (new_cost < dist[start_pos + 1]) {
                dist[start_pos + 1] = new_cost;
                segments[start_pos + 1] = fallback_segment;
            }
        }
    }

    if (dist[n] == std.math.maxInt(usize)) {
        return Error.UnsupportedInput;
    }

    // Backtrack to get optimal segmentation
    var optimal_segments = ArrayList(Segment).init(allocator);
    var current_pos = n;
    while (current_pos > 0) {
        if (segments[current_pos]) |segment| {
            try optimal_segments.append(segment);
            current_pos = segment.start_idx;
        } else {
            return Error.UnsupportedInput;
        }
    }

    // Reverse to get correct order
    std.mem.reverse(Segment, optimal_segments.items);
    return optimal_segments;
}

fn segmentLessThan(context: void, a: Segment, b: Segment) bool {
    _ = context;
    return a.start_idx < b.start_idx;
}

// Helper functions - updated for f32
fn appendF64(value: f64, list: *ArrayList(u8)) !void {
    const bytes: [8]u8 = @bitCast(value);
    try list.appendSlice(&bytes);
}

fn readF64(bytes: []const u8) f64 {
    return @bitCast(bytes[0..8].*);
}

fn appendF32(value: f32, list: *ArrayList(u8)) !void {
    const bytes: [4]u8 = @bitCast(value);
    try list.appendSlice(&bytes);
}

fn readF32(bytes: []const u8) f32 {
    return @bitCast(bytes[0..4].*);
}

fn appendU32(value: u32, list: *ArrayList(u8)) !void {
    const bytes: [4]u8 = @bitCast(value);
    try list.appendSlice(&bytes);
}

fn readU32(bytes: []const u8) u32 {
    return @bitCast(bytes[0..4].*);
}

// Tests with f32 error bounds
test "neats handles linear data" {
    const allocator = testing.allocator;
    const error_bound: f32 = tester.generateBoundedRandomValue(f32, 0, 1, undefined);

    // Linear data: y = 2x + 3 for x = 1,2,3,4,5,6,7,8,9,10 (1-based)
    const uncompressed_values = [_]f64{ 5, 7, 9, 11, 13, 15, 17, 19, 21, 23 }; // y = 2x + 3

    var compressed_values = ArrayList(u8).init(allocator);
    defer compressed_values.deinit();

    try compress(&uncompressed_values, &compressed_values, allocator, error_bound);

    // Inspect compressed stream
    var offset: usize = 0;
    offset += 4;

    // Read number of segments
    const num_segments = readU32(compressed_values.items[offset..]);
    offset += 4;

    try testing.expect(num_segments == 1); // Expect exactly 1 segment

    // Read segment's type
    const function_type_val: u8 = compressed_values.items[offset];
    offset += 1;

    const function_type: FunctionType = @enumFromInt(function_type_val);

    const p1 = readF64(compressed_values.items[offset..]);
    offset += 8;
    const p2 = readF64(compressed_values.items[offset..]);
    offset += 8;

    if (compressed_values.items[offset] == 1) {
        offset += 1;
        offset += 8; // shift value
    } else {
        offset += 1;
    }
    offset += 4; // end_idx

    try testing.expect(function_type == .Linear);
    try testing.expect(@abs(p1 - (2.0)) <= error_bound);
    try testing.expect(@abs(p2 - 3.0) <= error_bound);

    var decompressed_values = ArrayList(f64).init(allocator);
    defer decompressed_values.deinit();

    try decompress(compressed_values.items, &decompressed_values);

    try testing.expectEqual(uncompressed_values.len, decompressed_values.items.len);

    for (uncompressed_values, decompressed_values.items) |original, decompressed| {
        const abs_error = @abs(original - decompressed);
        try testing.expect(abs_error <= error_bound);
    }
}

test "neats handles quadratic data" {
    const allocator = testing.allocator;
    const error_bound: f32 = tester.generateBoundedRandomValue(f32, 0, 1, undefined);

    // Quadratic data: y = x²+1 for x = 1,2,3,4,5,6,7,8,9,10 (1-based)
    const uncompressed_values = [_]f64{ 2, 5, 10, 17, 26, 37, 50, 65, 82, 101 }; // y = x² + 1

    var compressed_values = ArrayList(u8).init(allocator);
    defer compressed_values.deinit();

    try compress(&uncompressed_values, &compressed_values, allocator, error_bound);

    // Inspect compressed stream
    var offset: usize = 0;
    offset += 4;

    // Read number of segments
    const num_segments = readU32(compressed_values.items[offset..]);
    offset += 4;

    try testing.expect(num_segments == 1); // Expect exactly 1 segment

    // Read segment's type
    const function_type_val: u8 = compressed_values.items[offset];
    offset += 1;

    const function_type: FunctionType = @enumFromInt(function_type_val);

    const p1 = readF64(compressed_values.items[offset..]);
    offset += 8;
    const p2 = readF64(compressed_values.items[offset..]);
    offset += 8;

    if (compressed_values.items[offset] == 1) {
        offset += 1;
        offset += 8; // shift value
    } else {
        offset += 1;
    }
    offset += 4; // end_idx

    try testing.expect(function_type == .Quadratic);
    try testing.expect(@abs(p1 - (1.0)) <= error_bound);
    try testing.expect(@abs(p2 - 1.0) <= error_bound);

    var decompressed_values = ArrayList(f64).init(allocator);
    defer decompressed_values.deinit();

    try decompress(compressed_values.items, &decompressed_values);

    try testing.expectEqual(uncompressed_values.len, decompressed_values.items.len);

    for (uncompressed_values, decompressed_values.items) |original, decompressed| {
        const abs_error = @abs(original - decompressed);
        try testing.expect(abs_error <= error_bound);
    }
}

test "neats handles exponential data" {
    const allocator = testing.allocator;
    const error_bound: f32 = tester.generateBoundedRandomValue(f32, 0, 1, undefined);

    // Exponential data: y = 2 * e^(0.5*x) for x = 1,2,3,4,5,6,7,8,9,10 (1-based)
    const uncompressed_values = [_]f64{ 3.3, 5.4, 8.9, 14.8, 24.5, 40.7, 67.6, 112.2, 186.2, 309.6 };

    var compressed_values = ArrayList(u8).init(allocator);
    defer compressed_values.deinit();

    try compress(&uncompressed_values, &compressed_values, allocator, error_bound);

    // Inspect compressed stream
    var offset: usize = 0;
    offset += 4;

    // Read number of segments
    const num_segments = readU32(compressed_values.items[offset..]);
    offset += 4;

    try testing.expect(num_segments == 1); // Expect exactly 1 segment

    // Read segment's type
    const function_type_val: u8 = compressed_values.items[offset];
    offset += 1;

    const function_type: FunctionType = @enumFromInt(function_type_val);

    const p1 = readF64(compressed_values.items[offset..]);
    offset += 8;
    const p2 = readF64(compressed_values.items[offset..]);
    offset += 8;

    if (compressed_values.items[offset] == 1) {
        offset += 1;
        offset += 8; // shift value
    } else {
        offset += 1;
    }
    offset += 4; // end_idx

    try testing.expect(function_type == .Exponential);
    try testing.expect(@abs(p1 - (0.5)) <= error_bound);
    try testing.expect(@abs(p2 - 2.0) <= error_bound);

    var decompressed_values = ArrayList(f64).init(allocator);
    defer decompressed_values.deinit();

    try decompress(compressed_values.items, &decompressed_values);

    try testing.expectEqual(uncompressed_values.len, decompressed_values.items.len);

    for (uncompressed_values, decompressed_values.items) |original, decompressed| {
        const abs_error = @abs(original - decompressed);
        try testing.expect(abs_error <= error_bound);
    }
}

test "neats handles power data" {
    const allocator = testing.allocator;
    const error_bound: f32 = tester.generateBoundedRandomValue(f32, 0, 1, undefined);

    // Power data: y = 2 * x^1.5 for x = 1,2,3,4,5,6,7,8,9,10 (1-based)
    const uncompressed_values = [_]f64{ 2.0, 5.66, 10.39, 16.0, 22.36, 29.39, 37.01, 45.25, 54.0, 63.25 };

    var compressed_values = ArrayList(u8).init(allocator);
    defer compressed_values.deinit();

    try compress(&uncompressed_values, &compressed_values, allocator, error_bound);

    // Inspect compressed stream
    var offset: usize = 0;
    offset += 4;

    // Read number of segments
    const num_segments = readU32(compressed_values.items[offset..]);
    offset += 4;

    try testing.expect(num_segments == 1); // Expect exactly 1 segment

    // Read segment's type
    const function_type_val: u8 = compressed_values.items[offset];
    offset += 1;

    const function_type: FunctionType = @enumFromInt(function_type_val);

    const p1 = readF64(compressed_values.items[offset..]);
    offset += 8;
    const p2 = readF64(compressed_values.items[offset..]);
    offset += 8;

    if (compressed_values.items[offset] == 1) {
        offset += 1;
        offset += 8; // shift value
    } else {
        offset += 1;
    }
    offset += 4; // end_idx

    try testing.expect(function_type == .Power);
    try testing.expect(@abs(p1 - (1.5)) <= error_bound);
    try testing.expect(@abs(p2 - 2.0) <= error_bound);

    var decompressed_values = ArrayList(f64).init(allocator);
    defer decompressed_values.deinit();

    try decompress(compressed_values.items, &decompressed_values);

    try testing.expectEqual(uncompressed_values.len, decompressed_values.items.len);

    for (uncompressed_values, decompressed_values.items) |original, decompressed| { // Start from 1
        const abs_error = @abs(original - decompressed);
        try testing.expect(abs_error <= error_bound);
    }
}

test "neats handles two-segment data" {
    const allocator = testing.allocator;
    const error_bound: f32 = 0.1;

    // First half: y = 2x for x=1..5 -> 2,4,6,8,10
    // Second half: y = 5 for x=6..10 -> 5,5,5,5,5 (should get exactly θ₁=0, θ₂=5)
    const uncompressed_values = [_]f64{ 2, 4, 6, 8, 10, 5, 5, 5, 5, 5 };

    var compressed_values = ArrayList(u8).init(allocator);
    defer compressed_values.deinit();

    try compress(&uncompressed_values, &compressed_values, allocator, error_bound);

    // Inspect compressed stream
    var offset: usize = 0;
    offset += 4;

    // Read number of segments
    const num_segments = readU32(compressed_values.items[offset..]);
    offset += 4;

    try testing.expect(num_segments == 2); // Expect exactly 2 segments

    // Now read each segment's type
    for (0..num_segments) |seg_idx| {
        const function_type_val: u8 = compressed_values.items[offset];
        offset += 1;

        const function_type: FunctionType = @enumFromInt(function_type_val);

        const p1 = readF64(compressed_values.items[offset..]);
        offset += 8;
        const p2 = readF64(compressed_values.items[offset..]);
        offset += 8;

        if (compressed_values.items[offset] == 1) {
            offset += 1;
            offset += 8; // shift value
        } else {
            offset += 1;
        }
        offset += 4; // end_idx

        if (seg_idx == 0) {
            try testing.expect(function_type == .Linear);
            try testing.expect(@abs(p1 - 2.0) <= error_bound);
            try testing.expect(@abs(p2 - 0.0) <= error_bound);
        } else if (seg_idx == 1) {
            try testing.expect(function_type == .Linear);
            try testing.expect(@abs(p1 - (0.0)) <= error_bound);
            try testing.expect(@abs(p2 - 5.0) <= error_bound);
        }
    }

    var decompressed_values = ArrayList(f64).init(allocator);
    defer decompressed_values.deinit();

    try decompress(compressed_values.items, &decompressed_values);

    try testing.expectEqual(uncompressed_values.len, decompressed_values.items.len);

    for (uncompressed_values, decompressed_values.items) |original, decompressed| {
        const abs_error = @abs(original - decompressed);
        try testing.expect(abs_error <= error_bound);
    }
}

test "neats handles quadratic then linear segments" {
    const allocator = testing.allocator;
    const error_bound: f32 = tester.generateBoundedRandomValue(f32, 0, 1, undefined);

    // Quadratic: y = x*x for x=1..5 -> 1,4,9,16,25
    // Linear: y = -x + 12 for x=6..10 -> 6,5,4,3,2
    const uncompressed_values = [_]f64{ 1, 4, 9, 16, 25, 6, 5, 4, 3, 2 };

    var compressed_values = ArrayList(u8).init(allocator);
    defer compressed_values.deinit();

    try compress(&uncompressed_values, &compressed_values, allocator, error_bound);

    // Inspect compressed stream
    var offset: usize = 0;
    offset += 4;

    // Read number of segments
    const num_segments = readU32(compressed_values.items[offset..]);
    offset += 4;

    try testing.expect(num_segments == 2); // Expect exactly 2 segments

    // Now read each segment's type
    for (0..num_segments) |seg_idx| {
        const function_type_val: u8 = compressed_values.items[offset];
        offset += 1;

        const function_type: FunctionType = @enumFromInt(function_type_val);

        const p1 = readF64(compressed_values.items[offset..]);
        offset += 8;
        const p2 = readF64(compressed_values.items[offset..]);
        offset += 8;

        if (compressed_values.items[offset] == 1) {
            offset += 1;
            offset += 8; // shift value
        } else {
            offset += 1;
        }
        offset += 4; // end_idx

        if (seg_idx == 0) {
            try testing.expect(function_type == .Quadratic);
            try testing.expect(@abs(p1 - 1.0) <= error_bound);
            try testing.expect(@abs(p2 - 0.0) <= error_bound);
        } else if (seg_idx == 1) {
            try testing.expect(function_type == .Linear);
            try testing.expect(@abs(p1 - (-1.0)) <= error_bound);
            try testing.expect(@abs(p2 - 12.0) <= error_bound);
        }
    }

    var decompressed_values = ArrayList(f64).init(allocator);
    defer decompressed_values.deinit();

    try decompress(compressed_values.items, &decompressed_values);

    try testing.expectEqual(uncompressed_values.len, decompressed_values.items.len);

    for (uncompressed_values, decompressed_values.items) |orig, decomp| {
        const abs_error = @abs(orig - decomp);
        try testing.expect(abs_error <= error_bound);
    }
}

test "neats compressor can compress and decompress random data" {
    const allocator = std.testing.allocator;
    const error_bound: f32 = tester.generateBoundedRandomValue(f32, 0, 1, undefined);

    var uncompressed_values = ArrayList(f64).init(allocator);
    defer uncompressed_values.deinit();
    try tester.generateBoundedRandomValues(&uncompressed_values, 0.0, 1.0, undefined);

    try uncompressed_values.append(tester.generateBoundedRandomValue(f64, 0, 1, undefined));

    try tester.testCompressAndDecompress(
        uncompressed_values.items,
        allocator,
        Method.NeaCompression,
        error_bound,
        tersets.isWithinErrorBound,
    );
}
