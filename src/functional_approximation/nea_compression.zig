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

/// Compresses `data` using NeaTS algorithm by partitioning the time series into optimal segments
/// with different nonlinear function types and error-bounded approximations.
/// The algorithm uses dynamic programming to find the optimal segmentation that minimizes
/// the total compressed size while maintaining approximation error within `error_bound`.
pub fn compress(
    data: []const f64,
    compressed_values: *ArrayList(u8),
    allocator: mem.Allocator,
    error_bound: f32,
) Error!void {
    // Validates that the input contains at least 2 data points for meaningful compression
    if (data.len < 2) return Error.UnsupportedInput;

    // Calculates the preprocessing shift amount needed for functions that require positive values
    // (exponential and power functions need all y-values to be positive for mathematical validity)
    const shift_amount = calculateShiftAmount(data, error_bound);

    // Creates a working copy of the data only if preprocessing shift is needed (optimization)
    const working_data = if (shift_amount == 0.0)
        data // Uses original data directly when no shift is required
    else blk: {
        // Allocates memory for a shifted copy of the input data

        const mutable_data = try allocator.alloc(f64, data.len);
        @memcpy(mutable_data, data); // Copies the original data values
        applyShift(mutable_data, shift_amount); // Applies the preprocessing shift
        break :blk mutable_data;
    };
    defer if (shift_amount != 0.0) allocator.free(working_data); // Releases allocated memory if used

    // Stores the preprocessing information - shift amount is always written (0.0 indicates no shift)
    try appendF32(shift_amount, compressed_values);

    // Defines the set of function types available for approximating time series segments
    const function_types = [_]FunctionType{ .Linear, .Quadratic, .Exponential, .Power, .Sqrt };
    // Defines the error bounds to consider during optimization (currently uses the provided bound)
    const epsilon_values = [_]f32{error_bound};

    // Partitions the time series using dynamic programming to find the optimal segmentation
    var optimal_segments = try partitionTimeSeries(working_data, &function_types, &epsilon_values, allocator);
    defer optimal_segments.deinit();

    // Serializes the segments to the compressed output stream
    try appendU32(@intCast(optimal_segments.items.len), compressed_values);
    for (optimal_segments.items) |segment| {
        // Writes the function type as a single byte identifier
        try compressed_values.append(@intFromEnum(segment.function_type));
        // Writes the two main function parameters (θ1 and θ2)
        try appendF64(segment.parameters.param1, compressed_values);
        try appendF64(segment.parameters.param2, compressed_values);
        // Writes optional shift value for functions requiring domain shifting
        if (segment.parameters.shift_value) |shift| {
            try compressed_values.append(1);
            try appendF64(shift, compressed_values);
        } else {
            try compressed_values.append(0);
        }
        // Writes only the end index since start index can be inferred from previous segment
        try appendU32(@intCast(segment.end_idx), compressed_values);
    }
}

/// Decompresses `compressed_values` back into the original time series values using
/// the stored function parameters and segment boundaries.
pub fn decompress(
    compressed_values: []const u8,
    decompressed_values: *ArrayList(f64),
) Error!void {
    // Validates that the compressed data contains some bytes to process
    if (compressed_values.len == 0) return Error.UnsupportedInput;

    var offset: usize = 0; // Tracks the current position in the compressed stream

    // Reads the preprocessing shift amount from the compressed stream
    if (offset + 4 > compressed_values.len) return Error.UnsupportedInput;
    const shift_amount = readF32(compressed_values[offset..]);
    offset += 4;

    // Reads the number of segments that were used in the partitioning
    if (offset + 4 > compressed_values.len) return Error.UnsupportedInput;
    const num_segments = readU32(compressed_values[offset..]);
    offset += 4;

    // Parses all segment information from the compressed stream
    var segments = ArrayList(Segment).init(std.heap.page_allocator);
    defer segments.deinit();

    var current_start_idx: u32 = 0; // Tracks the inferred start index for sequential segments

    for (1..num_segments + 1) |_| { // Iterates through each segment (1-based for display clarity)
        // Validates that sufficient bytes remain for segment data
        if (offset + 21 > compressed_values.len) return Error.UnsupportedInput;

        // Reads the function type identifier
        const function_type: FunctionType = @enumFromInt(compressed_values[offset]);
        offset += 1;

        // Reads the main function parameters (θ1 and θ2)
        const param1 = readF64(compressed_values[offset..]);
        offset += 8;
        const param2 = readF64(compressed_values[offset..]);
        offset += 8;

        // Reads the optional shift value if present
        var shift_value: ?f64 = null;
        if (compressed_values[offset] == 1) {
            offset += 1;
            shift_value = readF64(compressed_values[offset..]);
            offset += 8;
        } else {
            offset += 1;
        }

        // Reads the end index of this segment
        const end_idx = readU32(compressed_values[offset..]);
        offset += 4;

        // Creates a segment with the inferred start index
        const segment = Segment{
            .start_idx = current_start_idx, // Uses the inferred start index
            .end_idx = end_idx,
            .function_type = function_type,
            .parameters = FunctionParameters{
                .param1 = param1,
                .param2 = param2,
                .shift_value = shift_value,
            },
        };

        try segments.append(segment);

        // Updates the start index for the next segment (segments are contiguous)
        current_start_idx = end_idx;
    }

    // Validates and sorts segments to ensure they are properly ordered
    std.sort.heap(Segment, segments.items, {}, segmentLessThan);

    // Validates that the partitioning is complete and well-formed
    if (segments.items.len == 0 or segments.items[0].start_idx != 0) {
        return Error.UnsupportedInput;
    }

    // Verifies that segments are contiguous without gaps or overlaps
    for (0..segments.items.len - 1) |i| {
        if (segments.items[i].end_idx != segments.items[i + 1].start_idx) {
            return Error.UnsupportedInput;
        }
    }

    // Remembers the first segment type for postprocessing purposes
    const first_type = segments.items[0].function_type;

    // Generates the reconstructed values using the function approximations
    for (segments.items) |segment| {
        for (segment.start_idx..segment.end_idx) |i| {
            // Uses 1-based indexing for evaluation
            const value = segment.evaluate(@as(f64, @floatFromInt(i + 1))); // Use 1-based indexing for evaluation
            try decompressed_values.append(value);
        }
    }

    // Applies postprocessing to remove any preprocessing shifts that were applied
    postprocessData(decompressed_values.items, shift_amount, first_type);
}

/// Represents the different function types available for approximating time series segments,
/// as defined in Table I of the NeaTS paper. Each type corresponds to a specific mathematical form.
const FunctionType = enum(u8) {
    Linear = 1, // θ1x + θ2
    Quadratic = 2, // θ1x² + θ2
    Exponential = 3, // θ2e^(θ1x)
    Sqrt = 4, // θ1√x + θ2
    Power = 5, // θ2x^θ1

    /// Returns a string representation of the function type for debugging and display.
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

/// Represents a point in the 2D parameter space (m, b coordinates) used in O'Rourke's algorithm.
/// The coordinates correspond to transformed function parameters after applying the parameter
/// space transformation that converts nonlinear constraints into linear half-planes.
const ParameterPoint = struct {
    m: f64, // corresponds to θ1 after transformation
    b: f64, // corresponds to θ2 after transformation
};

/// Represents a half-plane constraint in the parameter space, defining the feasible region
/// for function parameters. The constraint has the form: b >= slope * m + intercept or
/// b <= slope * m + intercept, depending on the `is_lower` flag.
const HalfPlane = struct {
    slope: f64, // Slope coefficient of the half-plane boundary line
    intercept: f64, // Y-intercept of the half-plane boundary line
    is_lower: bool, // true for >=, false for <=
};

/// Represents a convex polygon in parameter space that defines the feasible region
/// for function parameters. Uses the Sutherland-Hodgman clipping algorithm to
/// incrementally intersect half-plane constraints.
const ConvexPolygon = struct {
    vertices: ArrayList(ParameterPoint), // Vertices defining the polygon boundary
    allocator: mem.Allocator,

    /// Initializes an empty convex polygon with the given allocator.
    fn init(allocator: mem.Allocator) ConvexPolygon {
        return ConvexPolygon{
            .vertices = ArrayList(ParameterPoint).init(allocator),
            .allocator = allocator,
        };
    }

    /// Releases all allocated memory used by the polygon.
    fn deinit(self: *ConvexPolygon) void {
        self.vertices.deinit();
    }

    /// Checks whether the polygon has any vertices (represents an empty feasible region).
    fn isEmpty(self: *const ConvexPolygon) bool {
        return self.vertices.items.len == 0;
    }

    /// Initializes the polygon with a large bounding box to represent an initial
    /// feasible region before any constraints are applied.
    fn initializeLargeFeasibleRegion(self: *ConvexPolygon) !void {
        const bound = 1000.0; // Large enough bound for most practical cases
        // Creates a square bounding box in parameter space
        try self.vertices.append(ParameterPoint{ .m = -bound, .b = -bound });
        try self.vertices.append(ParameterPoint{ .m = bound, .b = -bound });
        try self.vertices.append(ParameterPoint{ .m = bound, .b = bound });
        try self.vertices.append(ParameterPoint{ .m = -bound, .b = bound });
    }

    /// Adds a new half-plane constraint to the polygon using the Sutherland-Hodgman
    /// clipping algorithm, which efficiently maintains the convex feasible region.
    fn addConstraint(self: *ConvexPolygon, constraint: HalfPlane) !void {
        // Initializes with a large region if the polygon is currently empty
        if (self.vertices.items.len == 0) {
            try self.initializeLargeFeasibleRegion();
        }

        // Creates a new vertex list for the clipped polygon
        var new_vertices = ArrayList(ParameterPoint).init(self.allocator);
        defer {
            self.vertices.deinit(); // Creates a new vertex list for the clipped polygon
            self.vertices = new_vertices; // Replaces the old vertices with the new clipped ones
        }

        if (self.vertices.items.len == 0) return;

        // Implements the Sutherland-Hodgman clipping algorithm
        for (0..self.vertices.items.len) |i| {
            const curr = self.vertices.items[i];
            const prev_idx = if (i == 0) self.vertices.items.len - 1 else i - 1;
            const prev = self.vertices.items[prev_idx];

            // Determines which vertices satisfy the constraint
            const curr_inside = self.isPointInside(curr, constraint);
            const prev_inside = self.isPointInside(prev, constraint);

            if (curr_inside) {
                if (!prev_inside) {
                    // Crosses from outside to inside - adds intersection point
                    if (self.intersectLine(prev, curr, constraint)) |intersection| {
                        try new_vertices.append(intersection);
                    }
                }
                // Adds the current vertex since it satisfies the constraint
                try new_vertices.append(curr);
            } else if (prev_inside) {
                // Crosses from inside to outside - adds only the intersection point
                if (self.intersectLine(prev, curr, constraint)) |intersection| {
                    try new_vertices.append(intersection);
                }
            }
        }
    }

    /// Determines whether a point satisfies the given half-plane constraint,
    /// using a small tolerance for numerical stability.
    fn isPointInside(self: *const ConvexPolygon, point: ParameterPoint, constraint: HalfPlane) bool {
        _ = self;
        // Evaluates the linear constraint at the given point
        const value = point.b - constraint.slope * point.m - constraint.intercept;
        // Applies the constraint with numerical tolerance
        return if (constraint.is_lower) value >= -1e-10 else value <= 1e-10;
    }

    /// Computes the intersection of a line segment with the half-plane boundary,
    /// returning the intersection point if it lies within the segment bounds.
    fn intersectLine(self: *const ConvexPolygon, p1: ParameterPoint, p2: ParameterPoint, constraint: HalfPlane) ?ParameterPoint {
        _ = self;
        const dx = p2.m - p1.m; // Change in the m-coordinate
        const dy = p2.b - p1.b; // Change in the b-coordinate

        // Handles the special case of vertical lines
        if (@abs(dx) < 1e-10) {
            // Vertical line
            const m = p1.m;
            const b = constraint.slope * m + constraint.intercept;
            // Verifies that the intersection lies within the segment bounds
            if (b >= @min(p1.b, p2.b) and b <= @max(p1.b, p2.b)) {
                return ParameterPoint{ .m = m, .b = b };
            }
            return null;
        }

        // Computes the line equation parameters
        const line_slope = dy / dx;
        const line_intercept = p1.b - line_slope * p1.m;

        // Handles parallel lines (no intersection)
        if (@abs(line_slope - constraint.slope) < 1e-10) return null; // Parallel

        // Finds the intersection point of the two lines
        const m = (constraint.intercept - line_intercept) / (line_slope - constraint.slope);
        const b = constraint.slope * m + constraint.intercept;

        // Verifies that the intersection lies within the line segment
        const t = (m - p1.m) / dx;
        if (t >= 0.0 and t <= 1.0) {
            return ParameterPoint{ .m = m, .b = b };
        }

        return null;
    }

    /// Computes a representative point from the feasible region by calculating
    /// the centroid of the polygon vertices.
    fn getFeasiblePoint(self: *const ConvexPolygon) ?ParameterPoint {
        if (self.vertices.items.len == 0) return null;

        // Computes the average coordinates of all vertices
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

/// Stores the parameters of a function after transformation back from parameter space.
/// The parameters define the mathematical function used to approximate a time series segment.
const FunctionParameters = struct {
    param1: f64, // θ1
    param2: f64, // θ2
    shift_value: ?f64 = null, // for functions that need domain shifting
};

/// Transforms parameters from the (m,b) parameter space back to the original function parameters.
fn transformParameters(point: ParameterPoint, function_type: FunctionType, shift_value: ?f64) FunctionParameters {
    return switch (function_type) {
        .Linear => FunctionParameters{ .param1 = point.m, .param2 = point.b },
        .Quadratic => FunctionParameters{ .param1 = point.m, .param2 = point.b },
        .Exponential => FunctionParameters{ .param1 = point.m, .param2 = @exp(point.b) },
        .Power => FunctionParameters{ .param1 = point.m, .param2 = @exp(point.b) },
        .Sqrt => FunctionParameters{ .param1 = point.m, .param2 = point.b, .shift_value = shift_value },
    };
}

/// Transforms error-bound constraints from the original function space to linear half-planes
/// in parameter space, implementing the key mathematical insight from Table I of the paper.
/// This transformation enables the use of O'Rourke's linear algorithm for nonlinear functions.
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

/// Represents a time series segment with its approximating function and boundary indices.
/// Each segment covers a contiguous range of data points and is approximated by a single
/// mathematical function of the specified type.
const Segment = struct {
    start_idx: usize,
    end_idx: usize, // exclusive
    function_type: FunctionType,
    parameters: FunctionParameters,

    /// Evaluates the approximating function at the given x-coordinate,
    /// using absolute positioning to match the reference implementation.
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

    /// Calculates the storage cost of representing this segment in the compressed format.
    /// Returns a very high cost if the segment violates the error bound constraint.
    fn getCost(self: *const Segment, data: []const f64, epsilon: f32) usize {
        // Verify segment meets error bound (validation)
        var max_error: f64 = 0;
        for (self.start_idx..self.end_idx) |i| {
            const predicted = self.evaluate(@as(f64, @floatFromInt(i + 1))); // 1-based indexing
            const abs_error = @abs(predicted - data[i]);
            max_error = @max(max_error, abs_error);
        }

        const eps = @as(f64, epsilon);

        // Returns prohibitive cost if the error bound is exceeded
        if (max_error > eps) {
            return std.math.maxInt(usize);
        }

        // Calculates the actual storage cost: two f64 parameters + function type + end_idx
        const model_cost = 64 + 64 + 3 + 32; // = 163 bits

        return model_cost;
    }
};

/// Validates that a segment satisfies the error bound constraint by checking
/// the maximum approximation error across all data points in the segment.
fn validateSegment(segment: *const Segment, data: []const f64, epsilon: f32) bool {
    var max_error: f64 = 0;
    // Computes the maximum absolute error over all points in the segment
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

/// Finds the longest possible segment starting from `start_idx` that can be approximated
/// by the given `function_type` while maintaining the error bound. Uses O'Rourke's algorithm
/// generalized to nonlinear functions through parameter space transformation.
fn findLongestSegment(
    data: []const f64,
    start_idx: usize,
    function_type: FunctionType,
    epsilon: f32,
    shift_value: ?f64,
    allocator: mem.Allocator,
) !?Segment {
    // Requires at least two data points to define a meaningful segment
    if (start_idx >= data.len or start_idx + 1 >= data.len) {
        return null;
    }

    // Initializes the convex polygon representing the feasible parameter region
    var polygon = ConvexPolygon.init(allocator);
    defer polygon.deinit();

    var longest_valid_end: ?usize = null; // Tracks the end of the longest valid segment found
    var best_params: ?FunctionParameters = null; // Stores the best parameters for the longest segment
    var points_processed: usize = 0; // Counter for debugging and analysis

    // Implements O'Rourke's algorithm: processes data points from left to right
    for (start_idx..data.len) |end_idx| {
        const k = end_idx;
        const x_k = @as(f64, @floatFromInt(k + 1)); // 1-based indexing
        const y_k = data[k];
        points_processed += 1;

        // Gets constraints for this data point
        const constraints = getConstraints(x_k, y_k, epsilon, function_type) orelse {
            break;
        };

        // Intersects the new constraints with the existing feasible region
        try polygon.addConstraint(constraints.lower);
        try polygon.addConstraint(constraints.upper);

        // Checks if the feasible region is still non-empty after adding constraints
        if (!polygon.isEmpty()) {
            longest_valid_end = end_idx + 1; // +1 because end_idx is exclusive
            // Extracts feasible parameters from the polygon's centroid
            if (polygon.getFeasiblePoint()) |point| {
                best_params = transformParameters(point, function_type, shift_value);
            }
        } else {
            break; // Stops when the feasible region becomes empty (no valid parameters exist)
        }
    }

    // Constructs and validates the final segment if a valid one was found
    if (longest_valid_end) |end_idx| {
        if (best_params) |params| {
            const segment = Segment{
                .start_idx = start_idx,
                .end_idx = end_idx,
                .function_type = function_type,
                .parameters = params,
            };

            // Performs final validation to ensure the segment meets error bounds
            if (!validateSegment(&segment, data, epsilon)) {
                return null;
            }
            return segment;
        }
    }

    return null;
}

/// Calculates the preprocessing shift amount needed to ensure all data values are positive,
/// which is required for exponential and power functions. Returns 0.0 if no shift is needed.
fn calculateShiftAmount(data: []const f64, epsilon: f32) f32 {
    var min_val = std.math.inf(f64);
    for (data) |val| {
        min_val = @min(min_val, val);
    }

    const eps = @as(f64, epsilon);
    // Determines if a shift is needed to keep all values positive within the error bound
    if (min_val - eps <= 0) {
        return @as(f32, @floatCast(eps + 1.0 - min_val));
    }

    return 0.0;
}

/// Applies a preprocessing shift to all values in the mutable data array,
/// ensuring that all values become positive for mathematical functions that require it.
fn applyShift(data: []f64, shift_amount: f32) void {
    const shift_f64 = @as(f64, shift_amount);
    for (data) |*val| {
        val.* += shift_f64;
    }
}

/// Removes the preprocessing shift from decompressed data to restore original value ranges.
/// Handles special cases for certain function types that may require different treatment.
fn postprocessData(data: []f64, shift_amount: f32, first_segment_type: FunctionType) void {
    if (shift_amount == 0.0) return;

    const shift_f64 = @as(f64, shift_amount);
    for (data, 1..) |*val, idx| { // Start from 1 for consistency
        if (idx == 1 and first_segment_type == .Power) continue;
        val.* -= shift_f64;
    }
}

/// Partitions the time series into optimal segments using dynamic programming to minimize
/// the total compressed size while maintaining error bounds. Considers multiple function
/// types and selects the best combination for each segment.
fn partitionTimeSeries(
    data: []const f64,
    function_types: []const FunctionType,
    epsilon_values: []const f32,
    allocator: mem.Allocator,
) !ArrayList(Segment) {
    const n = data.len;

    // Allocates dynamic programming arrays for the shortest path algorithm
    var dist = try allocator.alloc(usize, n + 1); // Minimum cost to reach each position
    var segments = try allocator.alloc(?Segment, n + 1); // Best segment ending at each position
    defer allocator.free(dist);
    defer allocator.free(segments);

    // Initializes the dynamic programming arrays
    for (dist) |*d| d.* = std.math.maxInt(usize); // Sets all distances to infinity initially
    for (segments) |*s| s.* = null; // Initializes all segments as undefined
    dist[0] = 0; // Sets the cost of reaching the start position to zero

    // Executes the main dynamic programming loop over all starting positions
    for (0..n) |start_pos| {
        if (dist[start_pos] == std.math.maxInt(usize)) continue;

        var found_any_segment = false;

        // Tries each combination of function type and error bound
        for (function_types) |function_type| {
            for (epsilon_values) |epsilon| {
                // Finds the longest segment of this type starting at the current position
                if (try findLongestSegment(data, start_pos, function_type, epsilon, null, allocator)) |segment| {
                    const end_pos = segment.end_idx;
                    const segment_cost = segment.getCost(data, epsilon);
                    const new_cost = dist[start_pos] + segment_cost;

                    // Updates the optimal path if this segment provides a better solution
                    if (new_cost < dist[end_pos]) {
                        dist[end_pos] = new_cost;
                        segments[end_pos] = segment;
                        found_any_segment = true;
                    }
                }
            }
        }
        // Creates a fallback single-point segment if no other segments were found
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

            // Updates the path if the fallback segment provides a better solution
            if (new_cost < dist[start_pos + 1]) {
                dist[start_pos + 1] = new_cost;
                segments[start_pos + 1] = fallback_segment;
            }
        }
    }

    if (dist[n] == std.math.maxInt(usize)) {
        return Error.UnsupportedInput;
    }

    // Reconstructs the optimal segmentation by backtracking through the DP solution
    var optimal_segments = ArrayList(Segment).init(allocator);
    var current_pos = n;
    while (current_pos > 0) {
        if (segments[current_pos]) |segment| {
            try optimal_segments.append(segment);
            current_pos = segment.start_idx; // Moves to the start of the current segment
        } else {
            return Error.UnsupportedInput;
        }
    }

    // Reverses the segment order since backtracking produces segments in reverse order
    std.mem.reverse(Segment, optimal_segments.items);
    return optimal_segments;
}

/// Compares two segments by their starting indices for sorting purposes.
/// Used to ensure segments are processed in the correct chronological order.
fn segmentLessThan(context: void, a: Segment, b: Segment) bool {
    _ = context;
    return a.start_idx < b.start_idx;
}

// Binary serialization helper functions for compact storage format

/// Appends a 64-bit floating-point value to the compressed stream in binary format.
/// Uses bit-casting to convert the float to its exact binary representation.
fn appendF64(value: f64, list: *ArrayList(u8)) !void {
    const bytes: [8]u8 = @bitCast(value);
    try list.appendSlice(&bytes);
}

/// Reads a 64-bit floating-point value from the compressed stream.
/// Uses bit-casting to convert the binary representation back to a float.
fn readF64(bytes: []const u8) f64 {
    return @bitCast(bytes[0..8].*);
}

/// Appends a 32-bit floating-point value to the compressed stream in binary format.
/// Used for storing epsilon values and shift amounts with reduced precision.
fn appendF32(value: f32, list: *ArrayList(u8)) !void {
    const bytes: [4]u8 = @bitCast(value);
    try list.appendSlice(&bytes);
}

/// Reads a 32-bit floating-point value from the compressed stream.
/// Used for retrieving epsilon values and shift amounts.
fn readF32(bytes: []const u8) f32 {
    return @bitCast(bytes[0..4].*);
}

/// Appends a 32-bit unsigned integer to the compressed stream in binary format.
/// Used for storing segment counts and array indices.
fn appendU32(value: u32, list: *ArrayList(u8)) !void {
    const bytes: [4]u8 = @bitCast(value);
    try list.appendSlice(&bytes);
}

/// Reads a 32-bit unsigned integer from the compressed stream.
/// Used for retrieving segment counts and array indices.
fn readU32(bytes: []const u8) u32 {
    return @bitCast(bytes[0..4].*);
}

// Tests the compression and decompression of perfectly linear data (y = 2x + 3).
// Verifies that the algorithm correctly identifies and parameterizes linear relationships.
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

// Tests the compression and decompression of quadratic data (y = x² + 1).
// Verifies that the algorithm correctly identifies and parameterizes quadratic relationships.
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

// Tests the compression and decompression of exponential data (y = 2 * e^(0.5*x)).
// Verifies that the algorithm correctly handles the parameter space transformation for exponential functions.
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

// Tests the compression and decompression of power function data (y = 2 * x^1.5).
// Verifies that the algorithm correctly handles the logarithmic parameter space transformation for power functions.
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

// Tests the compression of data requiring two different segments with distinct function types.
// Verifies that the dynamic programming algorithm correctly partitions heterogeneous data.
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

// Tests the compression of data with mixed quadratic and linear segments.
// Verifies that the algorithm can handle transitions between different function types.
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

// Tests the robustness of the compression algorithm with randomly generated data.
// Ensures that the algorithm can handle arbitrary input patterns without failure.
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
