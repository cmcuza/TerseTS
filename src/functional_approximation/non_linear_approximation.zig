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

//! Implementation of Nonlinear Error-bounded Approximation for Time Series (NeaTS) from the paper:
//! "Andrea Guerra, Giorgio Vinciguerra, Antonio Boffa and Paolo Ferragina.
//! Learned Compression of Nonlinear Time Series With Random Access.
//! https://doi.org/10.48550/arXiv.2412.16266.
//! The implementation is partially based on the authors implementation at
//! https://github.com/and-gue/NeaTS (accessed on 15-08-25).
//! Contrary to the original implementation, this version only implements the lossy compression
//! phase which accepts a single error bound.

const std = @import("std");
const mem = std.mem;
const math = std.math;
const ArrayList = std.ArrayList;
const testing = std.testing;

const tersets = @import("../tersets.zig");
const tester = @import("../tester.zig");
const Error = tersets.Error;
const Method = tersets.Method;

const shared_structs = @import("../utilities/shared_structs.zig");
const convex_polygon = @import("../utilities/convex_polygon.zig");
const shared_functions = @import("../utilities/shared_functions.zig");
const DiscretePoint = shared_structs.DiscretePoint;
const LinearFunction = shared_structs.LinearFunction;
const BorderLine = convex_polygon.BorderLine;

/// Represents the different function types available for approximating time series segments,
/// as defined in Table I of the NeaTS paper. Each type corresponds to a specific mathematical form.
const FunctionType = enum(u8) {
    Linear = 1, // slope x + intercept.
    Quadratic = 2, // slope x^2 + intercept.
    Exponential = 3, // intercept e^(slope x).
    Sqrt = 4, // slope sqrt(x) + intercept.
    Power = 5, // intercept x^slope.
    Undefined = 6, // Undefined fall back.
};

/// Set of function types available for approximating time series segments.
const function_types = [_]FunctionType{ .Linear, .Quadratic, .Exponential, .Power, .Sqrt };

/// Compresses `uncompressed_data` using NeaTS algorithm by partitioning the time series into
/// optimal segments with different nonlinear function types and error-bounded approximations.
/// The algorithm uses dynamic programming to find the optimal segmentation that minimizes the
/// total compressed size while maintaining approximation error within `error_bound`. The
/// `allocator` is used for dynamic memory allocation during the compression process. If an error
/// occurs, the function returns an appropriate value.
pub fn compress(
    allocator: mem.Allocator,
    uncompressed_data: []const f64,
    compressed_values: *ArrayList(u8),
    error_bound: f32,
) Error!void {
    // Validates that the input contains at least 2 data points for meaningful compression.
    if (uncompressed_data.len < 2) return Error.UnsupportedInput;

    // Calculates the preprocessing shift amount needed for functions that require positive values
    // (exponential and power functions need all y-values to be positive for mathematical validity).
    const shift_amount = calculateShiftAmount(uncompressed_data, error_bound);

    // Creates a working copy of the data only if preprocessing shift is needed (optimization).
    const working_data = if (shift_amount == 0.0)
        uncompressed_data // Uses original data directly when no shift is required.
    else blk: {
        // Allocates memory for a shifted copy of the input data.
        const mutable_data = try allocator.alloc(f64, uncompressed_data.len);
        @memcpy(mutable_data, uncompressed_data); // Copies the original data values.
        applyShift(mutable_data, shift_amount); // Applies the preprocessing shift.
        break :blk mutable_data;
    };
    defer if (shift_amount != 0.0) allocator.free(working_data); // Releases allocated memory if used.

    // Stores the preprocessing information - shift amount is always written (0.0 indicates no shift).
    try shared_functions.appendValue(f32, shift_amount, compressed_values);

    // Defines the error bounds to consider during optimization (currently uses the provided bound).
    // Partitions the time series using dynamic programming to find the optimal segmentation.
    var optimal_approximation = try findOptimalFunctionalApproximation(
        allocator,
        working_data,
        error_bound,
    );
    defer optimal_approximation.deinit();

    // Serializes the segments to the compressed output stream.
    try shared_functions.appendValue(u32, @intCast(optimal_approximation.items.len), compressed_values);

    for (optimal_approximation.items) |segment| {
        // Writes the function type as a single byte identifier.
        try compressed_values.append(@intFromEnum(segment.function_type));
        // Writes the two main function parameters (slope and intercept).
        try shared_functions.appendValue(
            f64,
            segment.definition.slope,
            compressed_values,
        );
        try shared_functions.appendValue(
            f64,
            segment.definition.intercept,
            compressed_values,
        );
        // Writes only the end index since start index can be inferred from previous segment.
        try shared_functions.appendValue(u32, @intCast(segment.end_idx), compressed_values);
    }
}

/// Decompress `compressed_values` produced by "NeaTS". The function writes the result to
/// `decompressed_values`. If an error occurs it is returned.
pub fn decompress(
    compressed_values: []const u8,
    decompressed_values: *ArrayList(f64),
) Error!void {
    // Validates that the compressed data contains some bytes to process.
    if (compressed_values.len == 0) return Error.UnsupportedInput;

    var offset: usize = 0; // Tracks the current position in the compressed stream.

    // Reads the preprocessing shift amount from the compressed stream.
    if (offset + 4 > compressed_values.len) return Error.UnsupportedInput;
    const shift_amount = try shared_functions.readValue(f32, compressed_values[offset..]);
    offset += 4;

    // Reads the number of segments that were used in the partitioning.
    if (offset + 4 > compressed_values.len) return Error.UnsupportedInput;
    const num_segments = try shared_functions.readValue(u32, compressed_values[offset..]);
    offset += 4;

    // Parses all segment information from the compressed stream.
    var segments = ArrayList(FunctionalApproximation).init(std.heap.page_allocator);
    defer segments.deinit();

    var current_start_idx: u32 = 0; // Tracks the inferred start index for sequential segments.

    for (1..num_segments + 1) |_| { // Iterates through each segment (1-based for display clarity).
        // Validates that sufficient bytes remain for segment data.
        if (offset + 21 > compressed_values.len) return Error.UnsupportedInput;

        // Reads the function type identifier.
        const function_type: FunctionType = @enumFromInt(compressed_values[offset]);
        offset += 1;

        // Reads the main function parameters (slope and intercept).
        const slope = try shared_functions.readValue(f64, compressed_values[offset..]);
        offset += 8;
        const intercept = try shared_functions.readValue(f64, compressed_values[offset..]);
        offset += 8;

        // Reads the end index of this segment.
        const end_idx = try shared_functions.readValue(u32, compressed_values[offset..]);
        offset += 4;

        // Creates a segment with the inferred start index.
        const segment = FunctionalApproximation{
            .start_idx = current_start_idx, // Uses the inferred start index.
            .end_idx = end_idx,
            .function_type = function_type,
            .definition = LinearFunction{
                .slope = slope,
                .intercept = intercept,
            },
        };

        try segments.append(segment);

        // Updates the start index for the next segment (segments are contiguous).
        current_start_idx = end_idx;
    }

    // Validates and sorts segments to ensure they are properly ordered.
    std.sort.heap(
        FunctionalApproximation,
        segments.items,
        {},
        FunctionalApproximation.lessThan,
    );

    // Validates that the partitioning is complete and well-formed.
    if (segments.items.len == 0 or segments.items[0].start_idx != 0) {
        return Error.UnsupportedInput;
    }

    // Verifies that segments are contiguous without gaps or overlaps.
    for (0..segments.items.len - 1) |i| {
        if (segments.items[i].end_idx != segments.items[i + 1].start_idx) {
            return Error.UnsupportedInput;
        }
    }

    // Remembers the first segment type for postprocessing purposes.
    const first_type = segments.items[0].function_type;

    // Generates the reconstructed values using the function approximations.
    for (segments.items) |segment| {
        for (segment.start_idx..segment.end_idx) |i| {
            // Uses 1-based indexing for evaluation.
            const value = try segment.evaluate(@as(f64, @floatFromInt(i + 1)));
            try decompressed_values.append(value);
        }
    }

    // Applies postprocessing to remove any preprocessing shifts that were applied.
    postprocessData(decompressed_values.items, shift_amount, first_type);
}

/// Represents a segment of a time series that is approximated by a mathematical function.
/// Each `FunctionalApproximation` instance describes a contiguous range of data points,
/// defined by `start_idx` (inclusive) and `end_idx` (exclusive), and associates it with a
/// specific `function_type` and its `definition` for approximation.
const FunctionalApproximation = struct {
    start_idx: usize,
    end_idx: usize,
    function_type: FunctionType,
    definition: LinearFunction,

    /// Evaluates the approximating function at the given x-coordinate,
    /// using absolute positioning to match the reference implementation.
    pub fn evaluate(self: *const FunctionalApproximation, x_axis: f64) !f64 {
        return switch (self.function_type) {
            .Linear => self.definition.slope * x_axis + self.definition.intercept,
            .Quadratic => self.definition.slope * x_axis * x_axis + self.definition.intercept,
            .Exponential => self.definition.intercept * @exp(self.definition.slope * x_axis),
            .Power => blk: {
                if (x_axis <= 0) break :blk self.definition.intercept;
                break :blk self.definition.intercept * std.math.pow(
                    f64,
                    x_axis,
                    self.definition.slope,
                );
            },
            .Sqrt => blk: {
                if (x_axis < 0) break :blk self.definition.intercept;
                break :blk self.definition.slope * @sqrt(x_axis) + self.definition.intercept;
            },
            .Undefined => return Error.UnsupportedInput, // Undefined function type cannot be evaluated.
        };
    }

    /// Returns the cost associated with the current functional approximation type in `self`.
    /// The cost is determined based on the `function_type` field of the `FunctionalApproximation`.
    /// Returns 3 if the function type is `.Quadratic`. Returns 2 for all other function types.
    pub fn getCost(self: *const FunctionalApproximation) usize {
        // `.Quadratic` is returns 3 because it needs to store also the initial start_idx.
        // While the rest of `function_type` only need to store the LinearFunction in`definition`.
        // We ensure that the `.Undefined` function is not selected we set the cost to +inf.
        return switch (self.function_type) {
            .Quadratic => 3,
            .Undefined => 10000,
            else => 2,
        };
    }

    pub fn lessThan(_: void, self: FunctionalApproximation, other_app: FunctionalApproximation) bool {
        return self.start_idx < other_app.end_idx;
    }
};

/// Calculates the preprocessing shift amount needed to ensure all data values are positive,
/// which is required for exponential and power functions. Returns 0.0 if no shift is needed.
fn calculateShiftAmount(data: []const f64, error_bound: f32) f32 {
    var min_val = std.math.inf(f64);
    for (data) |val| {
        min_val = @min(min_val, val);
    }

    const eps = @as(f64, error_bound);
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

/// Partitions the time series into optimal segments using dynamic programming to minimize
/// the total compressed size while maintaining error bounds. Considers multiple function
/// types and selects the best combination for each segment.
fn findOptimalFunctionalApproximation(
    allocator: mem.Allocator,
    uncompressed_data: []const f64,
    error_bound: f32,
) !ArrayList(FunctionalApproximation) {
    const n = uncompressed_data.len;

    // Allocates dynamic programming arrays for the shortest path algorithm.
    var distances = try allocator.alloc(usize, n + 1);
    defer allocator.free(distances);

    // Best segment that ends exactly at position k (optional until filled).
    var previous_approximation = try allocator.alloc(?FunctionalApproximation, n + 1);
    defer allocator.free(previous_approximation);

    // One frontier per function type: longest segment starting at the last queried start.
    var current_approximation = try allocator.alloc(FunctionalApproximation, function_types.len);
    defer allocator.free(current_approximation);

    // Initializes the dynamic programming arrays.
    for (0..n + 1) |i| {
        distances[i] = std.math.maxInt(usize); // Sets all distances to infinity initially.
        previous_approximation[i] = null;
    }

    for (0..function_types.len) |i| {
        current_approximation[i] = FunctionalApproximation{
            .start_idx = 0,
            .end_idx = 0,
            .function_type = .Undefined,
            .definition = .{
                .slope = 0.0,
                .intercept = 0.0,
            },
        }; // Initializes all model frontiers as undefined.
    }

    distances[0] = 0; // Sets the cost of reaching the start position to zero.

    // Executes the main dynamic programming loop over all starting positions.
    for (0..n) |current_position| {
        if (distances[current_position] == std.math.maxInt(usize)) continue;

        // Tries each combination of function type and error bound.
        // In this case, there is only one error bound, but multiple function types.
        for (function_types, 0..) |function_type, model_idx| {
            if (current_approximation[model_idx].end_idx <= current_position) {
                // Finds the longest approximation that can be done with function_type
                // starting at the current position.
                const functional_approximation = try computeApproximation(
                    allocator,
                    uncompressed_data,
                    current_position,
                    function_type,
                    error_bound,
                );

                current_approximation[model_idx] = functional_approximation;
            } else {
                const model_start_idx = current_approximation[model_idx].start_idx;
                const cost = current_approximation[model_idx].getCost();
                if (distances[current_position] > distances[model_start_idx] + cost) {
                    distances[current_position] = distances[model_start_idx] + cost;
                    previous_approximation[current_position] = current_approximation[model_idx];
                    previous_approximation[current_position].?.end_idx = current_position;
                }
            }
        }

        // Relax edges from current_position to the end of each available frontier segment.
        for (0..function_types.len) |model_idx| {
            std.debug.assert(current_approximation[model_idx].function_type != .Undefined);
            const end_idx = current_approximation[model_idx].end_idx;
            const cost = current_approximation[model_idx].getCost();
            if (distances[end_idx] > distances[current_position] + cost) {
                // Updates the distance to the end position if a better segment is found.
                distances[end_idx] = distances[current_position] + cost;
                previous_approximation[end_idx] = current_approximation[model_idx];
            }
        }
    }

    std.debug.assert(distances[n] != std.math.maxInt(usize));

    // Reconstructs the optimal segmentation by backtracking through the DP solution.
    var current_position = n;
    var optimal_segments = ArrayList(FunctionalApproximation).init(allocator);
    while (current_position != 0) {
        const segment = previous_approximation[current_position];
        try optimal_segments.append(segment.?);
        current_position = segment.?.start_idx; // Moves to the start of the current segment.
    }

    // Reverses the segment order since backtracking produces segments in reverse order.
    std.mem.reverse(FunctionalApproximation, optimal_segments.items);
    return optimal_segments;
}

/// Finds the longest possible segment starting from `start_idx` that can be approximated
/// by the given `function_type` while maintaining the error bound. Uses O'Rourke's algorithm
/// generalized to nonlinear functions through parameter space transformation.
fn computeApproximation(
    allocator: mem.Allocator,
    uncompressed_data: []const f64,
    start_idx: usize,
    function_type: FunctionType,
    error_bound: f32,
) !FunctionalApproximation {
    const n = uncompressed_data.len;

    // If only one point left, fit a Linear Function with slope 0, intercept = point.
    if (start_idx + 1 >= n) {
        const intercept = uncompressed_data[start_idx];
        return FunctionalApproximation{
            .start_idx = start_idx,
            .end_idx = start_idx + 1, // Only covers one point
            .function_type = .Linear,
            .definition = LinearFunction{
                .slope = 0.0,
                .intercept = intercept,
            },
        };
    }

    // Initializes the convex polygon representing the feasible parameter region.
    var polygon = convex_polygon.ConvexPolygon.init(allocator);
    defer polygon.deinit();

    // Tracks the end of the longest valid segment found.
    var longest_valid_end: usize = start_idx + 2; // Always can fit 2 points!

    // Stores the best approximation for the longest segment.
    var best_approximation: LinearFunction = .{ .slope = 0, .intercept = 0 };

    // Implements O'Rourke's algorithm: processes data points from left to right.
    for (start_idx..n) |end_idx| {
        const x_axis: usize = end_idx + 1;
        const y_axis: f64 = uncompressed_data[end_idx];

        // Gets constraints for this data point.
        const constraints = getConstraints(
            x_axis,
            y_axis,
            error_bound,
            function_type,
        );

        const new_upper_line: BorderLine = .{
            .definition = constraints.upper,
            .x_axis_domain = .{ .start = -std.math.inf(f64), .end = std.math.inf(f64) },
        };

        const new_lower_line: BorderLine = .{
            .definition = constraints.lower,
            .x_axis_domain = .{ .start = -std.math.inf(f64), .end = std.math.inf(f64) },
        };

        // Intersects the new constraints with the existing feasible region.
        const intercept = try polygon.update(
            new_upper_line,
            new_lower_line,
        );

        // Checks if the feasible region is still non-empty after adding constraints.
        if (!intercept) {
            const feasible_solution = polygon.computeFeasibleSolution();
            best_approximation = transformParameters(feasible_solution, function_type);
            longest_valid_end = end_idx; // Updates the longest valid segment end.
            break;
        }
    }

    // Constructs and validates the final segment if a valid one was found.
    const functional_approximation = FunctionalApproximation{
        .start_idx = start_idx,
        .end_idx = longest_valid_end,
        .function_type = function_type,
        .definition = best_approximation,
    };

    return functional_approximation;
}

/// Transforms error-bound constraints from the original function space to linear functions
/// in parameter space, implementing the key mathematical insight from Table I of the paper.
/// This transformation enables the use of O'Rourke's linear algorithm for nonlinear functions.
fn getConstraints(
    x_axis: usize,
    y_axis: f64,
    error_bound: f32,
    function_type: FunctionType,
) struct {
    lower: LinearFunction,
    upper: LinearFunction,
} {
    const eps = @as(f64, error_bound); // Convert f32 to f64 for calculations.
    const slope: f64 = @floatFromInt(x_axis);

    std.debug.assert((slope > 0) and (y_axis - eps > 0) and (y_axis + eps > 0));

    return switch (function_type) {
        .Linear => .{
            .lower = LinearFunction{ .slope = -slope, .intercept = y_axis - eps },
            .upper = LinearFunction{ .slope = -slope, .intercept = y_axis + eps },
        },
        .Quadratic => .{
            .lower = LinearFunction{ .slope = -(slope * slope), .intercept = y_axis - eps },
            .upper = LinearFunction{ .slope = -(slope * slope), .intercept = y_axis + eps },
        },
        .Exponential => .{
            .lower = LinearFunction{ .slope = -slope, .intercept = @log(y_axis - eps) },
            .upper = LinearFunction{ .slope = -slope, .intercept = @log(y_axis + eps) },
        },
        .Power => .{
            .lower = LinearFunction{ .slope = -@log(slope), .intercept = @log(y_axis - eps) },
            .upper = LinearFunction{ .slope = -@log(slope), .intercept = @log(y_axis + eps) },
        },
        .Sqrt => .{
            .lower = LinearFunction{ .slope = -@sqrt(slope), .intercept = y_axis - eps },
            .upper = LinearFunction{ .slope = -@sqrt(slope), .intercept = y_axis + eps },
        },
        .Undefined => .{
            .lower = LinearFunction{ .slope = math.inf(f64), .intercept = math.inf(f64) },
            .upper = LinearFunction{ .slope = math.inf(f64), .intercept = math.inf(f64) },
        },
    };
}

/// Transforms parameters from the (m,b) parameter space back to the original function parameters.
fn transformParameters(
    linear_function: LinearFunction,
    function_type: FunctionType,
) LinearFunction {
    return switch (function_type) {
        .Linear => LinearFunction{
            .slope = linear_function.slope,
            .intercept = linear_function.intercept,
        },
        .Quadratic => LinearFunction{
            .slope = linear_function.slope,
            .intercept = linear_function.intercept,
        },
        .Exponential => LinearFunction{
            .slope = linear_function.slope,
            .intercept = @exp(linear_function.intercept),
        },
        .Power => LinearFunction{
            .slope = linear_function.slope,
            .intercept = @exp(linear_function.intercept),
        },
        .Sqrt => LinearFunction{
            .slope = linear_function.slope,
            .intercept = linear_function.intercept,
        },
        .Undefined => LinearFunction{
            .slope = math.inf(f64),
            .intercept = math.inf(f64),
        },
    };
}

/// Removes the preprocessing shift from decompressed data to restore original value ranges.
/// Handles special cases for certain function types that may require different treatment.
fn postprocessData(data: []f64, shift_amount: f32, first_segment_type: FunctionType) void {
    if (shift_amount == 0.0) return;

    const shift_f64 = @as(f64, shift_amount);
    for (data, 1..) |*val, idx| { // Start from 1 for consistency.
        if (idx == 1 and first_segment_type == .Power) continue;
        val.* -= shift_f64;
    }
}

test "find optimal approximation with random linear sequences with slope break" {
    const allocator = testing.allocator;
    const random = tester.getDefaultRandomGenerator();

    const error_bound = 0.8;

    const m1: f64 = tester.generateBoundedRandomValue(f64, 1, 10, undefined);
    const b1: f64 = tester.generateBoundedRandomValue(f64, 1, 10, undefined);

    const uncompressed_values: []f64 = try allocator.alloc(f64, 40);
    defer allocator.free(uncompressed_values);

    for (0..20) |i| {
        const y = m1 * @as(f64, @floatFromInt(i)) + b1 + random.float(f64) * 0.1;
        uncompressed_values[i] = y;
    }

    // Now second line: slope -1, intercept 10
    const m2: f64 = tester.generateBoundedRandomValue(f64, 1, 10, undefined);
    const b2: f64 = tester.generateBoundedRandomValue(f64, 1, 10, undefined);

    for (20..40) |i| {
        const y = m2 * @as(f64, @floatFromInt(i)) + b2 + random.float(f64) * 0.1;
        uncompressed_values[i] = y;
    }

    const optimal = try findOptimalFunctionalApproximation(
        allocator,
        uncompressed_values,
        error_bound,
    );

    for (optimal.items) |segment| {
        std.debug.print("{}\n", .{segment});
    }
}
