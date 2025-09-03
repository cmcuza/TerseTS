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
//! phase and accepts a single error bound.

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
    Linear = 1, // slope * x + intercept.
    Quadratic = 2, // slope * x ^ 2 + intercept.
    Exponential = 3, // intercept * e ^ (slope * x).
    Sqrt = 4, // slope * sqrt(x) + intercept.
    Power = 5, // intercept * x ^ slope.
    Undefined = 6, // Undefined fall back.
};

/// Set of function types available for approximating time series segments.
/// TODO: Make this configurable at running time.
const function_types = [5]FunctionType{ .Linear, .Quadratic, .Exponential, .Power, .Sqrt };

/// Compresses `uncompressed_data` using NeaTS algorithm by partitioning the time series into
/// optimal segments with different nonlinear function types and error-bounded approximations.
/// The algorithm uses dynamic programming to find the optimal segmentation that minimizes the
/// total compressed size while maintaining approximation error within `error_bound`. The
/// `allocator` is used for dynamic memory allocation during the compression process. If an error
/// occurs, the function returns an error.
pub fn compress(
    allocator: mem.Allocator,
    uncompressed_data: []const f64,
    compressed_values: *ArrayList(u8),
    error_bound: f32,
) Error!void {
    // Validates that the input contains at least 2 data points for meaningful compression.
    if (uncompressed_data.len < 2) return Error.UnsupportedInput;

    if (error_bound <= 0.0) return Error.UnsupportedErrorBound;

    // Calculates the preprocessing shift amount needed for functions that require positive values
    // (exponential and power functions need all y-values to be positive for mathematical validity).
    const shift_amount = try calculateShiftAmount(uncompressed_data, error_bound);

    // Creates a working copy of the data only if preprocessing shift is needed (optimization).
    const working_data = if (shift_amount == 0.0)
        uncompressed_data // Uses original data directly when no shift is required.
    else blk: {
        // Allocates memory for a shifted copy of the input data.
        const mutable_data = try allocator.alloc(f64, uncompressed_data.len);
        @memcpy(mutable_data, uncompressed_data); // Copies the original data values.
        for (mutable_data) |*val| {
            val.* += shift_amount; // Applies the shift to each data point.
        }
        break :blk mutable_data;
    };
    defer if (shift_amount != 0.0) allocator.free(working_data); // Releases allocated memory if used.

    // Stores the preprocessing information - shift amount is always written (0.0 indicates no shift).
    try shared_functions.appendValue(f64, shift_amount, compressed_values);

    var optimal_approximation = ArrayList(FunctionalApproximation).init(allocator);
    defer optimal_approximation.deinit();

    // Defines the error bounds to consider during optimization (currently uses the provided bound).
    // Partitions the time series using dynamic programming to find the optimal segmentation.
    try findOptimalFunctionalApproximation(
        allocator,
        working_data,
        error_bound,
        &optimal_approximation,
    );

    const segments_count = optimal_approximation.items.len;
    // Store the number of segments used in the partitioning.
    try shared_functions.appendValue(u32, @intCast(segments_count), compressed_values);

    // All function types are stored using 4 bits each, so we can pack 2 per byte.
    // This saves space in the compressed representation.
    // For it, we first calculate the number of bytes needed to store all function types.
    const packed_len: usize = (segments_count + 1) / 2; // ceil(count/2).

    // Allocates space for packed function types (2 per byte).
    var packed_function_types = try allocator.alloc(u8, packed_len);
    defer allocator.free(packed_function_types);
    @memset(packed_function_types, 0);
    for (optimal_approximation.items, 0..) |approximation, idx| {
        const code: u8 = @intCast(@intFromEnum(approximation.function_type));
        const byte_idx: usize = idx / 2;
        const is_high_nibble: bool = (idx % 2) == 0;
        // The high nibble stores the function type of the first segment, and the low
        // nibble stores the second approximation's type.
        if (is_high_nibble) {
            packed_function_types[byte_idx] |= @as(u8, code) << 4;
        } else {
            packed_function_types[byte_idx] |= @as(u8, code) & 0x0F;
        }
    }

    try compressed_values.appendSlice(packed_function_types);

    for (optimal_approximation.items) |segment| {
        // Writes the two main function parameters (slope and intercept) and the end point.
        try shared_functions.appendValue(f64, segment.definition.slope, compressed_values);
        try shared_functions.appendValue(f64, segment.definition.intercept, compressed_values);
        try shared_functions.appendValue(usize, segment.end_idx, compressed_values);
    }
}

/// Decompress `compressed_values` produced by "NeaTS". The function writes the result to
/// `decompressed_values`. If an error occurs it is returned.
pub fn decompress(
    allocator: mem.Allocator,
    compressed_values: []const u8,
    decompressed_values: *ArrayList(f64),
) Error!void {
    // Validates that the compressed data contains some bytes to process.
    if (compressed_values.len < 12) return Error.CorruptedCompressedData;

    var offset: usize = 0; // Tracks the current position in the compressed stream.

    // Reads the preprocessing shift amount from the compressed stream.
    const shift_amount = try shared_functions.readValue(f64, compressed_values[offset..]);
    offset += @sizeOf(f64);

    // Reads the number of segments that were used in the partitioning.
    const num_segments: usize = try shared_functions.readValue(u32, compressed_values[offset..]);
    offset += @sizeOf(u32);

    // Read packed function types (2 per byte, low nibble = even index, high nibble = odd).
    const type_bytes_len: usize = (num_segments + 1) / 2;

    // Validate that the compressed stream contains exactly the expected number of bytes.
    // Each segment stores: 2 * f64 (slope, intercept) + usize (end_idx).
    const bytes_per_segment = @sizeOf(f64) * 2 + @sizeOf(usize);
    const expected_total_bytes =
        @sizeOf(f64) + // shift_amount
        @sizeOf(u32) + // num_segments
        type_bytes_len + // packed function types
        num_segments * bytes_per_segment;

    if (compressed_values.len != expected_total_bytes)
        return Error.CorruptedCompressedData;

    const packed_function_types = compressed_values[offset .. offset + type_bytes_len];
    offset += type_bytes_len;

    var optimal_approximation = ArrayList(FunctionalApproximation).init(allocator);
    defer optimal_approximation.deinit();

    var current_start_idx: usize = 0; // Tracks the inferred start index for sequential segments.
    for (0..num_segments) |segment_idx| { // Iterates through each segment.
        const packed_code = packed_function_types[segment_idx / 2];
        const code: u4 = if (segment_idx % 2 != 0)
            @truncate(packed_code & 0x0F)
        else
            @truncate((packed_code >> 4) & 0x0F);

        const function_type: FunctionType = @enumFromInt(@as(u8, code));

        // Reads the main function parameters (slope and intercept) and end index.
        const slope = try shared_functions.readValue(f64, compressed_values[offset..]);
        offset += @sizeOf(f64);
        const intercept = try shared_functions.readValue(f64, compressed_values[offset..]);
        offset += @sizeOf(f64);
        const end_idx: usize = try shared_functions.readValue(usize, compressed_values[offset..]);
        offset += @sizeOf(usize);

        // Creates a segment with the inferred start index.
        const functional_approximation = FunctionalApproximation{
            .start_idx = current_start_idx, // Uses the inferred start index.
            .end_idx = end_idx,
            .function_type = function_type,
            .definition = LinearFunction{
                .slope = slope,
                .intercept = intercept,
            },
        };

        try optimal_approximation.append(functional_approximation);

        // Updates the start index for the next segment (segments are contiguous).
        current_start_idx = end_idx;
    }

    // Generates the reconstructed values using the function approximations.
    for (optimal_approximation.items) |functional_approximation| {
        for (functional_approximation.start_idx..functional_approximation.end_idx) |idx| {
            // Uses 1-based indexing for evaluation.
            const value = try functional_approximation.evaluate(@as(f64, @floatFromInt(idx + 1)));
            try decompressed_values.append(value);
        }
    }

    if (shift_amount == 0.0) return;

    // Apply postprocessing to remove any preprocessing shifts.
    for (0..decompressed_values.items.len) |idx| {
        decompressed_values.items[idx] -= shift_amount;
    }
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

    /// Evaluates the function defined in `self` at the given `x_axis` position.
    /// The `x_axis` is expected to be 1-based index relative to the start of the segment.
    /// Returns the computed value or an error if the function type is unsupported.
    pub fn evaluate(self: *const FunctionalApproximation, x_axis: f64) !f64 {
        const x_rel = x_axis - @as(f64, @floatFromInt(self.start_idx));
        return switch (self.function_type) {
            .Linear => @mulAdd(f64, self.definition.slope, x_rel, self.definition.intercept),
            .Quadratic => @mulAdd(f64, self.definition.slope, (x_rel * x_rel), self.definition.intercept),
            .Exponential => self.definition.intercept * @exp(self.definition.slope * x_rel),
            .Power => self.definition.intercept * math.pow(f64, x_rel, self.definition.slope),
            .Sqrt => @mulAdd(f64, self.definition.slope, @sqrt(x_rel), self.definition.intercept),
            .Undefined => return Error.UnsupportedInput, // Undefined function type cannot be evaluated.
        };
    }

    /// Returns the cost associated with the current functional approximation type in `self`.
    /// The cost is determined based on the `function_type` field of the `FunctionalApproximation`.
    /// Currently, all defined function types have a uniform cost of 2, while the `.Undefined`
    /// type is assigned a high cost to prevent its selection in the optimization process.
    pub fn getCost(self: *const FunctionalApproximation) usize {
        // `.Undefined` returns a big number to unsure that it is not selected.
        return switch (self.function_type) {
            .Undefined => 10000,
            else => 2,
        };
    }
};

/// Calculates the preprocessing shift amount needed to ensure all data values are positive.
/// This is required for exponential and power functions. The function first inspects the
/// `uncompressed_data` and shifts all values by the minimum value plus the `error_bound`.
/// If all values are already positive, no shift is applied (returns 0.0).
fn calculateShiftAmount(uncompressed_data: []const f64, error_bound: f32) !f64 {
    var min_val = std.math.inf(f64);
    for (uncompressed_data) |val| {
        if (!math.isFinite(val) or @abs(val) > tester.max_test_value) {
            return Error.UnsupportedInput;
        }
        min_val = @min(min_val, val);
    }

    const eps = @as(f64, error_bound);
    // Determines if a shift is needed to keep all values positive within the error bound.
    // The shift amount is calculated as (eps + 1.0 - min_val) to ensure positivity of the border
    // line y = min_val - eps.
    if (min_val - eps <= 0) {
        return eps + 1.0 - min_val;
    }

    return 0.0;
}

/// Finds the optimal segmentation of `uncompressed_data` into segments approximated by
/// different function types while maintaining the `error_bound`. Uses dynamic programming
/// to minimize the total cost of the segmentation. The resulting optimal segments are
/// appended to `optimal_approximation`. The `allocator` is used for dynamic memory allocation.
/// If an error occurs, it is returned.
fn findOptimalFunctionalApproximation(
    allocator: mem.Allocator,
    uncompressed_data: []const f64,
    error_bound: f32,
    optimal_approximation: *ArrayList(FunctionalApproximation),
) !void {
    const n = uncompressed_data.len;

    // Initializes the convex polygon representing the feasible parameter region.
    var polygon = convex_polygon.ConvexPolygon.init(allocator);
    defer polygon.deinit();

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
        // Tries each combination of function type and error bound.
        // In this case, there is only one error bound, but multiple function types.
        for (function_types, 0..) |function_type, model_idx| {
            if (current_approximation[model_idx].end_idx <= current_position) {
                // Finds the longest approximation that can be done with function_type
                // starting at the current position.
                const functional_approximation = try computeApproximation(
                    &polygon,
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
    while (current_position != 0) {
        const approximation = previous_approximation[current_position];
        try optimal_approximation.append(approximation.?);
        // Move to the start of the current segment.
        current_position = approximation.?.start_idx;
    }

    // Reverses the segment order since backtracking produces segments in reverse order.
    std.mem.reverse(FunctionalApproximation, optimal_approximation.items);
}

/// Computes the longest segment starting from `start_idx` that can be approximated by
/// the specified `function_type` while maintaining the `error_bound`. Uses O'Rourke's
/// algorithm generalized to nonlinear functions through parameter space transformation.
/// The function returns a `FunctionalApproximation` representing the best segment found.
/// The `polygon` parameter is used to maintain the feasible region of parameters.
/// If an error occurs, it is returned.
fn computeApproximation(
    polygon: *convex_polygon.ConvexPolygon,
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
            .end_idx = start_idx + 1, // Only covers one point.
            .function_type = function_type,
            .definition = LinearFunction{
                .slope = 0.0,
                .intercept = intercept,
            },
        };
    }

    // Tracks the end of the longest valid segment found.
    var longest_valid_end: usize = start_idx + 2; // Always can fit 2 points.

    // Stores the best approximation for the longest segment.
    var best_approximation: LinearFunction = .{ .slope = 0, .intercept = 0 };

    // Implements O'Rourke's algorithm: processes data points from left to right.
    for (start_idx..n) |end_idx| {
        const x_axis: usize = end_idx - start_idx + 1;
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
        if (!intercept) break;

        const feasible_solution = polygon.computeFeasibleSolution();
        best_approximation = transformParameters(feasible_solution, function_type);
        longest_valid_end = start_idx + x_axis; // Updates the longest valid segment end.
    }

    polygon.clear();

    return FunctionalApproximation{
        .start_idx = start_idx,
        .end_idx = longest_valid_end,
        .function_type = function_type,
        .definition = best_approximation,
    };
}

/// Transforms and returns the lower and upper border line constraints for the given
/// `function_type`, `x_axis`, `y_axis`, and `error_bound`. This function maps the
/// the nonlinear function constraints into linear constraints in parameter space,
/// enabling the use of O'Rourke's algorithm for linear functions. The transformation
/// is based on the mathematical formulations provided in Table I of the NeaTS paper.
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

/// Transforms the parameters of a `linear_function` obtained in the transformed parameter space
/// back to the original parameter space corresponding to the specified `function_type`.
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

test "non linear approximator can compress and decompress various function types with positive error bounds" {
    const allocator = testing.allocator;
    const data_distributions = &[_]tester.DataDistribution{
        .MixedBoundedValuesFunctions,
    };

    try tester.testErrorBoundedCompressionMethod(
        allocator,
        Method.NonLinearApproximation,
        data_distributions,
    );
}

test "non linear approximator cannot compress NaN values" {
    const allocator = testing.allocator;

    const uncompressed_values = &[4]f64{ 19.0, 48.0, math.nan(f64), 3.0 };

    var compressed_values = ArrayList(u8).init(allocator);
    defer compressed_values.deinit();

    compress(
        allocator,
        uncompressed_values,
        &compressed_values,
        0.1,
    ) catch |err| {
        try testing.expectEqual(Error.UnsupportedInput, err);
        return;
    };

    try testing.expectFmt(
        "",
        "The non linear approximator algorithm cannot compress NaN values",
        .{},
    );
}

test "non linear approximator cannot compress inf values" {
    const allocator = testing.allocator;

    const uncompressed_values = &[4]f64{ 19.0, 48.0, math.inf(f64), 3.0 };

    var compressed_values = ArrayList(u8).init(allocator);
    defer compressed_values.deinit();

    compress(
        allocator,
        uncompressed_values,
        &compressed_values,
        0.1,
    ) catch |err| {
        try testing.expectEqual(Error.UnsupportedInput, err);
        return;
    };

    try testing.expectFmt(
        "",
        "The non linear approximator algorithm cannot compress inf values",
        .{},
    );
}

test "non linear approximator cannot compress f64 with reduced precision" {
    const allocator = testing.allocator;

    const uncompressed_values = &[4]f64{ 19.0, 48.0, 1e17, 3.0 };

    var compressed_values = ArrayList(u8).init(allocator);
    defer compressed_values.deinit();

    compress(
        allocator,
        uncompressed_values,
        &compressed_values,
        0.1,
    ) catch |err| {
        try testing.expectEqual(Error.UnsupportedInput, err);
        return;
    };

    try testing.expectFmt(
        "",
        "The non linear approximator algorithm cannot compress reduced precision floating point values",
        .{},
    );
}
