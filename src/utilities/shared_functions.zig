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

//! Contains all shared functions used across TerseTS.

const std = @import("std");
const math = std.math;
const ArrayList = std.ArrayList;
const tersets = @import("../tersets.zig");
const tester = @import("../tester.zig");
const Error = tersets.Error;
const testing = std.testing;

const shared_structs = @import("shared_structs.zig");

/// Computes the Root-Mean-Squared-Errors (RMSE) for a segment of the `uncompressed_values`.
/// This function calculates the error between the actual values and the predicted values
/// based on a linear regression model fitted to the segment defined by `seg_start` and `seg_end`.
pub fn computeRMSE(uncompressed_values: []const f64, seg_start: usize, seg_end: usize) Error!f64 {
    const seg_len: f64 = @floatFromInt(seg_end - seg_start + 1);
    if (seg_len <= 1) return 0.0; // If the segment has one or no points, return zero error.

    // Check if the elements of the segment are within the valid range.
    if (uncompressed_values[seg_start] > tester.max_test_value or
        !math.isFinite(uncompressed_values[seg_start]) or
        uncompressed_values[seg_end] > tester.max_test_value or
        !math.isFinite(uncompressed_values[seg_end]))
    {
        return Error.UnsupportedInput;
    }
    // Calculate the slope and intercept of the line connecting the start and end points.
    const slope: f64 = (uncompressed_values[seg_end] - uncompressed_values[seg_start]) / (seg_len - 1);
    const intercept = uncompressed_values[seg_start];
    var sse: f64 = 0;
    var i: usize = seg_start;
    while (i <= seg_end) : (i += 1) {
        if (uncompressed_values[i] > tester.max_test_value or
            !math.isFinite(uncompressed_values[i])) return Error.UnsupportedInput;

        const scaled_time = @as(f64, @floatFromInt(i - seg_start)); // small numbers: 0,1,2,...
        const pred = intercept + slope * scaled_time;
        const diff = uncompressed_values[i] - pred;
        sse += diff * diff;
    }
    return math.sqrt(sse / seg_len);
}

/// Append `value` of `type` determined at compile time to `compressed_values`.
pub fn appendValue(comptime T: type, value: T, compressed_values: *ArrayList(u8)) !void {
    // Compile-time type check
    switch (@TypeOf(value)) {
        u64, i64, f64, usize => {
            const value_as_bytes: [8]u8 = @bitCast(value);
            try compressed_values.appendSlice(value_as_bytes[0..]);
        },
        u32, i32, f32 => {
            const value_as_bytes: [4]u8 = @bitCast(value);
            try compressed_values.appendSlice(value_as_bytes[0..]);
        },
        else => @compileError("Unsupported type for append value function"),
    }
}

/// Append `compressed_value` and `index` to `compressed_values`.
pub fn appendValueAndIndexToArrayList(
    compressed_value: f64,
    index: usize,
    compressed_values: *ArrayList(u8),
) !void {
    const valueAsBytes: [8]u8 = @bitCast(compressed_value);
    try compressed_values.appendSlice(valueAsBytes[0..]);
    const indexAsBytes: [8]u8 = @bitCast(index); // No -1 due to 0 indexing.
    try compressed_values.appendSlice(indexAsBytes[0..]);
}

/// Test if the RMSE of the linear regression line that fits the points in the segment in `values`
/// is within the `error_bound`.
pub fn testRMSEisWithinErrorBound(
    values: []const f64,
    error_bound: f32,
) !void {
    // At least two points are needed to form a line.
    if (values.len < 2) return;

    const rmse = try computeRMSE(values, 0, values.len - 1);
    try testing.expect(rmse <= error_bound);
}

/// Computes the maximum absolute (Chebyshev, L-inf) error between the actual values and the
/// linear interpolation over a segment of the input array. This function fits a straight
/// line between the values at `seg_start` and `seg_end` in `uncompressed_values`, then
/// calculates the maximum absolute difference between the actual values and the predicted
/// values (from the fitted line) for all indices in the segment `[seg_start, seg_end]`.
pub fn computeMaxAbsoluteError(uncompressed_values: []const f64, seg_start: usize, seg_end: usize) f64 {
    const seg_len: f64 = @floatFromInt(seg_end - seg_start + 1);
    if (seg_len <= 2) return 0.0; // If the segment has less than 3 points, return zero error.

    const slope: f64 = (uncompressed_values[seg_end] - uncompressed_values[seg_start]) / (seg_len - 1);
    const intercept: f64 = uncompressed_values[seg_start] - slope * @as(f64, @floatFromInt(seg_start));

    // Calculate the maximum absolute error of the segment.
    var linf: f64 = 0;
    var i = seg_start;
    while (i <= seg_end) : (i += 1) {
        const pred = slope * @as(f64, @floatFromInt(i)) + intercept; // Predicted value.
        const diff = @abs(uncompressed_values[i] - pred); // Difference between actual and predicted.
        linf = @max(diff, linf);
    }

    // Return max abs.
    return linf;
}

/// Auxiliary function to validate of the decompressed time series is within the error bound of the
/// uncompressed time series. The function returns true if all elements are within the error bound,
/// false otherwise.
pub fn isWithinErrorBound(
    uncompressed_values: []const f64,
    decompressed_values: []const f64,
    error_bound: f32,
) bool {
    if (uncompressed_values.len != decompressed_values.len) {
        return false;
    }

    for (0..uncompressed_values.len) |index| {
        const uncompressed_value = uncompressed_values[index];
        const decompressed_value = decompressed_values[index];
        if (@abs(uncompressed_value - decompressed_value) > error_bound) return false;
    }
    return true;
}

/// Reads a value of compile-time known type `T` from the beginning of the `compressed_values` byte
/// slice. Returns the value if the `compressed_values` contains at least `@sizeOf(T)` bytes. Return
/// an error otherwise.
pub fn readValue(comptime T: type, compressed_values: []const u8) Error!T {
    const size = @sizeOf(T);
    if (size > compressed_values.len) {
        return Error.UnsupportedInput; // Not enough bytes to read the value.
    }
    return @bitCast(compressed_values[0..size].*);
}

/// Returns `true` if two floating-point `value_a` and `value_b` numbers are approximately equal,
/// using both absolute and relative tolerances to account for rounding errors. This function is
/// necessary because direct comparison of floating-point values can fail due to rounding errors
/// and representation limitations inherent in floating-point arithmetic. Absolute tolerance is used
/// for values close to zero, while relative tolerance is used for larger magnitude values to ensure
/// a meaningful comparison. The values are fixed to 1e-12 and 1e-15 for absolute and relative
/// tolerances, respectively, which are suitable for f64 values.
pub fn isApproximatelyEqual(value_a: f64, value_b: f64) bool {
    if (value_a == value_b) return true;
    if (!math.isFinite(value_a) or !math.isFinite(value_b))
        return value_a == value_b; // Handle NaN and infinities.
    const abs_diff = @abs(value_a - value_b);
    const max_abs = @max(@abs(value_a), @abs(value_b));
    return abs_diff <= shared_structs.ABS_EPS or abs_diff <= max_abs * shared_structs.REL_EPS;
}
