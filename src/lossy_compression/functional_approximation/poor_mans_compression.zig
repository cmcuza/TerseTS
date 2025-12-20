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

//! Implementation of "Poor Man’s Compression - Midrange" and "Poor Man’s Compression - Mean" from
//! the paper "Iosif Lazaridis, Sharad Mehrotra: Capturing Sensor-Generated Time Series with Quality
//! Guarantees. ICDE 2003: 429-440
//! https://doi.org/10.1109/ICDE.2003.1260811".

const std = @import("std");
const ArrayList = std.ArrayList;
const math = std.math;
const mem = std.mem;
const testing = std.testing;
const Allocator = std.mem.Allocator;

const tersets = @import("../../tersets.zig");
const Method = tersets.Method;
const Error = tersets.Error;
const tester = @import("../../tester.zig");

const shared_functions = @import("../../utilities/shared_functions.zig");
const configuration = @import("../../configuration.zig");

/// Compress `uncompressed_values` within `error_bound` using "Poor Man’s Compression - Midrange".
/// The function writes the result to `compressed_values`. The `allocator` is used to allocate
/// memory for the `method_configuration` parser. The `method_configuration` is expected to
/// be of `AbsoluteErrorBound` type otherwise an `InvalidConfiguration` error is return.
/// If any other error occurs during the execution of the method, it is returned.
pub fn compressMidrange(
    allocator: Allocator,
    uncompressed_values: []const f64,
    compressed_values: *ArrayList(u8),
    method_configuration: []const u8,
) Error!void {
    var index: usize = 0; // n.

    const parsed_configuration = try configuration.parse(
        allocator,
        configuration.AbsoluteErrorBound,
        method_configuration,
    );

    const error_bound: f32 = parsed_configuration.abs_error_bound;

    var minimum: f80 = uncompressed_values[0]; // m.
    var maximum: f80 = uncompressed_values[0]; // M.

    for (uncompressed_values) |value| {
        const nextMinimum = @min(value, minimum);
        const nextMaximum = @max(value, maximum);

        if ((nextMaximum - nextMinimum) > 2 * error_bound) {
            const compressed_value: f64 = @floatCast((maximum + minimum) / 2);
            try shared_functions.appendValueAndIndexToArrayList(
                allocator,
                compressed_value,
                index,
                compressed_values,
            );
            minimum = value;
            maximum = value;
        } else {
            minimum = nextMinimum;
            maximum = nextMaximum;
        }
        index += 1;
    }

    const compressed_value: f64 = @floatCast((maximum + minimum) / 2);
    try shared_functions.appendValueAndIndexToArrayList(
        allocator,
        compressed_value,
        index,
        compressed_values,
    );
}

/// Compress `uncompressed_values` within `error_bound` using "Poor Man’s Compression - Mean".
/// The function writes the result to `compressed_values`. he `allocator` is used to
/// allocate memory for the `method_configuration` parser. The `method_configuration` is expected
/// to be of `AbsoluteErrorBound` type otherwise an `InvalidConfiguration` error is return.
/// If any other error occurs during the execution of the method, it is returned.
pub fn compressMean(
    allocator: Allocator,
    uncompressed_values: []const f64,
    compressed_values: *ArrayList(u8),
    method_configuration: []const u8,
) Error!void {
    var index: usize = 0; // n.
    var minimum = math.nan(f64); // m.
    var maximum = math.nan(f64); // M.
    var length: f64 = 0;
    var average: f80 = 0.0;

    const parsed_configuration = try configuration.parse(
        allocator,
        configuration.AbsoluteErrorBound,
        method_configuration,
    );

    const error_bound: f32 = parsed_configuration.abs_error_bound;

    if (error_bound < 0)
        return Error.UnsupportedErrorBound;

    for (uncompressed_values) |value| {
        const nextMinimum = @min(value, minimum);
        const nextMaximum = @max(value, maximum);
        const nextLength = length + 1;
        const nextAverage = (average * length + value) / nextLength;

        if ((nextMaximum - nextAverage > error_bound) or (nextAverage - nextMinimum > error_bound)) {
            try shared_functions.appendValueAndIndexToArrayList(
                allocator,
                @floatCast(average),
                index,
                compressed_values,
            );
            minimum = value;
            maximum = value;
            length = 1;
            average = value;
        } else {
            minimum = nextMinimum;
            maximum = nextMaximum;
            length = nextLength;
            average = nextAverage;
        }
        index += 1;
    }

    try shared_functions.appendValueAndIndexToArrayList(
        allocator,
        @floatCast(average),
        index,
        compressed_values,
    );
}

/// Decompress `compressed_values` produced by "Poor Man’s Compression - Midrange" and
/// "Poor Man’s Compression - Mean". The function writes the result to `decompressed_values`.
/// If an error occurs it is returned.
pub fn decompress(
    allocator: Allocator,
    compressed_values: []const u8,
    decompressed_values: *ArrayList(f64),
) Error!void {
    // The compressed representation is pairs containing a 64-bit float value and 64-bit end index.
    if (compressed_values.len % 16 != 0) return Error.UnsupportedInput;

    const compressed_values_and_index = mem.bytesAsSlice(f64, compressed_values);

    var compressed_index: usize = 0;
    var uncompressed_index: usize = 0;
    while (compressed_index < compressed_values_and_index.len) : (compressed_index += 2) {
        const value = compressed_values_and_index[compressed_index];
        const index: usize = @bitCast(compressed_values_and_index[compressed_index + 1]);
        for (uncompressed_index..index) |_| {
            try decompressed_values.append(allocator, value);
        }
        uncompressed_index = index;
    }
}

test "midrange can always compress and decompress with zero error bound" {
    const allocator = testing.allocator;
    try tester.testGenerateCompressAndDecompress(
        allocator,
        tester.generateRandomValues,
        Method.PoorMansCompressionMidrange,
        0,
        shared_functions.isWithinErrorBound,
    );
}

test "midrange can always compress and decompress with positive error bound" {
    const allocator = testing.allocator;
    const data_distributions = &[_]tester.DataDistribution{
        .FiniteRandomValues,
        .LinearFunctions,
        .BoundedRandomValues,
        .SinusoidalFunction,
        .LinearFunctionsWithNansAndInfinities,
        .RandomValuesWithNansAndInfinities,
        .SinusoidalFunctionWithNansAndInfinities,
        .BoundedRandomValuesWithNansAndInfinities,
    };

    // This function evaluates PoorMansCompressionMidrange using all data distribution stored in
    // `data_distribution` with a positive error bound ranging from [1e-4, 1)*range
    // of the generated uncompressed time series.
    try tester.testErrorBoundedCompressionMethod(
        allocator,
        Method.PoorMansCompressionMidrange,
        data_distributions,
    );
}

test "mean can always compress and decompress with zero error bound" {
    const allocator = testing.allocator;
    try tester.testGenerateCompressAndDecompress(
        allocator,
        tester.generateRandomValues,
        Method.PoorMansCompressionMean,
        0,
        shared_functions.isWithinErrorBound,
    );
}

test "mean can always compress and decompress with positive error bound" {
    const allocator = testing.allocator;
    const data_distributions = &[_]tester.DataDistribution{
        .FiniteRandomValues,
        .LinearFunctions,
        .BoundedRandomValues,
        .SinusoidalFunction,
        .LinearFunctionsWithNansAndInfinities,
        .RandomValuesWithNansAndInfinities,
        .SinusoidalFunctionWithNansAndInfinities,
        .BoundedRandomValuesWithNansAndInfinities,
    };

    // This function evaluates PoorMansCompressionMean using all data distribution stored in
    // `data_distribution` with a positive error bound ranging from [1e-4, 1)*range
    // of the generated uncompressed time series.
    try tester.testErrorBoundedCompressionMethod(
        allocator,
        Method.PoorMansCompressionMean,
        data_distributions,
    );
}

test "check pmc-mean configuration parsing" {
    // Tests the configuration parsing and functionality of the `compressMean` function.
    // The test verifies that the provided configuration is correctly interpreted and
    // that the `configuration.AbsoluteErrorBound` is expected in the function.
    const allocator = testing.allocator;

    const uncompressed_values = &[4]f64{ 19.0, 48.0, 29.0, 3.0 };

    var compressed_values = ArrayList(u8).init(allocator);
    defer compressed_values.deinit();

    const method_configuration =
        \\ {"abs_error_bound": 0.1}
    ;

    // The configuration is properly defined. No error expected.
    try compressMean(
        allocator,
        uncompressed_values,
        &compressed_values,
        method_configuration,
    );
}

test "check pmc-midrange configuration parsing" {
    // Tests the configuration parsing and functionality of the `compressMidrange` function.
    // The test verifies that the provided configuration is correctly interpreted and
    // that the `configuration.AbsoluteErrorBound` is expected in the function.
    const allocator = testing.allocator;

    const uncompressed_values = &[4]f64{ 19.0, 48.0, 29.0, 3.0 };

    var compressed_values = ArrayList(u8).init(allocator);
    defer compressed_values.deinit();

    const method_configuration =
        \\ {"abs_error_bound": 0.1}
    ;

    // The configuration is properly defined. No error expected.
    try compressMidrange(
        allocator,
        uncompressed_values,
        &compressed_values,
        method_configuration,
    );
}
