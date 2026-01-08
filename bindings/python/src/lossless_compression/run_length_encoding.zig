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

//! Implementation of "Run-Length-Encoding" for compressing and decompressing time series data.
//! This implementation compresses sequences of repeated values into a single value and a count,
//! allowing for efficient storage and transmission of data with many repeated values.

const std = @import("std");
const mem = std.mem;
const Allocator = mem.Allocator;
const ArrayList = std.ArrayList;
const testing = std.testing;

const tersets = @import("../tersets.zig");
const configuration = @import("../configuration.zig");
const Method = tersets.Method;
const Error = tersets.Error;

const tester = @import("../tester.zig");
const shared_functions = @import("../utilities/shared_functions.zig");

/// Compresses the `uncompressed_values` using "Run-Length-Enconding". The function writes the
/// result to `compressed_values`. The `method_configuration` is expected to be `EmptyConfiguration`,
/// otherwise an error is returned instead of ignoring the configuration. The `allocator` is used
/// to allocate the `method_configuration` parser's memory. If an error occurs it is returned.
pub fn compress(
    allocator: Allocator,
    uncompressed_values: []const f64,
    compressed_values: *ArrayList(u8),
    method_configuration: []const u8,
) Error!void {
    _ = try configuration.parse(
        allocator,
        configuration.EmptyConfiguration,
        method_configuration,
    );

    var counter: usize = 0;
    var current_value: f64 = uncompressed_values[0];

    // Append the first value to the compressed values.
    try shared_functions.appendValue(allocator, f64, uncompressed_values[0], compressed_values);

    for (uncompressed_values) |value| {
        if (value == current_value) {
            counter += 1;
        } else {
            // Append the count of the previous value.
            try shared_functions.appendValue(allocator, usize, counter, compressed_values);
            // Reset for the new value.
            current_value = value;
            counter = 1;
            // Append the new value.
            try shared_functions.appendValue(allocator, f64, value, compressed_values);
        }
    }

    // Append the count of the last value.
    try shared_functions.appendValue(allocator, usize, counter, compressed_values);
}

/// Decompress `compressed_values` produced by "Run-Length-Encoding" and write the
/// result to `decompressed_values`. If an error occurs it is returned.
pub fn decompress(allocator: Allocator, compressed_values: []const u8, decompressed_values: *ArrayList(f64)) Error!void {
    if (compressed_values.len % 16 != 0) return Error.UnsupportedInput;

    const compressed_representation = mem.bytesAsSlice(f64, compressed_values);

    var index: usize = 0;
    while (index < compressed_representation.len) : (index += 2) {
        const value: f64 = compressed_representation[index];
        const count: usize = @bitCast(compressed_representation[index + 1]);

        // Append the value `count` times to the decompressed values.
        for (0..count) |_| {
            try decompressed_values.append(allocator, value);
        }
    }
}

test "rle can always compress and decompress" {
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

    // This function evaluates RunLengthEncoding using all data distribution stored in
    // `data_distribution`. The error bound is ignored as RLE does not use it.
    try tester.testLosslessMethod(
        allocator,
        Method.RunLengthEncoding,
        data_distributions,
    );
}

test "run length encoding compresses repeated values" {
    const allocator = testing.allocator;

    var uncompressed_values = ArrayList(f64).empty;
    defer uncompressed_values.deinit(allocator);

    // Generate a random number of `distinct_elements` that will be repeated a random number of times
    // to test that RLE can compress repeated values.
    const distinct_elements: usize = tester.generateBoundRandomInteger(
        usize,
        tester.global_at_least,
        tester.global_at_most,
        null,
    );

    for (0..distinct_elements) |_| {
        const random_value = tester.generateRandomValue(null);
        const repeat: usize = tester.generateBoundRandomInteger(
            usize,
            tester.global_at_least,
            tester.global_at_most,
            null,
        );
        for (0..repeat) |_| {
            try uncompressed_values.append(allocator, random_value);
        }
    }

    const method_configuration = "{}";

    var compressed_values = try tersets.compress(
        allocator,
        uncompressed_values.items,
        Method.RunLengthEncoding,
        method_configuration,
    );
    defer compressed_values.deinit(allocator);

    var decompressed_values = try tersets.decompress(allocator, compressed_values.items);
    defer decompressed_values.deinit(allocator);

    try testing.expect(shared_functions.isWithinErrorBound(
        uncompressed_values.items,
        decompressed_values.items,
        0.0,
    ));
}

test "check rle configuration parsing" {
    // Tests the configuration parsing and functionality of the `compressMidrange` function.
    // The test verifies that the provided configuration is correctly interpreted and
    // that the `configuration.EmptyConfiguration` is expected in the function.
    const allocator = testing.allocator;

    const uncompressed_values = &[4]f64{ 19.0, 48.0, 29.0, 3.0 };

    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);

    const method_configuration =
        \\ {}
    ;

    // The configuration is properly defined. No error expected.
    try compress(
        allocator,
        uncompressed_values,
        &compressed_values,
        method_configuration,
    );
}
