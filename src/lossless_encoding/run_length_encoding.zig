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
const ArrayList = std.ArrayList;

const tersets = @import("../tersets.zig");
const Method = tersets.Method;
const Error = tersets.Error;

const tester = @import("../tester.zig");

const testing = std.testing;

/// Compresses the `uncompressed_values` using "Run-Length-Enconding". The function writes the
/// result to `compressed_values`. If an error occurs it is returned.
pub fn compress(
    uncompressed_values: []const f64,
    compressed_values: *ArrayList(u8),
) Error!void {
    var counter: usize = 0;
    var current_value: f64 = uncompressed_values[0];

    // Append the first value to the compressed values.
    try appendValue(f64, uncompressed_values[0], compressed_values);

    for (uncompressed_values) |value| {
        if (value == current_value) {
            counter += 1;
        } else {
            // Append the count of the previous value.
            try appendValue(usize, counter, compressed_values);
            // Reset for the new value.
            current_value = value;
            counter = 1;
            // Append the new value.
            try appendValue(f64, value, compressed_values);
        }
    }

    // Append the count of the last value.
    try appendValue(usize, counter, compressed_values);
}

/// Decompress `compressed_values` produced by "Run-Length-Encoding" and write the
/// result to `decompressed_values`. If an error occurs it is returned.
pub fn decompress(compressed_values: []const u8, decompressed_values: *ArrayList(f64)) Error!void {
    if (compressed_values.len % 16 != 0) return Error.UnsupportedInput;

    const compressed_representation = mem.bytesAsSlice(f64, compressed_values);

    var index: usize = 0;
    while (index < compressed_representation.len) : (index += 2) {
        const value: f64 = compressed_representation[index];
        const count: usize = @bitCast(compressed_representation[index + 1]);

        // Append the value `count` times to the decompressed values.
        for (0..count) |_| {
            try decompressed_values.append(value);
        }
    }
}

/// Append `value` of `type` determined at compile time to `compressed_values`.
fn appendValue(comptime T: type, value: T, compressed_values: *std.ArrayList(u8)) !void {
    // Compile-time type check.
    switch (@TypeOf(value)) {
        f64, usize => {
            const value_as_bytes: [8]u8 = @bitCast(value);
            try compressed_values.appendSlice(value_as_bytes[0..]);
        },
        else => @compileError("Unsupported type for append value function"),
    }
}

test "run length encoding can compress and decompress values" {
    const allocator = testing.allocator;
    try tester.testGenerateCompressAndDecompress(
        tester.generateRandomValues,
        allocator,
        Method.RunLengthEncoding,
        0,
        tersets.isWithinErrorBound,
    );
}

test "run length encoding compresses repeated values" {
    const allocator = testing.allocator;

    var uncompressed_values = ArrayList(f64).init(allocator);
    defer uncompressed_values.deinit();

    // Generate a random number of `distinct_elements` that will be repeated a random number of time
    // to test that RLE can compress repeated values.
    const distinct_elements: usize = tester.generateBoundRandomInteger(usize, 10, 50, undefined);

    for (0..distinct_elements) |_| {
        const random_value = tester.generateBoundedRandomValue(f64, -1e-16, 1e16, undefined);
        const repeat: usize = tester.generateBoundRandomInteger(usize, 10, 20, undefined);
        for (0..repeat) |_| {
            try uncompressed_values.append(random_value);
        }
    }

    try tester.testCompressAndDecompress(
        uncompressed_values.items,
        allocator,
        Method.RunLengthEncoding,
        0,
        tersets.isWithinErrorBound,
    );
}
