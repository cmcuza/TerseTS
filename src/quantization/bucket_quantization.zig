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

//! Implementation of the "Fixed-width Uniform Quantization Method" based on the description at
//! https://en.wikipedia.org/wiki/Quantization_(signal_processing). Quantization is the process
//! of mapping input values from a large set (often continuous) to output values in a smaller set.
//! It can be used to compress data by reducing the precision of the values, which is particularly
//! useful in time series compression. It can be in combination with other methods to achieve better
//! compression ratios.

const std = @import("std");
const ArrayList = std.ArrayList;
const math = std.math;
const mem = std.mem;
const testing = std.testing;

const tersets = @import("../tersets.zig");
const Method = tersets.Method;
const Error = tersets.Error;
const tester = @import("../tester.zig");

/// Compress `uncompressed_values` within `error_bound` using "Bucket Quantization".
/// The function writes the result to `compressed_values`. If an error occurs it is returned.
pub fn compress(
    uncompressed_values: []const f64,
    compressed_values: *ArrayList(u8),
    error_bound: f32,
) Error!void {
    if (error_bound < 0.0) return Error.UnsupportedErrorBound;

    // Multiply by 1.999 instead of 2 to avoid potential overflow or rounding issues
    // during quantization. Using 2 could cause values at the upper bound to exceed
    // the maximum representable value, while 1.999 ensures results stay within range.
    const buket_size = 1.9999 * error_bound;

    for (uncompressed_values) |value| {
        if (!std.math.isFinite(value)) return Error.UnsupportedInput;

        // Map the value to a quantized value within the error bound.
        const quantized_value = if (error_bound != 0.0)
            @floor(value / buket_size + 0.5) * buket_size
        else
            value;

        try appendValue(f64, quantized_value, compressed_values);
    }

    return;
}

/// Decompress `compressed_values` produced by "Bucket Quantization". The function writes the
/// result to `decompressed_values`. If an error occurs it is returned.
pub fn decompress(
    compressed_values: []const u8,
    decompressed_values: *ArrayList(f64),
) Error!void {
    // Ensure the compressed values are not empty.
    if (compressed_values.len == 0) return Error.UnsupportedInput;

    const compressed_representation = mem.bytesAsSlice(f64, compressed_values);
    // Iterate over the compressed values and convert them back to f64.
    for (compressed_representation) |value| {
        try decompressed_values.append(value);
    }
}

/// Append `value` of `type` determined at compile time to `compressed_values`.
fn appendValue(comptime T: type, value: T, compressed_values: *ArrayList(u8)) !void {
    // Compile-time type check.
    switch (@TypeOf(value)) {
        f64, usize => {
            const value_as_bytes: [8]u8 = @bitCast(value);
            try compressed_values.appendSlice(value_as_bytes[0..]);
        },
        else => @compileError("Unsupported type for append value function"),
    }
}

test "bucket quantization can always compress and decompress" {
    const allocator = testing.allocator;
    try tester.testGenerateCompressAndDecompress(
        tester.generateFiniteRandomValues,
        allocator,
        Method.BucketQuantization,
        0,
        tersets.isWithinErrorBound,
    );
}

test "bucket quantization can compress and decompress bounded values" {
    const allocator = testing.allocator;
    const error_bound = tester.generateBoundedRandomValue(f32, 0, 1e5, undefined);

    var uncompressed_values = ArrayList(f64).init(allocator);
    defer uncompressed_values.deinit();

    // Generate 500 random values within the range of -1e7 to 1e7.
    for (0..5) |_| {
        try tester.generateBoundedRandomValues(&uncompressed_values, -1e7, 1e7, undefined);
    }

    try tester.testCompressAndDecompress(
        uncompressed_values.items,
        allocator,
        Method.BucketQuantization,
        error_bound,
        tersets.isWithinErrorBound,
    );
}
