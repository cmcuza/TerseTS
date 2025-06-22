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
//! https://en.wikipedia.org/wiki/Quantization_(signal_processing). Quantization is the process of mapping input values
//! from a large set (often continuous) to output values in a smaller set (often discrete).
//! https://en.wikipedia.org/wiki/Quantization_(signal_processing)

const std = @import("std");
const ArrayList = std.ArrayList;
const math = std.math;
const mem = std.mem;
const testing = std.testing;

const tersets = @import("../tersets.zig");
const Method = tersets.Method;
const Error = tersets.Error;
const tester = @import("../tester.zig");

/// Compress `uncompressed_values` within `error_bound` using "Quantization".
/// The function writes the result to `compressed_values`. If an error occurs it is returned.
pub fn compress(
    uncompressed_values: []const f64,
    compressed_values: *ArrayList(u8),
    error_bound: f32,
) Error!ArrayList(f64) {
    if (error_bound < 0.0) return Error.UnsupportedErrorBound;

    for (uncompressed_values) |value| {
        if (!std.math.isFinite(value)) return Error.UnsupportedInput;

        // Map the value to a quantized value within the error bound.
        const quantized_value = if (error_bound != 0.0)
            @floor(value / error_bound) * error_bound
        else
            value;

        try appendValue(f64, quantized_value, &compressed_values);
    }
}

/// Append `value` of `type` determined at compile time to `compressed_values`.
fn appendValue(comptime T: type, value: T, compressed_values: *std.ArrayList(u8)) !void {
    // Compile-time type check
    switch (@TypeOf(value)) {
        f64, usize => {
            const value_as_bytes: [8]u8 = @bitCast(value);
            try compressed_values.appendSlice(value_as_bytes[0..]);
        },
        else => @compileError("Unsupported type for append value function"),
    }
}

test "quantized values are within the error bound" {
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0);
    const random = prng.random();
    const error_bound: f32 = 0.5;

    var values = ArrayList(f64).init(allocator);
    defer values.deinit();
    for (0..100) |_| {
        try values.append(random.float(f64) * 100.0);
    }

    const quantized = try quantizeValues(values.items, allocator, error_bound);
    defer quantized.deinit();

    try testing.expectEqual(values.items.len, quantized.items.len);

    for (quantized.items, 0..) |q, i| {
        const original = values.items[i];
        try testing.expect(@abs(original - q) <= error_bound);
    }
}
