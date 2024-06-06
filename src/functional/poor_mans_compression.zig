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

const tersets = @import("../tersets.zig");
const Method = tersets.Method;
const Error = tersets.Error;
const tester = @import("../tester.zig");

/// Compress `uncompressed_values` within `error_bound` using "Poor Man’s Compression - Midrange"
/// and write the result to `compressed_values`. If an error occurs it is returned.
pub fn compressMidrange(
    uncompressed_values: []const f64,
    compressed_values: *ArrayList(u8),
    error_bound: f32,
) Error!void {
    var index: usize = 0; // n.
    var minimum: f80 = uncompressed_values[0]; // m.
    var maximum: f80 = uncompressed_values[0]; // M.

    for (uncompressed_values) |value| {
        const nextMinimum = @min(value, minimum);
        const nextMaximum = @max(value, maximum);

        if ((nextMaximum - nextMinimum) > 2 * error_bound) {
            const compressed_value = (maximum + minimum) / 2;
            try appendValueAndIndexToArrayList(compressed_value, index, compressed_values);
            minimum = value;
            maximum = value;
        } else {
            minimum = nextMinimum;
            maximum = nextMaximum;
        }
        index += 1;
    }

    const compressed_value = (maximum + minimum) / 2;
    try appendValueAndIndexToArrayList(compressed_value, index, compressed_values);
}

/// Compress `uncompressed_values` within `error_bound` using "Poor Man’s Compression - Mean"
/// and write the result to `compressed_values`. If an error occurs it is returned.
pub fn compressMean(
    uncompressed_values: []const f64,
    compressed_values: *ArrayList(u8),
    error_bound: f32,
) Error!void {
    var index: usize = 0; // n.
    var minimum = math.nan(f64); // m.
    var maximum = math.nan(f64); // M.
    var length: f64 = 0;
    var average: f80 = 0.0;

    for (uncompressed_values) |value| {
        const nextMinimum = @min(value, minimum);
        const nextMaximum = @max(value, maximum);
        const nextLength = length + 1;
        const nextAverage = (average * length + value) / nextLength;

        if ((nextMaximum - nextAverage > error_bound) or (nextAverage - nextMinimum > error_bound)) {
            try appendValueAndIndexToArrayList(average, index, compressed_values);
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

    try appendValueAndIndexToArrayList(average, index, compressed_values);
}

/// Decompress `compressed_values` produced by "Poor Man’s Compression - Midrange" and
/// "Poor Man’s Compression - Mean" and write the result to `decompressed_values`. If an error
/// occurs it is returned.
pub fn decompress(
    compressed_values: []const u8,
    decompressed_values: *ArrayList(f64),
) Error!void {
    // The compressed representation is pairs containing a 64-bit float value and 64-bit end index.
    if (compressed_values.len % 16 != 0) return Error.IncorrectInput;

    const compressed_values_and_index = mem.bytesAsSlice(f64, compressed_values);

    var compressed_index: usize = 0;
    var uncompressed_index: usize = 0;
    while (compressed_index < compressed_values_and_index.len) : (compressed_index += 2) {
        const value = compressed_values_and_index[compressed_index];
        const index: usize = @bitCast(compressed_values_and_index[compressed_index + 1]);
        for (uncompressed_index..index) |_| {
            try decompressed_values.append(value);
        }
        uncompressed_index = index;
    }
}

/// Append `compressed_value` and `index` to `compressed_values`.
fn appendValueAndIndexToArrayList(
    compressed_value: f80,
    index: usize,
    compressed_values: *ArrayList(u8),
) !void {
    const value: f64 = @floatCast(compressed_value);
    const valueAsBytes: [8]u8 = @bitCast(value);
    try compressed_values.appendSlice(valueAsBytes[0..]);
    const indexAsBytes: [8]u8 = @bitCast(index); // No -1 due to 0 indexing.
    try compressed_values.appendSlice(indexAsBytes[0..]);
}

test "midrange can always compress and decompress" {
    const allocator = testing.allocator;
    const uncompressed_values = try tester.generateRandomValues(allocator);
    defer uncompressed_values.deinit();

    try tester.testCompressionAndDecompression(
        uncompressed_values.items,
        allocator,
        Method.PoorMansCompressionMidrange,
        0,
        tersets.isWithinErrorBound,
    );
}

test "mean can always compress and decompress" {
    const allocator = testing.allocator;
    const uncompressed_values = try tester.generateRandomValues(allocator);
    defer uncompressed_values.deinit();

    try tester.testCompressionAndDecompression(
        uncompressed_values.items,
        allocator,
        Method.PoorMansCompressionMean,
        0,
        tersets.isWithinErrorBound,
    );
}
