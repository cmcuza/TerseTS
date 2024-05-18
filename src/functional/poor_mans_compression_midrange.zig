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

//! Implementation of "Poor Man’s Compression - Midrange" and "Poor Man’s
//! Compression - Mean" from the paper "Iosif Lazaridis, Sharad Mehrotra:
//! Capturing Sensor-Generated Time Series with Quality Guarantees. ICDE 2003:
//! 429-440".

const std = @import("std");
const math = std.math;
const mem = std.mem;
const testing = std.testing;
const ArrayList = std.ArrayList;

const tersets = @import("../tersets.zig");
const Error = tersets.Error;

/// Compress `uncompressed_values` within `error_bound` using "Poor Man’s Compression - Midrange"
/// and write the result to `compressed_values`. If an error occurs it is returned.
pub fn compress(
    uncompressed_values: []const f64,
    compressed_values: *ArrayList(u8),
    error_bound: f32,
) Error!void {
    var index: usize = 0; // n.
    var minimum = uncompressed_values[0]; // m.
    var maximum = uncompressed_values[0]; // M.

    for (uncompressed_values) |value| {
        if ((@max(value, maximum) - @min(value, minimum)) > 2 * error_bound) {
            const compressed_value = (maximum + minimum) / 2;
            try appendValueAndIndexToArrayList(compressed_value, index, compressed_values);
            minimum = value;
            maximum = value;
        } else {
            minimum = @min(value, minimum);
            maximum = @max(value, maximum);
        }
        index += 1;
    }

    const compressed_value = (maximum + minimum) / 2;
    try appendValueAndIndexToArrayList(compressed_value, index, compressed_values);
}

fn appendValueAndIndexToArrayList(
    value: f64,
    index: usize,
    compressed_values: *ArrayList(u8),
) !void {
    const valueAsBytes: [8]u8 = @bitCast(value);
    try compressed_values.appendSlice(valueAsBytes[0..]);
    const indexAsBytes: [8]u8 = @bitCast(index); // No -1 due to 0 indexing.
    try compressed_values.appendSlice(indexAsBytes[0..]);
}

/// Decompress `compressed_values` using "Poor Man’s Compression - Midrange" and write the result
/// to `decompressed_values`. If an error occurs it is returned.
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

test "PMC-MR" {
    const alloc = testing.allocator;
    const uncompressed_values = [_]f64{ 1.0, 2.0, 2.0, 3.0, 3.0, 3.0 };
    var compressed_values = ArrayList(u8).init(alloc);
    defer compressed_values.deinit();
    var decompressed_values = ArrayList(f64).init(alloc);
    defer decompressed_values.deinit();

    try compress(uncompressed_values[0..], &compressed_values, 0);
    try decompress(compressed_values.items, &decompressed_values);

    try testing.expect(mem.eql(f64, uncompressed_values[0..], decompressed_values.items));
}
