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

const params = @import("../params.zig");

/// Compress `uncompressed_values` within `error_bound` using "Poor Man’s Compression - Midrange".
/// The function writes the result to `compressed_values`. If an error occurs it is returned.
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

/// Compress `uncompressed_values` within `error_bound` using "Poor Man’s Compression - Mean".
/// The function writes the result to `compressed_values`. If an error occurs it is returned.
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

/// Compress `uncompressed_values` using "Poor Man’s Compression - Midrange" with a
/// `relative_error_bound`. The error bound is a ratio (e.g., 0.01 = 1%), which ensures that every
/// decompressed value is within `relative_error_bound` * |uncompressed_values|. The function writes
/// the result to `compressed_values`. If an error occurs it is returned.
/// Compress using PMC with relative error bound.
/// Guarantees that each reconstructed value is within `±|v| * relative_bound`.
pub fn compressMidrangeRelative(
    uncompressed_values: []const f64,
    compressed_values: *ArrayList(u8),
    relative_bound: f32,
) Error!void {
    if (uncompressed_values.len == 0) return;

    var minimum: f80 = uncompressed_values[0];
    var maximum: f80 = uncompressed_values[0];
    var tightest_epsilon: f64 = @abs(uncompressed_values[0]) * relative_bound;

    var index: usize = 1;
    for (uncompressed_values[1..]) |value| {
        const next_min = @min(value, minimum);
        const next_max = @max(value, maximum);
        const new_epsilon = @abs(value) * relative_bound;
        const updated_tightest = @min(tightest_epsilon, new_epsilon);

        if ((next_max - next_min) > 2 * updated_tightest) {
            const compressed_value = (maximum + minimum) / 2;
            try appendValueAndIndexToArrayList(compressed_value, index, compressed_values);

            // Start new segment
            minimum = value;
            maximum = value;
            tightest_epsilon = new_epsilon;
        } else {
            minimum = next_min;
            maximum = next_max;
            tightest_epsilon = updated_tightest;
        }

        index += 1;
    }

    const compressed_value = (maximum + minimum) / 2;
    try appendValueAndIndexToArrayList(compressed_value, index, compressed_values);
}

/// Decompress `compressed_values` produced by "Poor Man’s Compression - Midrange" and
/// "Poor Man’s Compression - Mean". The function writes the result to `decompressed_values`.
/// If an error occurs it is returned.
pub fn decompress(
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
    try tester.testGenerateCompressAndDecompress(
        tester.generateRandomValues,
        allocator,
        Method.PoorMansCompressionMidrange,
        0,
        tersets.isWithinErrorBound,
    );
}

test "mean can always compress and decompress" {
    const allocator = testing.allocator;
    try tester.testGenerateCompressAndDecompress(
        tester.generateRandomValues,
        allocator,
        Method.PoorMansCompressionMean,
        0,
        tersets.isWithinErrorBound,
    );
}

test "midrange relative can always compress and decompress" {
    // Test that the compressed values are within the relative error bound.
    // The function is different from the ones above, as it uses a relative error bound and
    // the testGenerateCompressAndDecompress function does not support relative error bounds yet.
    const allocator = testing.allocator;

    var uncompressed_values = ArrayList(f64).init(allocator);
    defer uncompressed_values.deinit();

    // Generate a random relative error bound between 0.0 and 0.1, (i.e, 0% and 10%).
    const relative_error_bound: f32 = tester.generateBoundedRandomValue(f32, 0.0, 0.1, undefined);

    // Specifically for this test, we need to use the relative error bound.
    const parameters = params.FunctionalParams{
        .error_bound = relative_error_bound,
        .error_bound_type = .relative_error_bound,
    };

    try tester.generateBoundedRandomValues(&uncompressed_values, -100, 100, undefined);

    var compressed_values = try tersets.compress(
        uncompressed_values.items,
        allocator,
        Method.PoorMansCompressionMidrange,
        &parameters,
    );
    defer compressed_values.deinit();

    var decompressed_values = try tersets.decompress(
        compressed_values.items,
        allocator,
    );
    defer decompressed_values.deinit();

    try testing.expect(tersets.isWithinRelativeErrorBound(
        uncompressed_values.items,
        decompressed_values.items,
        relative_error_bound,
    ));
}
