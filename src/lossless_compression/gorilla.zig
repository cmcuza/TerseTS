// Copyright 2026 TerseTS Contributors
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

//! Implementation of Gorilla floating-point time series compression method.
//! This implements the data value compression for floating-point data, which
//! XOR's each value with the previous one. The result is stored as a compact
//! tuple including the leading-zero count, meaningful-bit count, followed by the
//! meaningful bits.
//!
//! The XOR value compression is described in the original publication:
//! Pelkonen et al., "Gorilla: A Fast, Scalable, In-Memory Time Series Database", VLDB 2015.
//! https://doi.org/10.14778/2824032.2824078

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
const shared_structs = @import("../utilities/shared_structs.zig");

/// Compresses the `uncompressed_values` using "Gorilla". The function writes the
/// result to `compressed_values`. The `method_configuration` is expected to be `EmptyConfiguration`,
/// otherwise an error is returned instead of ignoring the configuration. The `allocator` is used
/// to allocate the `method_configuration` parser's memory. `uncompressed_values` must contain at
/// least two values; the caller is responsible for handling the empty and single-value cases. If
/// an error occurs it is returned.
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

    // Store the value count so decompression knows where the bitstream ends before the zero padding.
    try shared_functions.appendValue(allocator, u64, @intCast(uncompressed_values.len), compressed_values);

    // Store first value uncompressed.
    const first_value: f64 = uncompressed_values[0];
    try shared_functions.appendValue(allocator, f64, first_value, compressed_values);

    var previous_value_bits: u64 = @bitCast(first_value);

    var bit_writer = try shared_structs.BulkBitWriter.init(allocator, compressed_values);

    // Gorilla tracks the number of leading zeros and trailing zeros to set the header.
    const dtype_bit_len: u7 = 64;
    var leading_zeros: u8 = dtype_bit_len + 1;
    var trailing_zeros: u8 = dtype_bit_len + 1;
    var meaningful_len: u8 = 0;
    for (uncompressed_values[1..]) |value| {
        const current_value_bits: u64 = @bitCast(value);
        const xor_value = previous_value_bits ^ current_value_bits;
        if (xor_value == 0) {
            // Write 0b0 for repeated values.
            try bit_writer.writeBits(@as(u1, 0), 1);
        } else {
            try bit_writer.writeBits(@as(u1, 1), 1);
            // Compute leading and trailing zeros.
            const l: u5 = @min(@clz(xor_value), 31);
            const t: u8 = @ctz(xor_value);
            if (l >= leading_zeros and t >= trailing_zeros) {
                // Nested region, don't change leading/trailing count.
                try bit_writer.writeBits(@as(u1, 0), 1);
            } else {
                // Not nested, need to first record the new leading/trailing count.
                try bit_writer.writeBits(@as(u1, 1), 1);
                meaningful_len = @intCast(dtype_bit_len - l - t);
                try bit_writer.writeBits(l, 5);
                try bit_writer.writeBits(meaningful_len - 1, 6);
                leading_zeros = l;
                trailing_zeros = t;
            }
            // Store meaningful bits from xor value.
            const meaningful = xor_value >> @intCast(trailing_zeros);
            try bit_writer.writeBits(meaningful, meaningful_len);
            previous_value_bits = current_value_bits;
        }
    }
    try bit_writer.flushBits();
}

/// Decompress a Gorilla-encoded `compressed_values` stream into `decompressed_values`. `allocator`
/// reserves capacity in `decompressed_values` up front from the count header, so recovered values
/// are appended without further growth checks. `compressed_values` must start with the
/// `[count: u64][first_value: f64]` header written by `compress`, which guarantees a count of at
/// least 2; malformed or truncated streams return `Error.CorruptedCompressedData` or
/// `Error.ByteStreamError` rather than trapping. If an error occurs it is returned.
pub fn decompress(allocator: Allocator, compressed_values: []const u8, decompressed_values: *ArrayList(f64)) Error!void {
    var offset: usize = 0;
    const value_count = try shared_functions.readOffsetValue(u64, compressed_values, &offset);

    // A zero count is not produced by `compress`, so it indicates a corrupt stream.
    if (value_count == 0) return Error.CorruptedCompressedData;

    // Every non-empty Gorilla stream must contain the count header and first uncompressed value.
    if (compressed_values.len < 16) return Error.CorruptedCompressedData;

    // The header gives the exact output length, so reserve it once and append without growth checks.
    try decompressed_values.ensureTotalCapacity(allocator, @intCast(value_count));

    const first_value = try shared_functions.readOffsetValue(f64, compressed_values, &offset);

    // Number of bits for f64.
    const dtype_bit_len: u64 = 64;
    decompressed_values.appendAssumeCapacity(first_value);

    var bit_reader = shared_structs.BulkBitReader.init(compressed_values[offset..]);

    var current_value_bits: u64 = @bitCast(first_value);
    var leading: u6 = 0;
    var trailing: u6 = 0;
    var window: u7 = 0;
    while (decompressed_values.items.len < value_count) {
        const identical_bit = bit_reader.readBitsNoEof(u1, 1) catch return Error.ByteStreamError;
        if (identical_bit != 0) {
            // Change current value if header begins with 1.
            const window_bit = bit_reader.readBitsNoEof(u1, 1) catch return Error.ByteStreamError;
            if (window_bit != 0) {
                // Change current leading and trailing count.
                leading = bit_reader.readBitsNoEof(u6, 5) catch return Error.ByteStreamError;
                window = (bit_reader.readBitsNoEof(u7, 6) catch return Error.ByteStreamError) + 1;
                if (@as(u8, leading) + @as(u8, window) > dtype_bit_len) {
                    // Leading zeros and window should be <= dtype bits, otherwise header is corrupted.
                    return Error.CorruptedCompressedData;
                }
                trailing = @intCast(dtype_bit_len - leading - window);
            }
            const xor_bits = (bit_reader.readBitsNoEof(u64, window) catch return Error.ByteStreamError) << trailing;
            current_value_bits ^= xor_bits;
        }
        const value: f64 = @bitCast(current_value_bits);
        decompressed_values.appendAssumeCapacity(value);
    }
}

test "gorilla can always compress and decompress" {
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

    // This function evaluates Gorilla using all data distribution stored in
    // `data_distribution`. The error bound is ignored as Gorilla does not use it.
    try tester.testLosslessMethod(
        allocator,
        Method.Gorilla,
        data_distributions,
    );
}

test "gorilla compresses repeated values" {
    const allocator = testing.allocator;

    var uncompressed_values = ArrayList(f64).empty;
    defer uncompressed_values.deinit(allocator);

    // Generate a random number of `distinct_elements` that will be repeated a random number of times
    // to test that Gorilla can compress repeated values.
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
        Method.Gorilla,
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

test "gorilla correctly errors for malformed headers" {
    const allocator = testing.allocator;

    const corrupt_compressed = &[_]u8{
        0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // vlen bytes.
        0xDE, 0xAD, 0xFA, 0xDE, 0x00, 0x00, 0x00, 0x00, // first literal.
        0xFF, 0xFF, 0xFF, 0xFF, // corrupt header.
    };
    var uncompressed_values = ArrayList(f64).empty;
    defer uncompressed_values.deinit(allocator);

    try std.testing.expectError(Error.CorruptedCompressedData, decompress(allocator, corrupt_compressed, &uncompressed_values));
}

test "check gorilla configuration parsing" {
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
