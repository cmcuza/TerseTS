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

/// Bit width of the leading-zero count in a new-window header.
const leading_zero_bits = 5;
/// Bit width of the meaningful-bit count in a new-window header, which stores the count minus one
/// because a new window always holds at least one meaningful bit.
const meaningful_count_bits = 6;
/// Largest leading-zero count `leading_zero_bits` can hold. A longer run is recorded as this value
/// and the extra zeros travel as meaningful bits.
const maximum_stored_leading_zeros = (1 << leading_zero_bits) - 1;
/// Placed in the window counts before the first window is written. It exceeds any real count, so
/// the first XOR-coded value always takes the new-window path.
const no_reusable_window = shared_structs.bits_per_value + 1;

/// End-of-stream marker, written as a new-window header after the last value. A real header keeps
/// `leading + count <= bits_per_value` because the remainder is the trailing-zero count, so this
/// pair cannot occur in data and the decoder needs no explicit value count.
const end_marker_leading_zeros: u5 = maximum_stored_leading_zeros;
const end_marker_meaningful_bit_count: u7 = shared_structs.bits_per_value;

/// Compress `uncompressed_values` into `compressed_values` using Gorilla, allocating with
/// `allocator`. `method_configuration` must be empty (`{}`), otherwise
/// `Error.InvalidConfiguration` is returned. `uncompressed_values` must not be empty;
/// `tersets.compress` guarantees this. On success `compressed_values` holds
/// `[first_value: f64][XOR bits][end-of-stream marker]`.
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

    // The first value has nothing to XOR against, so it is stored raw.
    const first_value: f64 = uncompressed_values[0];
    try shared_functions.appendValue(allocator, f64, first_value, compressed_values);

    var previous_value_bits: u64 = @bitCast(first_value);

    var bit_writer = try shared_structs.BulkBitWriter.init(allocator, compressed_values);

    // The window of meaningful bits carried over from the previous value.
    var window_leading_zeros: u8 = no_reusable_window;
    var window_trailing_zeros: u8 = no_reusable_window;
    var meaningful_bit_count: u8 = 0;

    for (uncompressed_values[1..]) |value| {
        const current_value_bits: u64 = @bitCast(value);
        const xor_value = previous_value_bits ^ current_value_bits;

        if (xor_value == 0) {
            // Marker `0`: repeated value.
            try bit_writer.writeBits(@as(u1, 0), 1);
        } else {
            try bit_writer.writeBits(@as(u1, 1), 1);

            const value_leading_zeros: u5 = @min(@clz(xor_value), maximum_stored_leading_zeros);
            const value_trailing_zeros: u8 = @ctz(xor_value);

            if (value_leading_zeros >= window_leading_zeros and
                value_trailing_zeros >= window_trailing_zeros)
            {
                // Marker `0`: the meaningful bits fit inside the previous window, so reuse it.
                try bit_writer.writeBits(@as(u1, 0), 1);
            } else {
                // Marker `1`: store a new window for this and later values to reuse.
                try bit_writer.writeBits(@as(u1, 1), 1);
                meaningful_bit_count = @intCast(shared_structs.bits_per_value -
                    @as(u16, value_leading_zeros) - @as(u16, value_trailing_zeros));
                try bit_writer.writeBits(value_leading_zeros, leading_zero_bits);
                try bit_writer.writeBits(meaningful_bit_count - 1, meaningful_count_bits);
                window_leading_zeros = value_leading_zeros;
                window_trailing_zeros = value_trailing_zeros;
            }

            // The window always fits the XOR here, so shifting drops only zero bits.
            const meaningful = xor_value >> @intCast(window_trailing_zeros);
            try bit_writer.writeBits(meaningful, meaningful_bit_count);
        }

        previous_value_bits = current_value_bits;
    }

    // The end marker tells the decoder where to stop; flushed padding bits are never read.
    try writeEndMarker(&bit_writer);
    try bit_writer.flushBits();
}

/// Decompress a Gorilla `compressed_values` stream into `decompressed_values`, allocating with
/// `allocator`. The stream must start with the raw `[first_value: f64]` written by `compress`;
/// malformed or truncated input returns `Error.CorruptedCompressedData`.
pub fn decompress(
    allocator: Allocator,
    compressed_values: []const u8,
    decompressed_values: *ArrayList(f64),
) Error!void {
    var offset: usize = 0;

    // The stream starts with the raw 8-byte first value.
    if (compressed_values.len < 8) return Error.CorruptedCompressedData;

    const first_value = try shared_functions.readOffsetValue(f64, compressed_values, &offset);
    try decompressed_values.append(allocator, first_value);

    var bit_reader = shared_structs.BulkBitReader.init(compressed_values[offset..]);

    var current_value_bits: u64 = @bitCast(first_value);
    var window_leading_zeros: u6 = 0;
    var window_trailing_zeros: u6 = 0;
    var meaningful_bit_count: u7 = 0;

    while (true) {
        const changed_bit = bit_reader.readBitsNoEof(u1, 1) catch return Error.CorruptedCompressedData;

        if (changed_bit != 0) {
            const new_window_bit = bit_reader.readBitsNoEof(u1, 1) catch return Error.CorruptedCompressedData;
            if (new_window_bit != 0) {
                const new_leading_zeros = bit_reader.readBitsNoEof(u6, leading_zero_bits) catch
                    return Error.CorruptedCompressedData;
                const new_meaningful_bit_count = 1 + (bit_reader.readBitsNoEof(u7, meaningful_count_bits) catch
                    return Error.CorruptedCompressedData);
                // The end-of-stream marker; see `end_marker_leading_zeros`.
                if (new_leading_zeros == end_marker_leading_zeros and
                    new_meaningful_bit_count == end_marker_meaningful_bit_count) break;
                // Any other window running past the end of the value means the header is corrupted.
                if (@as(u16, new_leading_zeros) + @as(u16, new_meaningful_bit_count) >
                    shared_structs.bits_per_value)
                {
                    return Error.CorruptedCompressedData;
                }
                window_leading_zeros = new_leading_zeros;
                meaningful_bit_count = new_meaningful_bit_count;
                window_trailing_zeros = @intCast(shared_structs.bits_per_value -
                    @as(u16, window_leading_zeros) - @as(u16, meaningful_bit_count));
            }
            // The encoder always writes a window before reusing one, so a count still at its
            // initial 0 means the stream reused a window that was never defined.
            if (meaningful_bit_count == 0) return Error.CorruptedCompressedData;
            const meaningful = bit_reader.readBitsNoEof(u64, meaningful_bit_count) catch
                return Error.CorruptedCompressedData;
            current_value_bits ^= meaningful << window_trailing_zeros;
        }

        try decompressed_values.append(allocator, @bitCast(current_value_bits));
    }
}

/// Write the end-of-stream marker to `bit_writer`.
fn writeEndMarker(bit_writer: *shared_structs.BulkBitWriter) Error!void {
    try bit_writer.writeBits(@as(u1, 1), 1);
    try bit_writer.writeBits(@as(u1, 1), 1);
    try bit_writer.writeBits(end_marker_leading_zeros, leading_zero_bits);
    try bit_writer.writeBits(end_marker_meaningful_bit_count - 1, meaningful_count_bits);
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

    // Gorilla is bitwise lossless, so it must recover any f64.
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

    // Gorilla is bitwise lossless, so compare bit patterns rather than numeric equality: the
    // latter cannot tell +0.0 from -0.0 and reports every NaN as equal to every other.
    try testing.expectEqual(uncompressed_values.items.len, decompressed_values.items.len);
    for (uncompressed_values.items, decompressed_values.items) |expected, actual| {
        try testing.expectEqual(@as(u64, @bitCast(expected)), @as(u64, @bitCast(actual)));
    }
}

test "gorilla roundtrips special floating-point values" {
    // Gorilla is bitwise lossless, so NaN payloads, infinities, and huge finite values survive.
    const payload_nan: f64 = @bitCast(@as(u64, 0x7ff8000000000001));
    const uncompressed_values = &[_]f64{
        1.0,
        std.math.nan(f64),
        payload_nan,
        std.math.inf(f64),
        -std.math.inf(f64),
        std.math.floatMax(f64),
        -std.math.floatMax(f64),
    };

    try tester.expectLosslessRoundTrip(testing.allocator, compress, decompress, uncompressed_values);
}

test "gorilla roundtrips edge floats" {
    // +0.0 and -0.0 compare numerically equal but differ in the sign bit, so only a bitwise
    // codec preserves them. `nextAfter` pairs produce the smallest possible XOR, exercising the
    // maximum leading-zeros path where the count is clamped to `maximum_stored_leading_zeros`.
    const uncompressed_values = &[_]f64{
        0.0,
        -0.0,
        std.math.floatMin(f64),
        std.math.floatTrueMin(f64),
        1.0,
        std.math.nextAfter(f64, 1.0, std.math.inf(f64)),
        std.math.nextAfter(f64, 1.0, -std.math.inf(f64)),
    };

    try tester.expectLosslessRoundTrip(testing.allocator, compress, decompress, uncompressed_values);
}

test "gorilla compresses repeated values below raw size" {
    // A constant signal collapses to a single marker bit per repeat after the first raw value,
    // so the byte stream must be far smaller than the raw f64 array.
    const allocator = testing.allocator;

    var uncompressed_values: [10]f64 = undefined;
    @memset(&uncompressed_values, 42.0);

    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);

    try compress(allocator, &uncompressed_values, &compressed_values, "{}");

    try testing.expect(compressed_values.items.len < uncompressed_values.len * @sizeOf(f64));
}

test "gorilla correctly errors for malformed headers" {
    const allocator = testing.allocator;

    // After the raw first value, the bits `1 1 11111 111110` declare a window of 31 leading zeros
    // and 63 meaningful bits. Those sum past the 64-bit value width, and the pair is not the
    // end-of-stream marker, so the header must be rejected.
    const corrupt_compressed = &[_]u8{
        0xDE, 0xAD, 0xFA, 0xDE, 0x00, 0x00, 0x00, 0x00, // First value, stored raw.
        0xFF, 0xF0, // Window header that runs past the end of the value.
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
