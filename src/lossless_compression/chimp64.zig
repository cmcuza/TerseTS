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

//! Implementation of the Chimp64 lossless floating-point time series compression method.
//! The method is described in:
//! Liakos et al., "Chimp: Efficient Lossless Floating Point Compression for Time Series Databases", VLDB 2022.
//! https://doi.org/10.14778/3551793.3551852
//! The code follows the official Java implementation published by the paper's authors at:
//! https://github.com/panagiotisl/chimp.

const std = @import("std");
const math = std.math;
const mem = std.mem;
const testing = std.testing;
const ArrayList = std.ArrayList;
const Allocator = mem.Allocator;

const tersets = @import("../tersets.zig");
const configuration = @import("../configuration.zig");
const shared_functions = @import("../utilities/shared_functions.zig");
const shared_structs = @import("../utilities/shared_structs.zig");
const tester = @import("../tester.zig");

const Error = tersets.Error;
const Method = tersets.Method;

/// Minimum trailing-zero run for the "store only meaningful bits" path (marker `01`). Shorter
/// runs are cheaper to write as part of the payload than as a separate count.
const trailing_zero_threshold = 6;

/// Compress `uncompressed_values` into `compressed_values` using Chimp64, allocating with
/// `allocator`. `method_configuration` must be empty (`{}`), otherwise
/// `Error.InvalidConfiguration` is returned. `uncompressed_values` must not be empty;
/// `tersets.compress` guarantees this. On
/// success `compressed_values` holds `[first_value: f64][XOR marker bits][end-of-stream marker]`.
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
    const first_value = uncompressed_values[0];
    try shared_functions.appendValue(allocator, f64, first_value, compressed_values);

    var previous_value_bits: u64 = @bitCast(first_value);
    var previous_leading_bucket: u7 = shared_structs.no_reusable_leading_bucket;

    var bit_writer = try shared_structs.BulkBitWriter.init(allocator, compressed_values);

    for (uncompressed_values[1..]) |value| {
        const current_value_bits: u64 = @bitCast(value);
        const xor = previous_value_bits ^ current_value_bits;

        // Marker `00`: repeated value.
        if (xor == 0) {
            try bit_writer.writeBits(@as(u2, 0b00), 2);
            previous_leading_bucket = shared_structs.no_reusable_leading_bucket;
            previous_value_bits = current_value_bits;
            continue;
        }

        const leading_zeros: u6 = @intCast(@clz(xor));
        const trailing_zeros: u6 = @intCast(@ctz(xor));
        const leading_bucket_index = shared_functions.leadingZeroBucketIndex(leading_zeros);
        const leading_bucket = shared_structs.leading_zero_buckets[leading_bucket_index];

        if (trailing_zeros > trailing_zero_threshold) {
            // Marker `01`: store a leading-zero bucket, meaningful-bit count, and meaningful bits.
            try bit_writer.writeBits(@as(u2, 0b01), 2);
            try bit_writer.writeBits(leading_bucket_index, shared_structs.leading_zero_bucket_bits);

            const meaningful_bit_count: u16 =
                shared_structs.bits_per_value - @as(u16, leading_bucket) - @as(u16, trailing_zeros);
            const meaningful_bits = xor >> trailing_zeros;

            try bit_writer.writeBits(@as(u6, @intCast(meaningful_bit_count)), 6);
            try bit_writer.writeBits(meaningful_bits, meaningful_bit_count);

            previous_leading_bucket = shared_structs.no_reusable_leading_bucket;
        } else {
            // Markers `10`/`11` write the whole non-leading XOR; their first bit `1` is shared.
            try bit_writer.writeBits(@as(u1, 0b1), 1);

            const non_leading_bit_count: u16 = shared_structs.bits_per_value - @as(u16, leading_bucket);
            if (@as(u7, leading_bucket) == previous_leading_bucket) {
                // Marker `10`: reuse the previous leading-zero bucket.
                try bit_writer.writeBits(@as(u1, 0b0), 1);
            } else {
                // Marker `11`: store a new leading-zero bucket, which the next value may reuse.
                try bit_writer.writeBits(@as(u1, 0b1), 1);
                try bit_writer.writeBits(leading_bucket_index, shared_structs.leading_zero_bucket_bits);
                previous_leading_bucket = leading_bucket;
            }

            try bit_writer.writeBits(xor, non_leading_bit_count);
        }

        previous_value_bits = current_value_bits;
    }

    // The end marker tells the decoder where to stop; flushed padding bits are never read.
    try shared_functions.writeChimpEndMarker(&bit_writer, 0);
    try bit_writer.flushBits();
}

/// Decompress a Chimp64 `compressed_values` stream into `decompressed_values`, allocating with
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

    var previous_value_bits: u64 = @bitCast(first_value);
    // Never read before a marker `11` sets it: the encoder cannot emit a reuse marker `10` first.
    var previous_leading_bucket: u6 = shared_structs.leading_zero_buckets[0];

    var bit_reader = shared_structs.BulkBitReader.init(compressed_values[offset..]);

    while (true) {
        const first_marker_bit = bit_reader.readBitsNoEof(u1, 1) catch return Error.CorruptedCompressedData;
        const second_marker_bit = bit_reader.readBitsNoEof(u1, 1) catch return Error.CorruptedCompressedData;

        // Marker `00`: repeated value.
        if (first_marker_bit == 0 and second_marker_bit == 0) {
            const value: f64 = @bitCast(previous_value_bits);
            try decompressed_values.append(allocator, value);
            continue;
        }

        var xor: u64 = undefined;
        var leading_bucket: u6 = undefined;

        // Marker `01`: read the bucket, meaningful-bit count, and meaningful bits.
        if (first_marker_bit == 0 and second_marker_bit == 1) {
            const leading_bucket_index = bit_reader.readBitsNoEof(u3, shared_structs.leading_zero_bucket_bits) catch return Error.CorruptedCompressedData;
            leading_bucket = shared_structs.leading_zero_buckets[leading_bucket_index];

            const meaningful_bit_count = bit_reader.readBitsNoEof(u6, 6) catch return Error.CorruptedCompressedData;
            // A count of 0 is the end-of-stream marker.
            if (meaningful_bit_count == 0) break;
            // Reject counts that leave no room for trailing zeros before the cast below.
            const occupied: u16 = @as(u16, leading_bucket) + @as(u16, meaningful_bit_count);
            if (occupied > shared_structs.bits_per_value) return Error.CorruptedCompressedData;
            const trailing_zeros: u6 = @intCast(shared_structs.bits_per_value - occupied);
            const meaningful_bits = bit_reader.readBitsNoEof(u64, meaningful_bit_count) catch return Error.CorruptedCompressedData;

            xor = meaningful_bits << trailing_zeros;
        } else {
            if (second_marker_bit == 0) {
                // Marker `10`: reuse the previous leading-zero bucket.
                leading_bucket = previous_leading_bucket;
            } else {
                // Marker `11`: read a new leading-zero bucket.
                const leading_bucket_index = bit_reader.readBitsNoEof(u3, shared_structs.leading_zero_bucket_bits) catch return Error.CorruptedCompressedData;
                leading_bucket = shared_structs.leading_zero_buckets[leading_bucket_index];
            }

            const non_leading_bit_count: u16 = shared_structs.bits_per_value - @as(u16, leading_bucket);
            xor = bit_reader.readBitsNoEof(u64, non_leading_bit_count) catch return Error.CorruptedCompressedData;
        }

        previous_leading_bucket = leading_bucket;
        previous_value_bits ^= xor;

        const value: f64 = @bitCast(previous_value_bits);
        try decompressed_values.append(allocator, value);
    }
}

test "chimp64 roundtrips generated values across all distributions" {
    const allocator = testing.allocator;

    // Chimp64 is bitwise lossless, so it must recover any f64. Test every distribution the tester offers.
    const data_distributions = &[_]tester.DataDistribution{
        .TightlyBoundedRandomValues,
        .LinearFunctions,
        .QuadraticFunctions,
        .ExponentialFunctions,
        .PowerFunctions,
        .SqrtFunctions,
        .BoundedRandomValues,
        .SinusoidalFunction,
        .MixedBoundedValuesFunctions,
        .FiniteRandomValues,
        .RandomValuesWithNansAndInfinities,
        .LinearFunctionsWithNansAndInfinities,
        .BoundedRandomValuesWithNansAndInfinities,
        .SinusoidalFunctionWithNansAndInfinities,
    };

    try tester.testLosslessMethod(
        allocator,
        Method.Chimp64,
        data_distributions,
    );
}

test "chimp64 roundtrips repeated values" {
    // Repeated values should use marker 00 after the first raw value.
    const uncompressed_values = &[_]f64{ 7.25, 7.25, 7.25, 7.25, 7.25 };

    try tester.expectLosslessRoundTrip(testing.allocator, compress, decompress, uncompressed_values);
}

test "chimp64 roundtrips changing values" {
    // Changing values cover bucket changes, bucket reuse, and meaningful-bit paths.
    const uncompressed_values = &[_]f64{ 100.0, 100.01, 100.02, 99.99, -3.5, 0.0, 2048.125 };

    try tester.expectLosslessRoundTrip(testing.allocator, compress, decompress, uncompressed_values);
}

test "chimp64 roundtrips special floating-point values" {
    // Chimp64 is bitwise lossless, so NaN payloads, infinities, and huge finite values are preserved.
    // The non-canonical NaN below exercises payload preservation explicitly.
    const payload_nan: f64 = @bitCast(@as(u64, 0x7ff8000000000001));
    const uncompressed_values = &[_]f64{
        1.0,
        math.nan(f64),
        payload_nan,
        math.inf(f64),
        -math.inf(f64),
        math.floatMax(f64),
        -math.floatMax(f64),
    };

    try tester.expectLosslessRoundTrip(testing.allocator, compress, decompress, uncompressed_values);
}

test "chimp64 roundtrips two values" {
    // Two values exercise exactly one XOR marker right after the first raw value.
    const uncompressed_values = &[_]f64{ 3.5, 9.0 };

    try tester.expectLosslessRoundTrip(testing.allocator, compress, decompress, uncompressed_values);
}

test "chimp64 roundtrips edge floats" {
    // +0.0 and -0.0 compare numerically equal but differ in the sign bit, so only a bitwise
    // codec preserves them. Subnormals use a distinct exponent encoding, and `nextAfter` pairs
    // produce the smallest possible XOR — exercising the maximum-leading-zeros bucket path.
    const uncompressed_values = &[_]f64{
        0.0,
        -0.0,
        math.floatMin(f64),
        math.floatTrueMin(f64),
        1.0,
        math.nextAfter(f64, 1.0, math.inf(f64)),
        math.nextAfter(f64, 1.0, -math.inf(f64)),
    };

    try tester.expectLosslessRoundTrip(testing.allocator, compress, decompress, uncompressed_values);
}

test "chimp64 compresses repeated values below raw size" {
    // A constant signal is maximally compressible: every repeat after the first raw value
    // collapses to a 2-bit marker, so the byte stream must be far smaller than the raw f64 array.
    const allocator = testing.allocator;

    var uncompressed_values: [500]f64 = undefined;
    @memset(&uncompressed_values, 42.0);

    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);

    try compress(allocator, &uncompressed_values, &compressed_values, "{}");

    try testing.expect(compressed_values.items.len < uncompressed_values.len * @sizeOf(f64));
}

test "check chimp64 configuration parsing" {
    // Chimp64 takes no parameters: an empty configuration must parse, and a configuration
    // carrying unexpected fields must be rejected with InvalidConfiguration.
    const allocator = testing.allocator;
    const uncompressed_values = &[_]f64{ 1.0, 2.0, 3.0 };

    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);

    // An empty configuration is valid.
    try compress(allocator, uncompressed_values, &compressed_values, "{}");

    // A configuration with unexpected fields is rejected.
    const invalid_configuration = "{ \"abs_error_bound\": 0.1 }";
    try testing.expectError(
        Error.InvalidConfiguration,
        compress(allocator, uncompressed_values, &compressed_values, invalid_configuration),
    );
}

test "chimp64 rejects corrupted compressed data" {
    const allocator = testing.allocator;
    const uncompressed_values = &[_]f64{ 100.0, 100.01, 100.02, 99.99, -3.5 };

    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);
    try compress(allocator, uncompressed_values, &compressed_values, "{}");

    var decompressed_values = ArrayList(f64).empty;
    defer decompressed_values.deinit(allocator);

    // Shorter than the raw first value.
    try testing.expectError(
        Error.CorruptedCompressedData,
        decompress(allocator, compressed_values.items[0..4], &decompressed_values),
    );

    // Truncated bit stream: a single byte after the raw first value cannot hold the
    // end-of-stream marker, so the decoder must report end-of-stream instead of
    // terminating cleanly.
    decompressed_values.clearRetainingCapacity();
    try testing.expectError(
        Error.CorruptedCompressedData,
        decompress(allocator, compressed_values.items[0..9], &decompressed_values),
    );
}
