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

const std = @import("std");
const math = std.math;
const mem = std.mem;
const testing = std.testing;
const Reader = std.Io.Reader;
const ArrayList = std.ArrayList;
const Allocator = mem.Allocator;

const tersets = @import("../tersets.zig");
const configuration = @import("../configuration.zig");
const shared_functions = @import("../utilities/shared_functions.zig");
const shared_structs = @import("../utilities/shared_structs.zig");
const tester = @import("../tester.zig");

const Error = tersets.Error;
const Method = tersets.Method;

const bits_per_value = 64;
const leading_zero_bucket_bits = 3;
const trailing_zero_threshold = 6;
const generated_test_rounds = 5;

const leading_zero_buckets = [_]u6{ 0, 8, 12, 16, 18, 20, 22, 24 };

/// Compress `uncompressed_values` using Chimp64's value codec.
/// The stream stores `[count: u64][first_value: f64][xor marker bits...]`.
/// Later values are encoded as XORs against the previous value with Chimp's
/// 3-bit leading-zero buckets and 6-bit trailing-zero threshold.
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

    // Store the value count so decompression can ignore padding bits after the stream is flushed.
    try shared_functions.appendValue(allocator, u64, @intCast(uncompressed_values.len), compressed_values);
    if (uncompressed_values.len == 0) return;

    // Chimp64 encodes later values as XORs against the previous value, so the first value is stored raw.
    const first_value = uncompressed_values[0];
    try shared_functions.appendValue(allocator, f64, first_value, compressed_values);

    // Chimp64 keeps the previous value for XOR prediction and the previous leading-zero bucket for reuse.
    var previous_value_bits: u64 = @bitCast(first_value);
    var previous_leading_zeros: u6 = leading_zero_buckets[0];

    var bit_writer = try shared_structs.BitWriter.init(allocator, compressed_values);

    for (uncompressed_values[1..]) |value| {
        const current_value_bits: u64 = @bitCast(value);
        const xor = previous_value_bits ^ current_value_bits;

        // Marker `00`: repeated value.
        if (xor == 0) {
            try bit_writer.writeBits(@as(u2, 0b00), 2);
            previous_value_bits = current_value_bits;
            continue;
        }

        const leading_zeros: u6 = @intCast(@clz(xor));
        const trailing_zeros: u6 = @intCast(@ctz(xor));
        const leading_bucket_index = leadingZeroBucketIndex(leading_zeros);
        const leading_bucket = leading_zero_buckets[leading_bucket_index];

        // Like Gorilla, this path stores only the meaningful bits when the XOR has enough trailing zeros.
        // Chimp64 differs by using leading-zero buckets and a fixed trailing-zero threshold.
        if (trailing_zeros > trailing_zero_threshold) {
            // Marker `01`: store a leading-zero bucket, meaningful-bit count, and meaningful bits
            try bit_writer.writeBits(@as(u2, 0b01), 2);
            try bit_writer.writeBits(leading_bucket_index, leading_zero_bucket_bits);

            const meaningful_bit_count: u16 =
                bits_per_value - @as(u16, leading_bucket) - @as(u16, trailing_zeros);
            const meaningful_bits = xor >> trailing_zeros;

            try bit_writer.writeBits(@as(u6, @intCast(meaningful_bit_count)), 6);
            try bit_writer.writeBits(meaningful_bits, meaningful_bit_count);
        } else {
            // Chimp64 optimization: if the trailing-zero run is small, write those zeros directly in the payload.
            // This avoids spending more metadata bits than the zeros would cost.
            // The first marker bit is shared by `10` and `11`.
            try bit_writer.writeBits(@as(u1, 0b1), 1);

            const non_leading_bit_count: u16 = bits_per_value - @as(u16, leading_bucket);
            if (leading_bucket == previous_leading_zeros) {
                // Marker `10`: reuse the previous leading-zero bucket.
                try bit_writer.writeBits(@as(u1, 0b0), 1);
            } else {
                // Marker `11`: store a new leading-zero bucket.
                try bit_writer.writeBits(@as(u1, 0b1), 1);
                try bit_writer.writeBits(leading_bucket_index, leading_zero_bucket_bits);
            }

            try bit_writer.writeBits(xor, non_leading_bit_count);
        }

        previous_leading_zeros = leading_bucket;
        previous_value_bits = current_value_bits;
    }

    try bit_writer.flushBits();
}

/// Decompress a Chimp64-encoded stream into `decompressed_values`.
/// Reads the `[count: u64][first_value: f64]` header, then reconstructs each later value by
/// applying its XOR delta against the previous value: marker `00` repeats it, marker `01` reads
/// a leading-zero bucket plus meaningful-bit count and bits, marker `10` reuses the previous
/// leading-zero bucket, and marker `11` reads a new bucket before the non-leading XOR bits.
pub fn decompress(
    allocator: Allocator,
    compressed_values: []const u8,
    decompressed_values: *ArrayList(f64),
) Error!void {
    var offset: usize = 0;
    const value_count = try shared_functions.readOffsetValue(u64, compressed_values, &offset);

    if (value_count == 0) return;

    // Every non-empty Chimp64 stream must contain the count header and first raw value.
    if (compressed_values.len < 16) return Error.UnsupportedInput;

    const first_value = try shared_functions.readOffsetValue(f64, compressed_values, &offset);
    try decompressed_values.append(allocator, first_value);

    var previous_value_bits: u64 = @bitCast(first_value);
    var previous_leading_zeros: u6 = leading_zero_buckets[0];

    // BitReader expects a reader interface, so wrap the remaining bytes in a stream.
    const reader = Reader.fixed(compressed_values[offset..]);
    var bit_reader = shared_structs.BitReader.init(reader);

    while (decompressed_values.items.len < value_count) {
        const first_marker_bit = bit_reader.readBitsNoEof(u1, 1) catch return Error.ByteStreamError;
        const second_marker_bit = bit_reader.readBitsNoEof(u1, 1) catch return Error.ByteStreamError;

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
            const leading_bucket_index = bit_reader.readBitsNoEof(u3, leading_zero_bucket_bits) catch return Error.ByteStreamError;
            leading_bucket = leading_zero_buckets[leading_bucket_index];

            const meaningful_bit_count = bit_reader.readBitsNoEof(u6, 6) catch return Error.ByteStreamError;
            // Validate the geometry before casting: leading + meaningful must leave room for
            // a non-negative trailing-zero count that still fits in u6.
            const occupied: u16 = @as(u16, leading_bucket) + @as(u16, meaningful_bit_count);
            if (occupied == 0 or occupied > bits_per_value) return Error.UnsupportedInput;
            const trailing_zeros: u6 = @intCast(bits_per_value - occupied);
            const meaningful_bits = bit_reader.readBitsNoEof(u64, meaningful_bit_count) catch return Error.ByteStreamError;

            xor = meaningful_bits << trailing_zeros;
        } else {
            if (second_marker_bit == 0) {
                // Marker `10`: reuse the previous leading-zero bucket.
                leading_bucket = previous_leading_zeros;
            } else {
                // Marker `11`: read a new leading-zero bucket.
                const leading_bucket_index = bit_reader.readBitsNoEof(u3, leading_zero_bucket_bits) catch return Error.ByteStreamError;
                leading_bucket = leading_zero_buckets[leading_bucket_index];
            }

            const non_leading_bit_count: u16 = bits_per_value - @as(u16, leading_bucket);
            xor = bit_reader.readBitsNoEof(u64, non_leading_bit_count) catch return Error.ByteStreamError;
        }

        previous_leading_zeros = leading_bucket;
        previous_value_bits ^= xor;

        const value: f64 = @bitCast(previous_value_bits);
        try decompressed_values.append(allocator, value);
    }
}

/// Map exact leading zeros to a Chimp64 bucket index.
fn leadingZeroBucketIndex(leading_zeros: u6) u3 {
    var selected_index: u3 = 0;

    for (leading_zero_buckets[1..], 1..) |bucket, index| {
        if (bucket > leading_zeros) break;

        selected_index = @intCast(index);
    }

    return selected_index;
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

    for (0..generated_test_rounds) |_| {
        try tester.testLosslessMethod(
            allocator,
            Method.Chimp64,
            data_distributions,
        );
    }
}

test "chimp64 roundtrips empty input" {
    // Empty input uses only the count header and no bit stream.
    const uncompressed_values = &[_]f64{};

    try tester.expectLosslessRoundTrip(testing.allocator, compress, decompress, uncompressed_values);
}

test "chimp64 roundtrips single value" {
    // A single value stores the count and first raw value without any XOR markers.
    const uncompressed_values = &[_]f64{42.5};

    try tester.expectLosslessRoundTrip(testing.allocator, compress, decompress, uncompressed_values);
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
    const uncompressed_values = &[_]f64{
        1.0,
        math.nan(f64),
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
