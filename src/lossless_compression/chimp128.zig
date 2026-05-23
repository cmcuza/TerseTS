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

//! Implementation of the Chimp128 lossless floating-point time series compression method.
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
const previous_values = 128;
// Higher than Chimp64's threshold of 6 because encoding a ring-buffer index costs 7 extra bits.
const trailing_zero_threshold = 6 + std.math.log2(previous_values);
const generated_test_rounds = 5;

// 14-bit LSB mask indexes into the fast-lookup table; two values sharing these bits are likely good predictors.
const lsb_bits = 14;
const lsb_mask: u64 = (1 << lsb_bits) - 1;

const leading_zero_buckets = [_]u6{ 0, 8, 12, 16, 18, 20, 22, 24 };

/// Compress `uncompressed_values` using Chimp128's value codec.
/// The stream stores `[count: u64][first_value: f64][xor marker bits...]`.
/// Later values are XOR-encoded against the best predictor from a 128-value ring buffer.
/// When the ring-buffer predictor is used, the 7-bit ring slot is stored so the decoder can look it up.
/// When no good ring-buffer predictor is found, the immediately previous value is used instead (markers 10/11).
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

    // Chimp128 encodes later values as XORs against a predictor, so the first value is stored raw.
    const first_value = uncompressed_values[0];
    try shared_functions.appendValue(allocator, f64, first_value, compressed_values);

    // Ring buffer of the last `previous_values` encoded values for predictor lookup.
    const stored_values = try allocator.alloc(u64, previous_values);
    defer allocator.free(stored_values);
    @memset(stored_values, 0);

    // Fast lookup: maps 14 LSBs of any value to the global index of the last value with those LSBs.
    // Initialized to 0; `current_index` starts at `previous_values` so that unseen keys (index 0)
    // fail the staleness check `current_index - indices[key] < previous_values`.
    const indices = try allocator.alloc(usize, 1 << lsb_bits);
    defer allocator.free(indices);
    @memset(indices, 0);

    var current_index: usize = previous_values;

    const first_value_bits: u64 = @bitCast(first_value);
    stored_values[current_index % previous_values] = first_value_bits;
    indices[first_value_bits & lsb_mask] = current_index;
    current_index += 1;

    var previous_value_bits: u64 = first_value_bits;
    var previous_leading_zeros: u6 = leading_zero_buckets[0];

    var bit_writer = try shared_structs.BitWriter.init(allocator, compressed_values);

    for (uncompressed_values[1..]) |value| {
        const current_value_bits: u64 = @bitCast(value);
        const key = current_value_bits & lsb_mask;
        const prev_index = indices[key];

        // Check if the LSB-matched ring-buffer entry is still within the active window.
        if (current_index - prev_index < previous_values) {
            const predictor_bits = stored_values[prev_index % previous_values];
            const xor = predictor_bits ^ current_value_bits;
            const ring_slot: u7 = @intCast(prev_index % previous_values);

            if (xor == 0) {
                // Marker `00`: ring-buffer match, value is identical.
                // 9 bits total: 2-bit marker + 7-bit ring slot so the decoder knows which stored value to reuse.
                try bit_writer.writeBits(@as(u2, 0b00), 2);
                try bit_writer.writeBits(ring_slot, 7);

                stored_values[current_index % previous_values] = current_value_bits;
                indices[key] = current_index;
                current_index += 1;
                previous_value_bits = current_value_bits;
                continue;
            }

            const trailing_zeros: u6 = @intCast(@ctz(xor));
            if (trailing_zeros > trailing_zero_threshold) {
                // Marker `01`: ring-buffer predictor with meaningful bits.
                // 7-bit ring slot is required so the decoder XORs against the right stored value.
                const leading_zeros: u6 = @intCast(@clz(xor));
                const leading_bucket_index = leadingZeroBucketIndex(leading_zeros);
                const leading_bucket = leading_zero_buckets[leading_bucket_index];
                const meaningful_bit_count: u16 =
                    bits_per_value - @as(u16, leading_bucket) - @as(u16, trailing_zeros);
                const meaningful_bits = xor >> trailing_zeros;

                try bit_writer.writeBits(@as(u2, 0b01), 2);
                try bit_writer.writeBits(ring_slot, 7);
                try bit_writer.writeBits(leading_bucket_index, leading_zero_bucket_bits);
                try bit_writer.writeBits(@as(u6, @intCast(meaningful_bit_count)), 6);
                try bit_writer.writeBits(meaningful_bits, meaningful_bit_count);

                previous_leading_zeros = leading_bucket;
                stored_values[current_index % previous_values] = current_value_bits;
                indices[key] = current_index;
                current_index += 1;
                previous_value_bits = current_value_bits;
                continue;
            }
        }

        // No good ring-buffer predictor: fall back to the immediately previous value (markers `10`/`11`).
        // No ring slot is stored because the decoder always has the previous value available.
        const xor = previous_value_bits ^ current_value_bits;
        const leading_zeros: u6 = @intCast(@clz(xor));
        const leading_bucket_index = leadingZeroBucketIndex(leading_zeros);
        const leading_bucket = leading_zero_buckets[leading_bucket_index];
        const non_leading_bit_count: u16 = bits_per_value - @as(u16, leading_bucket);

        try bit_writer.writeBits(@as(u1, 0b1), 1);
        if (leading_bucket == previous_leading_zeros) {
            // Marker `10`: reuse the previous leading-zero bucket.
            try bit_writer.writeBits(@as(u1, 0b0), 1);
        } else {
            // Marker `11`: store a new leading-zero bucket.
            try bit_writer.writeBits(@as(u1, 0b1), 1);
            try bit_writer.writeBits(leading_bucket_index, leading_zero_bucket_bits);
        }
        try bit_writer.writeBits(xor, non_leading_bit_count);

        previous_leading_zeros = leading_bucket;
        stored_values[current_index % previous_values] = current_value_bits;
        indices[key] = current_index;
        current_index += 1;
        previous_value_bits = current_value_bits;
    }

    try bit_writer.flushBits();
}

/// Decompress a Chimp128-encoded stream into `decompressed_values`.
/// Maintains the same 128-value ring buffer as the encoder so predictor lookups stay in sync.
/// Reads the `[count: u64][first_value: f64]` header, then per value: marker `00` copies the
/// stored value at the given 7-bit ring slot, marker `01` XORs that stored value with the
/// reconstructed meaningful bits, and markers `10`/`11` XOR against the immediately previous value.
pub fn decompress(
    allocator: Allocator,
    compressed_values: []const u8,
    decompressed_values: *ArrayList(f64),
) Error!void {
    var offset: usize = 0;
    const value_count = try shared_functions.readOffsetValue(u64, compressed_values, &offset);

    if (value_count == 0) return;

    // Every non-empty Chimp128 stream must contain the count header and first raw value.
    if (compressed_values.len < 16) return Error.UnsupportedInput;

    const first_value = try shared_functions.readOffsetValue(f64, compressed_values, &offset);
    try decompressed_values.append(allocator, first_value);

    const stored_values = try allocator.alloc(u64, previous_values);
    defer allocator.free(stored_values);
    @memset(stored_values, 0);

    const indices = try allocator.alloc(usize, 1 << lsb_bits);
    defer allocator.free(indices);
    @memset(indices, 0);

    var current_index: usize = previous_values;

    const first_value_bits: u64 = @bitCast(first_value);
    stored_values[current_index % previous_values] = first_value_bits;
    indices[first_value_bits & lsb_mask] = current_index;
    current_index += 1;

    var previous_value_bits: u64 = first_value_bits;
    var previous_leading_zeros: u6 = leading_zero_buckets[0];

    // BitReader expects a reader interface, so wrap the remaining bytes in a stream.
    const reader = Reader.fixed(compressed_values[offset..]);
    var bit_reader = shared_structs.BitReader.init(reader);

    while (decompressed_values.items.len < value_count) {
        const first_marker_bit = bit_reader.readBitsNoEof(u1, 1) catch return Error.ByteStreamError;
        const second_marker_bit = bit_reader.readBitsNoEof(u1, 1) catch return Error.ByteStreamError;

        var current_value_bits: u64 = undefined;

        if (first_marker_bit == 0 and second_marker_bit == 0) {
            // Marker `00`: read the 7-bit ring slot and reuse that stored value directly.
            const ring_slot = bit_reader.readBitsNoEof(u7, 7) catch return Error.ByteStreamError;
            current_value_bits = stored_values[ring_slot];
        } else if (first_marker_bit == 0 and second_marker_bit == 1) {
            // Marker `01`: read the 7-bit ring slot, reconstruct XOR, apply to the stored value.
            const ring_slot = bit_reader.readBitsNoEof(u7, 7) catch return Error.ByteStreamError;
            const leading_bucket_index = bit_reader.readBitsNoEof(u3, leading_zero_bucket_bits) catch return Error.ByteStreamError;
            const leading_bucket = leading_zero_buckets[leading_bucket_index];

            const meaningful_bit_count = bit_reader.readBitsNoEof(u6, 6) catch return Error.ByteStreamError;
            const trailing_zeros: u6 =
                @intCast(bits_per_value - @as(u16, leading_bucket) - @as(u16, meaningful_bit_count));
            const meaningful_bits = bit_reader.readBitsNoEof(u64, meaningful_bit_count) catch return Error.ByteStreamError;
            const xor = meaningful_bits << trailing_zeros;

            current_value_bits = stored_values[ring_slot] ^ xor;
            previous_leading_zeros = leading_bucket;
        } else {
            var leading_bucket: u6 = undefined;
            if (second_marker_bit == 0) {
                // Marker `10`: reuse the previous leading-zero bucket, XOR against previous value.
                leading_bucket = previous_leading_zeros;
            } else {
                // Marker `11`: read a new leading-zero bucket, XOR against previous value.
                const leading_bucket_index = bit_reader.readBitsNoEof(u3, leading_zero_bucket_bits) catch return Error.ByteStreamError;
                leading_bucket = leading_zero_buckets[leading_bucket_index];
                previous_leading_zeros = leading_bucket;
            }
            const non_leading_bit_count: u16 = bits_per_value - @as(u16, leading_bucket);
            const xor = bit_reader.readBitsNoEof(u64, non_leading_bit_count) catch return Error.ByteStreamError;
            current_value_bits = previous_value_bits ^ xor;
        }

        const decoded_key = current_value_bits & lsb_mask;
        stored_values[current_index % previous_values] = current_value_bits;
        indices[decoded_key] = current_index;
        current_index += 1;
        previous_value_bits = current_value_bits;

        const value: f64 = @bitCast(current_value_bits);
        try decompressed_values.append(allocator, value);
    }
}

/// Map exact leading zeros to a Chimp128 bucket index.
fn leadingZeroBucketIndex(leading_zeros: u6) u3 {
    var selected_index: u3 = 0;

    for (leading_zero_buckets[1..], 1..) |bucket, index| {
        if (bucket > leading_zeros) break;

        selected_index = @intCast(index);
    }

    return selected_index;
}

test "chimp128 roundtrips generated values across all distributions" {
    const allocator = testing.allocator;

    // Chimp128 is bitwise lossless, so it must recover any f64. Test every distribution the tester offers.
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
            Method.Chimp128,
            data_distributions,
        );
    }
}

test "chimp128 roundtrips empty input" {
    // Empty input uses only the count header and no bit stream.
    const uncompressed_values = &[_]f64{};

    try tester.expectLosslessRoundTrip(testing.allocator, compress, decompress, uncompressed_values);
}

test "chimp128 roundtrips single value" {
    // A single value stores the count and first raw value without any XOR markers.
    const uncompressed_values = &[_]f64{42.5};

    try tester.expectLosslessRoundTrip(testing.allocator, compress, decompress, uncompressed_values);
}

test "chimp128 roundtrips repeated values" {
    // Repeated values should use marker 00 with a ring-buffer slot after the first raw value.
    const uncompressed_values = &[_]f64{ 7.25, 7.25, 7.25, 7.25, 7.25 };

    try tester.expectLosslessRoundTrip(testing.allocator, compress, decompress, uncompressed_values);
}

test "chimp128 roundtrips changing values" {
    // Changing values cover bucket changes, bucket reuse, and meaningful-bit paths.
    const uncompressed_values = &[_]f64{ 100.0, 100.01, 100.02, 99.99, -3.5, 0.0, 2048.125 };

    try tester.expectLosslessRoundTrip(testing.allocator, compress, decompress, uncompressed_values);
}

test "chimp128 roundtrips special floating-point values" {
    // Chimp128 is bitwise lossless, so NaN payloads, infinities, and huge finite values are preserved.
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

test "chimp128 roundtrips two values" {
    // Two values exercise exactly one XOR marker right after the first raw value.
    const uncompressed_values = &[_]f64{ 3.5, 9.0 };

    try tester.expectLosslessRoundTrip(testing.allocator, compress, decompress, uncompressed_values);
}

test "chimp128 roundtrips edge floats" {
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

test "chimp128 roundtrips sequences longer than the ring buffer" {
    // More than 128 values makes the ring buffer wrap. The repeating pattern means earlier
    // values get reused as predictors, exercising the ring-buffer marker paths.
    var uncompressed_values: [400]f64 = undefined;
    for (&uncompressed_values, 0..) |*value, index| {
        value.* = @floatFromInt(index % 100);
    }

    try tester.expectLosslessRoundTrip(testing.allocator, compress, decompress, &uncompressed_values);
}

test "chimp128 compresses repeated values below raw size" {
    // A constant signal is maximally compressible: every repeat after the first raw value
    // collapses to a 9-bit marker, so the byte stream must be far smaller than the raw f64 array.
    const allocator = testing.allocator;

    var uncompressed_values: [500]f64 = undefined;
    @memset(&uncompressed_values, 1.0);

    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);

    try compress(allocator, &uncompressed_values, &compressed_values, "{}");

    try testing.expect(compressed_values.items.len < uncompressed_values.len * @sizeOf(f64));
}

test "check chimp128 configuration parsing" {
    // Chimp128 takes no parameters: an empty configuration must parse, and a configuration
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
