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

/// Size of the ring buffer of recently seen values that serve as XOR predictors. The "128" in
/// Chimp128 refers to this window length.
const previous_values = 128;
/// Bit width of a ring-buffer slot index (`log2(128) = 7`). The ring-buffer marker paths store a
/// slot of this width so the decoder can address the same stored value.
const ring_slot_bits = math.log2_int(usize, previous_values);
/// Minimum trailing-zero run for the "store only meaningful bits" path (marker `01`). Higher than
/// Chimp64's 6 because this path also stores a `ring_slot_bits` ring-buffer index.
const trailing_zero_threshold = 6 + ring_slot_bits;
/// Number of low bits used as the predictor lookup key: values sharing these bits usually differ
/// only in higher bits, so one is a good XOR predictor for the other.
const lsb_bits = 14;
/// Mask selecting the low `lsb_bits` of a value's bit pattern, i.e. its predictor lookup-table key.
const lsb_mask: u64 = (1 << lsb_bits) - 1;

/// Compress `uncompressed_values` into `compressed_values` using Chimp128, allocating with
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

    // Ring buffer of the last `previous_values` values, used as XOR predictors.
    var stored_values: [previous_values]u64 = @splat(0);

    // Maps the low `lsb_bits` of a value to the index of the last value with those bits. Encoder
    // only: it picks which predictor to try, and the chosen ring slot is written to the stream.
    // Fresh and zeroed per call like the reference; unseen keys read 0 and resolve to the first
    // stored value. `u32` keeps the table at 64 KB and limits one call to 2^32 values.
    const indices = try allocator.alloc(u32, 1 << lsb_bits);
    defer allocator.free(indices);
    @memset(indices, 0);

    // Position of the most recently stored value. Starting at 0 makes the window test below
    // span all `previous_values` live ring slots.
    var index: usize = 0;

    const first_value_bits: u64 = @bitCast(first_value);
    stored_values[index % previous_values] = first_value_bits;
    indices[@intCast(first_value_bits & lsb_mask)] = 0;

    var previous_leading_bucket: u7 = shared_structs.no_reusable_leading_bucket;

    var bit_writer = try shared_structs.BulkBitWriter.init(allocator, compressed_values);

    for (uncompressed_values[1..]) |value| {
        const current_value_bits: u64 = @bitCast(value);
        const key: usize = @intCast(current_value_bits & lsb_mask);
        const candidate_index: usize = indices[key];

        // Pick the predictor first, then classify the XOR once. `trailing_zeros` stays 0 unless
        // the ring-buffer candidate is used, so marker `01` below only fires for it. `u7` because
        // `@ctz` of a zero XOR is 64.
        var xor: u64 = undefined;
        var ring_slot: u7 = undefined;
        var trailing_zeros: u7 = 0;

        // Accept the LSB-matched entry only while its value still occupies its ring slot.
        if (index - candidate_index < previous_values) {
            const candidate_xor = stored_values[candidate_index % previous_values] ^ current_value_bits;
            trailing_zeros = @intCast(@ctz(candidate_xor));
            if (trailing_zeros > trailing_zero_threshold) {
                ring_slot = @intCast(candidate_index % previous_values);
                xor = candidate_xor;
            } else {
                ring_slot = @intCast(index % previous_values);
                xor = stored_values[ring_slot] ^ current_value_bits;
            }
        } else {
            ring_slot = @intCast(index % previous_values);
            xor = stored_values[ring_slot] ^ current_value_bits;
        }

        if (xor == 0) {
            // Marker `00`: the predictor already holds this value; store only its ring slot.
            try bit_writer.writeBits(@as(u2, 0b00), 2);
            try bit_writer.writeBits(ring_slot, ring_slot_bits);
            previous_leading_bucket = shared_structs.no_reusable_leading_bucket;
        } else {
            const leading_bucket_index = shared_functions.leadingZeroBucketIndex(@intCast(@clz(xor)));
            const leading_bucket = shared_structs.leading_zero_buckets[leading_bucket_index];

            if (trailing_zeros > trailing_zero_threshold) {
                // Marker `01`: ring-buffer predictor, meaningful bits only.
                const meaningful_bit_count: u16 =
                    shared_structs.bits_per_value - @as(u16, leading_bucket) - @as(u16, trailing_zeros);

                try bit_writer.writeBits(@as(u2, 0b01), 2);
                try bit_writer.writeBits(ring_slot, ring_slot_bits);
                try bit_writer.writeBits(leading_bucket_index, shared_structs.leading_zero_bucket_bits);
                try bit_writer.writeBits(@as(u6, @intCast(meaningful_bit_count)), 6);
                try bit_writer.writeBits(xor >> @intCast(trailing_zeros), meaningful_bit_count);

                previous_leading_bucket = shared_structs.no_reusable_leading_bucket;
            } else {
                // Markers `10`/`11` XOR against the previous value, so no ring slot is stored;
                // their first bit `1` is shared.
                try bit_writer.writeBits(@as(u1, 0b1), 1);
                if (@as(u7, leading_bucket) == previous_leading_bucket) {
                    // Marker `10`: reuse the previous leading-zero bucket.
                    try bit_writer.writeBits(@as(u1, 0b0), 1);
                } else {
                    // Marker `11`: store a new leading-zero bucket, which the next value may reuse.
                    try bit_writer.writeBits(@as(u1, 0b1), 1);
                    try bit_writer.writeBits(leading_bucket_index, shared_structs.leading_zero_bucket_bits);
                    previous_leading_bucket = leading_bucket;
                }
                try bit_writer.writeBits(xor, shared_structs.bits_per_value - @as(u16, leading_bucket));
            }
        }

        index += 1;
        stored_values[index % previous_values] = current_value_bits;
        indices[key] = @intCast(index);
    }

    // The end marker tells the decoder where to stop; flushed padding bits are never read.
    try shared_functions.writeChimpEndMarker(&bit_writer, ring_slot_bits);
    try bit_writer.flushBits();
}

/// Decompress a Chimp128 `compressed_values` stream into `decompressed_values`, allocating with
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

    // Mirrors the encoder's ring buffer.
    var stored_values: [previous_values]u64 = @splat(0);

    // No lookup table here: ring slots are read directly from the stream.
    // `index` mirrors the encoder's: the position of the most recently stored value.
    var index: usize = 0;

    const first_value_bits: u64 = @bitCast(first_value);
    stored_values[index % previous_values] = first_value_bits;

    var previous_value_bits: u64 = first_value_bits;
    // Never read before a marker `11` sets it: the encoder cannot emit a reuse marker `10` first.
    var previous_leading_bucket: u6 = shared_structs.leading_zero_buckets[0];

    var bit_reader = shared_structs.BulkBitReader.init(compressed_values[offset..]);

    while (true) {
        const first_marker_bit = bit_reader.readBitsNoEof(u1, 1) catch return Error.CorruptedCompressedData;
        const second_marker_bit = bit_reader.readBitsNoEof(u1, 1) catch return Error.CorruptedCompressedData;

        var current_value_bits: u64 = undefined;

        if (first_marker_bit == 0 and second_marker_bit == 0) {
            // Marker `00`: read the 7-bit ring slot and reuse that stored value directly.
            const ring_slot = bit_reader.readBitsNoEof(u7, ring_slot_bits) catch return Error.CorruptedCompressedData;
            current_value_bits = stored_values[ring_slot];
        } else if (first_marker_bit == 0 and second_marker_bit == 1) {
            // Marker `01`: read the 7-bit ring slot, reconstruct XOR, apply to the stored value.
            const ring_slot = bit_reader.readBitsNoEof(u7, ring_slot_bits) catch return Error.CorruptedCompressedData;
            const leading_bucket_index = bit_reader.readBitsNoEof(u3, shared_structs.leading_zero_bucket_bits) catch return Error.CorruptedCompressedData;
            const leading_bucket = shared_structs.leading_zero_buckets[leading_bucket_index];

            const meaningful_bit_count = bit_reader.readBitsNoEof(u6, 6) catch return Error.CorruptedCompressedData;
            // A count of 0 is the end-of-stream marker.
            if (meaningful_bit_count == 0) break;
            // Reject counts that leave no room for trailing zeros before the cast below.
            const occupied: u16 = @as(u16, leading_bucket) + @as(u16, meaningful_bit_count);
            if (occupied > shared_structs.bits_per_value) return Error.CorruptedCompressedData;
            const trailing_zeros: u6 = @intCast(shared_structs.bits_per_value - occupied);
            const meaningful_bits = bit_reader.readBitsNoEof(u64, meaningful_bit_count) catch return Error.CorruptedCompressedData;
            const xor = meaningful_bits << trailing_zeros;

            current_value_bits = stored_values[ring_slot] ^ xor;
            previous_leading_bucket = leading_bucket;
        } else {
            var leading_bucket: u6 = undefined;
            if (second_marker_bit == 0) {
                // Marker `10`: reuse the previous leading-zero bucket, XOR against previous value.
                leading_bucket = previous_leading_bucket;
            } else {
                // Marker `11`: read a new leading-zero bucket, XOR against previous value.
                const leading_bucket_index = bit_reader.readBitsNoEof(u3, shared_structs.leading_zero_bucket_bits) catch return Error.CorruptedCompressedData;
                leading_bucket = shared_structs.leading_zero_buckets[leading_bucket_index];
                previous_leading_bucket = leading_bucket;
            }
            const non_leading_bit_count: u16 = shared_structs.bits_per_value - @as(u16, leading_bucket);
            const xor = bit_reader.readBitsNoEof(u64, non_leading_bit_count) catch return Error.CorruptedCompressedData;
            current_value_bits = previous_value_bits ^ xor;
        }

        index += 1;
        stored_values[index % previous_values] = current_value_bits;
        previous_value_bits = current_value_bits;

        const value: f64 = @bitCast(current_value_bits);
        try decompressed_values.append(allocator, value);
    }
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

    try tester.testLosslessMethod(
        allocator,
        Method.Chimp128,
        data_distributions,
    );
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

test "chimp128 rejects corrupted compressed data" {
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
