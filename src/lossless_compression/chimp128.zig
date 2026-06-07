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
//! The ring-buffer predictor scheme, bit-level layout, and leading-zero bucket boundaries follow
//! the authors' reference Java implementation in the ELF repository, package
//! `gr.aueb.delorean.chimp`: https://github.com/Spatio-Temporal-Lab/elf.

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

/// Number of bits in an IEEE-754 `f64`; the width of every value Chimp128 XOR-encodes.
const bits_per_value = 64;
/// Bit width of the leading-zero bucket index written to the stream. 3 bits index the 8 buckets
/// in `leading_zero_buckets`.
const leading_zero_bucket_bits = 3;
/// Size of the ring buffer of recently seen values that serve as XOR predictors. The "128" in
/// Chimp128 refers to this window length.
const previous_values = 128;
/// Bit width of a ring-buffer slot index (`log2(128) = 7`). The ring-buffer marker paths store a
/// slot of this width so the decoder can address the same stored value.
const ring_slot_bits = std.math.log2_int(usize, previous_values);
/// Minimum trailing-zero run that triggers the "store only meaningful bits" path (marker `01`).
/// Higher than Chimp64's threshold of 6 because this path additionally stores a `ring_slot_bits`
/// ring-buffer index, so the trailing-zero run must save at least those extra bits to be worth it.
const trailing_zero_threshold = 6 + ring_slot_bits;
/// Number of randomized rounds the generated-distribution round-trip test runs.
const generated_test_rounds = 5;

/// Width of the least-significant-bit key used to index the predictor lookup table. Two values
/// sharing these `lsb_bits` low bits usually differ only in higher bits, making one a good XOR
/// predictor for the other; the encoder keys its `indices` table on them to find a recent match.
const lsb_bits = 14;
/// Mask selecting the low `lsb_bits` of a value's bit pattern, i.e. its predictor lookup-table key.
const lsb_mask: u64 = (1 << lsb_bits) - 1;

/// Quantized leading-zero counts from the Chimp paper. `@clz(xor)` is rounded down to one of these
/// eight boundaries so the chosen bucket index fits in `leading_zero_bucket_bits`.
const leading_zero_buckets = [_]u6{ 0, 8, 12, 16, 18, 20, 22, 24 };

/// Per-thread scratch for the predictor lookup table. Allocating and zeroing the 64 KB `indices`
/// table on every block dominated compress time, so it is reused across calls: zeroed once, then kept
/// clean by resetting only the slots a block dirtied (recorded in `dirty`) instead of wiping all
/// 64 KB. Thread-local so concurrent encoders never share state; the allocation is retained until the
/// thread exits.
// `u32` slots (not `usize`) halve the table to 64 KB and the `dirty` list with it; this caps a single
// compress call at ~4.3 billion values, far beyond any realistic block.
const Scratch = struct {
    indices: []u32,
    dirty: ArrayList(u32),
};
threadlocal var scratch: ?Scratch = null;

/// Compress `uncompressed_values` into `compressed_values` using Chimp128's value codec.
/// `allocator` backs the configuration parser, the ring buffer and predictor lookup table, and the
/// bit writer's scratch buffer. `method_configuration` must be an empty configuration; any field
/// makes the call return `Error.InvalidConfiguration`. On success `compressed_values` holds
/// `[count: u64][first_value: f64][XOR marker bits...]`. If an error occurs it is returned. The
/// predictor selection and per-marker encoding logic is described inline in the function body.
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
    // Reused across calls via the per-thread `scratch`: the table is zeroed once, then kept clean by
    // resetting only the slots dirtied by the previous call. `current_index` starts at
    // `previous_values` so unseen keys (index 0) fail the staleness check
    // `current_index - indices[key] < previous_values`. `dirty` has room for every possible key, so
    // tracking and resetting touched slots never allocates.
    if (scratch == null) {
        // The scratch outlives any single call, so it is owned by a process-lifetime allocator rather
        // than the caller's (which may be transient); it is intentionally never freed.
        const scratch_allocator = std.heap.page_allocator;
        const table = try scratch_allocator.alloc(u32, 1 << lsb_bits);
        @memset(table, 0);
        var dirty = ArrayList(u32).empty;
        try dirty.ensureTotalCapacity(scratch_allocator, 1 << lsb_bits);
        scratch = .{ .indices = table, .dirty = dirty };
    }
    const indices = scratch.?.indices;
    const dirty = &scratch.?.dirty;
    for (dirty.items) |slot| indices[slot] = 0;
    dirty.clearRetainingCapacity();

    var current_index: usize = previous_values;

    const first_value_bits: u64 = @bitCast(first_value);
    const first_value_key: usize = @intCast(first_value_bits & lsb_mask);
    stored_values[current_index % previous_values] = first_value_bits;
    dirty.appendAssumeCapacity(@intCast(first_value_key));
    indices[first_value_key] = @intCast(current_index);
    current_index += 1;

    var previous_value_bits: u64 = first_value_bits;
    var previous_leading_zeros: u6 = leading_zero_buckets[0];

    var bit_writer = try shared_structs.BulkBitWriter.init(allocator, compressed_values);

    for (uncompressed_values[1..]) |value| {
        const current_value_bits: u64 = @bitCast(value);
        const key: usize = @intCast(current_value_bits & lsb_mask);
        const prev_index: usize = indices[key];

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
                if (indices[key] == 0) dirty.appendAssumeCapacity(@intCast(key));
                indices[key] = @intCast(current_index);
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
                if (indices[key] == 0) dirty.appendAssumeCapacity(@intCast(key));
                indices[key] = @intCast(current_index);
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
        if (indices[key] == 0) dirty.appendAssumeCapacity(@intCast(key));
        indices[key] = @intCast(current_index);
        current_index += 1;
        previous_value_bits = current_value_bits;
    }

    try bit_writer.flushBits();
}

/// Decompress a Chimp128-encoded `compressed_values` stream into `decompressed_values`.
/// `allocator` grows `decompressed_values` and backs the ring buffer that mirrors the encoder's so
/// predictor lookups stay in sync. `compressed_values` must start with the
/// `[count: u64][first_value: f64]` header written by `compress`; malformed or truncated streams
/// return `Error.ByteStreamError` or `Error.UnsupportedInput` rather than trapping. If an error
/// occurs it is returned. The per-marker decoding logic is described inline in the function body.
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

    // The decoder reads ring-slot indices directly from the bitstream (markers `00`/`01`),
    // so the LSB→slot hash table that the encoder maintains is not needed here.

    var current_index: usize = previous_values;

    const first_value_bits: u64 = @bitCast(first_value);
    stored_values[current_index % previous_values] = first_value_bits;
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
            // The encoder never emits a zero meaningful-bit count for this marker.
            // Reject corrupted streams explicitly instead of treating them as xor = 0.
            if (meaningful_bit_count == 0) return Error.UnsupportedInput;
            // Validate the geometry before casting: leading + meaningful must leave room for
            // a non-negative trailing-zero count that still fits in u6.
            const occupied: u16 = @as(u16, leading_bucket) + @as(u16, meaningful_bit_count);
            if (occupied > bits_per_value) return Error.UnsupportedInput;
            const trailing_zeros: u6 = @intCast(bits_per_value - occupied);
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

        stored_values[current_index % previous_values] = current_value_bits;
        current_index += 1;
        previous_value_bits = current_value_bits;

        const value: f64 = @bitCast(current_value_bits);
        try decompressed_values.append(allocator, value);
    }
}

/// Map an exact leading-zero count `leading_zeros` (as returned by `@clz`) to the index of the
/// largest `leading_zero_buckets` boundary that does not exceed it. The returned `u3` is the value
/// written to the stream so the decoder can recover the same bucket.
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
