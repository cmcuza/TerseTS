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

//! Implementation of a delta encoding scheme followed by fixed-length bit-packing to compress
//! floating-point time series. This method guarantees lossless recovery for all values.
//! If all values are "effectively" integers (i.e., have no meaningful fractional part), the
//! sequence is scaled and stored as delta-encoded integers. Otherwise, values are split into
//! parts. Those recoverable after scaling are stored as delta-encoded integers. Others are stored
//! in raw `f64` format. A bitmask tracks which encoding is used per element. This hybrid format
//! ensures full fidelity for all values while improving compression when possible. Bit-packing
//! follows common techniques for efficient integer encoding, as seen in:
//! Lemire et al., "SIMD Compression and the Intersection of Sorted Integers", 2016.
//! https://doi.org/10.1002/spe.2326.

const std = @import("std");
const ArrayList = std.ArrayList;
const math = std.math;
const mem = std.mem;
const testing = std.testing;

const tersets = @import("../tersets.zig");
const configuration = @import("../configuration.zig");
const tester = @import("../tester.zig");
const shared_structs = @import("../utilities/shared_structs.zig");
const shared_functions = @import("../utilities/shared_functions.zig");

const Method = tersets.Method;
const Error = tersets.Error;

/// Compress `uncompressed_values` using "Delta Encoding" and a "Fixed-length Bit-Packing".
/// The function writes the result to `compressed_values`. The `compressed_values` includes
/// all necessary metadata enabling correct decompression. The `allocator` is used for memory
/// management of intermediates containers. The `method_configuration` is expected to be
/// `EmptyConfiguration`,otherwise an error is returned instead of ignoring the configuration.
/// If an error occurs it is returned.
pub fn compress(
    allocator: mem.Allocator,
    uncompressed_values: []const f64,
    compressed_values: *ArrayList(u8),
    method_configuration: []const u8,
) Error!void {

    // Expects an empty configuration.
    _ = try configuration.parse(
        allocator,
        configuration.EmptyConfiguration,
        method_configuration,
    );

    // The first bit of the encoded `count` marks the encoding mode:
    // 1 = all values are "effectively" integers, 0 = some values require fallback
    // The remaining 31 bits store the length of the time series.
    // Max supported input length: 2^31 - 1 (≈ 2.14B values ~ 16GB memory).
    // Although a realistic upper bound, we check to avoid miss behaviour.
    const maximum_len: usize = 0x7FFF_FFFF; // 2,147,483,647
    if (uncompressed_values.len > maximum_len)
        return Error.UnsupportedInput;

    // To ensure lossless recovery and good compression ratio, values are scaled using a precision
    // factor (`scale`) large enough to preserve their decimal digits. The highest such scale is
    // chosen only if all values remain within the i64 range after scaling. Otherwise, the minimum
    // scale that preserves some values is used as fallback. Without scaling, compressing floats
    // directly using their u64 bit pattern (via @bitCast) can be extremely inefficient. Tiny float
    // differences (e.g., 0.00001) may cause large differences in the u64 representation, defeating
    // the purpose of delta encoding. Therefore, we attempt to preserve a representation where deltas
    // between scaled integers are small, enabling better compression.
    var all_effective_ints = true;
    var min_scale: f64 = math.floatMax(f64); // minimum required precision.
    var max_scale: f64 = 1.0; // maximum required precision.
    for (uncompressed_values) |val| {
        if (!math.isFinite(val) or val > 1e15)
            return Error.UnsupportedInput;

        if (!isEffectivelyInteger(val)) {
            const current_scale = try detectScaleFromPrecision(allocator, val);
            if (current_scale < min_scale) min_scale = current_scale;
            if (current_scale > max_scale) max_scale = current_scale;
            all_effective_ints = false;
        }
    }

    // Choose final scale. If all values can be directly cast to integers without lossing precision,
    // then the `final_scale` is 1.0.
    var final_scale = min_scale;
    if (!all_effective_ints) {
        // Check if max_scale can safely be used without overflow.
        var safe = true;
        for (uncompressed_values) |val| {
            const scaled = @abs(val * max_scale);
            if (scaled > @as(f64, @floatFromInt(std.math.maxInt(i64)))) {
                safe = false;
                break;
            }
        }
        if (safe) {
            final_scale = max_scale;
            all_effective_ints = true;
        }
    } else {
        final_scale = 1.0;
    }

    // Step 3: Store header with mode bit and time series length.
    const count: u32 = @intCast(uncompressed_values.len);
    const mode_flag: u32 = if (all_effective_ints) 1 else 0;
    const encoded_count: u32 = (count & 0x7FFF_FFFF) | (mode_flag << 31);

    try shared_functions.appendValue(
        allocator,
        u32,
        encoded_count,
        compressed_values,
    );
    try shared_functions.appendValue(
        allocator,
        f64,
        final_scale,
        compressed_values,
    );

    // Step 4: Delegate to specialized encoder.
    if (mode_flag == 1) {
        // All values can be stored as scaled integers.
        try compressAllAsIntegers(
            allocator,
            uncompressed_values,
            compressed_values,
            final_scale,
        );
    } else {
        // Mix of scalable and raw values: fallback stream + bitmask.
        try compressMixedEncoding(
            allocator,
            uncompressed_values,
            compressed_values,
            final_scale,
        );
    }
}

/// Decompress `compressed_values` produced by "Delta Encoding" and "Bit-Packing". The function
/// writes the result to  `decompressed_values`. If the encoding mode indicates that all values
/// were scaled integers, the stream is decoded using delta reconstruction. Otherwise, a bitmask
/// and fallback stream are used to reconstruct mixed `f64` values with full fidelity. The `allocator`
/// is used in mixed-mode decoding for intermediate results. If an error occurs it is returned.
pub fn decompress(
    allocator: mem.Allocator,
    compressed_values: []const u8,
    decompressed_values: *ArrayList(f64),
) Error!void {
    // Ensure compressed stream has enough bytes for count and scale.
    if (compressed_values.len < 9)
        return Error.UnsupportedInput;

    var cursor: usize = 0;

    // Step 1: Read mode bit + count from header.
    const count_packed: u32 = try shared_functions.readOffsetValue(
        u32,
        compressed_values,
        &cursor,
    );
    const mode_bit: u1 = @intCast(count_packed >> 31); // 1 = all-as-integers.
    const count: u32 = count_packed & 0x7FFFFFFF;
    const scale: f64 = try shared_functions.readOffsetValue(f64, compressed_values, &cursor);

    // Step 2: Dispatch to appropriate decoder.
    if (mode_bit == 1) {
        try decompressAllAsIntegers(
            allocator,
            compressed_values,
            decompressed_values,
            count,
            scale,
        );
    } else {
        // Read scale after count for mixed encoding.
        try decompressMixedEncoding(
            allocator,
            compressed_values,
            decompressed_values,
            count,
            scale,
        );
    }
}

/// Compress a time series where all `uncompressed_values` are effectively integers (no precision
/// loss on scaling). The function applies a delta encoding and bit-packing to scaled i64 values.
/// The `scale` is used to bring the values back to their original precision level.
/// The `allocator` is used for intermediate values. If n error occurs it is returned.
fn compressAllAsIntegers(
    allocator: mem.Allocator,
    uncompressed_values: []const f64,
    compressed_values: *ArrayList(u8),
    scale: f64,
) Error!void {
    var scaled_values = ArrayList(i64).empty;
    defer scaled_values.deinit(allocator);

    // Step 1: Scale and convert to i64.
    for (uncompressed_values) |val| {
        const scaled: i64 = @intFromFloat(@round(val * scale));
        try scaled_values.append(allocator, scaled);
    }

    // Step 2: Store first value.
    try shared_functions.appendValue(
        allocator,
        i64,
        scaled_values.items[0],
        compressed_values,
    );

    // Step 3: Find min_delta.
    var min_delta: i64 = std.math.maxInt(i64);
    for (1..scaled_values.items.len) |i| {
        const delta = scaled_values.items[i] - scaled_values.items[i - 1];
        if (delta < min_delta) min_delta = delta;
    }

    // Store min_delta.
    try shared_functions.appendValue(
        allocator,
        i64,
        min_delta,
        compressed_values,
    );

    // Step 4: Bit-pack quantized deltas (delta - min_delta).
    var bit_writer = shared_structs.bitWriter(
        .little,
        compressed_values.writer(allocator),
    );
    for (1..scaled_values.items.len) |i| {
        const delta = scaled_values.items[i] - scaled_values.items[i - 1];
        const quantized: u64 = @intCast(delta - min_delta);

        try bitpackU64(
            @TypeOf(compressed_values.writer(allocator)),
            quantized,
            &bit_writer,
        );
    }

    try bit_writer.flushBits();
}

/// Compress a time series where the `uncompress_values` have mixed-precision using delta encoding
/// and fallback bit-cast. This function compress each value by first scaling it by `scale` and
/// checking if it can be exactly recovered (losslessly). If so, it is stored as an integer and
/// delta-encoded in `compressed_values`. Otherwise, the original floating-point value is bit-cast
/// and stored separately in `compressed_values`. A bitmask is written to track which encoding was
/// used per element (0 = scaled int, 1 = fallback). This hybrid strategy ensures full lossless
/// reconstruction while enabling compression opportunities on recoverable values. The memory
/// `allocator` is used for intermediate results. If an error occurs it is returned.
fn compressMixedEncoding(
    allocator: mem.Allocator,
    uncompressed_values: []const f64,
    compressed_values: *ArrayList(u8),
    scale: f64,
) Error!void {
    // Step 1: Separate values into "recoverable as scaled integers" vs fallback bit-cast.
    var mask = ArrayList(u8).empty; // Bitmask: 1 bit per value.
    defer mask.deinit(allocator);

    var scaled_values = ArrayList(i64).empty; // Holds scaled and rounded integers.
    defer scaled_values.deinit(allocator);

    var fallback_bits = ArrayList(u64).empty; // Holds bit-casted fallback values.
    defer fallback_bits.deinit(allocator);

    var current_byte: u8 = 0;
    var bit_index: u3 = 0;

    for (uncompressed_values) |val| {
        const scaled_f64 = @round(val * scale);
        const recovered = scaled_f64 / scale;

        if (recovered == val) {
            // Value can be stored as an integer after scaling.
            const scaled_i: i64 = @intFromFloat(scaled_f64);
            try scaled_values.append(allocator, scaled_i);
            // Bit = 0.
        } else {
            // Value cannot be losslessly recovered, store bit-cast f64.
            const raw_bits: u64 = @bitCast(val);
            try fallback_bits.append(allocator, raw_bits);
            current_byte |= (@as(u8, 1) << bit_index); // Bit = 1.
        }

        // Write current_byte if full (8 bits), else keep accumulating.
        if (bit_index == 7) {
            try mask.append(allocator, current_byte);
            current_byte = 0;
            bit_index = 0;
        } else {
            bit_index += 1;
        }
    }

    // Flush any remaining bits not written yet.
    if (bit_index != 0) {
        try mask.append(allocator, current_byte);
    }

    // Step 2: Store the count of scaled values.
    try shared_functions.appendValue(
        allocator,
        u32,
        @intCast(scaled_values.items.len),
        compressed_values,
    );

    // Step 3: Write bitmask for value encoding types.
    try compressed_values.appendSlice(allocator, mask.items);

    // Step 4: Store scaled values using delta encoding.
    if (scaled_values.items.len > 0) {
        try shared_functions.appendValue(
            allocator,
            i64,
            scaled_values.items[0],
            compressed_values,
        ); // First value.

        // Compute deltas and track minimum.
        var deltas = ArrayList(i64).empty;
        defer deltas.deinit(allocator);

        var prev: i64 = scaled_values.items[0];
        var min_delta: i64 = std.math.maxInt(i64);
        for (scaled_values.items[1..]) |curr| {
            const delta = curr - prev;
            if (delta < min_delta) min_delta = delta;
            try deltas.append(allocator, delta);
            prev = curr;
        }

        // Write min_delta to reconstruct deltas later.
        try shared_functions.appendValue(allocator, i64, min_delta, compressed_values);

        // Bit-pack delta values using fixed-length prefix scheme.
        var bit_writer = shared_structs.bitWriter(
            .little,
            compressed_values.writer(allocator),
        );

        for (deltas.items) |d| {
            const val: u64 = @intCast(d - min_delta);
            try bitpackU64(@TypeOf(compressed_values.writer(allocator)), val, &bit_writer);
        }

        try bit_writer.flushBits();
    }

    // Step 5: Append raw fallback values to the end.
    for (fallback_bits.items) |bits| {
        try shared_functions.appendValue(allocator, u64, bits, compressed_values);
    }
}

/// Decompress a total of `count` values stored in `compressed_values` encoded as scaled integers
/// with delta and bit-packed representation. The function stores the final decompressed results
/// in `decompressed_values`. The function assumes all values were compressible as integers
/// (i.e., no fallback stream). The `scale` is used to bring them back to their original precision.
/// If an error occurs it is returned.
fn decompressAllAsIntegers(
    allocator: mem.Allocator,
    compressed_values: []const u8,
    decompressed_values: *ArrayList(f64),
    count: usize,
    scale: f64,
) Error!void {
    // Initialize the cursor at 12 to skip the header contain a u32 and a f64.
    var cursor: usize = 12;

    // Step 1: Read the first scaled value and the min_delta used in delta encoding.
    const first: i64 = try shared_functions.readOffsetValue(
        i64,
        compressed_values,
        &cursor,
    );
    const min_delta: i64 = try shared_functions.readOffsetValue(
        i64,
        compressed_values,
        &cursor,
    );

    var current: i64 = first;
    const first_decompressed_value: f64 = @as(f64, @floatFromInt(current)) / scale;
    try decompressed_values.append(allocator, first_decompressed_value);

    // Step 2: Set up a bit reader for the packed deltas.
    var stream = std.io.fixedBufferStream(compressed_values[cursor..]);
    var bit_reader = shared_structs.bitReader(.little, stream.reader());

    // Step 3: Read and decode each delta using the 2-bit length prefix.
    for (1..count) |_| {
        const len1 = bit_reader.readBitsNoEof(u1, 1) catch break;
        const len2 = bit_reader.readBitsNoEof(u1, 1) catch return Error.ByteStreamError;

        var val: u64 = 0;
        if (len1 == 0) {
            if (len2 == 0) {
                val = bit_reader.readBitsNoEof(u8, 8) catch return Error.ByteStreamError;
            } else {
                val = bit_reader.readBitsNoEof(u16, 16) catch return Error.ByteStreamError;
            }
        } else {
            if (len2 == 0) {
                val = bit_reader.readBitsNoEof(u32, 32) catch return Error.ByteStreamError;
            } else {
                val = bit_reader.readBitsNoEof(u64, 64) catch return Error.ByteStreamError;
            }
        }

        // Step 4: Reconstruct and append the original value.
        const delta = @as(i64, @intCast(val)) + min_delta;
        current += delta;
        const decompressed_value: f64 = @as(f64, @floatFromInt(current)) / scale;
        try decompressed_values.append(allocator, decompressed_value);
    }
}

/// Decompress a total of `count` values stored in `compressed_values` encoded using a hybrid
/// (mixed) strategy. The function stores the final decompressed results in `decompressed_values`.
/// The function uses the `scale` factor used during compression to transform f64 to i64.
/// The `allocator` is used for allocating temporary buffers for scaled and fallback values.
/// If an error occurs it is returned.
pub fn decompressMixedEncoding(
    allocator: mem.Allocator,
    compressed_values: []const u8,
    decompressed_values: *ArrayList(f64),
    count: u32,
    scale: f64,
) Error!void {
    // Initialize the cursor at 12 to skip the header contain a u32 and a f64.
    var cursor: usize = 12;

    // Step 1: Read number of scaled values (rest are fallback).
    if (cursor + 4 > compressed_values.len)
        return Error.UnsupportedInput;

    const n_scaled: u32 = try shared_functions.readOffsetValue(u32, compressed_values, &cursor);
    const n_fallback = count - n_scaled;

    // Step 2: Read the bitmask indicating fallback positions.
    const mask_bytes: u32 = (count + 7) / 8;
    if (compressed_values.len < cursor + mask_bytes)
        return Error.UnsupportedInput;

    const mask = compressed_values[cursor .. cursor + mask_bytes];
    cursor += mask_bytes;

    // Step 3: Read and reconstruct scaled values using delta encoding.
    var scaled = ArrayList(i64).empty;
    defer scaled.deinit(allocator);

    if (n_scaled > 0) {
        // Read first scaled value and the minimum delta used during compression.
        const first: i64 = try shared_functions.readOffsetValue(
            i64,
            compressed_values,
            &cursor,
        );
        const min_delta: i64 = try shared_functions.readOffsetValue(
            i64,
            compressed_values,
            &cursor,
        );
        try scaled.append(allocator, first);

        // Use a bit reader to parse delta values from compressed stream.
        var stream = std.io.fixedBufferStream(compressed_values[cursor..]);
        var bit_reader = shared_structs.bitReader(.little, stream.reader());

        for (1..n_scaled) |_| {
            const header_1: u8 = bit_reader.readBitsNoEof(u8, 1) catch break;
            const header_2: u8 = bit_reader.readBitsNoEof(u8, 1) catch break;

            var quantized: u64 = 0;
            if (header_1 == 0) {
                if (header_2 == 0) {
                    quantized = bit_reader.readBitsNoEof(u8, 8) catch return Error.ByteStreamError;
                } else {
                    quantized = bit_reader.readBitsNoEof(u16, 16) catch return Error.ByteStreamError;
                }
            } else {
                if (header_2 == 0) {
                    quantized = bit_reader.readBitsNoEof(u32, 32) catch return Error.ByteStreamError;
                } else {
                    quantized = bit_reader.readBitsNoEof(u64, 64) catch return Error.ByteStreamError;
                }
            }

            const delta = @as(i64, @intCast(quantized)) + min_delta;
            try scaled.append(allocator, scaled.items[scaled.items.len - 1] + delta);
        }
    }

    // Step 4: Read fallback values from the end of the compressed buffer.
    const fallback_start = compressed_values.len - (n_fallback * @sizeOf(u64));
    const fallback_bytes = compressed_values[fallback_start..];

    var fallback = ArrayList(f64).empty;
    defer fallback.deinit(allocator);

    for (0..n_fallback) |i| {
        const offset = i * @sizeOf(u64);
        cursor = offset;
        const bits: u64 = try shared_functions.readOffsetValue(
            u64,
            fallback_bytes,
            &cursor,
        );
        try fallback.append(allocator, @bitCast(bits));
    }

    // Step 5: Merge both streams based on the bitmask to reconstruct the original sequence.
    var scaled_idx: usize = 0;
    var fallback_idx: usize = 0;

    for (0..count) |i| {
        const byte = mask[i / 8];
        const bit = (byte >> @intCast(i % 8)) & 1;

        if (bit == 0) {
            const val: f64 = @as(f64, @floatFromInt(scaled.items[scaled_idx])) / scale;
            try decompressed_values.append(allocator, val);
            scaled_idx += 1;
        } else {
            try decompressed_values.append(allocator, fallback.items[fallback_idx]);
            fallback_idx += 1;
        }
    }
}

/// Write a u64 `value` with `bit_writer` to a bit stream using a compact bit-packed encoding scheme.
/// The encoding uses 2 header bits to indicate how many bits are required to represent `value`:
/// 00 is 8 bits, 01 is 16 bits, 10 is 32 bits, and 11 is 64 bits. This encoding reduces the total
/// number of bits when most values are small. The `WriterType` is the type of the underlying writer
/// used by `bit_writer`. `bit_writer` is a pointer to the initialized `BitWriter` that receives the
/// encoded bits. The function returns a error if any occurrs.
fn bitpackU64(
    comptime WriterType: type,
    value: u64,
    bit_writer: *shared_structs.BitWriter(.little, WriterType),
) !void {
    if (value <= 0xFF) { // 8-bits.
        // Header: 00.
        try bit_writer.writeBits(@as(u1, 0b0), 1);
        try bit_writer.writeBits(@as(u1, 0b0), 1);
        try bit_writer.writeBits(@as(u8, @intCast(value)), 8);
    } else if (value <= 0xFFFF) { // 16-bits.
        // Header: 01.
        try bit_writer.writeBits(@as(u1, 0b0), 1);
        try bit_writer.writeBits(@as(u1, 0b1), 1);
        try bit_writer.writeBits(@as(u16, @intCast(value)), 16);
    } else if (value <= 0xFFFFFFFF) { // 32-bits.
        // Header: 10.
        try bit_writer.writeBits(@as(u1, 0b1), 1);
        try bit_writer.writeBits(@as(u1, 0b0), 1);
        try bit_writer.writeBits(@as(u32, @intCast(value)), 32);
    } else { // 64-bits (no compression).
        // Header: 11.
        try bit_writer.writeBits(@as(u1, 0b1), 1);
        try bit_writer.writeBits(@as(u1, 0b1), 1);
        try bit_writer.writeBits(@as(u64, value), 64);
    }
}

/// Determines whether a 64-bit floating-point number (f64) has no fractional part, i.e., whether
/// it is an "effective" integer. This function works by inspecting the IEEE-754 binary layout
/// of the number: 1 sign bit | 11 exponent bits | 52 mantissa bits. For values >= 1.0, the exponent
/// indicates how many bits are used to represent the integer part of the number. If there are no
/// non-zero bits in the mantissa beyond the exponent range, then the number is an exact integer.
/// Returns `true` if `value` has no fractional part (i.e., can be exactly represented as an integer)
/// `false` otherwise.
fn isEffectivelyInteger(value: f64) bool {
    // Convert the float to its raw 64-bit binary representation.
    const bits: u64 = @bitCast(value);

    // Extract the exponent bits (bits 52..62) and un-bias them by subtracting 1023.
    const exponent: i32 = @as(i32, @intCast((bits >> 52) & 0x7FF)) - 1023;

    // Case 1: Large exponent (>= 52) - all mantissa bits are integer-representable.
    if (exponent >= 52) return true;

    // Case 2: Negative exponent (< 0) - number is fractional or between -1.0 and 1.0.
    if (exponent < 0) return false;

    // Case 3: Some lower bits of the mantissa represent fractional precision.
    // Compute how many bits in the mantissa are "below" the exponent (i.e., fractional).
    const frac_bits = 52 - exponent;

    // Create a bitmask to isolate just those fractional bits.
    // We cast to `u6` for shift safety - Zig requires shift amounts to be <= 63.
    const mantissa_mask = (@as(u64, 1) << @as(u6, @intCast(frac_bits))) - 1;

    // Extract the 52-bit mantissa (bits 0..51).
    const mantissa = bits & 0xFFFFFFFFFFFFF;

    // If any of the fractional bits in the mantissa are set, it's not an integer.
    return (mantissa & mantissa_mask) == 0;
}

/// Detects the scale factor required to represent the given floating-point `value` with its maximum
/// significant decimal precision. ignoring trailing zeros, and returns a scale factor as a power of 10.
/// For example, if `value` is `1.2300`, the scale factor will be `100` (for two significant digits).
/// The `allocator` is used for temporary string allocation. Returns any allocation errors encountered.
fn detectScaleFromPrecision(allocator: mem.Allocator, value: f64) !f64 {
    var max_precision: usize = 0;

    const str = try std.fmt.allocPrint(allocator, "{d}", .{value});
    defer allocator.free(str);

    if (mem.indexOfScalar(u8, str, '.')) |dot| {
        const fractional = str[(dot + 1)..];
        var digits = fractional.len;

        while (digits > 0 and fractional[digits - 1] == '0') {
            digits -= 1;
        }

        if (digits > max_precision) max_precision = digits;
    }

    return std.math.pow(f64, 10.0, @floatFromInt(max_precision));
}

test "bitpacked delta enconding can compress and decompress bounded values" {
    const allocator = testing.allocator;

    var uncompressed_values = ArrayList(f64).empty;
    defer uncompressed_values.deinit(allocator);

    // Generate 100 random values within the range of -1e13 to 1e13.
    try tester.generateBoundedRandomValues(
        allocator,
        &uncompressed_values,
        -1e1,
        1e1,
        null,
    );
    try tester.generateBoundedRandomValues(
        allocator,
        &uncompressed_values,
        -1e3,
        1e3,
        null,
    );
    try tester.generateBoundedRandomValues(
        allocator,
        &uncompressed_values,
        -1e9,
        1e9,
        null,
    );
    try tester.generateBoundedRandomValues(
        allocator,
        &uncompressed_values,
        -1e13,
        1e13,
        null,
    );
}

test "bitpacked delta encoding can compress and decompress monotocally increasing values" {
    // This test validates that if the input values are all monotocally increasing values
    // the delta enconding can achieve high compression.
    const allocator = testing.allocator;

    var uncompressed_values = ArrayList(f64).empty;
    defer uncompressed_values.deinit(allocator);

    var current_timestamp: f64 = @floatFromInt(tester.generateBoundRandomInteger(usize, 0, 1e7, null));
    for (0..100) |_| {
        try uncompressed_values.append(allocator, current_timestamp);
        const delta_value: f64 = @floatFromInt(tester.generateBoundRandomInteger(
            usize,
            10,
            100,
            null,
        ));
        current_timestamp += delta_value;
    }

    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);

    const method_configuration = "{}";
    try compress(
        allocator,
        uncompressed_values.items,
        &compressed_values,
        method_configuration,
    );

    // Considering the range of the input data, the compressed values should always be smaller.
    try testing.expect(uncompressed_values.items.len * 8 > compressed_values.items.len);
}

test "bitpacked delta encoding can compress and decompress integers at different scales" {
    const allocator = testing.allocator;

    var uncompressed_values = ArrayList(f64).empty;
    defer uncompressed_values.deinit(allocator);

    for (2..6) |i| {
        const at_most = std.math.pow(usize, 10, i * 2);
        for (0..100) |_| {
            const random_value: f64 = @floatFromInt(tester.generateBoundRandomInteger(
                usize,
                0,
                at_most,
                null,
            ));
            try uncompressed_values.append(allocator, random_value);
        }
    }
    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);

    const method_configuration = "{}";
    try compress(
        allocator,
        uncompressed_values.items,
        &compressed_values,
        method_configuration,
    );

    var decompressed_values = ArrayList(f64).empty;
    defer decompressed_values.deinit(allocator);

    try decompress(
        allocator,
        compressed_values.items,
        &decompressed_values,
    );

    for (uncompressed_values.items, 0..) |actual_value, i| {
        const decompressed_value = decompressed_values.items[i];
        try testing.expectEqual(decompressed_value, actual_value);
    }
}

test "bitpacked delta encoding can compress and decompress floating points of mixed and small precision" {
    const allocator = testing.allocator;

    var uncompressed_values = ArrayList(f64).empty;
    defer uncompressed_values.deinit(allocator);

    // Generating bounded precision f64 automatically is not possible
    // thus we insert them manually.
    try uncompressed_values.append(allocator, 3.4);
    try uncompressed_values.append(allocator, 1.80);
    try uncompressed_values.append(allocator, 5.48);
    try uncompressed_values.append(allocator, 6.49435);
    try uncompressed_values.append(allocator, 1233.445);
    try uncompressed_values.append(allocator, 312.456);
    try uncompressed_values.append(allocator, 463.4245);
    try uncompressed_values.append(allocator, 0.4445);

    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);

    const method_configuration = "{}";
    try compress(
        allocator,
        uncompressed_values.items,
        &compressed_values,
        method_configuration,
    );

    var decompressed_values = ArrayList(f64).empty;
    defer decompressed_values.deinit(allocator);

    try decompress(
        allocator,
        compressed_values.items,
        &decompressed_values,
    );

    for (uncompressed_values.items, 0..) |actual_value, i| {
        const decompressed_value = decompressed_values.items[i];
        try testing.expectEqual(decompressed_value, actual_value);
    }
}
