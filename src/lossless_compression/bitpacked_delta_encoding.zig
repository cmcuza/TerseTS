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

//! Lossless floating-point time series compression using delta encoding and a
//! fixed-length bit-packing scheme. If all values have a zero fractional part
//! ("effectively" integers), the sequence is scaled and stored entirely as delta-encoded
//! integers. Otherwise, a mixed encoding is used: values recoverable via scaling are stored
//! as delta integers, while the rest are stored in raw `f64` format. A bitmask tracks the
//! encoding format used for each element, ensuring lossless coompression.
//! Bit-packing techniques are based on:
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
/// `EmptyConfiguration`, otherwise an error is returned.
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

    // The first bit of the compressed representation marks the encoding mode:
    // 1 = all values are "effectively" integers, 0 = some values require fallback
    // The remaining 31 bits store the length of the time series.
    // Max supported input length: 2^31 - 1 (≈ 2.14B values ~ 16GB memory).
    // Although a realistic upper bound, we check to avoid miss behaviour.
    const maximum_length: usize = 0x7FFF_FFFF; // 2,147,483,647
    if (uncompressed_values.len > maximum_length)
        return Error.UnsupportedInput;

    // Step 1: To ensure lossless compression and good compression ratio, values are scaled using a
    // precision factor (`scale`) large enough to preserve their decimal digits. The highest such
    // scale is chosen only if all values remain within the i64 range after scaling. Otherwise, the
    // minimum scale that preserves some values is used as fallback. Without scaling, compressing
    // floats directly using their u64 bit pattern (via @bitCast) can be extremely inefficient. Tiny
    // float differences (e.g., 0.00001) may cause large differences in the u64 representation,
    // defeating the purpose of delta encoding. Therefore, we attempt to preserve a representation
    // where deltas between scaled integers are small, enabling better compression.
    var all_effective_ints = true;
    var minimum_scale: f64 = math.floatMax(f64); // minimum required precision.
    var maximum_scale: f64 = 1.0; // maximum required precision.
    for (uncompressed_values) |value| {
        if (!math.isFinite(value) or value > 1e15)
            return Error.UnsupportedInput;

        if (!isEffectivelyInteger(value)) {
            const current_scale = try detectScaleFromPrecision(allocator, value);
            if (current_scale < minimum_scale) minimum_scale = current_scale;
            if (current_scale > maximum_scale) maximum_scale = current_scale;
            all_effective_ints = false;
        }
    }

    // Step 2: Choose final scale. If all values can be directly cast to integers without
    // losing precision, then the `final_scale` is 1.0.
    var final_scale = minimum_scale;
    if (!all_effective_ints) {
        // Check if maximum_scale can safely be used without overflow.
        var is_without_overflow = true;
        for (uncompressed_values) |value| {
            const scaled = @abs(value * maximum_scale);
            if (scaled > @as(f64, @floatFromInt(math.maxInt(i64)))) {
                is_without_overflow = false;
                break;
            }
        }
        if (is_without_overflow) {
            final_scale = maximum_scale;
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
        // Mix of scalable and raw values: fallback stream and bitmask.
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

    var index: usize = 0;

    // Step 1: Read mode bit + count from header.
    const count_packed: u32 = try shared_functions.readOffsetValue(
        u32,
        compressed_values,
        &index,
    );
    const mode: u1 = @intCast(count_packed >> 31); // all-as-integers.
    const count: u32 = count_packed & 0x7FFFFFFF;
    const scale: f64 = try shared_functions.readOffsetValue(f64, compressed_values, &index);

    // Step 2: Dispatch to appropriate decoder.
    if (mode == 1) {
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

/// Compresses a floating-point time series into `compressed_values` by multiplying `uncompressed_values`
/// by `scale` to store them as exact `i64` integers via delta encoding and bit-packing. The `scale` acts
/// as a multiplier (e.g., 100.0) to convert fractional values into integers without truncation, allowing
/// exact lossless compression. Temporary memory for the bit-packing process is managed via the `allocator`.
/// Returns an error if memory allocation fails.
fn compressAllAsIntegers(
    allocator: mem.Allocator,
    uncompressed_values: []const f64,
    compressed_values: *ArrayList(u8),
    scale: f64,
) Error!void {
    // Step 1: Compute the first scaled value and find minimum_delta in a single pass.
    const first: i64 = @intFromFloat(@round(uncompressed_values[0] * scale));

    var minimum_delta: i64 = math.maxInt(i64);
    var previous: i64 = first;
    for (uncompressed_values[1..]) |val| {
        const current: i64 = @intFromFloat(@round(val * scale));
        const delta = current - previous;
        if (delta < minimum_delta) minimum_delta = delta;
        previous = current;
    }

    // Step 2: Store first value.
    try shared_functions.appendValue(allocator, i64, first, compressed_values);

    // Step 3: Store minimum_delta.
    try shared_functions.appendValue(allocator, i64, minimum_delta, compressed_values);

    // Step 4: Bit-pack quantized deltas (delta - minimum_delta).
    var bit_writer = try shared_structs.BitWriter.init(allocator, compressed_values);

    previous = first;
    for (uncompressed_values[1..]) |val| {
        const current: i64 = @intFromFloat(@round(val * scale));
        const delta = current - previous;
        const quantized: u64 = @intCast(delta - minimum_delta);
        try bitpackU64(quantized, &bit_writer);
        previous = current;
    }

    try bit_writer.flushBits();
}

/// Compress a time series where the `uncompressed_values` have mixed-precision using delta encoding
/// and fallback bit-cast. This function compress each value by first scaling it by `scale` and
/// checking if it can be exactly recovered (losslessly). If so, it is stored as an integer and
/// delta-encoded in `compressed_values`. Otherwise, the original floating-point value is bit-cast
/// and stored separately in `compressed_values`. A bitmask is written to track which encoding was
/// used per element (0 = scaled int, 1 = fallback). This mixed encoding ensures full lossless
/// reconstruction while enabling compression opportunities on recoverable values. The memory
/// `allocator` is used for intermediate results. If an error occurs it is returned.
fn compressMixedEncoding(
    allocator: mem.Allocator,
    uncompressed_values: []const f64,
    compressed_values: *ArrayList(u8),
    scale: f64,
) Error!void {
    // Step 1: First pass — build bitmask, count scaled values, and find the first scaled value
    // and minimum_delta. This avoids allocating separate scaled_values, deltas, and fallback_bits
    // arrays, at the cost of re-classifying each value in two additional passes below.
    var mask = ArrayList(u8).empty;
    defer mask.deinit(allocator);

    var current_byte: u8 = 0;
    var bit_index: u3 = 0;
    var n_scaled: u32 = 0;
    var first_scaled: i64 = 0;
    var minimum_delta: i64 = math.maxInt(i64);
    var prev_scaled: i64 = 0;
    var found_first_scaled = false;

    for (uncompressed_values) |value| {
        const scaled_f64 = @round(value * scale);
        const recovered = scaled_f64 / scale;

        if (recovered == value) {
            const scaled_i: i64 = @intFromFloat(scaled_f64);
            if (!found_first_scaled) {
                first_scaled = scaled_i;
                prev_scaled = scaled_i;
                found_first_scaled = true;
            } else {
                const delta = scaled_i - prev_scaled;
                if (delta < minimum_delta) minimum_delta = delta;
                prev_scaled = scaled_i;
            }
            n_scaled += 1;
        } else {
            current_byte |= (@as(u8, 1) << bit_index); // Bit for fallback.
        }

        if (bit_index == 7) {
            try mask.append(allocator, current_byte);
            current_byte = 0;
            bit_index = 0;
        } else {
            bit_index += 1;
        }
    }

    if (bit_index != 0) {
        try mask.append(allocator, current_byte);
    }

    // Step 2: Write n_scaled and bitmask.
    try shared_functions.appendValue(allocator, u32, n_scaled, compressed_values);
    try compressed_values.appendSlice(allocator, mask.items);

    // Step 3: Write first scaled value and minimum_delta, then second pass to bit-pack deltas.
    if (n_scaled > 0) {
        try shared_functions.appendValue(allocator, i64, first_scaled, compressed_values);
        try shared_functions.appendValue(allocator, i64, minimum_delta, compressed_values);

        var bit_writer = try shared_structs.BitWriter.init(allocator, compressed_values);
        var previous: i64 = first_scaled;
        var is_first = true;

        for (uncompressed_values) |val| {
            const scaled_f64 = @round(val * scale);
            const recovered = scaled_f64 / scale;
            if (recovered == val) {
                if (is_first) {
                    is_first = false;
                    continue; // First value already stored above.
                }
                const current: i64 = @intFromFloat(scaled_f64);
                const delta = current - previous;
                const quantized: u64 = @intCast(delta - minimum_delta);
                try bitpackU64(quantized, &bit_writer);
                previous = current;
            }
        }

        try bit_writer.flushBits();
    }

    // Step 4: Third pass — write fallback values directly, avoiding a fallback_bits array.
    for (uncompressed_values) |value| {
        const scaled_f64 = @round(value * scale);
        const recovered = scaled_f64 / scale;
        if (recovered != value) {
            const raw_bits: u64 = @bitCast(value);
            try shared_functions.appendValue(allocator, u64, raw_bits, compressed_values);
        }
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
    // Initialize the index at 12 to skip the header containing a u32 and a f64.
    var index: usize = 12;

    // Step 1: Read the first scaled value and the minimum_delta used in delta encoding.
    const first: i64 = try shared_functions.readOffsetValue(
        i64,
        compressed_values,
        &index,
    );
    const minimum_delta: i64 = try shared_functions.readOffsetValue(
        i64,
        compressed_values,
        &index,
    );

    var current: i64 = first;
    const first_decompressed_value: f64 = @as(f64, @floatFromInt(current)) / scale;
    try decompressed_values.append(allocator, first_decompressed_value);

    // Step 2: Set up a bit reader for the packed deltas.
    const reader = std.Io.Reader.fixed(compressed_values[index..]);
    var bit_reader = shared_structs.BitReader.init(reader);

    // Step 3: Read and decode each delta using the 2-bit length prefix.
    for (1..count) |_| {
        const length_1 = bit_reader.readBitsNoEof(u1, 1) catch break;
        const length_2 = bit_reader.readBitsNoEof(u1, 1) catch return Error.CorruptedCompressedData;

        var value: u64 = 0;
        if (length_1 == 0) {
            if (length_2 == 0) {
                value = bit_reader.readBitsNoEof(u8, 8) catch return Error.CorruptedCompressedData;
            } else {
                value = bit_reader.readBitsNoEof(u16, 16) catch return Error.CorruptedCompressedData;
            }
        } else {
            if (length_2 == 0) {
                value = bit_reader.readBitsNoEof(u32, 32) catch return Error.CorruptedCompressedData;
            } else {
                value = bit_reader.readBitsNoEof(u64, 64) catch return Error.CorruptedCompressedData;
            }
        }

        // Step 4: Reconstruct and append the original value.
        const delta = @as(i64, @intCast(value)) + minimum_delta;
        current += delta;
        const decompressed_value: f64 = @as(f64, @floatFromInt(current)) / scale;
        try decompressed_values.append(allocator, decompressed_value);
    }
}

/// Decompress a total of `count` values stored in `compressed_values` using a mixed encoding.
/// The function stores the final decompressed results in `decompressed_values`.
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
    // Initialize the index at 12 to skip the header containing a u32 and a f64.
    var index: usize = 12;

    // Step 1: Read number of scaled values (rest are fallback).
    if (index + 4 > compressed_values.len)
        return Error.UnsupportedInput;

    const n_scaled: u32 = try shared_functions.readOffsetValue(u32, compressed_values, &index);
    const n_fallback = count - n_scaled;

    // Step 2: Read the bitmask indicating fallback positions.
    const mask_bytes: u32 = (count + 7) / 8;
    if (compressed_values.len < index + mask_bytes)
        return Error.UnsupportedInput;

    const mask = compressed_values[index .. index + mask_bytes];
    index += mask_bytes;

    // Step 3: Read and reconstruct scaled values using delta encoding.
    var scaled = ArrayList(i64).empty;
    defer scaled.deinit(allocator);

    if (n_scaled > 0) {
        // Read first scaled value and the minimum delta used during compression.
        const first: i64 = try shared_functions.readOffsetValue(
            i64,
            compressed_values,
            &index,
        );
        const minimum_delta: i64 = try shared_functions.readOffsetValue(
            i64,
            compressed_values,
            &index,
        );
        try scaled.append(allocator, first);

        // Use a bit reader to parse delta values from compressed stream.
        const reader = std.Io.Reader.fixed(compressed_values[index..]);
        var bit_reader = shared_structs.BitReader.init(reader);

        for (1..n_scaled) |_| {
            const header_1: u8 = bit_reader.readBitsNoEof(u8, 1) catch break;
            const header_2: u8 = bit_reader.readBitsNoEof(u8, 1) catch break;

            var quantized: u64 = 0;
            if (header_1 == 0) {
                if (header_2 == 0) {
                    quantized = bit_reader.readBitsNoEof(u8, 8) catch return Error.CorruptedCompressedData;
                } else {
                    quantized = bit_reader.readBitsNoEof(u16, 16) catch return Error.CorruptedCompressedData;
                }
            } else {
                if (header_2 == 0) {
                    quantized = bit_reader.readBitsNoEof(u32, 32) catch return Error.CorruptedCompressedData;
                } else {
                    quantized = bit_reader.readBitsNoEof(u64, 64) catch return Error.CorruptedCompressedData;
                }
            }

            const delta = @as(i64, @intCast(quantized)) + minimum_delta;
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
        index = offset;
        const bits: u64 = try shared_functions.readOffsetValue(
            u64,
            fallback_bytes,
            &index,
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
/// encoded bits. The function returns an error if any occurs.
fn bitpackU64(
    value: u64,
    bit_writer: *shared_structs.BitWriter,
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

    return math.pow(f64, 10.0, @floatFromInt(max_precision));
}

test "bitpacked delta encoding can compress and decompress bounded values" {
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

test "bitpacked delta encoding can compress and decompress monotonically increasing values" {
    // This test validates that if the input values are all monotonically increasing values
    // the delta encoding can achieve high compression. Monotonically increasing values are
    // common in time series data, where timestamps or cumulative metrics often increase over time.
    // The test generates a sequence of 100 values that start from a random timestamp and increase
    // by random deltas. After compression, the test checks that the compressed size is smaller
    // than the original size, which is expected due to the efficiency of delta encoding on such data.
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
    // This test validates that the bitpacked delta encoding can effectively compress and decompress
    // integer values at different scales. The test generates 100 random integer values at various
    // scales (e.g., 1, 10, 100, 1000, etc.) and compresses them using the `compress` function.
    // After decompression, the test checks that the decompressed values match the original
    // uncompressed values, ensuring that the scaling and delta encoding processes work correctly
    // across a range of magnitudes. This is important for time series data that may contain values
    // with varying orders of magnitude, such as financial data, sensor readings, or cumulative metrics.
    const allocator = testing.allocator;

    var uncompressed_values = ArrayList(f64).empty;
    defer uncompressed_values.deinit(allocator);

    for (2..6) |i| {
        const at_most = math.pow(usize, 10, i * 2);
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

    // We hardcode f64 values because generating random base-10 bounded floats (e.g., 0.3)
    // introduces sub-normal binary noise (0.2999999...) that trips up our automated scale finder.
    // These inputs explicitly test mixed decimal precision depths ranging from 1 to 5 places.
    // This is not an exhaustive test of all possible precision
    // combinations, but it serves as a sanity check for the scaling logic.
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
