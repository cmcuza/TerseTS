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

//! Implementation of the Camel lossy/lossless floating-point time series compression method.
//! The method is described in:
//! Yuanyuan Yao, Lu Chen, Ziquan Fang, Yunjun Gao, Christian S. Jensen and Tianyi Liet al.,
//!  "Camel: Efficient Compression of Floating-Point Time Series", SIGMOD 2024.
//! https://dl.acm.org/doi/epdf/10.1145/3698802
//!
//! Camel compresses the integer and decimal parts of double-precision floats separately.
//! For the decimal part it uses a novel XOR‑based scheme that identifies a `dxor` value
//! which yields stable leading‑zero counts, thus improving compression ratio and speed.
//! The integer part is compressed with a simple difference encoder (Algorithm 1).
//! Decimal compression follows Algorithm 2; decompression follows Algorithms 3–5.
//!
//! The method is lossless when the input values have at most `decimal_places` significant
//! digits after the decimal point; otherwise trailing digits are discarded (lossy).
//! The user controls the trade‑off by setting `decimal_places` in the configuration.

const std = @import("std");
const math = std.math;
const mem = std.mem;
const testing = std.testing;
const ArrayList = std.ArrayList;
const Allocator = mem.Allocator;

const tersets = @import("../tersets.zig");
const shared_functions = @import("../utilities/shared_functions.zig");
const shared_structs = @import("../utilities/shared_structs.zig");
const tester = @import("../tester.zig");

const Error = tersets.Error;
const Method = tersets.Method;

/// Number of bits in an IEEE-754 `f64`.
const bits_per_value = 64;
/// Maximum decimal places supported by the algorithm (double can represent at most 17).
const max_decimal_places = 17;
/// Default decimal places when configuration is empty.
const default_decimal_places = 4;
/// Number of bits used to encode the decimal place count `l` in the bitstream.
/// The paper's Algorithm 2 writes `l` using only 2 bits ("out.write(l, 2)"),
/// but that can only represent values 0–3, while `l` (the decimal place count)
/// ranges up to `max_decimal_places` (17). 2 bits would silently truncate `l`
/// (e.g. the default `l = 4` becomes `0`), corrupting decompression. We use
/// enough bits to round-trip any value up to `max_decimal_places`.
const decimal_count_bits: u6 = 5;

/// Convert a double to its 64-bit bitwise representation.
/// Parameters:
/// - `x`: the floating-point value to reinterpret.
/// Returns the raw IEEE-754 bit pattern of `x` as a `u64`. This is a plain bit
/// reinterpretation (no rounding), so it is exact and fully reversible.
inline fn doubleToBits(x: f64) u64 {
    return @as(u64, @bitCast(x));
}

/// Convert a 64-bit bitwise representation back to a double.
/// Parameters:
/// - `bits`: a raw IEEE-754 bit pattern, normally produced by `doubleToBits`.
/// Returns the `f64` whose bit pattern equals `bits`. This is the inverse of
/// `doubleToBits`.
inline fn bitsToDouble(bits: u64) f64 {
    return @as(f64, @bitCast(bits));
}

/// `i64` can represent magnitudes up to `2^63`; `2^63` itself is exactly
/// representable as an `f64` (it is a power of two), which makes it a
/// convenient, exact threshold for the overflow checks below.
const i64_magnitude_limit: f64 = 9223372036854775808.0;

/// Integer part of `x`, truncated toward zero.
/// Parameters:
/// - `x`: the value to split. Caller must first check `fitsIntegerPart(x)`,
///   because `@trunc(x)` for huge finite magnitudes (e.g. `math.floatMax(f64)`)
///   lies far outside the `i64` range and `@intFromFloat` would panic.
/// Returns the truncated integer part as an `i64`, or `0` for NaN and Inf.
inline fn intPart(x: f64) i64 {
    if (math.isNan(x) or math.isInf(x)) return 0;
    return @intFromFloat(@trunc(x));
}

/// Fractional part of `x`, computed as `x - trunc(x)`.
/// Parameters:
/// - `x`: the value to split.
/// Returns the fractional remainder, or `0` for NaN and Inf. Because `@trunc`
/// rounds toward zero, the result keeps the sign of `x` (e.g.
/// `fracPart(-56.789) == -0.789`).
inline fn fracPart(x: f64) f64 {
    if (math.isNan(x) or math.isInf(x)) return 0.0;
    return x - @trunc(x);
}

/// Check whether `x` can be safely split into an integer part and a decimal part.
/// Parameters:
/// - `x`: the value to check.
/// Returns `true` when `x` is finite and its truncated integer part fits in an
/// `i64`, i.e. it is safe to call `intPart`/`fracPart` on it. Values that fail
/// this check (NaN, Inf, and finite values whose magnitude is `>= 2^63`, e.g.
/// `math.floatMax(f64)`) are stored verbatim as raw bits instead, see
/// `compress`/`decompress`.
inline fn fitsIntegerPart(x: f64) bool {
    if (math.isNan(x) or math.isInf(x)) return false;
    return @abs(@trunc(x)) < i64_magnitude_limit;
}

/// Check whether `x` is `-0.0`.
/// Parameters:
/// - `x`: the value to check.
/// Returns `true` only for negative zero. The integer/decimal split cannot
/// represent `-0.0`: `intPart(-0.0) == 0` and `fracPart(-0.0) == +0.0` (IEEE 754
/// subtraction of equal values always yields `+0.0`), so recombining the parts
/// always produces `+0.0` and silently flips the sign bit. `-0.0` is therefore
/// routed through the verbatim "special" raw-bits path, see `compress`/`decompress`.
inline fn isNegativeZero(x: f64) bool {
    return x == 0.0 and math.signbit(x);
}

/// Compute `dxor.ddec` from formula (3) of the paper: `v.ddec - 2^{-l} * floor(v.ddec / 2^{-l})`.
/// Parameters:
/// - `frac`: the (non-negative) decimal magnitude `v.ddec`, a value in `[0, 1)`.
/// - `l`: the number of decimal places used to choose the step `2^{-l}`.
/// Returns the `dxor` decimal part used as the XOR partner during compression,
/// or `0.0` when `l == 0` (there is no decimal information to preserve).
fn computeDxorFrac(frac: f64, l: u8) f64 {
    if (l == 0) return 0.0;
    const step = math.pow(f64, 2.0, -@as(f64, @floatFromInt(l)));
    const t = @floor(frac / step);
    return frac - step * t;
}

// ----------------------------------------------------------------------------
//  Integer part compression
// ----------------------------------------------------------------------------

/// Write the compressed integer part of one value into the bit writer.
/// Parameters:
/// - `prev_int`: the integer part of the previous normal value, or `null` when
///   this is the first value (or the previous values were all "special").
/// - `int_part`: the integer part of the current value to encode.
/// - `writer`: the bit-level sink the encoded representation is appended to.
/// Behavior:
/// When `prev_int` is `null` the full 64-bit double representation of
/// `int_part` is stored verbatim. Otherwise the difference `int_part - prev_int`
/// is delta-encoded: differences in `{-1, 0, 1}` use a 2-bit code, and larger
/// differences fall back to a sign bit, a range flag, and a 3- or 16-bit
/// magnitude, mirroring Algorithm 1 of the paper.
fn compressIntegerPart(prev_int: ?i64, int_part: i64, writer: *shared_structs.BulkBitWriter) !void {
    if (prev_int == null) {
        // First value: store full double bits.
        const bits = doubleToBits(@as(f64, @floatFromInt(int_part)));
        try writer.writeBits(bits, 64);
    } else {
        const diff = int_part - prev_int.?;
        const abs_diff = @abs(diff);
        if (abs_diff <= 1) {
            // Encode diff+1 in 2 bits (0: -1, 1: 0, 2: 1).
            const code = @as(u2, @intCast(@as(u3, @intCast(diff + 1)) & 0b11));
            try writer.writeBits(code, 2);
        } else {
            // Range marker `3` (0b11): tells the decoder (Algorithm 3, line 4-5,
            // which always reads a 2-bit `range` first) that this is the
            // sign+flag+magnitude encoding rather than the |diff| <= 1 code.
            try writer.writeBits(@as(u2, 0b11), 2);
            // Sign bit: 1 for positive, 0 for negative.
            try writer.writeBits(@as(u1, @intFromBool(diff >= 0)), 1);
            // Range flag: 0 if abs_diff < 8, 1 otherwise.
            const flag = @as(u1, @intFromBool(abs_diff >= 8));
            try writer.writeBits(flag, 1);
            const bits_needed: u6 = if (flag == 0) 3 else 16;
            try writer.writeBits(@as(u64, @intCast(abs_diff)), bits_needed);
        }
    }
}

// ----------------------------------------------------------------------------
//  Integer part decompression
// ----------------------------------------------------------------------------

/// Read and decompress the integer part of one value.
/// Parameters:
/// - `prev_int`: the integer part of the previously decoded normal value, or
///   `null` when no normal value has been decoded yet.
/// - `reader`: the bit-level source the encoded representation is read from.
/// Returns the decoded integer part as an `i64`, or `Error.ByteStreamError` if
/// the stream ends before the expected bits are available. This is the inverse
/// of `compressIntegerPart`: it mirrors the same `null`/delta cases and adds
/// the decoded difference back onto `prev_int`.
fn decompressIntegerPart(prev_int: ?i64, reader: *shared_structs.BulkBitReader) Error!i64 {
    if (prev_int == null) {
        const bits = reader.readBitsNoEof(u64, 64) catch return Error.ByteStreamError;
        const v = bitsToDouble(bits);
        return intPart(v);
    } else {
        const range = reader.readBitsNoEof(u2, 2) catch return Error.ByteStreamError;
        const diff = switch (range) {
            0, 1, 2 => @as(i64, @intCast(range)) - 1,
            3 => blk: {
                const sign_bit = reader.readBitsNoEof(u1, 1) catch return Error.ByteStreamError;
                const flag = reader.readBitsNoEof(u1, 1) catch return Error.ByteStreamError;
                const bits = reader.readBitsNoEof(u64, if (flag == 0) 3 else 16) catch return Error.ByteStreamError;
                var d = @as(i64, @intCast(bits));
                if (sign_bit == 0) d = -d;
                break :blk d;
            },
        };
        return prev_int.? + diff;
    }
}

// ----------------------------------------------------------------------------
//  dxor' encoding/decoding
// ----------------------------------------------------------------------------

/// Write the integer representation `dxor_prime = dxor_frac * 10^l`.
/// Parameters:
/// - `dxor_prime`: the quantized `l`-digit decimal value to encode.
/// - `l`: the number of decimal places, which selects the bit-width scheme.
/// - `writer`: the bit-level sink the encoded representation is appended to.
/// Behavior:
/// Uses the variable-length scheme from Algorithm 2 (lines 12-22): for
/// `l <= 1` the value is written directly in `l + 1` bits; for `l == 2` a
/// 1-bit "small" flag picks between 2 and 5 bits; for `l >= 3` a 2-bit index
/// selects one of four bit-width buckets derived from the maximum possible
/// value, and the value is written using that many bits.
fn writeDxorPrime(dxor_prime: u64, l: u8, writer: *shared_structs.BulkBitWriter) !void {
    if (l <= 1) {
        try writer.writeBits(dxor_prime, @as(u6, @intCast(l + 1)));
    } else if (l == 2) {
        const small = @as(u1, @intFromBool(dxor_prime <= 4));
        try writer.writeBits(small, 1);
        const bits_needed: u6 = if (small == 1) 2 else 5;
        try writer.writeBits(dxor_prime, bits_needed);
    } else {
        const max_val = @ceil(math.log2(math.pow(f64, 2.0, -@as(f64, @floatFromInt(l))) *
            math.pow(f64, 10.0, @as(f64, @floatFromInt(l)))));
        const max = @as(u64, @intFromFloat(max_val));
        const t1 = @as(u64, @intFromFloat(@exp2(@as(f64, @floatFromInt(max)) * 0.25)));
        const t2 = @as(u64, @intFromFloat(@exp2(@as(f64, @floatFromInt(max)) * 0.5)));
        const t3 = @as(u64, @intFromFloat(@exp2(@as(f64, @floatFromInt(max)) * 0.75)));
        var idx: u2 = 3;
        if (dxor_prime <= t1) {
            idx = 0;
        } else if (dxor_prime <= t2) {
            idx = 1;
        } else if (dxor_prime <= t3) {
            idx = 2;
        }
        try writer.writeBits(idx, 2);
        const bits_to_write = @as(u6, @intCast((@as(u64, idx) + 1) * max / 4));
        try writer.writeBits(dxor_prime, bits_to_write);
    }
}

/// Read `dxor_prime` that was written by `writeDxorPrime`.
/// Parameters:
/// - `l`: the number of decimal places that was used to encode the value;
///   it determines which bit-width scheme to decode with.
/// - `reader`: the bit-level source the encoded representation is read from.
/// Returns the decoded `dxor_prime` value, or `Error.ByteStreamError` if the
/// stream ends before the expected bits are available. This mirrors the three
/// cases of `writeDxorPrime` (`l <= 1`, `l == 2`, `l >= 3`) exactly.
fn readDxorPrime(l: u8, reader: *shared_structs.BulkBitReader) Error!u64 {
    if (l <= 1) {
        return reader.readBitsNoEof(u64, @as(u6, @intCast(l + 1))) catch return Error.ByteStreamError;
    } else if (l == 2) {
        const small = reader.readBitsNoEof(u1, 1) catch return Error.ByteStreamError;
        const bits_needed: u6 = if (small == 1) 2 else 5;
        return reader.readBitsNoEof(u64, bits_needed) catch return Error.ByteStreamError;
    } else {
        const max_val = @ceil(math.log2(math.pow(f64, 2.0, -@as(f64, @floatFromInt(l))) *
            math.pow(f64, 10.0, @as(f64, @floatFromInt(l)))));
        const max = @as(u64, @intFromFloat(max_val));
        const idx = reader.readBitsNoEof(u2, 2) catch return Error.ByteStreamError;
        const bits_to_read = @as(u6, @intCast((@as(u64, idx) + 1) * max / 4));
        return reader.readBitsNoEof(u64, bits_to_read) catch return Error.ByteStreamError;
    }
}

// ----------------------------------------------------------------------------
//  Decimal part compression and decompression
// ----------------------------------------------------------------------------

/// Compress the fractional part of one value using the Camel XOR scheme.
/// Parameters:
/// - `frac`: the fractional part `x - trunc(x)`. It can be negative for
///   negative `x`, since Zig's `@trunc` rounds toward zero.
/// - `l`: the number of decimal places to keep; controls precision and the
///   bit-widths used by the encoding below.
/// - `writer`: the bit-level sink the encoded representation is appended to.
/// Behavior:
/// The Camel paper models the decimal part as a non-negative digit sequence
/// with the sign stored separately (`DF(v) = +-d^int.d^dec`, Definition 2.2),
/// so a sign bit is written first and the rest of the encoding works on
/// `magnitude = |frac|`, which always lies in `[0, 1)`. When `magnitude` is at
/// least one quantization step, the XOR scheme of Algorithm 2 is used (flag 1);
/// otherwise the value is too small for the XOR step to help and the rounded
/// `l`-digit magnitude is stored directly (flag 0).
fn compressDecimalPart(frac: f64, l: u8, writer: *shared_structs.BulkBitWriter) !void {
    const is_negative = frac < 0.0;
    try writer.writeBits(@as(u1, @intFromBool(is_negative)), 1);
    const magnitude = @abs(frac);

    // Write the decimal count l.
    try writer.writeBits(l, decimal_count_bits);

    const step = math.pow(f64, 2.0, -@as(f64, @floatFromInt(l)));
    if (magnitude >= step) {
        // Normal path: use XOR.
        try writer.writeBits(@as(u1, 1), 1); // flag = 1
        const dxor_frac_raw = computeDxorFrac(magnitude, l);
        // Quantize dxor to its `l`-decimal-digit integer form `dxor_prime`, then
        // rebuild `dxor_frac` from that *same* integer the way the decompressor
        // will (`dxor_prime / 10^l`). The XOR cancellation in Theorem 3.1 only
        // produces an exact round-trip when both sides XOR against a *bit-identical*
        // `dxor`; reusing `dxor_frac_raw` here (computed via the binary-floor
        // formula of Eq. 3) while the decompressor rebuilds it from the
        // decimal-quantized `dxor_prime` would XOR against two slightly different
        // floats, leaking error into bits outside the stored `center_bits` range.
        const factor = math.pow(f64, 10.0, @as(f64, @floatFromInt(l)));
        const dxor_prime = @as(u64, @intFromFloat(@round(dxor_frac_raw * factor)));
        const dxor_frac = @as(f64, @floatFromInt(dxor_prime)) / factor;
        // Construct numbers with integer part = 1.
        const a = 1.0 + magnitude;
        const b = 1.0 + dxor_frac;
        const xor_bits = doubleToBits(a) ^ doubleToBits(b);
        // Center bits are the l bits starting at position 12 (after sign and exponent).
        const shift = @as(u6, @intCast(52 - l));
        const mask = (@as(u64, 1) << @as(u6, @intCast(l))) - 1;
        const center_bits = @as(u64, @truncate((xor_bits >> shift) & mask));
        try writer.writeBits(center_bits, @as(u6, @intCast(l)));
        // dxor' = dxor_frac * 10^l
        try writeDxorPrime(dxor_prime, l, writer);
    } else {
        // Tiny fraction: store the fraction directly (lossy but precise for small values).
        try writer.writeBits(@as(u1, 0), 1); // flag = 0
        const factor = math.pow(f64, 10.0, @as(f64, @floatFromInt(l)));
        // Round to the nearest integer rather than truncating: `magnitude * factor`
        // for an exact `l`-digit decimal value lands just below the true integer
        // (e.g. 0.0001 * 10^4 == 0.999999999999988…) due to binary floating-point
        // representation error, so truncation would silently drop the digit (-> 0).
        const dxor_prime = @as(u64, @intFromFloat(@round(magnitude * factor)));
        try writeDxorPrime(dxor_prime, l, writer);
    }
}

/// Decompress the fractional part that was compressed by `compressDecimalPart`.
/// Parameters:
/// - `reader`: the bit-level source the encoded representation is read from.
/// Returns the reconstructed fractional value (including its sign), or
/// `Error.ByteStreamError` if the stream ends before the expected bits are
/// available. Reads the sign bit, the decimal place count `l`, and the XOR
/// flag, then either reverses the XOR scheme (flag 1) or reads the stored
/// magnitude directly (flag 0) — the exact inverse of `compressDecimalPart`.
fn decompressDecimalPart(reader: *shared_structs.BulkBitReader) Error!f64 {
    const is_negative = reader.readBitsNoEof(u1, 1) catch return Error.ByteStreamError;
    const l = @as(u8, @intCast(reader.readBitsNoEof(u8, decimal_count_bits) catch return Error.ByteStreamError));
    const flag = reader.readBitsNoEof(u1, 1) catch return Error.ByteStreamError;
    var magnitude: f64 = undefined;
    if (flag == 1) {
        const center_bits = reader.readBitsNoEof(u64, @as(u6, @intCast(l))) catch return Error.ByteStreamError;
        // Reconstruct the XOR result: place the center bits back at the position
        // they were extracted from in compressDecimalPart, i.e. bits [52-l, 51]
        // (the most-significant `l` bits of the mantissa). This is the inverse of
        // `(xor_bits >> (52 - l)) & mask` and must use the same shift amount;
        // the paper's "<<< 12" uses an MSB-relative shift convention that is not
        // the same as Zig's LSB-relative `<<`.
        const shift = @as(u6, @intCast(52 - l));
        const vd = center_bits << shift;
        const dxor_prime = try readDxorPrime(l, reader);
        const factor = math.pow(f64, 10.0, @as(f64, @floatFromInt(l)));
        const dxor_frac = @as(f64, @floatFromInt(dxor_prime)) / factor;
        const a_bits = doubleToBits(1.0 + dxor_frac);
        const xor_bits = a_bits ^ vd;
        const b = bitsToDouble(xor_bits);
        magnitude = b - 1.0;
        if (magnitude < 0.0) magnitude = 0.0;
    } else {
        const dxor_prime = try readDxorPrime(l, reader);
        const factor = math.pow(f64, 10.0, @as(f64, @floatFromInt(l)));
        magnitude = @as(f64, @floatFromInt(dxor_prime)) / factor;
    }
    return if (is_negative == 1) -magnitude else magnitude;
}

/// Compress an array of double-precision floats using Camel.
/// Parameters:
/// - `allocator`: used both for the JSON configuration parsing and for any
///   growth of `compressed_values` and the internal bit writer.
/// - `uncompressed_values`: the input values to compress, in time order.
/// - `compressed_values`: output buffer that the compressed bytes are appended to.
/// - `method_configuration`: a JSON object such as `{"decimal_places": 4}`.
///   When empty, `default_decimal_places` is used; an out-of-range or malformed
///   value yields `Error.InvalidConfiguration`.
/// The compressed stream format is:
///   - 8 bytes:  u64 value count
///   - 8 bytes:  first value (raw f64)
///   - for each subsequent value:
///        1 bit:  0 = normal, 1 = special (NaN, Inf, `-0.0`, or a finite value
///                whose magnitude is too large to split into integer/decimal parts)
///        if special: 64 bits raw value
///        else: compressed integer part + compressed decimal part
pub fn compress(
    allocator: Allocator,
    uncompressed_values: []const f64,
    compressed_values: *ArrayList(u8),
    method_configuration: []const u8,
) Error!void {
    // Parse configuration manually.
    var decimal_places: u8 = default_decimal_places;
    if (method_configuration.len > 0) {
        var parsed = std.json.parseFromSlice(std.json.Value, allocator, method_configuration, .{}) catch return Error.InvalidConfiguration;
        defer parsed.deinit();
        if (parsed.value.object.get("decimal_places")) |v| {
            const int_val = v.integer;
            if (int_val < 0 or int_val > max_decimal_places) return Error.InvalidConfiguration;
            decimal_places = @intCast(int_val);
        } else if (parsed.value.object.count() > 0) {
            return Error.InvalidConfiguration;
        }
    }
    const l = decimal_places;

    // Write the value count.
    try shared_functions.appendValue(allocator, u64, @intCast(uncompressed_values.len), compressed_values);
    if (uncompressed_values.len == 0) return;

    // Write the first value raw.
    try shared_functions.appendValue(allocator, f64, uncompressed_values[0], compressed_values);

    // Use a bulk bit writer for the rest.
    var bit_writer = try shared_structs.BulkBitWriter.init(allocator, compressed_values);

    var prev_int: ?i64 = if (fitsIntegerPart(uncompressed_values[0])) intPart(uncompressed_values[0]) else null;
    for (uncompressed_values[1..]) |v| {
        const is_special = !fitsIntegerPart(v) or isNegativeZero(v);
        try bit_writer.writeBits(@as(u1, @intFromBool(is_special)), 1);
        if (is_special) {
            const bits = doubleToBits(v);
            try bit_writer.writeBits(bits, 64);
            // Special values (NaN, Inf, finite values too large to split into an
            // integer/decimal part such as `math.floatMax(f64)`, and `-0.0` whose
            // sign the split cannot preserve) have no meaningful integer part, so
            // we leave `prev_int` as-is for subsequent normal values.
            continue;
        }
        const int_part = intPart(v);
        const frac = fracPart(v);
        try compressIntegerPart(prev_int, int_part, &bit_writer);
        try compressDecimalPart(frac, l, &bit_writer);
        prev_int = int_part;
    }

    try bit_writer.flushBits();
}

/// Decompress a Camel-compressed byte stream back into an array of doubles.
/// Parameters:
/// - `allocator`: used to grow `decompressed_values` to the decoded value count.
/// - `compressed_values`: the byte stream produced by `compress`.
/// - `decompressed_values`: output buffer that decoded values are appended to,
///   in the same order they were compressed.
/// Returns `Error.UnsupportedInput` if the stream is shorter than the minimal
/// header, or `Error.ByteStreamError` if it ends before all values are decoded.
/// Reverses the format documented on `compress`: it reads the value count and
/// first raw value, then walks the bitstream decoding either a special raw
/// value or an integer/decimal pair for each remaining value.
pub fn decompress(
    allocator: Allocator,
    compressed_values: []const u8,
    decompressed_values: *ArrayList(f64),
) Error!void {
    var offset: usize = 0;
    const value_count = try shared_functions.readOffsetValue(u64, compressed_values, &offset);
    if (value_count == 0) return;

    if (compressed_values.len < 16) return Error.UnsupportedInput;
    try decompressed_values.ensureTotalCapacity(allocator, @intCast(value_count));

    const first_value = try shared_functions.readOffsetValue(f64, compressed_values, &offset);
    decompressed_values.appendAssumeCapacity(first_value);

    var prev_int: ?i64 = if (fitsIntegerPart(first_value)) intPart(first_value) else null;
    var bit_reader = shared_structs.BulkBitReader.init(compressed_values[offset..]);

    while (decompressed_values.items.len < value_count) {
        const is_special = bit_reader.readBitsNoEof(u1, 1) catch return Error.ByteStreamError;
        if (is_special == 1) {
            const bits = bit_reader.readBitsNoEof(u64, 64) catch return Error.ByteStreamError;
            const value = bitsToDouble(bits);
            decompressed_values.appendAssumeCapacity(value);
            // Do not update prev_int for special values.
            continue;
        }
        const int_part = try decompressIntegerPart(prev_int, &bit_reader);
        const frac = try decompressDecimalPart(&bit_reader);
        const value = @as(f64, @floatFromInt(int_part)) + frac;
        decompressed_values.appendAssumeCapacity(value);
        prev_int = int_part;
    }
}

// ----------------------------------------------------------------------------
//  Tests
// ----------------------------------------------------------------------------

test "camel round-trip fixed values" {
    const allocator = testing.allocator;

    const values = [_]f64{
        0.0,
        1.0,
        -1.0,
        12.34,
        -56.789,
        1000.0001,
        -9999.9999,
        3.14159,
        -2.5,
    };

    var compressed = ArrayList(u8).empty;
    defer compressed.deinit(allocator);

    // `3.14159` has 5 significant decimal digits, so `decimal_places` must be at
    // least 5 for Camel's lossless guarantee ("lossless when input values have at
    // most `decimal_places` significant decimal digits") to cover every value below.
    try compress(allocator, values[0..], &compressed, "{\"decimal_places\": 5}");

    var decompressed = ArrayList(f64).empty;
    defer decompressed.deinit(allocator);

    try decompress(allocator, compressed.items, &decompressed);

    try testing.expectEqual(values.len, decompressed.items.len);
    try testing.expect(shared_functions.isWithinErrorBound(
        values[0..],
        decompressed.items,
        1e-9,
    ));
}

test "camel roundtrips empty input" {
    const uncompressed_values = &[_]f64{};
    try tester.expectLosslessRoundTrip(testing.allocator, compress, decompress, uncompressed_values);
}

test "camel roundtrips single value" {
    const uncompressed_values = &[_]f64{42.5};
    try tester.expectLosslessRoundTrip(testing.allocator, compress, decompress, uncompressed_values);
}

test "camel roundtrips repeated values" {
    const uncompressed_values = &[_]f64{ 7.25, 7.25, 7.25, 7.25 };
    try tester.expectLosslessRoundTrip(testing.allocator, compress, decompress, uncompressed_values);
}

test "camel roundtrips simple changes" {
    const uncompressed_values = &[_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    try tester.expectLosslessRoundTrip(testing.allocator, compress, decompress, uncompressed_values);
}

test "camel roundtrips fractional examples from paper" {
    const uncompressed_values = &[_]f64{ 1.36, 1.11 };
    try tester.expectLosslessRoundTrip(testing.allocator, compress, decompress, uncompressed_values);
}

test "camel roundtrips with l=4" {
    const uncompressed_values = &[_]f64{ 1.4276, 1.0526 };
    try tester.expectLosslessRoundTrip(testing.allocator, compress, decompress, uncompressed_values);
}

test "camel roundtrips special floating-point values" {
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

test "camel round-trip integer-only, the edge cases and negative zero" {
    const uncompressed_values = &[_]f64{
        -65535.0, // the largest negative integer difference that fits in 16 bits
        0.0,
        -0.0,
        5.0,
        -7.0,
        65528.0, // maximal integer difference that fits in 16 bits
    };
    try tester.expectLosslessRoundTrip(testing.allocator, compress, decompress, uncompressed_values);
}
