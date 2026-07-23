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

//! Implementation of the Elf lossless floating-point time series compression method.
//! The method is described in:
//! Li et al., "Elf: Erasing-based Lossless Floating-Point Compression", VLDB 2023.
//! https://doi.org/10.14778/3587136.3587149
//! The bit-level layout, decimal-precision tables, and recovery formulas follow the authors'
//! reference Java implementation in the ELF repository: https://github.com/Spatio-Temporal-Lab/elf.

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

// Terms from the Elf paper (Theorem 3), defined once so later comments can stay short:
//   alpha                - decimal digits after the point.
//   beta                 - count of significant decimal digits.
//   beta_star            - beta stored per value in 4 bits; 0 is a sentinel for the
//                          exact negative-power-of-ten case (see `restorer`).
//   value_prime          - the value after `eraser` clears the low "noise" mantissa bits.
//   significand position - power-of-ten place of the most significant decimal digit.
//   f(alpha)             - binary bits needed for alpha decimal digits = ceil(alpha*log2(10)).
//   g(alpha)             - mantissa cut point = f(alpha) + exponent - 1023.

/// IEEE-754 `f64` layout used by the eraser: 52 mantissa bits and an exponent biased by 1023.
const mantissa_bits: u6 = 52;
const exponent_bias: i32 = 1023;
const exponent_mask: u64 = 0x7ff;

/// Digits needed to round-trip any `f64`. beta returns this to mean "no short exact decimal
/// form", which routes the value to no-erase.
const maximum_significant_digits: u8 = 17;

/// Cap on the `getSignificantCount` search. An `f64` carries ~16 significant digits, so beyond
/// this many steps the multiply is only chasing rounding noise.
const maximum_scale_iterations: u8 = 22;

/// Smallest value that `@intFromFloat` cannot truncate into an `i64`.
const maximum_safe_int_float: f64 = 0x1p63;

/// f(alpha) = ceil(alpha * log2(10)) for alpha in [0, 20]: the binary bits needed to hold alpha
/// decimal digits. The `eraser` reads it to find the mantissa cut point.
const f_alpha_table = [_]u8{
    0,  4,  7,  10, 14, 17, 20, 24, 27, 30,
    34, 37, 40, 44, 47, 50, 54, 57, 60, 64,
    67,
};

/// 10^i for i in [0, 20]. The significant-digit search scales by these until the product is an
/// exact integer.
const power_of_10_table = [_]f64{
    1.0,    1.0e1,  1.0e2,  1.0e3,  1.0e4,  1.0e5,  1.0e6,
    1.0e7,  1.0e8,  1.0e9,  1.0e10, 1.0e11, 1.0e12, 1.0e13,
    1.0e14, 1.0e15, 1.0e16, 1.0e17, 1.0e18, 1.0e19, 1.0e20,
};

/// 10^-i for i in [0, 20]. `restorer` looks the value up here when the original was an exact
/// negative power of ten.
const negative_power_of_10_table = [_]f64{
    1.0,     1.0e-1,  1.0e-2,  1.0e-3,  1.0e-4,  1.0e-5,  1.0e-6,
    1.0e-7,  1.0e-8,  1.0e-9,  1.0e-10, 1.0e-11, 1.0e-12, 1.0e-13,
    1.0e-14, 1.0e-15, 1.0e-16, 1.0e-17, 1.0e-18, 1.0e-19, 1.0e-20,
};

/// End-of-stream marker, written as a case-11 header behind a no-erase marker bit. The pair is
/// impossible for real data: bucket 7 means 24 leading zeros and a raw center of 0 means 64 center
/// bits, which do not fit in 64 together. The decoder needs no explicit value count.
const end_marker_lead_index: u3 = 7;
const end_marker_center_raw: u6 = 0;

/// State carried from one value to the next by `xorCompress` and `xorDecompress`. The bucket
/// fields are null until a bucket is written, so the first value can never reuse one.
const XorState = struct {
    stored_value_prime: u64,
    stored_leading_zeros: ?u6,
    stored_trailing_zeros: ?u6,
};

/// Compress `uncompressed_values` into `compressed_values` using Elf's eraser and XOR pipeline,
/// allocating with `allocator`. `method_configuration` must be empty (`{}`), otherwise
/// `Error.InvalidConfiguration` is returned. `uncompressed_values` must not be empty;
/// `tersets.compress` guarantees this. On success `compressed_values` holds
/// `[first_value: f64][bit stream][end-of-stream marker]`.
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

    var xor_state = XorState{
        .stored_value_prime = @bitCast(first_value),
        .stored_leading_zeros = null,
        .stored_trailing_zeros = null,
    };

    var bit_writer = try shared_structs.BulkBitWriter.init(allocator, compressed_values);

    for (uncompressed_values[1..]) |value| {
        const value_prime_bits = try eraser(&bit_writer, value);
        try xorCompress(&bit_writer, value_prime_bits, &xor_state);
    }

    // The end marker tells the decoder where to stop; flushed padding bits are never read.
    try writeEndMarker(&bit_writer);
    try bit_writer.flushBits();
}

/// Decompress an Elf `compressed_values` stream into `decompressed_values`, allocating with
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

    var xor_state = XorState{
        .stored_value_prime = @bitCast(first_value),
        .stored_leading_zeros = null,
        .stored_trailing_zeros = null,
    };

    var bit_reader = shared_structs.BulkBitReader.init(compressed_values[offset..]);

    while (true) {
        // `eraser` marker dispatch:
        //   "0"     (1 bit)  -> no erase.
        //   "1"+beta_star  (5 bits) -> erase, read beta_star (4 bits).
        const marker_bit = bit_reader.readBitsNoEof(u1, 1) catch return Error.CorruptedCompressedData;

        if (marker_bit == 0) {
            // No erase: value_prime equals the value here, so xorDecompress returns it directly (no restore).
            // A null result is the end-of-stream marker, so decoding stops here.
            const value_bits = (try xorDecompress(&bit_reader, &xor_state)) orelse break;
            try decompressed_values.append(allocator, @bitCast(value_bits));
            continue;
        }

        // Erase: read beta_star, then restore the original value.
        // The end-of-stream marker only appears on the no-erase path, so a null here means corruption.
        const beta_star = bit_reader.readBitsNoEof(u8, 4) catch return Error.CorruptedCompressedData;
        const value_prime_bits = (try xorDecompress(&bit_reader, &xor_state)) orelse
            return Error.CorruptedCompressedData;
        const value = try restorer(@bitCast(value_prime_bits), beta_star);
        try decompressed_values.append(allocator, value);
    }
}

/// Return the significand position of `value_abs`, plus `is_negative_power_of_ten` for the
/// 10^-i corner case where erasing would not preserve that position (paper Theorem 3).
fn significandPosition(value_abs: f64) struct { position: i16, is_negative_power_of_ten: bool } {
    if (value_abs >= 1.0) {
        // Find i such that 10^i <= value_abs < 10^(i+1), so significand position = i >= 0.
        for (0..power_of_10_table.len - 1) |i| {
            if (value_abs < power_of_10_table[i + 1]) {
                return .{ .position = @intCast(i), .is_negative_power_of_ten = false };
            }
        }
    } else {
        // Find i such that 10^-i <= value_abs < 10^-(i-1), so significand position = -i.
        // is_negative_power_of_ten fires when value_abs lands exactly on the lower boundary.
        for (1..negative_power_of_10_table.len) |i| {
            if (value_abs >= negative_power_of_10_table[i]) {
                return .{
                    .position = -@as(i16, @intCast(i)),
                    .is_negative_power_of_ten = (value_abs == negative_power_of_10_table[i]),
                };
            }
        }
    }
    // Fallback for values outside the tables (|value| >= 10^20 or |value| < 10^-20).
    // is_negative_power_of_ten must stay true only for exact negative powers (value = 10^-i, i > 0),
    // so guard on log10v < 0: a large positive power like 1e12 also has integral log10 but is NOT the
    // 10^-i corner case and must not be flagged (that would wrongly force beta_star = 0).
    const log10v = @log10(value_abs);
    return .{
        .position = @intFromFloat(@floor(log10v)),
        .is_negative_power_of_ten = (log10v < 0 and log10v == @floor(log10v)),
    };
}

/// Return f(alpha) for `alpha`, reading `f_alpha_table` when it is in range.
fn getFAlpha(alpha: i32) i32 {
    if (alpha >= f_alpha_table.len) {
        // Rare: alpha > 20 happens for very small values (|value| < 1e-10). When this hits,
        // the `eraser`'s downstream `eraseBits > 4` check usually routes to no-erase.
        return @intFromFloat(@ceil(@as(f64, @floatFromInt(alpha)) * @log2(@as(f64, 10.0))));
    }
    return @as(i32, f_alpha_table[@intCast(alpha)]);
}

/// Return 10^`i`, reading `power_of_10_table` when `i` is in range.
fn getPositivePowerOfTen(i: i32) f64 {
    if (i >= power_of_10_table.len) {
        return math.pow(f64, 10.0, @floatFromInt(i));
    }
    return power_of_10_table[@intCast(i)];
}

/// Return 10^-`i`, reading `negative_power_of_10_table` when `i` is in range.
fn getNegativePowerOfTen(i: i32) f64 {
    if (i >= negative_power_of_10_table.len) {
        return math.pow(f64, 10.0, -@as(f64, @floatFromInt(i)));
    }
    return negative_power_of_10_table[@intCast(i)];
}

/// Return how many significant decimal digits represent `value_abs` exactly, given that its
/// leading digit sits at `significand_position`. Returns `maximum_significant_digits` when there
/// is no short exact form.
fn getSignificantCount(value_abs: f64, significand_position: i16) u8 {
    // Smallest power-of-ten exponent worth trying: for `value_abs >= 1`, start at `exponent = 1`;
    // for `value_abs < 1`, start at `exponent = -significand_position` so the leading digit lands in
    // the ones place.
    var exponent: i32 = if (significand_position >= 0) 1 else -@as(i32, significand_position);

    var scaled: f64 = undefined;
    var scaled_int: i64 = undefined;

    // Increase the exponent until `value_abs * 10^exponent` is an exact integer, i.e. until every
    // significant digit has been shifted left of the decimal point.
    var iterations: u8 = 0;
    while (true) : (iterations += 1) {
        scaled = value_abs * getPositivePowerOfTen(exponent);
        // Check the bound before @intFromFloat: a value >= 2^63 would trap the cast.
        if (iterations >= maximum_scale_iterations or scaled >= maximum_safe_int_float) {
            return maximum_significant_digits;
        }
        scaled_int = @intFromFloat(scaled);
        if (@as(f64, @floatFromInt(scaled_int)) == scaled) break;
        exponent += 1;
    }

    // Confirm the scaling is exactly reversible. If `value_abs * 10^exponent` only "looked" integral
    // due to rounding in the multiply, dividing back out won't recover `value_abs` - meaning there is
    // no short exact form.
    if (scaled / getPositivePowerOfTen(exponent) != value_abs) return maximum_significant_digits;

    const significant_count = @as(i32, significand_position) + exponent + 1;
    return @intCast(@max(0, @min(significant_count, @as(i32, maximum_significant_digits))));
}

/// Return the `alpha` and `beta_star` that `eraser` needs for `value_abs`. `beta_star` is the
/// significant-digit count, except that 0 is reserved as the exact-negative-power-of-ten sentinel.
fn computeAlphaAndBetaStar(value_abs: f64) struct { alpha: i32, beta_star: u8 } {
    const significand_info = significandPosition(value_abs);
    const beta = getSignificantCount(value_abs, significand_info.position);
    const alpha: i32 = @as(i32, beta) - @as(i32, significand_info.position) - 1;
    const beta_star: u8 = if (significand_info.is_negative_power_of_ten) 0 else beta;
    return .{ .alpha = alpha, .beta_star = beta_star };
}

/// Return `value` rounded away from zero to `alpha` decimal places.
fn roundUp(value: f64, alpha: i32) f64 {
    const scale = getPositivePowerOfTen(alpha);
    if (value < 0) return @floor(value * scale) / scale;
    return @ceil(value * scale) / scale;
}

/// Write the prefix marker for `value` to `bit_writer` ("0" for no erase, "1" plus beta_star for
/// erase) and return the value_prime bits for `xorCompress`. Implements the paper's Eraser.
fn eraser(
    bit_writer: *shared_structs.BulkBitWriter,
    value: f64,
) Error!u64 {
    const value_bits: u64 = @bitCast(value);

    // Zero, infinities and NaN have no decimal form to erase; pass their bits through.
    if (value == 0.0 or math.isInf(value) or math.isNan(value)) {
        try bit_writer.writeBits(@as(u1, 0), 1);
        return value_bits;
    }

    const alpha_beta_star = computeAlphaAndBetaStar(@abs(value));

    // No-erase when alpha falls outside f_alpha_table's range: alpha < 0 means the magnitude is too
    // large to hold erasable digits (beta capped at 17), and alpha >= 21 is beyond the table (tiny or
    // subnormal values). Ordinary integers pass here and no-erase below via the delta == 0 check.
    if (alpha_beta_star.alpha < 0 or alpha_beta_star.alpha >= f_alpha_table.len) {
        try bit_writer.writeBits(@as(u1, 0), 1);
        return value_bits;
    }

    // Mantissa bits below g(alpha) are binary noise: not needed to represent the value exactly.
    const exponent: i32 = @intCast((value_bits >> mantissa_bits) & exponent_mask);
    const g_alpha: i32 = getFAlpha(alpha_beta_star.alpha) + exponent - exponent_bias;
    const erase_bits: i32 = @as(i32, mantissa_bits) - g_alpha;

    // At most 4 bits saved does not pay for the 5-bit erase prefix, and 64 or more would
    // overflow the shift below.
    if (erase_bits <= 4 or erase_bits >= 64) {
        try bit_writer.writeBits(@as(u1, 0), 1);
        return value_bits;
    }

    // If none of those low bits are set, erasing changes nothing, so skip the 4-bit prefix.
    const shift: u6 = @intCast(erase_bits);
    const mask: u64 = @as(u64, 0xffffffffffffffff) << shift;
    const delta: u64 = (~mask) & value_bits;
    if (delta == 0) {
        try bit_writer.writeBits(@as(u1, 0), 1);
        return value_bits;
    }

    // The `erase_bits > 4` test above bounds beta at 15, so beta_star fits the 4-bit field.
    // The narrowing cast traps instead of dropping high bits if that bound ever stops holding.
    try bit_writer.writeBits(@as(u1, 1), 1);
    try bit_writer.writeBits(@as(u4, @intCast(alpha_beta_star.beta_star)), 4);
    return mask & value_bits;
}

/// Return the original value for an erased `value_prime` and its `beta_star`, inverting `eraser`.
/// Returns `Error.CorruptedCompressedData` when the pair cannot come from a valid stream.
fn restorer(value_prime: f64, beta_star: u8) Error!f64 {
    // A corrupted stream can reconstruct 0, an infinity or NaN here, which would trap the
    // `@intFromFloat` in `significandPosition`.
    if (!math.isFinite(value_prime) or value_prime == 0.0) return Error.CorruptedCompressedData;

    const significand_position = significandPosition(@abs(value_prime)).position;
    if (beta_star == 0) {
        // The 10^-negative_exponent corner case: the significand position of value_prime is that of
        // value minus 1 (Theorem 3), so negative_exponent = -significand_position - 1. A corrupted
        // stream can pair beta_star = 0 with |value_prime| >= 1 (significand_position >= 0), making it
        // negative; `getNegativePowerOfTen` would trap on the negative @intCast. Reject instead.
        const negative_exponent: i32 = -@as(i32, significand_position) - 1;
        if (negative_exponent < 0) return Error.CorruptedCompressedData;
        const restored_value = getNegativePowerOfTen(negative_exponent);
        return if (value_prime < 0) -restored_value else restored_value;
    }
    // For valid streams alpha equals the encoder's alpha, which is in [0, 20]. A corrupted
    // stream can drive alpha negative, which would trap `roundUp`/`getPositivePowerOfTen`'s negative @intCast.
    const alpha: i32 = @as(i32, beta_star) - @as(i32, significand_position) - 1;
    if (alpha < 0) return Error.CorruptedCompressedData;
    return roundUp(value_prime, alpha);
}

/// Write `value_prime_bits` to `bit_writer` as one of four XOR cases, updating `state` in place.
/// Implements the paper's XOR_cmp.
fn xorCompress(
    bit_writer: *shared_structs.BulkBitWriter,
    value_prime_bits: u64,
    state: *XorState,
) Error!void {
    const xor_value = state.stored_value_prime ^ value_prime_bits;

    // Case 01: identical value.
    if (xor_value == 0) {
        try bit_writer.writeBits(@as(u2, 0b01), 2);
        return;
    }

    const exact_leading_zeros: u6 = @intCast(@clz(xor_value));
    const exact_trailing_zeros: u6 = @intCast(@ctz(xor_value));
    const leading_bucket_index = shared_functions.leadingZeroBucketIndex(exact_leading_zeros);
    const new_leading_zeros = shared_structs.leading_zero_buckets[leading_bucket_index];

    // Case 00: reuse the stored window, possible when the meaningful bits fit inside it.
    if (state.stored_leading_zeros) |bucket_leading_zeros| {
        if (state.stored_trailing_zeros) |bucket_trailing_zeros| {
            if (new_leading_zeros == bucket_leading_zeros and exact_trailing_zeros >= bucket_trailing_zeros) {
                const center_bits: u16 = shared_structs.bits_per_value - @as(u16, bucket_leading_zeros) - @as(u16, bucket_trailing_zeros);
                const meaningful: u64 = xor_value >> bucket_trailing_zeros;
                try bit_writer.writeBits(@as(u2, 0b00), 2);
                try bit_writer.writeBits(meaningful, center_bits);
                state.stored_value_prime = value_prime_bits;
                return;
            }
        }
    }

    // Cases 10/11: write a new bucket. The top meaningful bit is always 1, so only
    // `center_bits - 1` bits are stored and the decoder prepends it.
    const center_bits: u16 = shared_structs.bits_per_value - @as(u16, new_leading_zeros) - @as(u16, exact_trailing_zeros);
    const meaningful_bit_count: u16 = center_bits - 1;
    // Two-step shift avoids `xor_value >> 64` UB when exact_trailing_zeros = 63 (center_bits = 1).
    const meaningful: u64 = (xor_value >> exact_trailing_zeros) >> 1;

    if (center_bits <= 16) {
        // Case 10 (2 + 3 + 4 + (center-1) bits): center count fits in 4 bits.
        // The mask & 0xf wraps `center_bits = 16` to 0; the decoder remaps 0 -> 16.
        try bit_writer.writeBits(@as(u2, 0b10), 2);
        try bit_writer.writeBits(leading_bucket_index, shared_structs.leading_zero_bucket_bits);
        try bit_writer.writeBits(@as(u4, @intCast(center_bits & 0xf)), 4);
        try bit_writer.writeBits(meaningful, meaningful_bit_count);
    } else {
        // Case 11 (2 + 3 + 6 + (center-1) bits): center count fits in 6 bits.
        // The mask & 0x3f wraps `center_bits = 64` to 0; the decoder remaps 0 -> 64.
        try bit_writer.writeBits(@as(u2, 0b11), 2);
        try bit_writer.writeBits(leading_bucket_index, shared_structs.leading_zero_bucket_bits);
        try bit_writer.writeBits(@as(u6, @intCast(center_bits & 0x3f)), 6);
        try bit_writer.writeBits(meaningful, meaningful_bit_count);
    }

    state.stored_leading_zeros = new_leading_zeros;
    state.stored_trailing_zeros = exact_trailing_zeros;
    state.stored_value_prime = value_prime_bits;
}

/// Write the end-of-stream marker to `bit_writer`: a no-erase marker bit followed by a case-11
/// header carrying the impossible `(end_marker_lead_index, end_marker_center_raw)` pair.
fn writeEndMarker(bit_writer: *shared_structs.BulkBitWriter) Error!void {
    try bit_writer.writeBits(@as(u1, 0), 1);
    try bit_writer.writeBits(@as(u2, 0b11), 2);
    try bit_writer.writeBits(end_marker_lead_index, shared_structs.leading_zero_bucket_bits);
    try bit_writer.writeBits(end_marker_center_raw, 6);
}

/// Read one XOR case from `bit_reader` and return the reconstructed value_prime bits, updating
/// `state` in place. Returns null at the end-of-stream marker. Mirror of `xorCompress`.
fn xorDecompress(
    bit_reader: *shared_structs.BulkBitReader,
    state: *XorState,
) Error!?u64 {
    const flag = bit_reader.readBitsNoEof(u2, 2) catch return Error.CorruptedCompressedData;

    switch (flag) {
        // Case 01: repeated value. value_prime = stored_value_prime, no state change.
        0b01 => return state.stored_value_prime,

        // Case 00: bucket reuse. Read center_bits of XOR using the stored window.
        0b00 => {
            const bucket_leading_zeros = state.stored_leading_zeros orelse return Error.CorruptedCompressedData;
            const bucket_trailing_zeros = state.stored_trailing_zeros orelse return Error.CorruptedCompressedData;
            const center_bits: u16 = shared_structs.bits_per_value - @as(u16, bucket_leading_zeros) - @as(u16, bucket_trailing_zeros);
            const meaningful = bit_reader.readBitsNoEof(u64, center_bits) catch return Error.CorruptedCompressedData;
            // The encoder routes xor == 0 through case 01, never case 00, so meaningful is
            // always non-zero here. Reject a corrupted stream that encodes a zero-XOR reuse
            // instead of silently repeating the previous value.
            if (meaningful == 0) return Error.CorruptedCompressedData;
            const xor_value = meaningful << bucket_trailing_zeros;
            const value_prime_bits = state.stored_value_prime ^ xor_value;
            state.stored_value_prime = value_prime_bits;
            return value_prime_bits;
        },

        // Case 10: new bucket, center_bits <= 16.
        0b10 => {
            const leading_bucket_index = bit_reader.readBitsNoEof(u3, shared_structs.leading_zero_bucket_bits) catch return Error.CorruptedCompressedData;
            const center_raw = bit_reader.readBitsNoEof(u4, 4) catch return Error.CorruptedCompressedData;
            const new_leading_zeros = shared_structs.leading_zero_buckets[leading_bucket_index];
            // Encoder wrote `center_bits & 0xf`; the 0 sentinel decodes back to 16.
            const center_bits: u16 = if (center_raw == 0) 16 else @as(u16, center_raw);
            // No geometry check is needed, unlike case 11: any bucket is at most 24 and
            // `center_bits` at most 16 here, so the subtraction below cannot underflow.
            const new_trailing_zeros: u6 = @intCast(shared_structs.bits_per_value - @as(u16, new_leading_zeros) - center_bits);
            // Read center-1 meaningful bits; prepend the implicit top 1 and shift into place.
            const meaningful = bit_reader.readBitsNoEof(u64, center_bits - 1) catch return Error.CorruptedCompressedData;
            const xor_value = ((meaningful << 1) | 1) << new_trailing_zeros;
            const value_prime_bits = state.stored_value_prime ^ xor_value;
            state.stored_value_prime = value_prime_bits;
            state.stored_leading_zeros = new_leading_zeros;
            state.stored_trailing_zeros = new_trailing_zeros;
            return value_prime_bits;
        },

        // Case 11: new bucket, center_bits > 16.
        0b11 => {
            const leading_bucket_index = bit_reader.readBitsNoEof(u3, shared_structs.leading_zero_bucket_bits) catch return Error.CorruptedCompressedData;
            const center_raw = bit_reader.readBitsNoEof(u6, 6) catch return Error.CorruptedCompressedData;
            // The end-of-stream marker; see `end_marker_lead_index`.
            if (leading_bucket_index == end_marker_lead_index and center_raw == end_marker_center_raw) return null;
            const new_leading_zeros = shared_structs.leading_zero_buckets[leading_bucket_index];
            // Encoder wrote `center_bits & 0x3f`; the 0 sentinel decodes back to 64.
            const center_bits: u16 = if (center_raw == 0) 64 else @as(u16, center_raw);
            // Reject a bucket and center that do not fit in 64 bits before the cast below.
            if (@as(u16, new_leading_zeros) + center_bits > shared_structs.bits_per_value) return Error.CorruptedCompressedData;
            const new_trailing_zeros: u6 = @intCast(shared_structs.bits_per_value - @as(u16, new_leading_zeros) - center_bits);
            const meaningful = bit_reader.readBitsNoEof(u64, center_bits - 1) catch return Error.CorruptedCompressedData;
            const xor_value = ((meaningful << 1) | 1) << new_trailing_zeros;
            const value_prime_bits = state.stored_value_prime ^ xor_value;
            state.stored_value_prime = value_prime_bits;
            state.stored_leading_zeros = new_leading_zeros;
            state.stored_trailing_zeros = new_trailing_zeros;
            return value_prime_bits;
        },
    }
}

test "elf roundtrips generated values across all distributions" {
    const allocator = testing.allocator;

    // Elf is bitwise lossless, so it must recover any f64 input - including unbounded
    // random values, NaN payloads, and infinities. Test every distribution the tester offers.
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
        Method.Elf,
        data_distributions,
    );
}

test "elf roundtrips two values" {
    // Two values exercise exactly one `eraser`+`xorCompress` marker right after the first raw value.
    const uncompressed_values = &[_]f64{ 3.5, 9.0 };

    try tester.expectLosslessRoundTrip(testing.allocator, compress, decompress, uncompressed_values);
}

test "elf roundtrips repeated values" {
    // Repeated values exercise `xorCompress` case 01 (xor = 0) after the first raw value.
    const uncompressed_values = &[_]f64{ 7.25, 7.25, 7.25, 7.25, 7.25 };

    try tester.expectLosslessRoundTrip(testing.allocator, compress, decompress, uncompressed_values);
}

test "elf roundtrips changing values" {
    // Changing values cover bucket transitions, bucket reuse, and meaningful-bit paths.
    const uncompressed_values = &[_]f64{ 100.0, 100.01, 100.02, 99.99, -3.5, 0.0, 2048.125 };

    try tester.expectLosslessRoundTrip(testing.allocator, compress, decompress, uncompressed_values);
}

test "elf roundtrips special floating-point values" {
    // Special values route through the no-erase path (marker 0) and preserve raw bits.
    // We keep NaN payload bits intact (no canonicalization).
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

test "elf roundtrips edge floats" {
    // +0.0 and -0.0 compare numerically equal but differ in the sign bit, so only a
    // bitwise codec preserves them. Subnormals use a distinct exponent encoding, and
    // `nextAfter` pairs produce the smallest possible XOR - exercising the maximum
    // leading-zeros bucket path.
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

test "elf roundtrips decimal-originated values" {
    // Sensor-style values with limited decimal precision exercise the erase path
    // (beta_star in [1, 4]) plus the pow10 corner case. Every value writes a fresh beta_star -
    // the paper version doesn't reuse beta_star across consecutive same-precision values.
    const uncompressed_values = &[_]f64{ 0.1, 3.17, 2.5, 100.01, 0.001, -42.42, 1e-5, 0.5 };

    try tester.expectLosslessRoundTrip(testing.allocator, compress, decompress, uncompressed_values);
}

test "elf roundtrips pow10 boundary values" {
    // Negative powers of 10 trigger the corner case where the significand position shifts during
    // erasure. Encoder writes the beta_star = 0 sentinel; decoder uses the `getNegativePowerOfTen` restore formula.
    const uncompressed_values = &[_]f64{ 0.1, 0.01, 0.001, 0.0001, 0.00001 };

    try tester.expectLosslessRoundTrip(testing.allocator, compress, decompress, uncompressed_values);
}

test "elf roundtrips integer values" {
    // Integer-valued floats have alpha = beta - significand_position - 1 <= 0, routed to no-erase.
    // Verifies the alpha < 0 guard in `eraser` doesn't break integer round-trips.
    const uncompressed_values = &[_]f64{ 0.0, 1.0, 10.0, 100.0, 1000.0, 1e10 };

    try tester.expectLosslessRoundTrip(testing.allocator, compress, decompress, uncompressed_values);
}

test "elf compresses repeated values below raw size" {
    // A constant signal maximally exercises `xorCompress` case 01 (xor = 0):
    // every repeat is at most 1 + 2 = 3 bits. Output must be far smaller than the raw f64 array.
    const allocator = testing.allocator;

    var uncompressed_values: [500]f64 = undefined;
    @memset(&uncompressed_values, 42.0);

    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);

    try compress(allocator, &uncompressed_values, &compressed_values, "{}");

    try testing.expect(compressed_values.items.len < uncompressed_values.len * @sizeOf(f64));
}

test "elf compresses decimal data below raw size" {
    // Sensor-style values with consistent 2-decimal precision exercise the eraser
    // sweet spot - even without beta_star reuse, the erased mantissa noise shrinks XOR
    // outputs enough to fit under the raw byte size.
    const allocator = testing.allocator;

    var uncompressed_values: [500]f64 = undefined;
    var prng = std.Random.DefaultPrng.init(42);
    const rand = prng.random();
    for (&uncompressed_values) |*value| {
        // Generate values like 53.27, 91.04, ... - bounded with exactly 2 decimal places.
        value.* = @floor(rand.float(f64) * 10000.0) / 100.0;
    }

    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);

    try compress(allocator, &uncompressed_values, &compressed_values, "{}");

    try testing.expect(compressed_values.items.len < uncompressed_values.len * @sizeOf(f64));
}

test "elf rejects corrupted compressed data" {
    const allocator = testing.allocator;
    // A mix of erased and no-erase values so the stream covers all four `xorCompress` cases.
    const uncompressed_values = &[_]f64{ 100.01, 100.02, 0.001, 42.0, -3.5, 100.02, 1e10 };

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

    // The first value alone: the bit stream cannot hold the end-of-stream marker.
    decompressed_values.clearRetainingCapacity();
    try testing.expectError(
        Error.CorruptedCompressedData,
        decompress(allocator, compressed_values.items[0..8], &decompressed_values),
    );

    // Every truncation of the bit stream must report an error rather than trap.
    for (9..compressed_values.items.len) |length| {
        decompressed_values.clearRetainingCapacity();
        try testing.expectError(
            Error.CorruptedCompressedData,
            decompress(allocator, compressed_values.items[0..length], &decompressed_values),
        );
    }

    // Flipping any single bit must error or decode, never trap.
    for (8..compressed_values.items.len) |index| {
        for (0..8) |bit| {
            var corrupted = try allocator.dupe(u8, compressed_values.items);
            defer allocator.free(corrupted);
            corrupted[index] ^= @as(u8, 1) << @intCast(bit);

            decompressed_values.clearRetainingCapacity();
            _ = decompress(allocator, corrupted, &decompressed_values) catch continue;
        }
    }
}

test "check elf configuration parsing" {
    // Elf takes no parameters: an empty configuration must parse, and a configuration
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
