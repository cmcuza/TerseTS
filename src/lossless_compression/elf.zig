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
//!
//! Elf erases the noise bits a decimal value (e.g., 23.45) carries in its binary
//! mantissa, producing a near-equal v' with many trailing zeros. v' is then XOR-
//! encoded against the previous v' (4 cases by leading/trailing-zero buckets);
//! the decoder recovers v from v' by rounding up to beta_star decimal digits.
//!
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

/// Number of bits in an IEEE-754 `f64`; the width of every value Elf XOR-encodes.
const bits_per_value = 64;
/// Number of randomized rounds the generated-distribution round-trip test runs.
const generated_test_rounds = 5;

/// log2(10) - bridges decimal-digit counts and binary-bit counts. Evaluated at compile
/// time so we don't recompute log2 at runtime; matches Java's `Math.log(10)/Math.log(2)`.
const log_2_10: f64 = @log2(@as(f64, 10.0));

/// Compress `uncompressed_values` into `compressed_values` using Elf's two-layer codec.
/// `allocator` backs the configuration parser and the bit writer's scratch buffer.
/// `method_configuration` must be an empty configuration; any field makes the call return
/// `Error.InvalidConfiguration`. On success `compressed_values` holds
/// `[count: u64][first_value: f64][bit stream...]`, where each value is an Eraser marker
/// (1 or 5 bits) followed by the XOR_cmp encoding of v'. If an error occurs it is returned.
/// The two-layer encoding logic is described inline in the function body.
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

    // 8-byte count header lets the decoder know exactly when to stop, ignoring
    // any padding bits the BitWriter flushes after the last value.
    try shared_functions.appendValue(allocator, u64, @intCast(uncompressed_values.len), compressed_values);
    if (uncompressed_values.len == 0) return;

    // First value goes raw. Subsequent values XOR against the previous v', so we use
    // this raw first value as the baseline (no Layer 1 / Layer 2 needed for it).
    const first_value = uncompressed_values[0];
    try shared_functions.appendValue(allocator, f64, first_value, compressed_values);

    var xor_state = XorState{
        .stored_val = @bitCast(first_value),
        .stored_leading_zeros = null,
        .stored_trailing_zeros = null,
    };

    var bit_writer = try shared_structs.BulkBitWriter.init(allocator, compressed_values);

    for (uncompressed_values[1..]) |value| {
        const v_prime_bits = try eraseValue(&bit_writer, value);
        try xorCompressOther(&bit_writer, v_prime_bits, &xor_state);
    }

    try bit_writer.flushBits();
}

/// Decompress an Elf-encoded `compressed_values` stream into `decompressed_values`.
/// `allocator` grows `decompressed_values` as values are recovered. `compressed_values` must
/// start with the `[count: u64][first_value: f64]` header written by `compress`; malformed or
/// truncated streams return `Error.ByteStreamError` or `Error.UnsupportedInput` rather than
/// trapping. If an error occurs it is returned. The per-value decoding logic (Eraser marker
/// dispatch, beta_star read, XOR_dcmp, and recovery) is described inline in the function body.
pub fn decompress(
    allocator: Allocator,
    compressed_values: []const u8,
    decompressed_values: *ArrayList(f64),
) Error!void {
    var offset: usize = 0;
    const value_count = try shared_functions.readOffsetValue(u64, compressed_values, &offset);
    if (value_count == 0) return;

    // Every non-empty Elf stream needs the count (8 bytes) and the first raw value (8 bytes).
    if (compressed_values.len < 16) return Error.UnsupportedInput;

    // The header gives the exact output length, so reserve it once and append without growth checks.
    try decompressed_values.ensureTotalCapacity(allocator, @intCast(value_count));

    const first_value = try shared_functions.readOffsetValue(f64, compressed_values, &offset);
    decompressed_values.appendAssumeCapacity(first_value);

    var xor_state = XorState{
        .stored_val = @bitCast(first_value),
        .stored_leading_zeros = null,
        .stored_trailing_zeros = null,
    };

    // Read the bit stream straight from the remaining bytes with a buffered, byte-slice reader.
    var bit_reader = shared_structs.BulkBitReader.init(compressed_values[offset..]);

    while (decompressed_values.items.len < value_count) {
        // Layer 1 marker dispatch:
        //   "0"     (1 bit)  -> no erase
        //   "1"+beta_star  (5 bits) -> erase, read beta_star (4 bits)
        const marker_bit = bit_reader.readBitsNoEof(u1, 1) catch return Error.ByteStreamError;

        if (marker_bit == 0) {
            // No erase: v' is the raw v, so no Layer 1 recovery needed.
            const v_bits = try xorDecompressOther(&bit_reader, &xor_state);
            decompressed_values.appendAssumeCapacity(@bitCast(v_bits));
            continue;
        }

        // Erase: read beta_star (4 bits), recover from v'.
        const beta_star = bit_reader.readBitsNoEof(u8, 4) catch return Error.ByteStreamError;
        const v_prime_bits = try xorDecompressOther(&bit_reader, &xor_state);
        const v = try recoverByBetaStar(@bitCast(v_prime_bits), beta_star);
        decompressed_values.appendAssumeCapacity(v);
    }
}

/// f_alpha_table[alpha] = ceil(alpha * log2(10)), precomputed for alpha in [0, 20]. Tells us how
/// many binary bits are required to represent 10^alpha exactly. The Eraser uses this to find
/// where to slice the mantissa: bits below position `f(alpha) + e - 1023` are noise that
/// can be erased while still recovering v from the alpha (decimal-place-count) hint.
const f_alpha_table = [_]u8{
    0,  4,  7,  10, 14, 17, 20, 24, 27, 30,
    34, 37, 40, 44, 47, 50, 54, 57, 60, 64,
    67,
};

/// power_of_10_table[i] = 10^i for i in [0, 20]. Used to multiply a value by a power
/// of 10 when computing beta (we walk i upward until v * 10^i becomes an exact integer).
const power_of_10_table = [_]f64{
    1.0,    1.0e1,  1.0e2,  1.0e3,  1.0e4,  1.0e5,  1.0e6,
    1.0e7,  1.0e8,  1.0e9,  1.0e10, 1.0e11, 1.0e12, 1.0e13,
    1.0e14, 1.0e15, 1.0e16, 1.0e17, 1.0e18, 1.0e19, 1.0e20,
};

/// negative_power_of_10_table[i] = 10^-i for i in [0, 20]. Used in the Restorer for
/// the special case v = 10^-i (beta_star = 0 sentinel), where recovery is simply v = 10^-(u-1).
const negative_power_of_10_table = [_]f64{
    1.0,     1.0e-1,  1.0e-2,  1.0e-3,  1.0e-4,  1.0e-5,  1.0e-6,
    1.0e-7,  1.0e-8,  1.0e-9,  1.0e-10, 1.0e-11, 1.0e-12, 1.0e-13,
    1.0e-14, 1.0e-15, 1.0e-16, 1.0e-17, 1.0e-18, 1.0e-19, 1.0e-20,
};

/// sp_table_ge_one[i] = 10^i as f64. Used by getSP to bracket values v >= 1 without
/// calling log10: the lookup finds i such that 10^i <= v < 10^(i+1), so SP(v) = i.
const sp_table_ge_one = [_]f64{
    1.0,         10.0,         100.0,         1000.0,          10_000.0, 100_000.0,
    1_000_000.0, 10_000_000.0, 100_000_000.0, 1_000_000_000.0,
};

/// sp_table_lt_one[i] = 10^-i as f64. Used by getSP to bracket values 0 < v < 1:
/// the lookup finds i such that 10^-i <= v < 10^-(i-1), so SP(v) = -i.
const sp_table_lt_one = [_]f64{
    1.0,    1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4,  1.0e-5,
    1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9, 1.0e-10,
};

/// Chimp-style leading-zero buckets, also used by Elf's XOR_cmp. There are only 8
/// possible bucket values, so we encode the chosen one with a 3-bit index.
const leading_zero_bucket_values = [_]u6{ 0, 8, 12, 16, 18, 20, 22, 24 };

/// leading_zero_bucket_index[exact_lz] = bucket index (0..7) such that
/// leading_zero_bucket_values[index] is the largest bucket <= exact_lz.
/// Direct 64-entry lookup replaces Chimp64's linear-search `leadingZeroBucketIndex` function.
const leading_zero_bucket_index = [_]u3{
    0, 0, 0, 0, 0, 0, 0, 0, // 0..7  -> bucket 0  (value 0)
    1, 1, 1, 1, // 8..11   -> bucket 1  (value 8)
    2, 2, 2, 2, // 12..15  -> bucket 2  (value 12)
    3, 3, // 16..17  -> bucket 3  (value 16)
    4, 4, // 18..19  -> bucket 4  (value 18)
    5, 5, // 20..21  -> bucket 5  (value 20)
    6, 6, // 22..23  -> bucket 6  (value 22)
    7, 7, 7, 7, 7, 7, 7, 7, // 24..31  -> bucket 7  (value 24)
    7, 7, 7, 7, 7, 7, 7, 7, // 32..39  -> bucket 7
    7, 7, 7, 7, 7, 7, 7, 7, // 40..47  -> bucket 7
    7, 7, 7, 7, 7, 7, 7, 7, // 48..55  -> bucket 7
    7, 7, 7, 7, 7, 7, 7, 7, // 56..63  -> bucket 7
};

/// leading_zero_rounded[exact_lz] equals
/// leading_zero_bucket_values[leading_zero_bucket_index[exact_lz]] - the actual leading-zero count
/// rounded down to the nearest bucket boundary. Used to update the encoder's stored_leading_zeros
/// after a new-bucket case, and to compare against the previous value's stored_leading_zeros for
/// the bucket-reuse decision.
const leading_zero_rounded = [_]u6{
    0, 0, 0, 0, 0, 0, 0, 0, // 0..7   -> 0
    8, 8, 8, 8, // 8..11  -> 8
    12, 12, 12, 12, // 12..15 -> 12
    16, 16, // 16..17 -> 16
    18, 18, // 18..19 -> 18
    20, 20, // 20..21 -> 20
    22, 22, // 22..23 -> 22
    24, 24, 24, 24, 24, 24, 24, 24, // 24..31 -> 24
    24, 24, 24, 24, 24, 24, 24, 24, // 32..39 -> 24
    24, 24, 24, 24, 24, 24, 24, 24, // 40..47 -> 24
    24, 24, 24, 24, 24, 24, 24, 24, // 48..55 -> 24
    24, 24, 24, 24, 24, 24, 24, 24, // 56..63 -> 24
};

/// Returns the start decimal significand position of |v|: the index of v's most
/// significant non-zero decimal digit measured as a power of 10. Examples:
///   getSP(3.17)   = 0   (the '3' is in the 10^0 place)
///   getSP(314.0)  = 2   (the '3' is in the 10^2 place)
///   getSP(0.0314) = -2  (the '3' is in the 10^-2 place)
/// Caller must ensure v_abs is finite and > 0.
fn getSP(v_abs: f64) i16 {
    return getSPAnd10iNFlag(v_abs).sp;
}

/// Same as getSP, plus a flag indicating whether v is exactly 10^-i for some i > 0
/// (the corner case where erasing breaks SP-preservation: see paper Theorem 3).
/// The flag becomes beta_star = 0 in the bitstream, signalling the decoder to recover via
/// v = 10^-(u-1) instead of the usual roundUp formula.
fn getSPAnd10iNFlag(v_abs: f64) struct { sp: i16, is_pow10_neg: bool } {
    if (v_abs >= 1.0) {
        // Find i such that 10^i <= v_abs < 10^(i+1), so SP = i >= 0.
        for (0..sp_table_ge_one.len - 1) |i| {
            if (v_abs < sp_table_ge_one[i + 1]) {
                return .{ .sp = @intCast(i), .is_pow10_neg = false };
            }
        }
    } else {
        // Find i such that 10^-i <= v_abs < 10^-(i-1), so SP = -i.
        // is_pow10_neg fires when v_abs lands exactly on the lower boundary.
        for (1..sp_table_lt_one.len) |i| {
            if (v_abs >= sp_table_lt_one[i]) {
                return .{
                    .sp = -@as(i16, @intCast(i)),
                    .is_pow10_neg = (v_abs == sp_table_lt_one[i]),
                };
            }
        }
    }
    // Fallback for values outside the tables (|v| >= 10^9 or |v| < 10^-10).
    const log10v = @log10(v_abs);
    return .{
        .sp = @intFromFloat(@floor(log10v)),
        .is_pow10_neg = (log10v == @floor(log10v)),
    };
}

/// Returns ceil(alpha * log2(10)) - the number of binary mantissa bits needed to represent
/// 10^alpha exactly. Used in g(alpha) = getFAlpha(alpha) + exponent - 1023, which marks the
/// mantissa position below which bits are noise that can be erased.
/// Contract: alpha >= 0. The Java reference throws IllegalArgumentException on negative
/// input; we mirror that as a debug-time assertion (programmer-bug check).
fn getFAlpha(alpha: i32) i32 {
    std.debug.assert(alpha >= 0);
    if (alpha >= f_alpha_table.len) {
        // Rare: alpha > 20 happens for very small values (|v| < 1e-10). When this hits,
        // the eraser's downstream `eraseBits > 4` check usually routes to no-erase.
        return @intFromFloat(@ceil(@as(f64, @floatFromInt(alpha)) * log_2_10));
    }
    return @as(i32, f_alpha_table[@intCast(alpha)]);
}

/// Returns 10^i. Used by the beta-computation loop to walk v * 10^i until it becomes
/// an exact integer, and by roundUp during decompression.
/// Contract: i >= 0.
fn get10iP(i: i32) f64 {
    std.debug.assert(i >= 0);
    if (i >= power_of_10_table.len) {
        return std.math.pow(f64, 10.0, @floatFromInt(i));
    }
    return power_of_10_table[@intCast(i)];
}

/// Returns 10^-i. Used by the Restorer when beta_star = 0 (the v = 10^-i corner case):
/// recovery reduces to v = 10^-(sp+1), looked up directly here.
/// Contract: i >= 0.
fn get10iN(i: i32) f64 {
    std.debug.assert(i >= 0);
    if (i >= negative_power_of_10_table.len) {
        return std.math.pow(f64, 10.0, -@as(f64, @floatFromInt(i)));
    }
    return negative_power_of_10_table[@intCast(i)];
}

/// Returns beta (decimal significand count) of |v|: how many significant decimal digits
/// the value has. Examples:
///   getSignificantCount(3.17,   sp=0)  = 3   ("3", "1", "7")
///   getSignificantCount(0.0314, sp=-2) = 3   ("3", "1", "4")
///   getSignificantCount(5.20,   sp=0)  = 2   ("5", "2")
///
/// Algorithm (matches the reference): start at the SMALLEST candidate scale and walk i
/// upward to the first power of ten that makes v * 10^i an exact integer; beta = sp + i + 1.
/// Starting low lands directly on the minimal i, so no trailing-zero stripping is needed.
/// Returns 17 (the f64 decimal-precision ceiling) for values with no clean short decimal form
/// - the caller treats that as "give up, no erase". (A previous version started at the maximal
/// i = 17 - sp - 1 and stripped zeros; the large multiply lost precision and gave up far too
/// often, leaving values un-erased and inflating compressed size.)
fn getSignificantCount(v_abs: f64, sp: i16) u8 {
    // Smallest scale worth trying: for v >= 1 start at i = 1; for v < 1 start at i = -sp so the
    // leading non-zero digit reaches the ones place. Both keep get10iP's argument >= 1.
    var i: i32 = if (sp >= 0) 1 else -@as(i32, sp);

    // Walk i upward until v * 10^i is an exact integer (or we exhaust f64 precision).
    // `0x1p63` is 2^63 as an exact f64, the strict upper bound for a safe @intFromFloat -> i64.
    var temp: f64 = v_abs * get10iP(i);
    if (temp >= 0x1p63) return 17;
    var temp_long: i64 = @intFromFloat(temp);

    var iterations: u8 = 0;
    while (@as(f64, @floatFromInt(temp_long)) != temp) : (iterations += 1) {
        // Safety cap: f64 carries ~15.95 decimal digits, so after ~22 steps we are chasing noise.
        if (iterations >= 22) return 17;
        i += 1;
        temp = v_abs * get10iP(i);
        if (temp >= 0x1p63) return 17;
        temp_long = @intFromFloat(temp);
    }

    // Verify the scaling is exactly reversible. If v * 10^i looked integral only because of
    // rounding in the multiply, the round-trip won't recover v - then v has no clean short form.
    if (temp / get10iP(i) != v_abs) return 17;

    const beta = @as(i32, sp) + i + 1;
    return @intCast(@max(0, @min(beta, 17)));
}

/// Computes (alpha, beta_star) for the Eraser:
///   alpha = beta - SP - 1: the decimal-place count (digits past the decimal point)
///   beta_star = beta when v != 10^-i, else 0 (sentinel for the corner case)
/// The encoder writes beta_star (4 bits) into the bitstream; the decoder regenerates alpha
/// from beta_star and a fresh SP(v') computation.
/// Caller must ensure v_abs is finite and > 0.
fn computeAlphaAndBetaStar(v_abs: f64) struct { alpha: i32, beta_star: u8 } {
    const sp_and_flag = getSPAnd10iNFlag(v_abs);
    const beta = getSignificantCount(v_abs, sp_and_flag.sp);
    const alpha: i32 = @as(i32, beta) - @as(i32, sp_and_flag.sp) - 1;
    const beta_star: u8 = if (sp_and_flag.is_pow10_neg) 0 else beta;
    return .{ .alpha = alpha, .beta_star = beta_star };
}

/// Inverse of the Eraser: round `v` to exactly `alpha` decimal places, rounding
/// AWAY FROM ZERO. Recovers the original value from the erased one because the
/// Eraser produced v' = v - delta with 0 < delta < 10^-alpha, so rounding v' away from zero
/// to alpha places lands exactly back on v.
/// Caller must ensure alpha >= 0 (get10iP's contract).
fn roundUp(v: f64, alpha: i32) f64 {
    const scale = get10iP(alpha);
    if (v < 0) return @floor(v * scale) / scale;
    return @ceil(v * scale) / scale;
}

/// Layer 1 (Eraser) encoder. Writes the 1- or 5-bit prefix marker for one value
/// to `bit_writer`, decides whether to erase mantissa noise, and returns v_prime_bits
/// for the caller to feed into Layer 2 (XOR_cmp).
///
/// Paper marker layout (2 cases, stateless per-value):
///   "0"      (1 bit)   - no erase (special value OR unprofitable)
///   "1"<beta_star>  (5 bits)  - erase, beta_star (4 bits) follows the 1-bit marker
///
/// No state is carried across values: each erase decision writes a fresh beta_star.
///
/// NaN handling differs from the Java reference: we preserve NaN payload bits (route
/// through no-erase) rather than canonicalizing to 0x7ff8000000000000, so the codec
/// is bit-exact lossless on every input - matching chimp64/chimp128/gorilla/elf_plus.
fn eraseValue(
    bit_writer: *shared_structs.BulkBitWriter,
    v: f64,
) Error!u64 {
    const v_bits: u64 = @bitCast(v);

    // Special values: 0, +/-inf, NaN. Skip the decimal-precision machinery and pass
    // raw bits through XOR_cmp.
    if (v == 0.0 or std.math.isInf(v) or std.math.isNan(v)) {
        try bit_writer.writeBits(@as(u1, 0), 1);
        return v_bits;
    }

    // Decimal-precision analysis. beta_star will be written to the stream; alpha is used to
    // compute how many mantissa bits to erase.
    const v_abs = @abs(v);
    const ab = computeAlphaAndBetaStar(v_abs);

    // Bail to no-erase if alpha is outside the useful range:
    //   alpha < 0  -> integer-valued float (e.g. 100.0) - nothing to erase
    //   alpha >= 21 -> beyond the f_alpha_table (very small subnormals) - rare, skip
    if (ab.alpha < 0 or ab.alpha >= f_alpha_table.len) {
        try bit_writer.writeBits(@as(u1, 0), 1);
        return v_bits;
    }

    // g(alpha) tells us how many mantissa bits are needed to represent v exactly given
    // its decimal precision; everything below g(alpha) is binary noise we can erase.
    const exponent: i32 = @intCast((v_bits >> 52) & 0x7ff);
    const g_alpha: i32 = getFAlpha(ab.alpha) + exponent - 1023;
    const erase_bits: i32 = 52 - g_alpha;

    // Profitability + safety guard:
    //   <= 4 bits saved -> the 5-bit erase prefix overhead wipes the gain
    //   >= 64 bits     -> shift count would be UB on u64
    if (erase_bits <= 4 or erase_bits >= 64) {
        try bit_writer.writeBits(@as(u1, 0), 1);
        return v_bits;
    }

    // Build the mask, then check that v actually has any of those low bits set.
    // If not, "erasing" wouldn't change v_bits - skip to no-erase to save 4 bits.
    const shift: u6 = @intCast(erase_bits);
    const mask: u64 = @as(u64, 0xffffffffffffffff) << shift;
    const delta: u64 = (~mask) & v_bits;
    if (delta == 0) {
        try bit_writer.writeBits(@as(u1, 0), 1);
        return v_bits;
    }

    // beta_star fits in 4 bits (<= 15) by construction: the eraser rejects beta >= 16 above via
    // the `eraseBits > 4` profitability check. Assert as a sanity check.
    std.debug.assert(ab.beta_star <= 15);
    try bit_writer.writeBits(@as(u1, 1), 1);
    try bit_writer.writeBits(ab.beta_star, 4);
    return mask & v_bits;
}

/// Layer 1 (Restorer) decoder. Given an erased value `v_prime` (output of XOR_dcmp)
/// and the beta_star from the bitstream, returns the original v.
///
/// Two recovery paths:
///   beta_star > 0: standard case. Compute alpha = beta_star - SP(|v'|) - 1, then
///                  roundUp(v', alpha).
///   beta_star = 0: corner case (v was 10^-i). SP shifted during erasure, so recover via
///           v = 10^-(sp+1) (with sign of v' preserved).
///
/// `v_prime` and `beta_star` come from an untrusted stream, so this validates them before
/// feeding the decimal-recovery machinery: a corrupted stream could otherwise drive getSP,
/// get10iN, or get10iP with inputs that trap their @intFromFloat/@intCast. On any such input
/// it returns `Error.UnsupportedInput`. Valid streams never trigger these guards: the encoder
/// only erases finite, non-zero values, and the paper's SP relations keep alpha and i >= 0.
fn recoverByBetaStar(v_prime: f64, beta_star: u8) Error!f64 {
    // The erase path is only valid for finite, non-zero values. A corrupted XOR layer can
    // reconstruct 0/+-inf/NaN here, which would trap getSP's @intFromFloat fallback below.
    if (!std.math.isFinite(v_prime) or v_prime == 0.0) return Error.UnsupportedInput;

    const sp = getSP(@abs(v_prime));
    if (beta_star == 0) {
        // The 10^-i corner case: SP(v') = SP(v) - 1 (Theorem 3), so i = -SP(v') - 1.
        // A corrupted stream can pair beta_star = 0 with |v'| >= 1 (sp >= 0), making i
        // negative; get10iN would trap on the negative @intCast. Reject instead.
        const i: i32 = -@as(i32, sp) - 1;
        if (i < 0) return Error.UnsupportedInput;
        const recovered = get10iN(i);
        return if (v_prime < 0) -recovered else recovered;
    }
    // For valid streams alpha equals the encoder's alpha, which is in [0, 20]. A corrupted
    // stream can drive alpha negative, which would trap roundUp/get10iP's negative @intCast.
    const alpha: i32 = @as(i32, beta_star) - @as(i32, sp) - 1;
    if (alpha < 0) return Error.UnsupportedInput;
    return roundUp(v_prime, alpha);
}

/// State carried by the XOR_cmp encoder and matching decoder across consecutive values.
/// stored_val holds the previous v' (raw u64 bits). stored_leading/trailing_zeros hold
/// the previously-encoded bucket parameters, used to decide whether to reuse them
/// (case 00) or write new ones (cases 10/11). Both bucket fields are `?u6` so that
/// the "no previous bucket" sentinel maps cleanly to null on the first encoded XOR.
const XorState = struct {
    stored_val: u64,
    stored_leading_zeros: ?u6,
    stored_trailing_zeros: ?u6,
};

/// Layer 2 (XOR_cmp) encoder. Writes one of four cases to the bitstream based on
/// XOR(prev_v', v') and the bucket-reuse opportunity. State is updated in-place.
fn xorCompressOther(
    bit_writer: *shared_structs.BulkBitWriter,
    v_prime_bits: u64,
    state: *XorState,
) Error!void {
    const xor = state.stored_val ^ v_prime_bits;

    // Case 01 (2 bits): identical value. Nothing else to write.
    if (xor == 0) {
        try bit_writer.writeBits(@as(u2, 0b01), 2);
        return;
    }

    const exact_lz: u6 = @intCast(@clz(xor));
    const exact_tz: u6 = @intCast(@ctz(xor));
    const new_lead = leading_zero_rounded[exact_lz];

    // Case 00 (2 + center bits): bucket reuse. Triggers when the new XOR's lead matches
    // the stored bucket AND has at least as many trailing zeros - meaning the meaningful
    // bits fit inside the previously-defined window.
    if (state.stored_leading_zeros) |stored_lz| {
        if (state.stored_trailing_zeros) |stored_tz| {
            if (new_lead == stored_lz and exact_tz >= stored_tz) {
                const center_bits: u16 = bits_per_value - @as(u16, stored_lz) - @as(u16, stored_tz);
                const meaningful: u64 = xor >> stored_tz;
                try bit_writer.writeBits(@as(u2, 0b00), 2);
                try bit_writer.writeBits(meaningful, center_bits);
                state.stored_val = v_prime_bits;
                return;
            }
        }
    }

    // Cases 10/11 (new bucket): write the bucket index plus the meaningful bits.
    // The top meaningful bit is always 1 (otherwise leading_zeros would be larger),
    // so encode (center_bits - 1) bits and let the decoder prepend the implicit 1.
    const lead_idx = leading_zero_bucket_index[exact_lz];
    const center_bits: u16 = bits_per_value - @as(u16, new_lead) - @as(u16, exact_tz);
    const meaningful_bit_count: u16 = center_bits - 1;
    // Two-step shift avoids `xor >> 64` UB when exact_tz = 63 (center_bits = 1).
    const meaningful: u64 = (xor >> exact_tz) >> 1;

    if (center_bits <= 16) {
        // Case 10 (2 + 3 + 4 + (center-1) bits): center count fits in 4 bits.
        // The mask & 0xf wraps `center_bits = 16` to 0; the decoder remaps 0 -> 16.
        try bit_writer.writeBits(@as(u2, 0b10), 2);
        try bit_writer.writeBits(lead_idx, 3);
        try bit_writer.writeBits(@as(u4, @intCast(center_bits & 0xf)), 4);
        try bit_writer.writeBits(meaningful, meaningful_bit_count);
    } else {
        // Case 11 (2 + 3 + 6 + (center-1) bits): center count fits in 6 bits.
        // The mask & 0x3f wraps `center_bits = 64` to 0; the decoder remaps 0 -> 64.
        try bit_writer.writeBits(@as(u2, 0b11), 2);
        try bit_writer.writeBits(lead_idx, 3);
        try bit_writer.writeBits(@as(u6, @intCast(center_bits & 0x3f)), 6);
        try bit_writer.writeBits(meaningful, meaningful_bit_count);
    }

    state.stored_leading_zeros = new_lead;
    state.stored_trailing_zeros = exact_tz;
    state.stored_val = v_prime_bits;
}

/// Layer 2 (XOR_cmp) decoder. Reads one of the four XOR_cmp cases from the bitstream
/// and returns the reconstructed v' bits. Mirror of xorCompressOther.
fn xorDecompressOther(
    bit_reader: *shared_structs.BulkBitReader,
    state: *XorState,
) Error!u64 {
    const flag = bit_reader.readBitsNoEof(u2, 2) catch return Error.ByteStreamError;

    switch (flag) {
        // Case 01: repeated value. v' = stored_val, no state change.
        0b01 => return state.stored_val,

        // Case 00: bucket reuse. Read center_bits of XOR using the stored window.
        0b00 => {
            const stored_lz = state.stored_leading_zeros orelse return Error.UnsupportedInput;
            const stored_tz = state.stored_trailing_zeros orelse return Error.UnsupportedInput;
            const center_bits: u16 = bits_per_value - @as(u16, stored_lz) - @as(u16, stored_tz);
            const meaningful = bit_reader.readBitsNoEof(u64, center_bits) catch return Error.ByteStreamError;
            // The encoder routes xor == 0 through case 01, never case 00, so meaningful is
            // always non-zero here. Reject a corrupted stream that encodes a zero-XOR reuse
            // instead of silently repeating the previous value.
            if (meaningful == 0) return Error.UnsupportedInput;
            const xor = meaningful << stored_tz;
            const v_prime_bits = state.stored_val ^ xor;
            state.stored_val = v_prime_bits;
            return v_prime_bits;
        },

        // Case 10: new bucket, center_bits <= 16.
        0b10 => {
            const lead_idx = bit_reader.readBitsNoEof(u3, 3) catch return Error.ByteStreamError;
            const center_raw = bit_reader.readBitsNoEof(u4, 4) catch return Error.ByteStreamError;
            const new_lead = leading_zero_bucket_values[lead_idx];
            // Encoder wrote `center_bits & 0xf`; the 0 sentinel decodes back to 16.
            const center_bits: u16 = if (center_raw == 0) 16 else @as(u16, center_raw);
            const new_trail: u6 = @intCast(bits_per_value - @as(u16, new_lead) - center_bits);
            // Read center-1 meaningful bits; prepend the implicit top 1 and shift into place.
            const meaningful = bit_reader.readBitsNoEof(u64, center_bits - 1) catch return Error.ByteStreamError;
            const xor = ((meaningful << 1) | 1) << new_trail;
            const v_prime_bits = state.stored_val ^ xor;
            state.stored_val = v_prime_bits;
            state.stored_leading_zeros = new_lead;
            state.stored_trailing_zeros = new_trail;
            return v_prime_bits;
        },

        // Case 11: new bucket, center_bits > 16.
        0b11 => {
            const lead_idx = bit_reader.readBitsNoEof(u3, 3) catch return Error.ByteStreamError;
            const center_raw = bit_reader.readBitsNoEof(u6, 6) catch return Error.ByteStreamError;
            const new_lead = leading_zero_bucket_values[lead_idx];
            // Encoder wrote `center_bits & 0x3f`; the 0 sentinel decodes back to 64.
            const center_bits: u16 = if (center_raw == 0) 64 else @as(u16, center_raw);
            // Validate geometry before casting: a corrupted stream can pair a large center_bits
            // with a non-zero leading bucket, underflowing the trailing-zero count. Reject it
            // instead of trapping in the @intCast below.
            if (@as(u16, new_lead) + center_bits > bits_per_value) return Error.UnsupportedInput;
            const new_trail: u6 = @intCast(bits_per_value - @as(u16, new_lead) - center_bits);
            const meaningful = bit_reader.readBitsNoEof(u64, center_bits - 1) catch return Error.ByteStreamError;
            const xor = ((meaningful << 1) | 1) << new_trail;
            const v_prime_bits = state.stored_val ^ xor;
            state.stored_val = v_prime_bits;
            state.stored_leading_zeros = new_lead;
            state.stored_trailing_zeros = new_trail;
            return v_prime_bits;
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

    for (0..generated_test_rounds) |_| {
        try tester.testLosslessMethod(
            allocator,
            Method.Elf,
            data_distributions,
        );
    }
}

test "elf roundtrips empty input" {
    // Empty input uses only the count header and no bit stream.
    const uncompressed_values = &[_]f64{};

    try tester.expectLosslessRoundTrip(testing.allocator, compress, decompress, uncompressed_values);
}

test "elf roundtrips single value" {
    // A single value stores the count and first raw value without any markers.
    const uncompressed_values = &[_]f64{42.5};

    try tester.expectLosslessRoundTrip(testing.allocator, compress, decompress, uncompressed_values);
}

test "elf roundtrips two values" {
    // Two values exercise exactly one Eraser+XOR_cmp marker right after the first raw value.
    const uncompressed_values = &[_]f64{ 3.5, 9.0 };

    try tester.expectLosslessRoundTrip(testing.allocator, compress, decompress, uncompressed_values);
}

test "elf roundtrips repeated values" {
    // Repeated values exercise XOR_cmp case 01 (xor = 0) after the first raw value.
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
    // Unlike the Java reference's NaN canonicalization, we keep payload bits intact.
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
    // Negative powers of 10 trigger the corner case where SP shifts during erasure.
    // Encoder writes beta_star = 0 sentinel; decoder uses the get10iN(-sp-1) recovery formula.
    const uncompressed_values = &[_]f64{ 0.1, 0.01, 0.001, 0.0001, 0.00001 };

    try tester.expectLosslessRoundTrip(testing.allocator, compress, decompress, uncompressed_values);
}

test "elf roundtrips integer values" {
    // Integer-valued floats have alpha = beta - SP - 1 <= 0, which the eraser routes to no-erase.
    // Verifies the alpha < 0 guard in eraseValue doesn't break integer round-trips.
    const uncompressed_values = &[_]f64{ 0.0, 1.0, 10.0, 100.0, 1000.0, 1e10 };

    try tester.expectLosslessRoundTrip(testing.allocator, compress, decompress, uncompressed_values);
}

test "elf compresses repeated values below raw size" {
    // A constant signal maximally exercises XOR_cmp case 01 (xor = 0):
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
