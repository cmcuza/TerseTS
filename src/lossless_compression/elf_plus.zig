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

//! Implementation of the Elf+ lossless floating-point time series compression method.
//! Elf is described in:
//! Li et al., "Elf: Erasing-based Lossless Floating-Point Compression", VLDB 2023.
//! https://doi.org/10.14778/3587136.3587149
//!
//! Elf+ is the enhanced version of Elf maintained on the `dev` branch of the authors'
//! reference implementation at https://github.com/Spatio-Temporal-Lab/elf (under review for
//! VLDBJ, and the variant the reference README recommends over the paper version for both
//! ratio and speed). On top of the VLDB 2023 algorithm it adds beta_star state reuse (saving
//! bits when the decimal-significand count stays constant across consecutive values) and
//! lookup-table-driven beta computation.
//!
//! Like Elf, the method erases the noise bits a decimal value (e.g. 23.45) carries in its
//! binary mantissa, producing a near-equal value_prime with many trailing zeros. value_prime is
//! then XOR-encoded against the previous value_prime (four cases by leading/trailing-zero
//! buckets); the decoder recovers the value from value_prime by rounding up to beta_star decimal
//! digits. The bit-level layout, decimal-precision tables, and recovery formulas follow the
//! authors' reference Java implementation.
//!
//! Elf+'s marker scheme, beta_star-reuse threading, and significant-digit search live here; the
//! decimal-precision tables, XOR encoding, and value restoration are shared with `elf.zig`.

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
const elf = @import("elf.zig");

const Error = tersets.Error;
const Method = tersets.Method;
const XorState = elf.XorState;

// Terms from the Elf paper (Theorem 3), defined once so later comments can stay short:
//   alpha                - decimal digits after the point.
//   beta                 - count of significant decimal digits.
//   beta_star            - beta stored per value in 4 bits; 0 is a sentinel for the
//                          exact negative-power-of-ten case (see `restorer`).
//   value_prime          - the value after `eraser` clears the low "noise" mantissa bits.
//   significand position - power-of-ten place of the most significant decimal digit.
//   f(alpha)             - binary bits needed for alpha decimal digits = ceil(alpha*log2(10)).
//   g(alpha)             - mantissa cut point = f(alpha) + exponent - 1023.

/// Number of randomized rounds the generated-distribution round-trip test runs.
const generated_test_rounds: usize = 5;

/// Compress `uncompressed_values` into `compressed_values` using Elf+'s `eraser` and Elf's shared
/// `xorCompress` pipeline, allocating with `allocator`. `method_configuration` must be empty
/// (`{}`), otherwise `Error.InvalidConfiguration` is returned. `uncompressed_values` must not be
/// empty; `tersets.compress` guarantees this. On success `compressed_values` holds
/// `[first_value: f64][bit stream][end-of-stream marker]`, where each value is an eraser marker
/// (1, 2, or 6 bits) followed by the XOR encoding of value_prime.
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

    const first_value = uncompressed_values[0];
    try shared_functions.appendValue(allocator, f64, first_value, compressed_values);

    // beta_star reuse state: the beta_star of the most recent erased value, threaded across
    // values so a value with matching precision can be encoded with the 1-bit reuse marker.
    var last_beta_star: ?u8 = null;
    var xor_state = XorState{
        .stored_value_prime = @bitCast(first_value),
        .stored_leading_zeros = null,
        .stored_trailing_zeros = null,
    };

    var bit_writer = try shared_structs.BulkBitWriter.init(allocator, compressed_values);

    for (uncompressed_values[1..]) |value| {
        const erase_result = try eraser(&bit_writer, value, last_beta_star);
        last_beta_star = erase_result.new_last_beta_star;
        try elf.xorCompress(&bit_writer, erase_result.value_prime_bits, &xor_state);
    }

    // The end marker tells the decoder where to stop; flushed padding bits are never read.
    try writeEndMarker(&bit_writer);
    try bit_writer.flushBits();
}

/// Decompress an Elf+-encoded `compressed_values` stream into `decompressed_values`, allocating
/// with `allocator`. The stream must start with the raw `[first_value: f64]` written by
/// `compress`; malformed or truncated input returns `Error.CorruptedCompressedData`.
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

    var last_beta_star: ?u8 = null;
    var xor_state = XorState{
        .stored_value_prime = @bitCast(first_value),
        .stored_leading_zeros = null,
        .stored_trailing_zeros = null,
    };

    var bit_reader = shared_structs.BulkBitReader.init(compressed_values[offset..]);

    while (true) {
        // `eraser` marker dispatch:
        //   "0"          (1 bit)   -> erase, beta_star reused from previous erased value.
        //   "10"         (2 bits)  -> no erase; carries the end-of-stream marker (see `writeEndMarker`).
        //   "11"+beta_star (6 bits) -> erase, new beta_star (4 bits follow the 2-bit marker).
        const first_marker_bit = bit_reader.readBitsNoEof(u1, 1) catch return Error.CorruptedCompressedData;

        if (first_marker_bit == 0) {
            // Reuse requires a previously-written beta_star; the encoder only emits this case
            // after a case-11 has set one.
            const beta_star = last_beta_star orelse return Error.CorruptedCompressedData;
            const value_prime_bits = (try elf.xorDecompress(&bit_reader, &xor_state)) orelse
                return Error.CorruptedCompressedData;
            const value = try elf.restorer(@bitCast(value_prime_bits), beta_star);
            try decompressed_values.append(allocator, value);
            continue;
        }

        const second_marker_bit = bit_reader.readBitsNoEof(u1, 1) catch return Error.CorruptedCompressedData;

        if (second_marker_bit == 0) {
            // No erase: value_prime equals the value here, so xorDecompress returns it directly
            // (no restore). A null result is the end-of-stream marker, so decoding stops here.
            const value_bits = (try elf.xorDecompress(&bit_reader, &xor_state)) orelse break;
            try decompressed_values.append(allocator, @bitCast(value_bits));
            continue;
        }

        // Erase with new beta_star: read beta_star, update state, then restore the original value.
        const new_beta_star = bit_reader.readBitsNoEof(u8, 4) catch return Error.CorruptedCompressedData;
        last_beta_star = new_beta_star;
        const value_prime_bits = (try elf.xorDecompress(&bit_reader, &xor_state)) orelse
            return Error.CorruptedCompressedData;
        const value = try elf.restorer(@bitCast(value_prime_bits), new_beta_star);
        try decompressed_values.append(allocator, value);
    }
}

/// Write the end-of-stream marker: elf_plus's "10" no-erase prefix followed by Elf's impossible
/// case-11 header. Mirrors `elf.zig`'s `writeEndMarker`, widened to elf_plus's 2-bit no-erase marker.
fn writeEndMarker(bit_writer: *shared_structs.BulkBitWriter) Error!void {
    try bit_writer.writeBits(@as(u2, 0b10), 2);
    try bit_writer.writeBits(@as(u2, 0b11), 2);
    try bit_writer.writeBits(elf.end_marker_lead_index, shared_structs.leading_zero_bucket_bits);
    try bit_writer.writeBits(elf.end_marker_center_raw, 6);
}

/// Return the count of significant decimal digits needed to represent `value_abs` exactly.
/// The leading digit sits at `significand_position`. Returns 17 when `value_abs` has no short
/// exact decimal form, or needs more digits than an `f64` can distinguish.
///
/// This is Elf+'s real algorithmic addition over `elf.getSignificantCount`: `last_beta_star`
/// seeds the starting exponent so a value with the same precision as the previous one converges
/// in one step, and because that seed can overshoot the minimal digit count, trailing decimal
/// zeros are stripped after a match is found. Kept separate from Elf's version for this reason.
fn getSignificantCount(value_abs: f64, significand_position: i16, last_beta_star: ?u8) u8 {
    // Seed exponent from the previous value's beta_star: a non-zero hint means this value likely
    // needs the same digit count, skipping early iterations below. No hint (first value, or the
    // previous value was the 10^-i sentinel) falls back to Elf's own starting exponent.
    var exponent: i32 = if (last_beta_star) |previous_beta_star|
        (if (previous_beta_star != 0)
            @as(i32, previous_beta_star) - @as(i32, significand_position) - 1
        else if (significand_position >= 0) 1 else -@as(i32, significand_position))
    else
        @as(i32, elf.maximum_significant_digits) - @as(i32, significand_position) - 1;

    // Clamp to >= 1: a very large significand_position (|value| >= ~1e17) can drive the formula
    // above to 0 or negative, and getPositivePowerOfTen requires a non-negative exponent.
    if (exponent < 1) exponent = 1;

    var scaled: f64 = undefined;
    var scaled_int: i64 = undefined;

    // Increase the exponent until `value_abs * 10^exponent` is an exact integer, i.e. until every
    // significant digit has been shifted left of the decimal point.
    var iterations: u8 = 0;
    while (true) : (iterations += 1) {
        scaled = value_abs * elf.getPositivePowerOfTen(exponent);
        // Check the bound before @intFromFloat: a value >= 2^63 would trap the cast.
        if (iterations >= elf.maximum_scale_iterations or scaled >= elf.maximum_safe_int_float) {
            return elf.maximum_significant_digits;
        }
        scaled_int = @intFromFloat(scaled);
        if (@as(f64, @floatFromInt(scaled_int)) == scaled) break;
        exponent += 1;
    }

    // Confirm the scaling is exactly reversible. If `value_abs * 10^exponent` only "looked" integral
    // due to rounding in the multiply, dividing back out won't recover `value_abs` - meaning there is
    // no short exact form.
    if (scaled / elf.getPositivePowerOfTen(exponent) != value_abs) return elf.maximum_significant_digits;

    // Strip trailing decimal zeros so we report the MINIMAL significand count.
    // Example: value_abs = 5.20 -> scaled_int = 520, strip one zero -> scaled_int = 52, beta = 2.
    while (exponent > 0 and @rem(scaled_int, 10) == 0) {
        exponent -= 1;
        scaled_int = @divTrunc(scaled_int, 10);
    }

    const significant_count = @as(i32, significand_position) + exponent + 1;
    return @intCast(@max(0, @min(significant_count, @as(i32, elf.maximum_significant_digits))));
}

/// Return the two quantities `eraser` needs: `alpha`, the number of decimal digits after the point,
/// and `beta_star`, the significant-digit count stored per value (4 bits) that `restorer` uses to
/// round value_prime back to the original. `beta_star` is the significant-digit count normally; the
/// value 0 is reserved as a sentinel for the exact-negative-power-of-ten corner case
/// (is_negative_power_of_ten), where `restorer` instead restores the value directly from
/// negative_power_of_10_table.
///
/// Not reused from `elf.computeAlphaAndBetaStar` for the same reason as `getSignificantCount`
/// above: this version threads `last_beta_star` through to seed that search. `last_beta_star` is
/// a hint that speeds up the beta iteration; pass null on the first value of the stream.
fn computeAlphaAndBetaStar(value_abs: f64, last_beta_star: ?u8) struct { alpha: i32, beta_star: u8 } {
    const significand_info = elf.significandPosition(value_abs);
    const beta = getSignificantCount(value_abs, significand_info.position, last_beta_star);
    const alpha: i32 = @as(i32, beta) - @as(i32, significand_info.position) - 1;
    const beta_star: u8 = if (significand_info.is_negative_power_of_ten) 0 else beta;
    return .{ .alpha = alpha, .beta_star = beta_star };
}

/// Write the prefix marker for `value` to `bit_writer` ("0" to erase reusing the previous
/// beta_star, "10" for no erase, or "11" plus a new beta_star to erase) and return the
/// value_prime bits plus the updated beta_star-reuse hint for `xorCompress`. Implements the
/// paper's Eraser, extended with Elf+'s beta_star reuse.
fn eraser(
    bit_writer: *shared_structs.BulkBitWriter,
    value: f64,
    last_beta_star: ?u8,
) Error!struct { value_prime_bits: u64, new_last_beta_star: ?u8 } {
    const value_bits: u64 = @bitCast(value);

    // Special values: 0, +/-inf, NaN. Skip the decimal-precision machinery and pass
    // raw bits through `xorCompress`. last_beta_star is preserved so the next decimal value can still reuse it.
    if (value == 0.0 or math.isInf(value) or math.isNan(value)) {
        try bit_writer.writeBits(@as(u2, 0b10), 2);
        return .{ .value_prime_bits = value_bits, .new_last_beta_star = last_beta_star };
    }

    // Decimal-precision analysis. beta_star will be written to the stream; alpha is used to
    // compute how many mantissa bits to erase.
    const value_abs = @abs(value);
    const alpha_beta_star = computeAlphaAndBetaStar(value_abs, last_beta_star);

    // Bail to no-erase if alpha is outside the useful range. Normally alpha equals the scale i
    // (>= 1), so these guards only catch the extremes where the significant-digit search saturated:
    //   alpha < 0   -> magnitude so large beta capped at 17 (|value| >~ 1e17) - nothing to erase.
    //   alpha >= 21 -> beyond the f_alpha_table (very small / subnormal values) - rare, skip.
    // Ordinary integers (e.g. 100.0) pass this guard with alpha >= 0; they route to no-erase a few
    // lines below via the delta == 0 check, which sees no erasable low mantissa bits.
    if (alpha_beta_star.alpha < 0 or alpha_beta_star.alpha >= elf.f_alpha_table.len) {
        try bit_writer.writeBits(@as(u2, 0b10), 2);
        return .{ .value_prime_bits = value_bits, .new_last_beta_star = last_beta_star };
    }
    // beta_star is encoded in 4 bits; values > 15 are not representable on the erase path.
    if (alpha_beta_star.beta_star > 15) {
        try bit_writer.writeBits(@as(u2, 0b10), 2);
        return .{ .value_prime_bits = value_bits, .new_last_beta_star = last_beta_star };
    }

    // g(alpha) tells us how many mantissa bits are needed to represent the value exactly given
    // its decimal precision; everything below g(alpha) is binary noise we can erase.
    const exponent: i32 = @intCast((value_bits >> elf.mantissa_bits) & elf.exponent_mask);
    const g_alpha: i32 = elf.getFAlpha(alpha_beta_star.alpha) + exponent - elf.exponent_bias;
    const erase_bits: i32 = @as(i32, elf.mantissa_bits) - g_alpha;

    // Profitability + safety guard:
    //   <= 4 bits saved -> the erase marker + beta_star overhead wipes the gain.
    //   >= 64 bits     -> shift count would be UB on u64.
    if (erase_bits <= 4 or erase_bits >= shared_structs.bits_per_value) {
        try bit_writer.writeBits(@as(u2, 0b10), 2);
        return .{ .value_prime_bits = value_bits, .new_last_beta_star = last_beta_star };
    }

    // Build the mask, then check that the value actually has any of those low bits set.
    // If not, "erasing" wouldn't change value_bits - skip to no-erase to save bits.
    const shift: u6 = @intCast(erase_bits);
    const mask: u64 = @as(u64, 0xffffffffffffffff) << shift;
    const delta: u64 = (~mask) & value_bits;
    if (delta == 0) {
        try bit_writer.writeBits(@as(u2, 0b10), 2);
        return .{ .value_prime_bits = value_bits, .new_last_beta_star = last_beta_star };
    }

    const value_prime_bits: u64 = mask & value_bits;

    // beta_star reuse: if it matches the previous erased value's beta_star, emit the 1-bit
    // case-0 marker instead of re-writing the 6-bit case-11 marker.
    if (last_beta_star) |previous_beta_star| {
        if (previous_beta_star == alpha_beta_star.beta_star) {
            try bit_writer.writeBits(@as(u1, 0), 1);
            return .{ .value_prime_bits = value_prime_bits, .new_last_beta_star = last_beta_star };
        }
    }

    try bit_writer.writeBits(@as(u2, 0b11), 2);
    try bit_writer.writeBits(alpha_beta_star.beta_star, 4);
    return .{ .value_prime_bits = value_prime_bits, .new_last_beta_star = alpha_beta_star.beta_star };
}

test "elf_plus roundtrips generated values across all distributions" {
    const allocator = testing.allocator;

    // Elf+ is bitwise lossless, so it must recover any f64 input - including unbounded
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
            Method.ElfPlus,
            data_distributions,
        );
    }
}

test "elf_plus roundtrips single value" {
    // A single value stores only the first raw value plus the end-of-stream marker.
    const uncompressed_values = &[_]f64{42.5};

    try tester.expectLosslessRoundTrip(testing.allocator, compress, decompress, uncompressed_values);
}

test "elf_plus roundtrips two values" {
    // Two values exercise exactly one `eraser`+`xorCompress` marker right after the first raw value.
    const uncompressed_values = &[_]f64{ 3.5, 9.0 };

    try tester.expectLosslessRoundTrip(testing.allocator, compress, decompress, uncompressed_values);
}

test "elf_plus roundtrips repeated values" {
    // Repeated values exercise `xorCompress` case 01 (xor = 0) after the first raw value.
    const uncompressed_values = &[_]f64{ 7.25, 7.25, 7.25, 7.25, 7.25 };

    try tester.expectLosslessRoundTrip(testing.allocator, compress, decompress, uncompressed_values);
}

test "elf_plus roundtrips changing values" {
    // Changing values cover bucket transitions, bucket reuse, and meaningful-bit paths.
    const uncompressed_values = &[_]f64{ 100.0, 100.01, 100.02, 99.99, -3.5, 0.0, 2048.125 };

    try tester.expectLosslessRoundTrip(testing.allocator, compress, decompress, uncompressed_values);
}

test "elf_plus roundtrips special floating-point values" {
    // Special values route through the no-erase path (marker 10) and preserve raw bits.
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

test "elf_plus roundtrips edge floats" {
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

test "elf_plus roundtrips decimal-originated values" {
    // Sensor-style values with limited decimal precision exercise the erase path
    // (beta_star in [1, 4]) plus the pow10 corner case.
    const uncompressed_values = &[_]f64{ 0.1, 3.17, 2.5, 100.01, 0.001, -42.42, 1e-5, 0.5 };

    try tester.expectLosslessRoundTrip(testing.allocator, compress, decompress, uncompressed_values);
}

test "elf_plus roundtrips beta_star state reuse" {
    // Ten values all with two decimal places (beta_star = 3 or 4). Once the encoder writes the
    // first one via case 11, every subsequent one with matching beta_star uses the 1-bit case 0.
    const uncompressed_values = &[_]f64{ 1.23, 4.56, 7.89, 2.34, 5.67, 8.90, 1.11, 2.22, 3.33, 4.44 };

    try tester.expectLosslessRoundTrip(testing.allocator, compress, decompress, uncompressed_values);
}

test "elf_plus roundtrips beta_star transitions" {
    // Each value has a different decimal place count, forcing case 11 (new beta_star) every
    // time - exercises the encoder's beta_star update logic and decoder's last_beta_star tracking.
    const uncompressed_values = &[_]f64{ 1.0, 2.5, 3.123, 4.0001, 5.5, 6.78, 7.99999 };

    try tester.expectLosslessRoundTrip(testing.allocator, compress, decompress, uncompressed_values);
}

test "elf_plus roundtrips pow10 boundary values" {
    // Negative powers of 10 trigger the corner case where the significand position shifts during
    // erasure. Encoder writes the beta_star = 0 sentinel; decoder uses the `getNegativePowerOfTen` restore formula.
    const uncompressed_values = &[_]f64{ 0.1, 0.01, 0.001, 0.0001, 0.00001 };

    try tester.expectLosslessRoundTrip(testing.allocator, compress, decompress, uncompressed_values);
}

test "elf_plus roundtrips integer values" {
    // Integer-valued floats route to no-erase via the delta == 0 check (no erasable low bits).
    // Verifies the no-erase guards in `eraser` don't break integer round-trips.
    const uncompressed_values = &[_]f64{ 0.0, 1.0, 10.0, 100.0, 1000.0, 1e10 };

    try tester.expectLosslessRoundTrip(testing.allocator, compress, decompress, uncompressed_values);
}

test "elf_plus compresses repeated values below raw size" {
    // A constant signal maximally exercises beta_star reuse + `xorCompress` case 01 (xor = 0):
    // every repeat is at most 1 + 2 = 3 bits. Output must be far smaller than the raw f64 array.
    const allocator = testing.allocator;

    var uncompressed_values: [500]f64 = undefined;
    @memset(&uncompressed_values, 42.0);

    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);

    try compress(allocator, &uncompressed_values, &compressed_values, "{}");

    try testing.expect(compressed_values.items.len < uncompressed_values.len * @sizeOf(f64));
}

test "elf_plus compresses decimal data below raw size" {
    // Sensor-style values with consistent 2-decimal precision exercise the eraser
    // sweet spot - the erased mantissa noise shrinks XOR outputs and beta_star reuse
    // keeps the per-value overhead at ~1 bit.
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

test "elf_plus rejects corrupted compressed data" {
    const allocator = testing.allocator;
    // A mix of beta_star reuse, no-erase, and fresh-beta_star values so the stream covers all
    // three `eraser` marker cases plus the four `xorCompress` cases.
    const uncompressed_values = &[_]f64{ 1.23, 4.56, 100.01, 42.0, -3.5, 7.89, 1e10 };

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

test "check elf_plus configuration parsing" {
    // Elf+ takes no parameters: an empty configuration must parse, and a configuration
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
