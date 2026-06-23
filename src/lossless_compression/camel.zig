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

//! Implementation of the Camel lossless floating-point time series compression method.
//! The method is described in:
//! Yao et al., "Camel: Efficient Compression of Floating-Point Time Series", SIGMOD 2024.
//! https://dl.acm.org/doi/epdf/10.1145/3698802

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

/// Upper bound for `calDecimalCount`'s search (a double has at most 17 significant decimal
/// digits). The per-value decimal count `lv` is additionally capped to `4` in `compress`.
const max_decimal_places = 17;
/// Default decimal places when configuration is empty.
const default_decimal_places = 4;
/// `2^63` as an `f64`; the exact threshold used by `fitsIntegerPart` to check that `@trunc(x)`
/// fits in an `i64` without overflow.
const i64_magnitude_limit: f64 = 9223372036854775808.0;

/// Compress `uncompressed_values` into `compressed_values` using "Camel". `allocator` backs
/// the configuration parser and the bit writer's scratch buffer. `method_configuration` accepts a
/// JSON object with a `decimal_places` field (1–4, default 4); values whose integer parts differ
/// by more than 65535 between consecutive normal values return `Error.UnsupportedInput`. If an
/// error occurs it is returned.
pub fn compress(
    allocator: Allocator,
    uncompressed_values: []const f64,
    compressed_values: *ArrayList(u8),
    method_configuration: []const u8,
) Error!void {
    var decimal_places: u8 = default_decimal_places;
    _ = try configuration.parse(
        allocator,
        configuration.EmptyConfiguration,
        method_configuration,
    );
    // The per-value decimal count is stored in 2 bits, so the cap is 1..4.
    const max_count: u8 = 4;
    decimal_places = @max(@as(u8, 1), @min(decimal_places, max_count));

    try shared_functions.appendValue(allocator, u64, @intCast(uncompressed_values.len), compressed_values);
    if (uncompressed_values.len == 0) return;

    const first = uncompressed_values[0];
    if (!fitsIntegerPart(first) or isNegativeZero(first)) return Error.UnsupportedInput;
    // Write the first value raw; the decimal count travels per-value as a 2-bit field.
    try shared_functions.appendValue(allocator, f64, first, compressed_values);

    var bit_writer = try shared_structs.BulkBitWriter.init(allocator, compressed_values);
    var prev_int: i64 = intPart(first);

    for (uncompressed_values[1..]) |v| {
        if (!fitsIntegerPart(v) or isNegativeZero(v)) return Error.UnsupportedInput;
        const int_part = intPart(v);
        const diff_overflow = @subWithOverflow(int_part, prev_int);
        if (diff_overflow[1] != 0 or diff_overflow[0] < -65535 or diff_overflow[0] > 65535) {
            return Error.UnsupportedInput;
        }
        const int_signal: u1 = @intFromBool(v >= 0.0);
        const lv = @min(calDecimalCount(v), decimal_places);
        // Value sign (intSignal) in integer part; decimal count as `lv - 1` in 2 bits (00→1…11→4).
        try compressIntegerPart(prev_int, int_part, int_signal, &bit_writer);
        try bit_writer.writeBits(@as(u2, @intCast(lv - 1)), 2);
        try compressDecimalPart(@abs(fracPart(v)), lv, &bit_writer);
        prev_int = int_part;
    }

    try bit_writer.flushBits();
}

/// Decompress `compressed_values` produced by "Camel" and write the result to
/// `decompressed_values`. `allocator` grows `decompressed_values` as values are recovered.
/// If an error occurs it is returned.
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

    var prev_int: i64 = if (fitsIntegerPart(first_value)) intPart(first_value) else 0;
    var bit_reader = shared_structs.BulkBitReader.init(compressed_values[offset..]);

    while (decompressed_values.items.len < value_count) {
        const result = try decompressIntegerPart(prev_int, &bit_reader);
        const lv: u8 = @as(u8, bit_reader.readBitsNoEof(u2, 2) catch return Error.ByteStreamError) + 1;
        const frac_magnitude = try decompressDecimalPart(lv, &bit_reader);
        const reconstructed: f64 = if (result.int_signal == 1)
            @as(f64, @floatFromInt(result.int_part)) + frac_magnitude
        else
            -(@as(f64, @floatFromInt(@abs(result.int_part))) + frac_magnitude);
        decompressed_values.appendAssumeCapacity(reconstructed);
        prev_int = result.int_part;
    }
}

/// Returns `true` when `x` is finite and its truncated integer part fits in an `i64`, i.e. it is
/// safe to call `intPart` on it.
fn fitsIntegerPart(x: f64) bool {
    if (math.isNan(x) or math.isInf(x)) return false;
    return @abs(@trunc(x)) < i64_magnitude_limit;
}

/// Returns the truncated integer part of `x` as an `i64`. Caller must first verify
/// `fitsIntegerPart(x)`.
fn intPart(x: f64) i64 {
    if (math.isNan(x) or math.isInf(x)) return 0;
    return @intFromFloat(@trunc(x));
}

/// Returns `true` only for negative zero (`-0.0`).
fn isNegativeZero(x: f64) bool {
    return x == 0.0 and math.signbit(x);
}

/// Returns the fractional remainder of `x` as `x - trunc(x)`. The sign matches `x`.
fn fracPart(x: f64) f64 {
    if (math.isNan(x) or math.isInf(x)) return 0.0;
    return x - @trunc(x);
}

/// Returns the smallest decimal count in `[1, max_decimal_places]` such that
/// `|value| * 10^count` is within epsilon of an integer; ported from Camel.java's `calDecimalCount`.
fn calDecimalCount(value: f64) u8 {
    const epsilon: f64 = 0.0000001;
    var factor: f64 = 1.0;
    var decimal_count: u8 = 0;
    const abs_value = @abs(value);

    while (@abs(abs_value * factor - @round(abs_value * factor)) > epsilon and
        decimal_count < max_decimal_places)
    {
        factor *= 10.0;
        decimal_count += 1;
    }

    if (decimal_count == 0) decimal_count = 1;
    return decimal_count;
}

/// Reinterprets an `f64` as its raw IEEE-754 `u64` bit pattern.
fn doubleToBits(x: f64) u64 {
    return @as(u64, @bitCast(x));
}

/// Writes the compressed integer part of one value into `writer` using delta-encoding.
/// First writes `int_signal` (1 = non-negative value, 0 = negative), then encodes
/// `int_part - prev_int` with a 2-bit code for `{-1, 0, 1}` and a sign+range+magnitude
/// encoding for larger differences (Algorithm 1 of the paper).
fn compressIntegerPart(prev_int: i64, int_part: i64, int_signal: u1, writer: *shared_structs.BulkBitWriter) !void {
    try writer.writeBits(int_signal, 1);
    const diff = int_part - prev_int;
    const abs_diff = @abs(diff);
    if (abs_diff <= 1) {
        // Encode diff+1 in 2 bits (0: -1, 1: 0, 2: 1).
        const code = @as(u2, @intCast(@as(u3, @intCast(diff + 1)) & 0b11));
        try writer.writeBits(code, 2);
    } else {
        // Range marker `3` (0b11) signals the sign+flag+magnitude encoding.
        try writer.writeBits(@as(u2, 0b11), 2);
        try writer.writeBits(@as(u1, @intFromBool(diff >= 0)), 1);
        const flag = @as(u1, @intFromBool(abs_diff >= 8));
        try writer.writeBits(flag, 1);
        try writer.writeBits(@as(u64, @intCast(abs_diff)), if (flag == 0) 3 else 16);
    }
}

/// Compresses the fractional magnitude of one value using the Camel XOR scheme (Algorithm 2).
/// `magnitude` must be non-negative (sign is carried by `int_signal` in the integer part).
/// Writes a flag; when the magnitude is at least one quantization step the XOR center bits and
/// `dxor_prime` are written (flag 1), otherwise the rounded magnitude is stored directly (flag 0).
fn compressDecimalPart(magnitude: f64, l: u8, writer: *shared_structs.BulkBitWriter) !void {
    const step = math.pow(f64, 2.0, -@as(f64, @floatFromInt(l)));
    if (magnitude >= step) {
        try writer.writeBits(@as(u1, 1), 1);
        const dxor_frac_raw = computeDxorFrac(magnitude, l);
        const factor = math.pow(f64, 10.0, @as(f64, @floatFromInt(l)));
        const dxor_prime = @as(u64, @intFromFloat(@round(dxor_frac_raw * factor)));
        // Rebuild dxor_frac from the quantized dxor_prime so that the decompressor XORs
        // against a bit-identical value (Theorem 3.1 of the paper).
        const dxor_frac = @as(f64, @floatFromInt(dxor_prime)) / factor;
        const xor_bits = doubleToBits(1.0 + magnitude) ^ doubleToBits(1.0 + dxor_frac);
        const shift = @as(u6, @intCast(52 - l));
        const mask = (@as(u64, 1) << @as(u6, @intCast(l))) - 1;
        try writer.writeBits(@as(u64, @truncate((xor_bits >> shift) & mask)), @as(u6, @intCast(l)));
        try writeDxorPrime(dxor_prime, l, writer);
    } else {
        try writer.writeBits(@as(u1, 0), 1);
        const factor = math.pow(f64, 10.0, @as(f64, @floatFromInt(l)));
        try writeDxorPrime(@as(u64, @intFromFloat(@round(magnitude * factor))), l, writer);
    }
}

/// Returns `dxor.ddec` from formula (3) of the paper: `frac - 2^{-l} * floor(frac / 2^{-l})`.
fn computeDxorFrac(frac: f64, l: u8) f64 {
    if (l == 0) return 0.0;
    const step = math.pow(f64, 2.0, -@as(f64, @floatFromInt(l)));
    return frac - step * @floor(frac / step);
}

/// Writes the quantized decimal integer `dxor_prime` using the per-`l` prefix+value bucket
/// tables from Camel.java's `mValueBits`.
fn writeDxorPrime(dxor_prime: u64, l: u8, writer: *shared_structs.BulkBitWriter) !void {
    switch (l) {
        0, 1 => {
            try writer.writeBits(dxor_prime, 3);
        },
        2 => {
            if (dxor_prime < 8) {
                try writer.writeBits(@as(u1, 0), 1);
                try writer.writeBits(dxor_prime, 3);
            } else {
                try writer.writeBits(@as(u1, 1), 1);
                try writer.writeBits(dxor_prime, 5);
            }
        },
        3 => {
            if (dxor_prime < 2) {
                try writer.writeBits(@as(u2, 0), 2);
                try writer.writeBits(dxor_prime, 1);
            } else if (dxor_prime < 8) {
                try writer.writeBits(@as(u2, 1), 2);
                try writer.writeBits(dxor_prime, 3);
            } else if (dxor_prime < 32) {
                try writer.writeBits(@as(u2, 2), 2);
                try writer.writeBits(dxor_prime, 5);
            } else {
                try writer.writeBits(@as(u2, 3), 2);
                try writer.writeBits(dxor_prime, 7);
            }
        },
        else => { // l == 4
            if (dxor_prime < 16) {
                try writer.writeBits(@as(u2, 0), 2);
                try writer.writeBits(dxor_prime, 4);
            } else if (dxor_prime < 64) {
                try writer.writeBits(@as(u2, 1), 2);
                try writer.writeBits(dxor_prime, 6);
            } else if (dxor_prime < 256) {
                try writer.writeBits(@as(u2, 2), 2);
                try writer.writeBits(dxor_prime, 8);
            } else {
                try writer.writeBits(@as(u2, 3), 2);
                try writer.writeBits(dxor_prime, 10);
            }
        },
    }
}

/// Reinterprets a raw IEEE-754 `u64` bit pattern as an `f64`. Inverse of `doubleToBits`.
fn bitsToDouble(bits: u64) f64 {
    return @as(f64, @bitCast(bits));
}

const IntPartResult = struct { int_part: i64, int_signal: u1 };

/// Reads and decodes the integer part of one value from `reader`, inverting `compressIntegerPart`.
/// Returns the decoded `int_part` and the `int_signal` (1 = non-negative value, 0 = negative).
fn decompressIntegerPart(prev_int: i64, reader: *shared_structs.BulkBitReader) Error!IntPartResult {
    const int_signal = reader.readBitsNoEof(u1, 1) catch return Error.ByteStreamError;
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
    return .{ .int_part = prev_int + diff, .int_signal = @intCast(int_signal) };
}

/// Decompresses the fractional magnitude produced by `compressDecimalPart`, inverting the XOR
/// scheme. Returns a non-negative value; the caller applies the sign from `int_signal`.
fn decompressDecimalPart(l: u8, reader: *shared_structs.BulkBitReader) Error!f64 {
    const flag = reader.readBitsNoEof(u1, 1) catch return Error.ByteStreamError;
    const factor = math.pow(f64, 10.0, @as(f64, @floatFromInt(l)));
    var magnitude: f64 = undefined;
    if (flag == 1) {
        const center_bits = reader.readBitsNoEof(u64, @as(u6, @intCast(l))) catch return Error.ByteStreamError;
        const shift = @as(u6, @intCast(52 - l));
        const vd = center_bits << shift;
        const dxor_prime = try readDxorPrime(l, reader);
        const dxor_frac = @as(f64, @floatFromInt(dxor_prime)) / factor;
        magnitude = bitsToDouble(doubleToBits(1.0 + dxor_frac) ^ vd) - 1.0;
        if (magnitude < 0.0) magnitude = 0.0;
        // Snap to l decimal digits to cancel residual XOR/(1+m)−1 noise.
        magnitude = @round(magnitude * factor) / factor;
    } else {
        const dxor_prime = try readDxorPrime(l, reader);
        magnitude = @as(f64, @floatFromInt(dxor_prime)) / factor;
    }
    return magnitude;
}

/// Reads `dxor_prime` encoded by `writeDxorPrime`, using the same per-`l` prefix+value tables.
fn readDxorPrime(l: u8, reader: *shared_structs.BulkBitReader) Error!u64 {
    switch (l) {
        0, 1 => return reader.readBitsNoEof(u64, 3) catch return Error.ByteStreamError,
        2 => {
            const t = reader.readBitsNoEof(u1, 1) catch return Error.ByteStreamError;
            return reader.readBitsNoEof(u64, if (t == 0) 3 else 5) catch return Error.ByteStreamError;
        },
        3 => {
            const t = reader.readBitsNoEof(u2, 2) catch return Error.ByteStreamError;
            const bits: u6 = switch (t) {
                0 => 1,
                1 => 3,
                2 => 5,
                3 => 7,
            };
            return reader.readBitsNoEof(u64, bits) catch return Error.ByteStreamError;
        },
        else => { // l == 4
            const t = reader.readBitsNoEof(u2, 2) catch return Error.ByteStreamError;
            const bits: u6 = switch (t) {
                0 => 4,
                1 => 6,
                2 => 8,
                3 => 10,
            };
            return reader.readBitsNoEof(u64, bits) catch return Error.ByteStreamError;
        },
    }
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

test "camel returns UnsupportedInput for special floating-point values" {
    const specials = [_]f64{
        math.nan(f64),
        math.inf(f64),
        -math.inf(f64),
        math.floatMax(f64),
        -math.floatMax(f64),
        -0.0,
    };
    for (specials) |special| {
        var buffer = ArrayList(u8).empty;
        defer buffer.deinit(testing.allocator);
        const result = compress(testing.allocator, &[_]f64{ 1.0, special }, &buffer, "{}");
        try testing.expectError(Error.UnsupportedInput, result);
    }
}

test "camel round-trip integer-only and edge cases" {
    const uncompressed_values = &[_]f64{
        -65535.0, // largest negative integer difference that fits in 16 bits
        0.0,
        5.0,
        -7.0,
        65528.0, // maximal integer difference that fits in 16 bits (65528 - (-7) = 65535)
    };
    try tester.expectLosslessRoundTrip(testing.allocator, compress, decompress, uncompressed_values);
}

test "camel cannot compress values with large integer differences" {
    const uncompressed_values = &[_]f64{ 0.0, 100000.0 };
    var buffer = ArrayList(u8).empty;
    defer buffer.deinit(testing.allocator);

    compress(testing.allocator, uncompressed_values, &buffer, "{}") catch |err| {
        try testing.expectEqual(Error.UnsupportedInput, err);
        return;
    };

    try testing.expectFmt("", "Expected error not triggered", .{});
}
