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

//! Implementation of the CAMEL lossless floating-point time series compression method.
//! The method is described in:
//! Messaoudi et al., "CAMEL: Efficient Lossless Floating Point Compression for Time Series Databases", 2022.
//! https://doi.org/10.1109/ICDE53745.2022.00135

const std = @import("std");
const ArrayList = std.ArrayList;
const Allocator = std.mem.Allocator;
const Print = std.debug.print;
const Math = std.math;
const testing = std.testing;

const tester = @import("../tester.zig");
const tersets = @import("../tersets.zig");
const shared_functions = @import("../utilities/shared_functions.zig");
const Method = tersets.Method;
const configuration = @import("../configuration.zig");
const Error = tersets.Error;

pub const Parts = struct {
    integer: i64, // integer part (with sign)
    decimal: u64, // fractional part as integer (0 .. 10^l - 1)
    decimal_digits: u8, // number of fractional digits (l)
    special: bool, // true if number cannot be represented (NaN, Inf, too large)
    raw_bits: u64, // raw bits for special cases
};

const BitWriter = struct {
    buffer: *ArrayList(u8),
    allocator: Allocator,
    bit_buffer: u8 = 0,
    bit_count: u8 = 0,

    // Initializes a bit writer over a byte buffer.
    fn init(allocator: Allocator, buffer: *ArrayList(u8)) BitWriter {
        return .{
            .allocator = allocator,
            .buffer = buffer,
        };
    }

    // Writes `bits` least-significant bits of `value` to the stream.
    fn writeBits(self: *BitWriter, value: anytype, bits: u8) !void {
        var v = @as(u64, @intCast(value));
        var remaining = bits;
        while (remaining > 0) {
            const take = @min(remaining, 8 - self.bit_count);
            const take_mask = (@as(u64, 1) << @as(u6, @intCast(take))) - 1;
            const take_bits = v & take_mask;
            const shifted = take_bits << @as(u3, @intCast(self.bit_count));
            self.bit_buffer |= @as(u8, @intCast(shifted));
            self.bit_count += take;
            if (self.bit_count == 8) {
                try self.buffer.append(self.allocator, self.bit_buffer);
                self.bit_buffer = 0;
                self.bit_count = 0;
            }
            v >>= @as(u6, @intCast(take));
            remaining -= take;
        }
    }

    // Writes a single bit to the stream.
    fn writeBit(self: *BitWriter, bit: u1) !void {
        try self.writeBits(bit, 1);
    }

    // Flushes the remaining partial byte to the buffer.
    fn finish(self: *BitWriter) !void {
        if (self.bit_count > 0) {
            try self.buffer.append(self.allocator, self.bit_buffer);
        }
    }
};

const BitReader = struct {
    buffer: []const u8,
    byte_index: usize = 0,
    bit_pos: u3 = 0,

    // Initializes a bit reader over a byte slice.
    fn init(buffer: []const u8) BitReader {
        return .{ .buffer = buffer };
    }

    // Reads `bits` bits from the stream (LSB-first).
    fn readBits(self: *BitReader, bits: u8) Error!u64 {
        if (bits > 64) return Error.CorruptedCompressedData;
        var result: u64 = 0;
        var remaining = bits;
        var shift: u7 = 0;
        while (remaining > 0) {
            if (self.byte_index >= self.buffer.len) return Error.CorruptedCompressedData;
            const take = @min(remaining, 8 - @as(u8, self.bit_pos));
            const mask = (@as(u64, self.buffer[self.byte_index]) >> @as(u3, @intCast(self.bit_pos))) &
                ((@as(u64, 1) << @as(u6, @intCast(take))) - 1);
            result |= (mask << @as(u6, @intCast(shift)));
            shift += @as(u7, @intCast(take));
            const new_bit_pos = @as(u8, self.bit_pos) + take;
            if (new_bit_pos == 8) {
                self.byte_index += 1;
                self.bit_pos = 0;
            } else {
                self.bit_pos = @as(u3, @intCast(new_bit_pos));
            }
            remaining -= take;
        }
        return result;
    }

    // Reads a single bit from the stream.
    fn readBit(self: *BitReader) Error!u1 {
        return @as(u1, @intCast(try self.readBits(1)));
    }
};

// Splits a float into integer/fractional parts and metadata.
pub fn splitNumber(number: f64, fixed_l: ?u8) Parts {
    // Preserve negative zero as a special case to keep the sign bit.
    if (number == 0.0 and (@as(u64, @bitCast(number)) & 0x8000_0000_0000_0000) != 0) {
        return .{
            .integer = 0,
            .decimal = 0,
            .decimal_digits = 0,
            .special = true,
            .raw_bits = @as(u64, @bitCast(number)),
        };
    }

    // Handle special values (NaN, Inf)
    if (!Math.isFinite(number)) {
        return .{
            .integer = 0,
            .decimal = 0,
            .decimal_digits = 0,
            .special = true,
            .raw_bits = @as(u64, @bitCast(number)),
        };
    }

    const l = fixed_l orelse computeDecimalDigits(number);
    if (l == 0) {
        // Integer number
        const floored = Math.floor(number);
        const max_i64 = @as(f64, @floatFromInt(Math.maxInt(i64)));
        const min_i64 = @as(f64, @floatFromInt(Math.minInt(i64)));
        if (floored > max_i64 or floored < min_i64) {
            // Too large integer – store as special
            return .{
                .integer = 0,
                .decimal = 0,
                .decimal_digits = 0,
                .special = true,
                .raw_bits = @as(u64, @bitCast(number)),
            };
        }
        const integer = @as(i64, @intFromFloat(Math.trunc(number)));
        return .{
            .integer = integer,
            .decimal = 0,
            .decimal_digits = 0,
            .special = false,
            .raw_bits = 0,
        };
    }

    // Scale: multiply by 10^l and round to the nearest integer
    const scale = Math.pow(f64, 10.0, @as(f64, @floatFromInt(l)));
    const scaled = Math.round(number * scale);
    // Check for i64 overflow (range ~ ±9.22e18)
    const max_i64 = @as(f64, @floatFromInt(Math.maxInt(i64)));
    const min_i64 = @as(f64, @floatFromInt(Math.minInt(i64)));
    if (scaled > max_i64 or scaled < min_i64) {
        // Too large to represent in i64 -> store as special
        return .{
            .integer = 0,
            .decimal = 0,
            .decimal_digits = 0,
            .special = true,
            .raw_bits = @as(u64, @bitCast(number)),
        };
    }

    const scaled_int = @as(i64, @intFromFloat(scaled));
    const scale_i64 = @as(i64, @intFromFloat(scale));
    const integer = @divFloor(scaled_int, scale_i64);
    const decimal = @abs(scaled_int - integer * scale_i64);

    // Ensure fractional part fits in u64 and does not exceed 10^l - 1
    const max_decimal = @as(u64, @intFromFloat(scale)) - 1;
    const decimal_u64 = @as(u64, @intCast(decimal));
    if (decimal_u64 > max_decimal) {
        // Should not happen, but keep a safe fallback
        return .{
            .integer = 0,
            .decimal = 0,
            .decimal_digits = 0,
            .special = true,
            .raw_bits = @as(u64, @bitCast(number)),
        };
    }

    return .{
        .integer = integer,
        .decimal = decimal_u64,
        .decimal_digits = l,
        .special = false,
        .raw_bits = 0,
    };
}

// Computes the number of decimal digits to keep.
fn computeDecimalDigits(value: f64) u8 {
    return calculateDecimalCount(value);
}

// Calculates decimal digit count up to a fixed precision.
fn calculateDecimalCount(value: f64) u8 {
    const eps = 1e-12;
    const abs_val = @abs(value);
    const int_part = @floor(abs_val);
    const frac = abs_val - int_part;
    if (frac < eps) return 0;
    var count: u8 = 0;
    var remaining = frac;
    while (count < 6 and remaining > eps) {
        remaining *= 10.0;
        const digit = @floor(remaining);
        remaining -= digit;
        count += 1;
        if (@abs(remaining) < eps) break;
    }
    return count;
}

// Computes dxor for the decimal fraction (Algorithm 2).
fn calculateDxor(dec_fraction: f64, l: u8) f64 {
    // Eq. 3: dxor.dec = v.dec - 2^-l * floor(v.dec / 2^-l)  (при v.dec != 2^-l)
    const step = Math.pow(f64, 2.0, -@as(f64, @floatFromInt(l)));
    const eps = 1e-15;

    if (@abs(dec_fraction - step) < eps) {
        return dec_fraction;
    }

    const t = Math.floor(dec_fraction / step);
    return dec_fraction - step * t;
}

// Encodes an integer with delta compression.
pub fn compressInteger(writer: *BitWriter, int_part: i64, prev_int: i64, index: usize) !void {
    if (index == 1) {
        try writer.writeBits(@as(u64, @bitCast(int_part)), 64);
        return;
    }

    const diff = int_part - prev_int;
    const abs_diff = @abs(diff);

    if (abs_diff <= 1) {
        const code = @as(u2, @intCast(diff + 1));
        try writer.writeBits(code, 2);
    } else {
        // Add prefix 11, as the decoder expects
        try writer.writeBits(@as(u2, 3), 2);

        const sign_bit: u1 = if (diff >= 0) 1 else 0;
        try writer.writeBit(sign_bit);

        const range_bit: u1 = if (abs_diff < 8) 0 else 1;
        try writer.writeBit(range_bit);

        const bits_count: u8 = if (abs_diff < 8) 3 else 16;
        try writer.writeBits(@as(u64, @intCast(abs_diff)), bits_count);
    }
}

// Decodes an integer with delta compression.
fn decompressInteger(reader: *BitReader, prev_int: i64, index: usize) !i64 {
    if (index == 1) {
        const raw = try reader.readBits(64);
        return @as(i64, @bitCast(raw));
    }

    const range = try reader.readBits(2);
    const diff: i64 = blk: {
        if (range != 3) {
            break :blk @as(i64, @intCast(range)) - 1;
        } else {
            const symbol = try reader.readBit(); // 1 -> положительный, 0 -> отрицательный
            const flag = try reader.readBit(); // 0 -> 3 бита, 1 -> 16 бит
            const bits_count = if (flag == 0) @as(u8, 3) else @as(u8, 16);
            const abs_diff = try reader.readBits(bits_count);
            const abs_i64 = @as(i64, @intCast(abs_diff));
            break :blk if (symbol == 1) abs_i64 else -abs_i64;
        }
    };
    return prev_int + diff;
}

// Encodes the decimal fraction using CAMEL rules.
fn compressDecimal(writer: *BitWriter, dec_fraction: f64, l: u8) !void {
    // extended encoding of l (3 bits)
    // try writer.writeBits(@as(u3, @intCast(l)), 3);

    // Algorithm 2: if v_t >= 2^-l then ...
    const threshold = Math.pow(f64, 2.0, -@as(f64, @floatFromInt(l)));
    var dxor: f64 = undefined;

    if (dec_fraction >= threshold) {
        try writer.writeBit(1);
        dxor = calculateDxor(dec_fraction, l);

        // v̂_d(v_t) = v_t.ddec ⊕ dxor(v_t)
        const v_full = 1.0 + dec_fraction;
        const dxor_full = 1.0 + dxor;
        const v_bits = @as(u64, @bitCast(v_full));
        const dxor_bits = @as(u64, @bitCast(dxor_full));
        const vd_hat = v_bits ^ dxor_bits;

        // out.write(v̂_d(v_t) >>> (52 − l), l)
        const shifted = vd_hat >> @as(u6, @intCast(52 - l));
        const center_bits = shifted & ((@as(u64, 1) << @as(u6, @intCast(l))) - 1);
        try writer.writeBits(center_bits, l);
    } else {
        try writer.writeBit(0);
        dxor = dec_fraction;
    }

    // dxor' = dxor * 10^l
    const scale = Math.pow(f64, 10.0, @as(f64, @floatFromInt(l)));
    const dxor_prime = @as(u64, @intFromFloat(@round(dxor * scale)));

    if (l <= 1) {
        const bits_count: u8 = if (l == 0) 1 else 4; // 10^1 = 10 -> 4 bits
        try writer.writeBits(dxor_prime, bits_count);
    } else if (l == 2) {
        const flag_bit: u1 = if (dxor_prime <= 4) 0 else 1;
        try writer.writeBit(flag_bit);
        const bits_count: u8 = if (dxor_prime <= 4) 2 else 5;
        try writer.writeBits(dxor_prime, bits_count);
    } else {
        // max = ceil(log2(2^-l * 10^l)) = ceil(l * log2(5))
        const max_val = @ceil(@as(f64, @floatFromInt(l)) * Math.log2(5.0));
        const max = @as(u64, @intFromFloat(max_val));

        const threshold1 = @as(u64, @intFromFloat(Math.pow(f64, 2.0, @as(f64, @floatFromInt(max)) / 4.0)));
        const threshold2 = @as(u64, @intFromFloat(Math.pow(f64, 2.0, 2.0 * @as(f64, @floatFromInt(max)) / 4.0)));
        const threshold3 = @as(u64, @intFromFloat(Math.pow(f64, 2.0, 3.0 * @as(f64, @floatFromInt(max)) / 4.0)));
        const thresholds = [_]u64{ threshold1, threshold2, threshold3 };

        var index: u8 = 3;
        for (thresholds, 0..) |th, i| {
            if (dxor_prime <= th) {
                index = @as(u8, @intCast(i));
                break;
            }
        }

        try writer.writeBits(index, 2);
        const bits_to_write = (index + 1) * max / 4;
        try writer.writeBits(dxor_prime, @as(u8, @intCast(bits_to_write)));
    }
}

// Decodes the decimal fraction (expects `l` encoded in stream).
fn decompressDecimal(reader: *BitReader) Error!f64 {
    const l = @as(u8, @intCast(reader.readBits(3) catch return Error.CorruptedCompressedData));
    const flag = reader.readBit() catch return Error.CorruptedCompressedData;

    if (flag == 1) {
        const center_bits = reader.readBits(l) catch return Error.CorruptedCompressedData;
        const vd_hat = center_bits << @as(u6, @intCast(52 - l));
        const dxor_prime = try restore(reader, l);
        const scale = Math.pow(f64, 10.0, @as(f64, @floatFromInt(l)));
        const dxor_f64 = @as(f64, @floatFromInt(dxor_prime)) / scale;

        // IE(1 + Restore(l,in)) ⊕ v̂_d(v_t) - 1
        const one_plus = 1.0 + dxor_f64;
        const bits_one_plus = @as(u64, @bitCast(one_plus));
        const xor_result = bits_one_plus ^ vd_hat;
        const reconstructed = @as(f64, @bitCast(xor_result));
        return reconstructed - 1.0;
    } else {
        const dxor_prime = try restore(reader, l);
        const scale = Math.pow(f64, 10.0, @as(f64, @floatFromInt(l)));
        return @as(f64, @floatFromInt(dxor_prime)) / scale;
    }
}

// Restores the variable-length decimal payload.
fn restore(reader: *BitReader, l: u8) Error!u64 {
    if (l <= 1) {
        const bits_count: u8 = if (l == 0) 1 else 4;
        return reader.readBits(bits_count) catch return Error.CorruptedCompressedData;
    } else if (l == 2) {
        const flag = reader.readBit() catch return Error.CorruptedCompressedData;
        const bits_count = if (flag == 0) @as(u8, 2) else @as(u8, 5);
        return reader.readBits(bits_count) catch return Error.CorruptedCompressedData;
    } else {
        const max_val = @ceil(@as(f64, @floatFromInt(l)) * Math.log2(5.0));
        const max = @as(u64, @intFromFloat(max_val));

        const flag = reader.readBits(2) catch return Error.CorruptedCompressedData;
        const index = @as(u8, @intCast(flag));
        const bits_to_read = (index + 1) * max / 4;
        return reader.readBits(@as(u8, @intCast(bits_to_read))) catch return Error.CorruptedCompressedData;
    }
}

// Decodes the decimal fraction given `l` from the caller.
fn decompressDecimalWithL(reader: *BitReader, l: u8) Error!f64 {
    const flag = reader.readBit() catch return Error.CorruptedCompressedData;

    if (flag == 1) {
        const center_bits = reader.readBits(l) catch return Error.CorruptedCompressedData;
        const vd_hat = center_bits << @as(u6, @intCast(52 - l));
        const dxor_prime = try restore(reader, l);
        const scale = Math.pow(f64, 10.0, @as(f64, @floatFromInt(l)));
        const dxor_f64 = @as(f64, @floatFromInt(dxor_prime)) / scale;
        const one_plus_dxor = 1.0 + dxor_f64;
        const bits_one_plus = @as(u64, @bitCast(one_plus_dxor));
        const xor_result = bits_one_plus ^ vd_hat;
        const reconstructed = @as(f64, @bitCast(xor_result));
        return reconstructed - 1.0;
    } else {
        const dxor_prime = try restore(reader, l);
        const scale = Math.pow(f64, 10.0, @as(f64, @floatFromInt(l)));
        return @as(f64, @floatFromInt(dxor_prime)) / scale;
    }
}

// Compresses a series of values using CAMEL.
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

    var writer = BitWriter.init(allocator, compressed_values);

    // write count first (64 bits)
    try writer.writeBits(@as(u64, @intCast(uncompressed_values.len)), 64);

    var prev_int: i64 = 0;
    var index: usize = 1;

    for (uncompressed_values) |v| {
        const parts = splitNumber(v, null);

        if (parts.special) {
            try writer.writeBits(31, 5);
            try writer.writeBit(0);
            try writer.writeBits(parts.raw_bits, 64);
        } else {
            try writer.writeBits(parts.decimal_digits, 5);
            try compressInteger(&writer, parts.integer, prev_int, index);
            prev_int = parts.integer;

            const scale = Math.pow(f64, 10.0, @as(f64, @floatFromInt(parts.decimal_digits)));
            const dec_fraction = @as(f64, @floatFromInt(parts.decimal)) / scale;
            try compressDecimal(&writer, dec_fraction, parts.decimal_digits);
        }
        index += 1;
    }

    try writer.finish();
}

// Decompresses a CAMEL-compressed byte stream.
pub fn decompress(
    allocator: Allocator,
    compressed_values: []const u8,
    decompressed_values: *ArrayList(f64),
) Error!void {
    var reader = BitReader.init(compressed_values);
    const count = reader.readBits(64) catch return Error.CorruptedCompressedData;
    if (count > 1 << 30) return Error.UnsupportedInput;

    var prev_int: i64 = 0;
    var index: usize = 1;

    for (0..@intCast(count)) |_| {
        const l = reader.readBits(5) catch return Error.CorruptedCompressedData;
        const l_u8 = @as(u8, @intCast(l));

        if (l_u8 == 31) {
            const flag = reader.readBit() catch return Error.CorruptedCompressedData;
            if (flag != 0) return Error.UnsupportedInput;
            const raw_bits = reader.readBits(64) catch return Error.CorruptedCompressedData;
            try decompressed_values.append(allocator, @as(f64, @bitCast(raw_bits)));
        } else {
            const int_part = try decompressInteger(&reader, prev_int, index);
            const dec_part = try decompressDecimalWithL(&reader, l_u8);
            const value = @as(f64, @floatFromInt(int_part)) + dec_part;
            try decompressed_values.append(allocator, value);
            prev_int = int_part;
        }
        index += 1;
    }
}

////////////////////////////////////////////////////////////////////////////
//// Tests //////////////////////////////////////////////////////////////////

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

    try compress(allocator, values[0..], &compressed, "{}");

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

test "camel round-trip small integer deltas" {
    const allocator = testing.allocator;

    var values_list = ArrayList(f64).empty;
    defer values_list.deinit(allocator);

    var current: i64 = 100;
    try values_list.append(allocator, @as(f64, @floatFromInt(current)));

    const deltas = [_]i64{ -1, 0, 1, -1, 1, 0, 1, -1, 0 };
    for (deltas) |d| {
        current += d;
        try values_list.append(allocator, @as(f64, @floatFromInt(current)));
    }

    var compressed = ArrayList(u8).empty;
    defer compressed.deinit(allocator);

    try compress(allocator, values_list.items, &compressed, "{}");

    var decompressed = ArrayList(f64).empty;
    defer decompressed.deinit(allocator);

    try decompress(allocator, compressed.items, &decompressed);

    try testing.expectEqual(values_list.items.len, decompressed.items.len);
    try testing.expect(shared_functions.isWithinErrorBound(
        values_list.items,
        decompressed.items,
        1e-9,
    ));
}

test "camel round-trip special values" {
    const allocator = testing.allocator;

    const values = [_]f64{
        Math.nan(f64),
        Math.inf(f64),
        -Math.inf(f64),
    };

    var compressed = ArrayList(u8).empty;
    defer compressed.deinit(allocator);

    try compress(allocator, values[0..], &compressed, "{}");

    var decompressed = ArrayList(f64).empty;
    defer decompressed.deinit(allocator);

    try decompress(allocator, compressed.items, &decompressed);

    try testing.expectEqual(values.len, decompressed.items.len);
    try testing.expect(Math.isNan(decompressed.items[0]));
    try testing.expect(Math.isInf(decompressed.items[1]) and decompressed.items[1] > 0);
    try testing.expect(Math.isInf(decompressed.items[2]) and decompressed.items[2] < 0);
}

test "camel round-trip empty input" {
    const allocator = testing.allocator;

    const values = [_]f64{};

    var compressed = ArrayList(u8).empty;
    defer compressed.deinit(allocator);

    try compress(allocator, values[0..], &compressed, "{}");

    var decompressed = ArrayList(f64).empty;
    defer decompressed.deinit(allocator);

    try decompress(allocator, compressed.items, &decompressed);

    try testing.expectEqual(values.len, decompressed.items.len);
}

test "camel round-trip decimal digit branches" {
    const allocator = testing.allocator;

    const values = [_]f64{
        1.2, // l=1
        -3.4, // l=1 (negative)
        1.23, // l=2
        -9.87, // l=2 (negative)
        1.001, // l=3, dec_fraction < 2^-l
        2.5, // l=1, dec_fraction >= 2^-l
        1.2345, // l=4
        -6.54321, // l=5
    };

    var compressed = ArrayList(u8).empty;
    defer compressed.deinit(allocator);

    try compress(allocator, values[0..], &compressed, "{}");

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

test "camel decompress corrupted input" {
    const allocator = testing.allocator;

    const compressed = [_]u8{}; // empty -> missing count
    var decompressed = ArrayList(f64).empty;
    defer decompressed.deinit(allocator);

    try testing.expectError(
        Error.CorruptedCompressedData,
        decompress(allocator, compressed[0..], &decompressed),
    );
}

test "camel round-trip integer-only, the edge cases and negative zero" {
    const allocator = testing.allocator;

    const values = [_]f64{
        -65535.0, // the largest negative integer difference that fits in 16 bits
        0.0,
        -0.0,
        5.0,
        -7.0,
        65528.0, // maximal integer difference that fits in 16 bits
    };

    var compressed = ArrayList(u8).empty;
    defer compressed.deinit(allocator);

    try compress(allocator, values[0..], &compressed, "{}");

    var decompressed = ArrayList(f64).empty;
    defer decompressed.deinit(allocator);

    try decompress(allocator, compressed.items, &decompressed);

    try testing.expectEqual(values.len, decompressed.items.len);
    try testing.expect(shared_functions.isWithinErrorBound(
        values[0..],
        decompressed.items,
        0.0,
    ));

    // Preserve -0.0 sign bit
    try testing.expect(
        @as(u64, @bitCast(values[1])) == @as(u64, @bitCast(decompressed.items[1])),
    );
}
