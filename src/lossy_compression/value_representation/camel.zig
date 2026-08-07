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

//! Implementation of the Camel floating-point time series compression method.
//! The method is described in:
//! Yao et al., "Camel: Efficient Compression of Floating-Point Time Series", SIGMOD 2024.
//! https://dl.acm.org/doi/epdf/10.1145/3698802
//! This implementation also follows the Java reference implementation published by the paper's
//! authors: https://github.com/yoyo185644/camel.
//!
//! The original paper treats Camel as lossless compression if a low (1-4) decimal precision
//! covers every input value. However, TerseTS accepts arbitrary binary `f64` values and
//! cannot assume that they all satisfy a particular decimal-precision bound. To solve this problem,
//! TerseTS leaves values within the selected precision unchanged and rounds values that exceed it
//! before applying Camel's representation. TerseTS therefore exposes Camel as a lossy value
//! representation method and requires callers to choose `decimal_precision` explicitly, with
//! `decimal_precision` in the range 1...4. Users requiring bitwise losslessness should use other
//! lossless compression methods in the library.
//!
//! Camel has a clear constraint on the integer part of each value: the truncated integer part must
//! fit in an `i64`, and the difference between consecutive integer parts must fit in 16-bit.
//! If that constraint is violated, `compress` returns `Error.UnsupportedInput`. If data contains highly
//! variable integer parts, users can still apply Camel by first transforming the data to a normalized range.
//! Moreover, values within the selected precision can still reconstruct to a neighboring `f64` bit pattern
//! because Camel rebuilds decimal components with binary floating-point arithmetic. The configured
//! decimal value is preserved, but bit-exact recovery is not guaranteed.

const std = @import("std");
const math = std.math;
const mem = std.mem;
const testing = std.testing;
const ArrayList = std.ArrayList;
const Allocator = mem.Allocator;

const tersets = @import("../../tersets.zig");
const configuration = @import("../../configuration.zig");
const shared_functions = @import("../../utilities/shared_functions.zig");
const shared_structs = @import("../../utilities/shared_structs.zig");
const tester = @import("../../tester.zig");

const Error = tersets.Error;

/// DecodedIntegerPart holds the integer part of a decompressed value and its sign. The sign is
/// kept separate so that negative zero can be reconstructed exactly.
const DecodedIntegerPart = struct {
    value: i64,
    is_non_negative: bool,
};

/// Number of bits used to store a value's decimal-place count minus one.
const decimal_place_count_bits: u6 = 2;

/// Largest decimal-place count representable by `decimal_place_count_bits`.
const maximum_encoded_decimal_places: u8 = 1 << decimal_place_count_bits;

/// Largest integer delta representable by Camel's 16-bit unsigned magnitude field.
const maximum_integer_delta: i64 = math.maxInt(u16);

/// Bit widths used by the two integer-delta magnitude encodings.
const narrow_integer_delta_bits: u6 = 3;
const wide_integer_delta_bits: u6 = 16;

/// First magnitude that does not fit in the narrow integer-delta encoding.
const wide_integer_delta_threshold: i64 = 1 << narrow_integer_delta_bits;

/// An impossible integer encoding that terminates the stream. Real extended deltas have magnitude
/// at least two, so marker `11` with zero sign, width, and magnitude cannot represent a value.
const end_marker_bits: u8 = 0b0110_0000;

/// Number of explicitly stored fraction bits in an IEEE-754 `f64`.
const f64_fraction_bits: u8 = 52;

/// Exclusive magnitude limit for converting a truncated `f64` to `i64` (`2^63`).
const maximum_i64_magnitude_exclusive: f64 = @floatFromInt(@as(u64, 1) << 63);

/// Tolerance used when deciding whether a scaled fractional value is effectively an integer.
const decimal_integer_epsilon: f64 = 0.0000001;

/// Compress non-empty `uncompressed_values` into `compressed_values` using Camel. `allocator` backs
/// the configuration parser and output growth. `method_configuration` must contain
/// `decimal_precision` in the range 1...4, for example `{ "decimal_precision": 4 }`. On success,
/// `compressed_values` contains `[first_value: f64][encoded value bits...][end marker]`. The first
/// value is stored verbatim. Every subsequent value whose detected decimal-place count exceeds the
/// configuration is rounded before Camel encodes it; already-conforming values are left untouched.
/// Values whose consecutive integer parts differ by more than `maximum_integer_delta`, or
/// whose rounded integer parts do not fit in `i64`, return `Error.UnsupportedInput`.
pub fn compress(
    allocator: Allocator,
    uncompressed_values: []const f64,
    compressed_values: *ArrayList(u8),
    method_configuration: []const u8,
) Error!void {
    const parsed_configuration = try configuration.parse(
        allocator,
        configuration.DecimalPrecision,
        method_configuration,
    );
    if (parsed_configuration.decimal_precision > maximum_encoded_decimal_places) {
        return Error.InvalidConfiguration;
    }

    // Later integer parts are delta-encoded, so the first value is stored raw as their baseline.
    const first_value = uncompressed_values[0];
    if (!fitsIntegerPart(first_value)) return Error.UnsupportedInput;
    try shared_functions.appendValue(allocator, f64, first_value, compressed_values);

    var bit_writer = try shared_structs.BulkBitWriter.init(allocator, compressed_values);
    var previous_integer = integerPart(first_value);

    for (uncompressed_values[1..]) |value| {
        const detected_decimal_place_count = decimalPlaceCount(
            value,
            parsed_configuration.decimal_precision,
        );
        const rounded_value = if (detected_decimal_place_count <=
            parsed_configuration.decimal_precision)
            value
        else
            roundToDecimalPrecision(value, parsed_configuration.decimal_precision);
        if (!fitsIntegerPart(rounded_value)) return Error.UnsupportedInput;

        const integer_part = integerPart(rounded_value);
        const integer_delta_result = @subWithOverflow(integer_part, previous_integer);
        if (integer_delta_result[1] != 0 or
            @abs(integer_delta_result[0]) > maximum_integer_delta)
        {
            return Error.UnsupportedInput;
        }
        const integer_delta = integer_delta_result[0];

        const is_non_negative = !math.signbit(rounded_value);
        const decimal_place_count = @min(
            detected_decimal_place_count,
            parsed_configuration.decimal_precision,
        );

        try compressIntegerPart(integer_delta, is_non_negative, &bit_writer);
        // Counts 1...4 are stored as 0...3 in the two-bit field.
        try bit_writer.writeBits(
            @as(u2, @intCast(decimal_place_count - 1)),
            decimal_place_count_bits,
        );
        try compressDecimalPart(
            @abs(fractionalPart(rounded_value)),
            decimal_place_count,
            &bit_writer,
        );
        previous_integer = integer_part;
    }

    // The end marker tells the decoder where to stop; flushed padding bits are never read.
    try writeEndMarker(&bit_writer);
    try bit_writer.flushBits();
}

/// Decompress a Camel-encoded `compressed_values` stream into `decompressed_values`. `allocator`
/// grows `decompressed_values` as values are recovered. Malformed or truncated streams return
/// `Error.CorruptedCompressedData`.
pub fn decompress(
    allocator: Allocator,
    compressed_values: []const u8,
    decompressed_values: *ArrayList(f64),
) Error!void {
    var offset: usize = 0;
    if (compressed_values.len < @sizeOf(f64)) return Error.CorruptedCompressedData;

    const first_value = try shared_functions.readOffsetValue(f64, compressed_values, &offset);
    if (!fitsIntegerPart(first_value)) return Error.CorruptedCompressedData;
    try decompressed_values.append(allocator, first_value);

    var previous_integer = integerPart(first_value);
    var bit_reader = shared_structs.BulkBitReader.init(compressed_values[offset..]);

    while (true) {
        const decoded_integer_part = try decompressIntegerPart(previous_integer, &bit_reader) orelse break;
        if (decoded_integer_part.value == math.minInt(i64)) {
            return Error.CorruptedCompressedData;
        }

        const decimal_place_count: u8 = @as(
            u8,
            bit_reader.readBitsNoEof(u2, decimal_place_count_bits) catch
                return Error.CorruptedCompressedData,
        ) + 1;

        const fractional_magnitude = try decompressDecimalPart(decimal_place_count, &bit_reader);
        var reconstructed_value =
            @abs(@as(f64, @floatFromInt(decoded_integer_part.value))) +
            fractional_magnitude;

        if (!decoded_integer_part.is_non_negative) reconstructed_value = -reconstructed_value;

        try decompressed_values.append(allocator, reconstructed_value);
        previous_integer = decoded_integer_part.value;
    }
}

/// Returns `true` when `value` is finite and its truncated integer part fits in an `i64`.
fn fitsIntegerPart(value: f64) bool {
    if (math.isNan(value) or math.isInf(value)) return false;
    return @abs(@trunc(value)) < maximum_i64_magnitude_exclusive;
}

/// Returns the truncated integer part of `value` as an `i64`. The caller must first verify
/// `fitsIntegerPart(value)`.
fn integerPart(value: f64) i64 {
    return @intFromFloat(@trunc(value));
}

/// Returns the signed fractional remainder of `value` after truncation toward zero.
fn fractionalPart(value: f64) f64 {
    return value - @trunc(value);
}

/// Round `value` to the nearest number with `decimal_precision` places after the decimal point.
fn roundToDecimalPrecision(value: f64, decimal_precision: u8) f64 {
    const decimal_scale = math.pow(
        f64,
        10.0,
        @as(f64, @floatFromInt(decimal_precision)),
    );
    return @round(value * decimal_scale) / decimal_scale;
}

/// Return the smallest count in `1...maximum_decimal_places` that places `value` on an integer
/// decimal grid, or `maximum_decimal_places + 1` when the value needs rounding.
fn decimalPlaceCount(value: f64, maximum_decimal_places: u8) u8 {
    var decimal_scale: f64 = 10.0;
    var decimal_place_count: u8 = 1;
    const abs_value = @abs(value);
    var scaled_value = abs_value * decimal_scale;
    while (decimal_place_count <= maximum_decimal_places and @abs(scaled_value - @round(scaled_value)) > decimal_integer_epsilon) {
        decimal_scale *= 10.0;
        decimal_place_count += 1;
        scaled_value = abs_value * decimal_scale;
    }

    return decimal_place_count;
}

/// Write `is_non_negative` and `integer_delta` to `writer`. Deltas `-1`, `0`, and `1` use a two-bit
/// code; larger deltas use marker `11`, a sign bit, a width bit, and a 3- or 16-bit magnitude.
fn compressIntegerPart(
    integer_delta: i64,
    is_non_negative: bool,
    writer: *shared_structs.BulkBitWriter,
) !void {
    try writer.writeBits(@as(u1, @intFromBool(is_non_negative)), 1);

    const delta_magnitude = @abs(integer_delta);
    if (delta_magnitude <= 1) {
        // Map deltas -1, 0, and 1 to codes 0, 1, and 2 respectively.
        const delta_code: u2 = @intCast(integer_delta + 1);
        try writer.writeBits(delta_code, 2);
    } else {
        // Marker `11` selects the sign + width + magnitude representation.
        try writer.writeBits(@as(u2, 0b11), 2);
        try writer.writeBits(@as(u1, @intFromBool(integer_delta >= 0)), 1);

        const uses_wide_magnitude = delta_magnitude >= wide_integer_delta_threshold;
        const magnitude_bit_count: u6 = if (uses_wide_magnitude)
            wide_integer_delta_bits
        else
            narrow_integer_delta_bits;
        try writer.writeBits(@as(u1, @intFromBool(uses_wide_magnitude)), 1);
        try writer.writeBits(@as(u64, @intCast(delta_magnitude)), magnitude_bit_count);
    }
}

/// Write `fractional_magnitude` to `writer` using `decimal_place_count`. A marker selects Camel's
/// XOR-reference representation or a directly quantized representation.
fn compressDecimalPart(
    fractional_magnitude: f64,
    decimal_place_count: u8,
    writer: *shared_structs.BulkBitWriter,
) !void {
    const binary_step = math.pow(
        f64,
        2.0,
        -@as(f64, @floatFromInt(decimal_place_count)),
    );
    const decimal_scale = math.pow(
        f64,
        10.0,
        @as(f64, @floatFromInt(decimal_place_count)),
    );

    if (fractional_magnitude >= binary_step) {
        // Marker 1: write the significant XOR-center bits and the quantized XOR reference.
        try writer.writeBits(@as(u1, 1), 1);
        const raw_xor_reference = fractional_magnitude -
            binary_step * @floor(fractional_magnitude / binary_step);
        const quantized_xor_reference: u64 = @intFromFloat(@round(
            raw_xor_reference * decimal_scale,
        ));
        // Reconstruct the reference after quantization so encoder and decoder XOR against exactly
        // the same floating-point value (Theorem 3.1 of the paper).
        const xor_reference = @as(f64, @floatFromInt(quantized_xor_reference)) / decimal_scale;
        const fractional_xor = floatToBits(1.0 + fractional_magnitude) ^
            floatToBits(1.0 + xor_reference);
        const center_bit_shift: u6 = @intCast(f64_fraction_bits - decimal_place_count);
        const center_bit_mask = (@as(u64, 1) << @as(u6, @intCast(decimal_place_count))) - 1;
        const xor_center_bits = (fractional_xor >> center_bit_shift) & center_bit_mask;

        try writer.writeBits(xor_center_bits, @as(u6, @intCast(decimal_place_count)));
        try writeQuantizedFraction(
            quantized_xor_reference,
            decimal_place_count,
            writer,
        );
    } else {
        // Marker 0: the quantized fractional magnitude is sufficient by itself.
        try writer.writeBits(@as(u1, 0), 1);
        const quantized_magnitude: u64 = @intFromFloat(@round(
            fractional_magnitude * decimal_scale,
        ));
        try writeQuantizedFraction(quantized_magnitude, decimal_place_count, writer);
    }
}

/// Reinterpret `value` as its raw IEEE-754 bits.
fn floatToBits(value: f64) u64 {
    return @as(u64, @bitCast(value));
}

/// Write a quantized fractional value using the prefix and value-width buckets from the Java
/// reference implementation's `mValueBits` table. The bucket table depends on the decimal-place
/// count, which is already encoded in the stream.
fn writeQuantizedFraction(
    quantized_value: u64,
    decimal_place_count: u8,
    writer: *shared_structs.BulkBitWriter,
) !void {
    switch (decimal_place_count) {
        1 => {
            try writer.writeBits(quantized_value, 3);
        },
        2 => {
            if (quantized_value < 8) {
                try writer.writeBits(@as(u1, 0), 1);
                try writer.writeBits(quantized_value, 3);
            } else {
                try writer.writeBits(@as(u1, 1), 1);
                try writer.writeBits(quantized_value, 5);
            }
        },
        3 => {
            if (quantized_value < 2) {
                try writer.writeBits(@as(u2, 0), 2);
                try writer.writeBits(quantized_value, 1);
            } else if (quantized_value < 8) {
                try writer.writeBits(@as(u2, 1), 2);
                try writer.writeBits(quantized_value, 3);
            } else if (quantized_value < 32) {
                try writer.writeBits(@as(u2, 2), 2);
                try writer.writeBits(quantized_value, 5);
            } else {
                try writer.writeBits(@as(u2, 3), 2);
                try writer.writeBits(quantized_value, 7);
            }
        },
        4 => {
            if (quantized_value < 16) {
                try writer.writeBits(@as(u2, 0), 2);
                try writer.writeBits(quantized_value, 4);
            } else if (quantized_value < 64) {
                try writer.writeBits(@as(u2, 1), 2);
                try writer.writeBits(quantized_value, 6);
            } else if (quantized_value < 256) {
                try writer.writeBits(@as(u2, 2), 2);
                try writer.writeBits(quantized_value, 8);
            } else {
                try writer.writeBits(@as(u2, 3), 2);
                try writer.writeBits(quantized_value, 10);
            }
        },
        else => return Error.UnsupportedInput,
    }
}

/// Write the impossible extended-delta encoding reserved as Camel's end-of-stream marker.
fn writeEndMarker(writer: *shared_structs.BulkBitWriter) Error!void {
    try writer.writeBits(end_marker_bits, @bitSizeOf(@TypeOf(end_marker_bits)));
}

/// Reinterpret `bits` as an IEEE-754 `f64`.
fn bitsToFloat(bits: u64) f64 {
    return @as(f64, @bitCast(bits));
}

/// Reads and decodes the integer part of one value from `reader`, inverting `compressIntegerPart`.
/// The returned sign is kept separate so negative zero can be reconstructed exactly.
fn decompressIntegerPart(
    previous_integer: i64,
    reader: *shared_structs.BulkBitReader,
) Error!?DecodedIntegerPart {
    const encoded_sign = reader.readBitsNoEof(u1, 1) catch
        return Error.CorruptedCompressedData;
    const delta_code = reader.readBitsNoEof(u2, 2) catch
        return Error.CorruptedCompressedData;
    const integer_delta: i64 = switch (delta_code) {
        0, 1, 2 => @as(i64, @intCast(delta_code)) - 1,
        3 => blk: {
            const encoded_delta_sign = reader.readBitsNoEof(u1, 1) catch
                return Error.CorruptedCompressedData;
            const encoded_magnitude_width = reader.readBitsNoEof(u1, 1) catch
                return Error.CorruptedCompressedData;
            const uses_wide_magnitude = encoded_magnitude_width == 1;
            const magnitude_bit_count: u6 = if (uses_wide_magnitude)
                wide_integer_delta_bits
            else
                narrow_integer_delta_bits;
            const delta_magnitude = reader.readBitsNoEof(u64, magnitude_bit_count) catch
                return Error.CorruptedCompressedData;

            if (encoded_sign == 0 and
                encoded_delta_sign == 0 and
                encoded_magnitude_width == 0 and
                delta_magnitude == 0)
            {
                return null;
            }

            // Marker `11` is only emitted for magnitudes greater than one, and the wide form is
            // only emitted when the magnitude no longer fits in the narrow field.
            if (delta_magnitude <= 1 or
                (uses_wide_magnitude and delta_magnitude < wide_integer_delta_threshold))
            {
                return Error.CorruptedCompressedData;
            }

            var decoded_delta: i64 = @intCast(delta_magnitude);
            if (encoded_delta_sign == 0) decoded_delta = -decoded_delta;
            break :blk decoded_delta;
        },
    };

    const decoded_integer = @addWithOverflow(previous_integer, integer_delta);
    if (decoded_integer[1] != 0) return Error.CorruptedCompressedData;

    return .{
        .value = decoded_integer[0],
        .is_non_negative = encoded_sign == 1,
    };
}

/// Decompresses the fractional magnitude produced by `compressDecimalPart`, inverting the XOR
/// scheme. Returns a non-negative value; the caller applies the separately encoded sign.
fn decompressDecimalPart(
    decimal_place_count: u8,
    reader: *shared_structs.BulkBitReader,
) Error!f64 {
    const uses_xor_reference = reader.readBitsNoEof(u1, 1) catch
        return Error.CorruptedCompressedData;
    const decimal_scale = math.pow(
        f64,
        10.0,
        @as(f64, @floatFromInt(decimal_place_count)),
    );

    if (uses_xor_reference == 0) {
        const quantized_magnitude = try readQuantizedFraction(decimal_place_count, reader);
        return @as(f64, @floatFromInt(quantized_magnitude)) / decimal_scale;
    }

    const xor_center_bits = reader.readBitsNoEof(
        u64,
        @as(u6, @intCast(decimal_place_count)),
    ) catch return Error.CorruptedCompressedData;
    const center_bit_shift: u6 = @intCast(f64_fraction_bits - decimal_place_count);
    const positioned_xor_bits = xor_center_bits << center_bit_shift;
    const quantized_xor_reference = try readQuantizedFraction(decimal_place_count, reader);
    const xor_reference = @as(f64, @floatFromInt(quantized_xor_reference)) / decimal_scale;
    const fractional_magnitude = bitsToFloat(
        floatToBits(1.0 + xor_reference) ^ positioned_xor_bits,
    ) - 1.0;

    // Snap to the encoded decimal-place count to remove residual XOR arithmetic noise.
    return @round(fractional_magnitude * decimal_scale) / decimal_scale;
}

/// Read a quantized fractional value written by `writeQuantizedFraction`. The `decimal_place_count`
/// specifies the bucket table used to encode the value. The `reader` writes the value to a `u64` and returns it.
fn readQuantizedFraction(
    decimal_place_count: u8,
    reader: *shared_structs.BulkBitReader,
) Error!u64 {
    switch (decimal_place_count) {
        1 => return reader.readBitsNoEof(u64, 3) catch
            return Error.CorruptedCompressedData,
        2 => {
            const bucket = reader.readBitsNoEof(u1, 1) catch
                return Error.CorruptedCompressedData;
            const value_bit_count: u6 = if (bucket == 0) 3 else 5;
            return reader.readBitsNoEof(u64, value_bit_count) catch
                return Error.CorruptedCompressedData;
        },
        3 => {
            const bucket = reader.readBitsNoEof(u2, 2) catch
                return Error.CorruptedCompressedData;
            const value_bit_count: u6 = switch (bucket) {
                0 => 1,
                1 => 3,
                2 => 5,
                3 => 7,
            };
            return reader.readBitsNoEof(u64, value_bit_count) catch
                return Error.CorruptedCompressedData;
        },
        4 => {
            const bucket = reader.readBitsNoEof(u2, 2) catch
                return Error.CorruptedCompressedData;
            const value_bit_count: u6 = switch (bucket) {
                0 => 4,
                1 => 6,
                2 => 8,
                3 => 10,
            };
            return reader.readBitsNoEof(u64, value_bit_count) catch
                return Error.CorruptedCompressedData;
        },
        else => return Error.CorruptedCompressedData,
    }
}

test "camel detects decimal places only within the configured precision" {
    try testing.expectEqual(@as(u8, 1), decimalPlaceCount(12.0, 4));
    try testing.expectEqual(@as(u8, 1), decimalPlaceCount(1.2, 4));
    try testing.expectEqual(@as(u8, 2), decimalPlaceCount(-4.56, 4));
    try testing.expectEqual(@as(u8, 4), decimalPlaceCount(1.2345, 4));
    try testing.expectEqual(@as(u8, 2), decimalPlaceCount(1.23, 1));
    try testing.expectEqual(@as(u8, 5), decimalPlaceCount(1.23456, 4));
}

test "camel does roundtrips on representative integer and decimal encodings" {
    const maximum_delta: f64 = @floatFromInt(maximum_integer_delta);
    const cases = [_]struct {
        values: []const f64,
        decimal_precision: u8,
    }{
        .{ .values = &[_]f64{ 7.25, 7.25, 7.25, 7.25 }, .decimal_precision = 2 },
        .{ .values = &[_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 }, .decimal_precision = 1 },
        .{ .values = &[_]f64{ 1.36, 1.11 }, .decimal_precision = 2 },
        .{ .values = &[_]f64{ 1.4276, 1.0526 }, .decimal_precision = 4 },
        .{ .values = &[_]f64{ -0.0, 0.0, -0.0, 1.0, -0.0 }, .decimal_precision = 1 },
        .{ .values = &[_]f64{ 0.0, 0.3 }, .decimal_precision = 1 },
        .{
            .values = &[_]f64{ -maximum_delta, 0.0, 5.0, -7.0, maximum_delta - 7.0 },
            .decimal_precision = 1,
        },
    };

    for (cases) |case| {
        var configuration_buffer: [32]u8 = undefined;
        const method_configuration = try std.fmt.bufPrint(
            &configuration_buffer,
            "{{\"decimal_precision\": {d}}}",
            .{case.decimal_precision},
        );
        try tester.expectExactRoundTrip(
            testing.allocator,
            compress,
            decompress,
            case.values,
            method_configuration,
        );
    }
}

test "camel preserves random values on configured decimal grids" {
    const random = tester.getDefaultRandomGenerator();

    var generated_values = ArrayList(f64).empty;
    defer generated_values.deinit(testing.allocator);
    try generated_values.appendSlice(testing.allocator, &[_]f64{ 0.0, -4.56 });
    // A range narrower than 65,535 keeps every possible consecutive integer delta encodable.
    try tester.generateBoundedRandomValues(
        testing.allocator,
        &generated_values,
        -30_000.0,
        30_000.0,
        random,
    );

    var rounded_values = ArrayList(f64).empty;
    defer rounded_values.deinit(testing.allocator);
    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(testing.allocator);
    var decompressed_values = ArrayList(f64).empty;
    defer decompressed_values.deinit(testing.allocator);

    for (1..maximum_encoded_decimal_places + 1) |precision_index| {
        const decimal_precision: u8 = @intCast(precision_index);
        rounded_values.clearRetainingCapacity();
        compressed_values.clearRetainingCapacity();
        decompressed_values.clearRetainingCapacity();

        for (generated_values.items) |value| {
            try rounded_values.append(
                testing.allocator,
                roundToDecimalPrecision(value, decimal_precision),
            );
        }

        var configuration_buffer: [32]u8 = undefined;
        const method_configuration = try std.fmt.bufPrint(
            &configuration_buffer,
            "{{\"decimal_precision\": {d}}}",
            .{decimal_precision},
        );
        try compress(
            testing.allocator,
            rounded_values.items,
            &compressed_values,
            method_configuration,
        );
        try decompress(testing.allocator, compressed_values.items, &decompressed_values);

        try testing.expectEqual(rounded_values.items.len, decompressed_values.items.len);
        const decimal_scale = math.pow(
            f64,
            10.0,
            @as(f64, @floatFromInt(decimal_precision)),
        );
        for (rounded_values.items, decompressed_values.items) |original, reconstructed| {
            const original_decimal_units: i64 = @intFromFloat(@round(original * decimal_scale));
            const reconstructed_decimal_units: i64 = @intFromFloat(@round(
                reconstructed * decimal_scale,
            ));
            try testing.expectEqual(original_decimal_units, reconstructed_decimal_units);

            const floating_point_tolerance = @max(
                @as(f64, 1.0),
                @max(@abs(original), @abs(reconstructed)),
            ) * math.floatEps(f64);
            try testing.expectApproxEqAbs(original, reconstructed, floating_point_tolerance);
        }
    }
}

test "camel random arbitrary values use the correct precision error bound" {
    var uncompressed_values = ArrayList(f64).empty;
    defer uncompressed_values.deinit(testing.allocator);

    const random = tester.getDefaultRandomGenerator();
    const mean_value = tester.generateBoundedRandomValue(f64, -1e10, 1e10, random);
    // Camel stores each consecutive integer-part difference in 16 bits. Keeping the complete
    // random window narrower than 65,535 guarantees that every generated sequence is supported,
    // regardless of the order in which its values are generated.
    const half_span = tester.generateBoundedRandomValue(f64, 1e3, 30_000.0, random);

    try tester.generateBoundedRandomValues(
        testing.allocator,
        &uncompressed_values,
        mean_value - half_span,
        mean_value + half_span,
        random,
    );

    for (1..maximum_encoded_decimal_places + 1) |precision_index| {
        const decimal_precision: u8 = @intCast(precision_index);
        var configuration_buffer: [32]u8 = undefined;
        const method_configuration = try std.fmt.bufPrint(
            &configuration_buffer,
            "{{\"decimal_precision\": {d}}}",
            .{decimal_precision},
        );

        var compressed_values = ArrayList(u8).empty;
        defer compressed_values.deinit(testing.allocator);
        try compress(
            testing.allocator,
            uncompressed_values.items,
            &compressed_values,
            method_configuration,
        );

        var decompressed_values = ArrayList(f64).empty;
        defer decompressed_values.deinit(testing.allocator);
        try decompress(testing.allocator, compressed_values.items, &decompressed_values);

        try testing.expectEqual(uncompressed_values.items.len, decompressed_values.items.len);

        // Rounding to p decimal places selects the nearest multiple of 10^-p. Therefore, the
        // greatest possible distance between the original and rounded values is half that step:
        //
        //   decimal_precision = 4
        //   quantization step  = 0.0001
        //   maximum error      = 0.00005
        //
        // For example, 1.23456 becomes 1.2346 and differs by 0.00004, which is within the bound.
        const decimal_quantization_step = math.pow(
            f64,
            10.0,
            -@as(f64, @floatFromInt(decimal_precision)),
        );
        const maximum_decimal_rounding_error = decimal_quantization_step / 2.0;

        for (uncompressed_values.items, decompressed_values.items, 0..) |original, reconstructed, index| {
            // Decimal values and the arithmetic above are represented in binary by f64. Add one
            // magnitude-scaled machine epsilon so those representation effects are not mistaken
            // for a violation of Camel's decimal-rounding guarantee.
            const floating_point_slack = @max(
                @as(f64, 1.0),
                @max(@abs(original), @abs(reconstructed)),
            ) * math.floatEps(f64);
            // The first value is stored verbatim. Later values receive the half-step allowance only
            // when they exceed the configured precision and Camel intentionally quantizes them.
            // Already-conforming values must pass with floating-point reconstruction slack alone.
            const was_quantized = index != 0 and
                decimalPlaceCount(original, decimal_precision) > decimal_precision;
            const intentional_rounding_error = if (was_quantized)
                maximum_decimal_rounding_error
            else
                0.0;
            const reconstruction_error = @abs(original - reconstructed);
            const allowed_error = intentional_rounding_error + floating_point_slack;
            if (reconstruction_error > allowed_error) {
                try testing.expectFmt(
                    "",
                    "precision {}, quantized {}, original {}, reconstructed {}, error {}, allowed {}",
                    .{
                        decimal_precision,
                        was_quantized,
                        original,
                        reconstructed,
                        reconstruction_error,
                        allowed_error,
                    },
                );
            }
        }
    }
}

test "camel requires decimal precision between one and four" {
    const uncompressed_values = &[_]f64{ 0.0, 1.2345 };
    const invalid_configurations = [_][]const u8{
        "{}",
        "{\"decimal_precision\": 0}",
        "{\"decimal_precision\": 5}",
        "{\"target_precision\": 4}",
    };

    for (invalid_configurations) |method_configuration| {
        var compressed_values = ArrayList(u8).empty;
        defer compressed_values.deinit(testing.allocator);
        try testing.expectError(
            Error.InvalidConfiguration,
            compress(
                testing.allocator,
                uncompressed_values,
                &compressed_values,
                method_configuration,
            ),
        );
    }

    for ([_]u8{ 1, maximum_encoded_decimal_places }) |decimal_precision| {
        var configuration_buffer: [32]u8 = undefined;
        const method_configuration = try std.fmt.bufPrint(
            &configuration_buffer,
            "{{\"decimal_precision\": {d}}}",
            .{decimal_precision},
        );
        var compressed_values = ArrayList(u8).empty;
        defer compressed_values.deinit(testing.allocator);
        try compress(
            testing.allocator,
            uncompressed_values,
            &compressed_values,
            method_configuration,
        );
    }
}

test "camel rejects unsupported floating-point values in any position" {
    const unsupported_values = [_]f64{
        math.nan(f64),
        math.inf(f64),
        -math.inf(f64),
        math.floatMax(f64),
        -math.floatMax(f64),
    };

    for (unsupported_values) |unsupported_value| {
        const inputs = [_][2]f64{
            .{ unsupported_value, 1.0 },
            .{ 1.0, unsupported_value },
        };
        for (inputs) |input| {
            var compressed_values = ArrayList(u8).empty;
            defer compressed_values.deinit(testing.allocator);

            try testing.expectError(
                Error.UnsupportedInput,
                compress(
                    testing.allocator,
                    &input,
                    &compressed_values,
                    "{\"decimal_precision\": 4}",
                ),
            );
        }
    }
}

test "camel cannot compress values with large integer differences" {
    const first_unsupported_delta: f64 = @floatFromInt(maximum_integer_delta + 1);
    const uncompressed_values = &[_]f64{ 0.0, first_unsupported_delta };
    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(testing.allocator);

    try testing.expectError(
        Error.UnsupportedInput,
        compress(
            testing.allocator,
            uncompressed_values,
            &compressed_values,
            "{\"decimal_precision\": 4}",
        ),
    );
}
