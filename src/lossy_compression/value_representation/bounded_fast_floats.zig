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

//! Implementation of the method BUFF (BoUnded Fast Float) from the paper
//! "Liu, C., Jiang, H., Paparrizos, J., & Elmore, A. J."
//! Decomposed bounded floats for fast compression and queries.
//! Proceedings of the VLDB Endowment, 14(11), 2586-2598 (2021).
//! https://doi.org/10.14778/3476249.3476305.
//!
//! In this file we implement two variants of the BUFF method:
//! `BitPackedBUFF` and `BytePackedBUFF`.
//! The `BitPackedBUFF` method packs the fixed-point representation
//! of the floating-point values into a compact bit-level format,
//! while the `BytePackedBUFF` method aligns the fixed-point representation
//! to byte boundaries for easier storage and retrieval. More details about
//! the byte representation can be found in the paper:
//! https://doi.org/10.1145/2723372.2747642.
//!
//! Both variants have similar compression ratio performance but the `BytePackedBUFF`
//! variant should offer faster compression and decompression speeds. In the original
//! paper the authors only implement the `BytePackedBUFF` variant. But Section 3.2 lays
//! the foundation for the `BitPackedBUFF` variant which we implement here as well.

const std = @import("std");
const math = std.math;
const io = std.io;
const testing = std.testing;
const ArrayList = std.ArrayList;
const Allocator = std.mem.Allocator;

const tersets = @import("../../tersets.zig");
const shared_functions = @import("../../utilities/shared_functions.zig");
const shared_structs = @import("../../utilities/shared_structs.zig");
const configuration = @import("../../configuration.zig");
const tester = @import("../../tester.zig");

const Error = tersets.Error;
const Method = tersets.Method;

/// Decimal precision lookup table from Table 2 in the paper. Each entry corresponds to the
/// number of bits needed in the fractional part to achieve the specified decimal digit precision.
/// Index 0 corresponds to 1 decimal digit precision, index 1 to 2 decimal digits, and so on.
/// Up to 10 decimal digits of precision are supported.
const decimal_precision_lookup: [10]u6 = .{ 5, 8, 11, 15, 18, 21, 25, 28, 31, 35 };

/// Normalization factor for converting fractional part to f64.
const normalization_factor: f64 = math.pow(f64, 2.0, 53.0);

/// f64 decomposition constants.
const mantissa_bits: u6 = 52; // Mantissa bits.
const total_fractional_bits: u6 = 53; // 53 bits for 1.F.
const exponent_bias: i32 = 1023; // Bias for f64 exponent.

/// Fixed-point representation structure. The structure holds the `sign`, `integer_part`,
/// and `fractional_part` of the fixed-point number.
const FixedPointRepresentation = struct {
    sign: u1, // 0 or 1.
    integer_part: u64, // Integer bits as an integer.
    fractional_part: u64, // Fractional bits as an integer.
};

/// Compresses an array of f64 `uncompressed_values` using the BitPackedBUFF variant of the BUFF
/// method with the specified `method_configuration`. The compressed data is stored in
/// `compressed_values`. The function utilizes an `allocator` for memory management during the
/// compression process. This variant of BUFF uses bit-packing to store the fixed-point
/// representations of the floating-point values directly. The integer parts are delta and zigzag
/// encoded before bit-packing. The fractional part is truncated based on the decimal precision
/// before storing the fixed-point representations. The `method_configuration` should include the
/// decimal precision for compression. The function returns an error if the configuration is invalid
/// or if the input values contain unsupported values (e.g., NaN, infinity).
pub fn compressBitPackedBUFF(
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

    if (parsed_configuration.decimal_precision > 10) {
        // Limiting to 10 decimal digits of precision based on Table 2.
        // Precision 11 and above do not provide significant
        // compression benefits compared to the overhead of storing them.
        return Error.InvalidConfiguration;
    }

    if (uncompressed_values.len >= math.maxInt(u32)) {
        // We use u32 to store the number of values in the header, so we need to ensure that we do not exceed that limit.
        // Seems like a reasonable limit for most use-cases and allows us to keep the header size smaller.
        // Still, we check this explicitly to provide a clear error message instead of silently truncating the number of values.
        return Error.UnsupportedInput;
    }

    var fixed_point_representation_array =
        try ArrayList(FixedPointRepresentation).initCapacity(
            allocator,
            uncompressed_values.len,
        );
    defer fixed_point_representation_array.deinit(allocator);

    // Track minimum integer part for optimization explained in Section 3.1.2.
    // Basically, we can reduce the integer parts by the minimum integer part to save bits when encoding.
    var minimum_integer_part: u64 = std.math.maxInt(u64);

    const decoded_decimal_precision: u6 =
        decimal_precision_lookup[parsed_configuration.decimal_precision - 1];

    // Number of bits to drop from the fractional part based on decimal precision.
    const bits_to_drop: u6 = 53 - decoded_decimal_precision;

    // Decompose each value into a fixed point representation. Truncate the fractional
    // part based on the decimal precision and update the minimum integer part.
    for (uncompressed_values) |value| {
        var fixed_point_representation = try decomposeF64ToFixedPointRepresentation(value);
        // Apply the decimal precision by truncating the fractional part.
        const truncated_fractional_part: u64 = fixed_point_representation.fractional_part >> bits_to_drop;
        fixed_point_representation.fractional_part = truncated_fractional_part;

        try fixed_point_representation_array.append(allocator, fixed_point_representation);

        if (fixed_point_representation.integer_part < minimum_integer_part) {
            minimum_integer_part = fixed_point_representation.integer_part;
        }
    }

    // Reduce the integer parts by the minimum integer part.
    for (fixed_point_representation_array.items) |*fixed_point_representation| {
        fixed_point_representation.integer_part -= minimum_integer_part;
    }

    var delta_encoded_integer_parts: ArrayList(u64) = ArrayList(u64).empty;
    defer delta_encoded_integer_parts.deinit(allocator);

    // Apply delta and zigzag encoding for later bitpacking the integer parts.
    // Find the maximum number of bits needed to encode the integer part.
    // TerseTS API ensures that the number of values is at least 2, so the for loop is safe.
    var maximum_number_of_bits_needed: u8 = 0;
    for (1..fixed_point_representation_array.items.len) |index| {
        const current_element: i64 = @intCast(fixed_point_representation_array.items[index].integer_part);
        const previous_element: i64 = @intCast(fixed_point_representation_array.items[index - 1].integer_part);

        const delta_integer_part: i64 = current_element - previous_element;

        // Apply zigzag encoding to negative deltas.
        const encoded_delta_integer_part = shared_functions.encodeZigZag(delta_integer_part);
        try delta_encoded_integer_parts.append(allocator, encoded_delta_integer_part);

        maximum_number_of_bits_needed = @max(
            maximum_number_of_bits_needed,
            shared_functions.bitsNeededUnsigned(encoded_delta_integer_part),
        );
    }

    // Ensure we never use a bit-width of 0 when there are deltas to encode.
    // A bit-width of 0 would mean we cannot encode any information about the integer part changes.
    // However, even if all deltas are 0, we still need at least 1 bit to represent that information.
    if (delta_encoded_integer_parts.items.len > 0 and maximum_number_of_bits_needed == 0) {
        maximum_number_of_bits_needed = 1;
    }

    // Prepare the header information:
    // 1) Decoded decimal precision (u8).
    // 2) Maximum number of bits needed for integer part (u8).
    // 3) Number of values (u32).
    // 4) Minimum integer part (u64).
    // 5) First value (fixed-point representation).
    // Total header size = 1 + 1 + 4 + 8 + 8 = 22 bytes.
    try shared_functions.appendValue(
        allocator,
        u8,
        decoded_decimal_precision,
        compressed_values,
    );

    try shared_functions.appendValue(
        allocator,
        u8,
        maximum_number_of_bits_needed,
        compressed_values,
    );
    // By using u32 we limit the number of values to 4,294,967,295. This should be sufficient
    // for most use-cases and keeps the header size smaller.
    try shared_functions.appendValue(
        allocator,
        u32,
        @as(u32, @intCast(uncompressed_values.len)),
        compressed_values,
    );
    // Store the minimum integer part for later use during decompression.
    try shared_functions.appendValue(
        allocator,
        u64,
        minimum_integer_part,
        compressed_values,
    );

    // Store the first value's fixed-point representation.
    // Starting with the integer part.
    const first_fixed_point_representation = fixed_point_representation_array.items[0];
    try shared_functions.appendValue(
        allocator,
        u64,
        first_fixed_point_representation.integer_part,
        compressed_values,
    );

    // Prepare the bit writer for compressed output. From now on, we write using only the bit writer
    // instead of the shared_functions.appendValue.
    const writer = compressed_values.writer(allocator);
    var bit_writer = shared_structs.bitWriter(.little, writer);

    // Write the first value's fractional part and sign.
    try bit_writer.writeBits(
        first_fixed_point_representation.fractional_part,
        decoded_decimal_precision,
    );
    try bit_writer.writeBits(@as(u1, @intCast(first_fixed_point_representation.sign)), 1);

    // Apply bitpacking encoding for the rest of the values.
    // 1) Delta encoded integer part with maximum_number_of_bits_needed bits.
    // 2) Fractional part with decoded_decimal_precision bits.
    // 3) Sign with 1 bit.
    for (1..fixed_point_representation_array.items.len) |index| {
        const fixed_point_representation = fixed_point_representation_array.items[index];
        const encoded_delta_integer_part: u64 = delta_encoded_integer_parts.items[index - 1];

        try bit_writer.writeBits(encoded_delta_integer_part, maximum_number_of_bits_needed);
        try bit_writer.writeBits(fixed_point_representation.fractional_part, decoded_decimal_precision);
        try bit_writer.writeBits(@as(u1, @intCast(fixed_point_representation.sign)), 1);
    }

    try bit_writer.flushBits();
}

/// Decompresses an array of `compressed_values` using the BitPackedBUFF variant of the BUFF
/// method. The decompressed f64 values are stored in `decompressed_values`. The function utilizes
/// an `allocator` for memory management during the decompression process. This variant of BUFF
/// uses bit-packing to read the fixed-point representations of the floating-point values directly.
/// If an error occurs it is returned.
pub fn decompressBitPackedBUFF(
    allocator: Allocator,
    compressed_values: []const u8,
    decompressed_values: *ArrayList(f64),
) Error!void {
    // Ensure the compressed values are not empty, i.e., at least the header is present.
    if (compressed_values.len < 22) return Error.UnsupportedInput;

    // Read the header information:
    // 1) decoded decimal precision (u8).
    // 2) maximum number of bits needed for integer part (u8).
    // 3) number of values (u32).
    // 4) minimum integer part (u64).
    // 5) first value (fixed-point representation).
    const decoded_decimal_precision: u8 = compressed_values[0];

    // Check if the decoded decimal precision has not been corrupted and is within the valid range.
    if (decoded_decimal_precision == 0 or decoded_decimal_precision > 53) {
        // Invalid decimal precision in the header.
        return Error.CorruptedCompressedData;
    }

    const maximum_number_of_bits_needed_for_integers: u8 = compressed_values[1];
    const number_of_values: u32 = @bitCast(compressed_values[2..6].*);
    const minimum_integer_part: u64 = @bitCast(compressed_values[6..14].*);
    const shifted_bits: u6 = @intCast(53 - decoded_decimal_precision);

    // Read the first value's integer part from the fixed-point representation.
    var integer_part: u64 = @bitCast(compressed_values[14..22].*);
    var stream = io.fixedBufferStream(compressed_values[22..]);
    var bit_reader = shared_structs.bitReader(.little, stream.reader());

    // Read the first value's fractional part and sign.
    var fractional_part: u64 = bit_reader.readBitsNoEof(
        u64,
        decoded_decimal_precision,
    ) catch return Error.ByteStreamError;
    fractional_part <<= shifted_bits; // Align to 53 bits.
    var sign: u1 = bit_reader.readBitsNoEof(u1, 1) catch return Error.ByteStreamError;

    // Reconstruct the first value and append to decompressed values.
    var fix_point_representation: FixedPointRepresentation = .{
        .integer_part = integer_part + minimum_integer_part,
        .fractional_part = fractional_part,
        .sign = sign,
    };
    var decompressed_value: f64 = constructF64FromFixedPointRepresentation(fix_point_representation);
    try decompressed_values.append(allocator, decompressed_value);

    // For-loop to read and reconstruct the rest of the values.
    // 1) delta encoded integer part with maximum_number_of_bits_needed bits.
    // 2) fractional part with decoded_decimal_precision bits.
    // 3) sign with 1 bit.
    var previous_integer_part: i64 = @intCast(integer_part);
    for (1..number_of_values) |_| {
        const delta_encoded_integer_part: u64 =
            bit_reader.readBitsNoEof(u64, maximum_number_of_bits_needed_for_integers) catch
                return Error.ByteStreamError;

        // Decode the zigzag encoded delta integer part.
        const signed_delta_integer: i64 = shared_functions.decodeZigZag(delta_encoded_integer_part);
        previous_integer_part += signed_delta_integer;

        fractional_part = bit_reader.readBitsNoEof(u64, decoded_decimal_precision) catch
            return Error.ByteStreamError;
        fractional_part <<= shifted_bits; // Align to 53 bits.

        sign = bit_reader.readBitsNoEof(u1, 1) catch return Error.ByteStreamError;

        // Add back the minimum integer part.
        integer_part = @as(u64, @intCast(previous_integer_part)) + minimum_integer_part;
        fix_point_representation = .{
            .integer_part = integer_part,
            .fractional_part = fractional_part,
            .sign = sign,
        };
        // Reconstruct the value and append to decompressed values.
        decompressed_value = constructF64FromFixedPointRepresentation(fix_point_representation);
        try decompressed_values.append(allocator, decompressed_value);
    }
}

/// Decomposes a f64 `value` into its fixed-point representation. The function returns a
/// `FixedPointRepresentation` structure containing the sign, integer part, and fractional
/// part of the fixed-point `value`. If the input value is NaN, or infinity, the function
/// returns `Error.UnsupportedInput`.
fn decomposeF64ToFixedPointRepresentation(value: f64) Error!FixedPointRepresentation {
    const exponent_mask: u64 = 0x7FF; // 11 bits.
    const mantissa_mask: u64 = 0x000F_FFFF_FFFF_FFFF; // 52 bits.
    const remove_sign_mask: u64 = 0x7FFF_FFFF_FFFF_FFFF; // // 63 bits.

    const value_as_u64: u64 = @bitCast(value);

    // Extract the sign of the float.
    const sign: u1 = @intCast(value_as_u64 >> 63);

    // Remove the sign-bit for the decomposition.
    const value_without_sign: u64 = value_as_u64 & remove_sign_mask;

    // Extract exponent and mantissa.
    const exponent: u64 = (value_without_sign >> mantissa_bits) & exponent_mask;
    const mantissa: u64 = value_without_sign & mantissa_mask;

    // Reject NaNs, and infinities (exponent all zeros is also used for subnormals).
    // If exponent is all zeros and mantissa is non-zero, it's a subnormal number.
    // Otherwise, it is zero which we can handle.
    // This can only happen if the compressed values are corrupted.
    // The compression phase checks should prevent this from happening.
    if (exponent == exponent_mask or exponent == 0) {
        if (mantissa == 0) {
            // Handle zero value.
            return FixedPointRepresentation{
                .sign = sign,
                .integer_part = 0,
                .fractional_part = 0,
            };
        }
        return Error.UnsupportedInput;
    }

    // Extract unbiased exponent.
    const unbiased_exponent: i32 = @as(i32, @intCast(exponent)) - exponent_bias;

    // Build significand equal to 1.F as 53 bits in a u64.
    const significand: u64 = (@as(u64, 1) << mantissa_bits) | mantissa;

    // Handle two cases:
    // 1) unbiased_exponent >= 0: `value` >= 1.0, split into integer and fractional bits.
    // 2) unbiased_exponent < 0 : `value` in (0, 1.0), integer part is 0.
    if (unbiased_exponent >= 0) {
        // Unsupported input if more than 53 bits are needed for the integer part.
        // This can only happen if the data has been corrupted since the
        // compression phase checks should prevent this from happening.
        if (unbiased_exponent + 1 > total_fractional_bits) {
            return Error.UnsupportedInput;
        }

        // Number of integer bits is equal to exponent plus 1.
        const number_of_integer_bits: u6 = @intCast(unbiased_exponent + 1);

        // Shift the significand according to the exponent based on the radix point.
        // Use the radix point to obtain the integer and fractional parts.
        const shift_for_radix_point: u6 = total_fractional_bits - number_of_integer_bits;

        // Shift right by `shift_for_radix_point` to get the integer part.
        const integer_part: u64 = significand >> shift_for_radix_point;

        // Mask to get the fractional part bits and shift left to align after integer part.
        const fractional_part_mask: u64 = (@as(u64, 1) << shift_for_radix_point) - 1;
        const fractional_part: u64 = (significand & fractional_part_mask) << number_of_integer_bits;

        return FixedPointRepresentation{
            .sign = sign,
            .integer_part = integer_part,
            .fractional_part = fractional_part,
        };
    } else {
        // For negative exponents the value lies in (0, 1.0). The integer part is 0.
        // We compute the fractional part as a fixed-point value with `total_fractional_bits`
        // bits after the radix point. The fractional part is:
        // fractional_part = significand * 2^(unbiased_exponent + 1)
        // Since unbiased_exponent is negative, we shift right by -(unbiased_exponent + 1).
        const scaling_exponent: i32 = @intCast(-(unbiased_exponent + 1));

        // If the scaling exponent is larger than or equal to 64, the value is too small
        // to be represented with fixed-point representation. This can only happen if the data has been corrupted.
        if (scaling_exponent >= 64)
            return Error.UnsupportedInput;

        var fractional_part: u64 = significand >> @as(u6, @intCast(scaling_exponent));

        // Mask to at most `total_fractional_bits` bits.
        const fractional_part_mask: u64 =
            (@as(u64, 1) << total_fractional_bits) - 1;
        fractional_part &= fractional_part_mask;

        return FixedPointRepresentation{
            .sign = sign,
            .integer_part = 0,
            .fractional_part = fractional_part,
        };
    }
}

/// Constructs a f64 `fixed_point_value` from its fixed-point representation. The function
/// utilizes the sign, integer part, and fractional part from the `FixedPointRepresentation`
/// structure to reconstruct the original f64 value. The constant `normalization_factor` is used
/// to properly scale the fractional part back to its floating-point representation.
fn constructF64FromFixedPointRepresentation(fixed_point_value: FixedPointRepresentation) f64 {
    // Determine the sign of the f64 value.
    const sign: f64 = if (fixed_point_value.sign == 0) 1.0 else -1.0;

    // Convert integer and fractional parts to f64.
    const integer_as_f64: f64 = @floatFromInt(fixed_point_value.integer_part);
    const fractional_as_f64: f64 = @floatFromInt(fixed_point_value.fractional_part);

    // Reconstruct the f64 value. The normalization factor is used to scale the fractional part.
    const value: f64 = sign * (integer_as_f64 + fractional_as_f64 / normalization_factor);

    return value;
}

test "decompose f64 array to fixed point representation works correctly with expected results" {
    const test_values: [6]f64 = .{ 0.125, 12.375, -0.1, 0.0, 3.14159, -256.75 };
    const expected_results: [6]FixedPointRepresentation = .{
        FixedPointRepresentation{
            .sign = 0,
            .integer_part = 0,
            .fractional_part = 0x04000000000000, // 0.125 in fixed-point.
        },
        FixedPointRepresentation{
            .sign = 0,
            .integer_part = 12,
            .fractional_part = 0xc000000000000, // 0.375 in fixed-point.
        },
        FixedPointRepresentation{
            .sign = 1,
            .integer_part = 0,
            .fractional_part = 0x3333333333333, // -0.1 in fixed-point.
        },
        FixedPointRepresentation{
            .sign = 0,
            .integer_part = 0,
            .fractional_part = 0, // 0.0 in fixed-point.
        },
        FixedPointRepresentation{
            .sign = 0,
            .integer_part = 3,
            .fractional_part = 0x487e7c06e19b8, // 0.14159 in fixed-point.
        },
        FixedPointRepresentation{
            .sign = 1,
            .integer_part = 256,
            .fractional_part = 0x18000000000000, // -0.75 in fixed-point.
        },
    };
    for (test_values, 0..) |value, index| {
        const result = try decomposeF64ToFixedPointRepresentation(value);
        const expected = expected_results[index];

        try testing.expect(result.sign == expected.sign);
        try testing.expect(result.integer_part == expected.integer_part);
        try testing.expect(result.fractional_part == expected.fractional_part);
    }
}

test "decompose and reconstruct f64 array to fixed point representation at a known precision level works correctly with fixed values" {
    const test_values: [6]f64 = .{ 0.125, 12.375, -0.1, 0.0, 3.14159, -256.75 };

    const decimal_precision: u8 = 5;
    const bits_to_drop: u6 = 53 - decimal_precision_lookup[decimal_precision - 1];

    for (test_values) |value| {
        const result = try decomposeF64ToFixedPointRepresentation(value);

        const truncated_fractional_part: u64 =
            (result.fractional_part >> bits_to_drop) << bits_to_drop;

        const new_result = FixedPointRepresentation{
            .sign = result.sign,
            .integer_part = result.integer_part,
            .fractional_part = truncated_fractional_part,
        };

        const truncated_reconstructed_value =
            constructF64FromFixedPointRepresentation(new_result);

        try testing.expectApproxEqAbs(truncated_reconstructed_value, value, 0.00001);
    }
}

test "decompose and reconstruct f64 array to fixed point representation at a known precision level works correctly with random values" {
    const allocator: Allocator = testing.allocator;
    var uncompressed_values: ArrayList(f64) = ArrayList(f64).empty;
    defer uncompressed_values.deinit(allocator);

    try tester.generateDefaultBoundedValues(
        allocator,
        &uncompressed_values,
        tester.getDefaultRandomGenerator(),
    );

    const decimal_precision: u8 = tester.generateBoundRandomInteger(u8, 1, 10, null);
    const bits_to_drop: u6 = 53 - decimal_precision_lookup[decimal_precision - 1];

    for (uncompressed_values.items) |value| {
        const result = try decomposeF64ToFixedPointRepresentation(value);

        const truncated_fractional_part: u64 =
            (result.fractional_part >> bits_to_drop) << bits_to_drop;

        const new_result = FixedPointRepresentation{
            .sign = result.sign,
            .integer_part = result.integer_part,
            .fractional_part = truncated_fractional_part,
        };

        const truncated_reconstructed_value =
            constructF64FromFixedPointRepresentation(new_result);

        const tolerance: f64 = math.pow(f64, 10.0, -@as(f64, @floatFromInt(decimal_precision)));

        try testing.expectApproxEqAbs(truncated_reconstructed_value, value, tolerance);
    }
}

test "buff bitpacked can compress and decompress with random values within decimal precision" {
    const allocator: Allocator = testing.allocator;
    var uncompressed_values: ArrayList(f64) = ArrayList(f64).empty;
    defer uncompressed_values.deinit(allocator);

    try tester.generateDefaultBoundedValues(
        allocator,
        &uncompressed_values,
        tester.getDefaultRandomGenerator(),
    );

    var compressed_values: ArrayList(u8) = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);

    const decimal_precision: u8 = tester.generateBoundRandomInteger(u8, 2, 10, null);

    const method_configuration = try std.fmt.allocPrint(
        allocator,
        "{{\"decimal_precision\": {d}}}",
        .{decimal_precision},
    );
    defer allocator.free(method_configuration);

    try compressBitPackedBUFF(
        allocator,
        uncompressed_values.items,
        &compressed_values,
        method_configuration,
    );

    var decompressed_values: ArrayList(f64) = ArrayList(f64).empty;
    defer decompressed_values.deinit(allocator);

    try decompressBitPackedBUFF(
        allocator,
        compressed_values.items,
        &decompressed_values,
    );

    const tolerance: f64 = math.pow(f64, 10.0, -@as(f64, @floatFromInt(decimal_precision)));
    for (uncompressed_values.items, 0..) |original_value, index| {
        const decompressed_value = decompressed_values.items[index];
        try testing.expectApproxEqAbs(original_value, decompressed_value, tolerance);
    }
}
