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

//! Implmentation of the method BUFF (BoUnded Fast Float) from the paper
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
//! Both variants have similar compression ratio performance but the
//! `BytePackedBUFF` variant offers faster compression and decompression speeds.

const std = @import("std");
const math = std.math;
const io = std.io;
const testing = std.testing;
const ArrayList = std.ArrayList;
const Allocator = std.mem.Allocator;

const tersets = @import("../../tersets.zig");
const shared_functions = @import("../../utilities/shared_functions.zig");
const configuration = @import("../../configuration.zig");
const tester = @import("../../tester.zig");

const Error = tersets.Error;
const Method = tersets.Method;

/// Target precision lookout table from Table 2 in the paper. Each entry corresponds to the
/// number of bits needed in the fractional part to achieve the specified decimal digit precision.
/// Index 0 corresponds to 1 decimal digit precision, index 1 to 2 decimal digits, and so on.
/// Up to 10 decimal digits of precision are supported.
const target_precision_lookout: [10]u6 = .{ 5, 8, 11, 15, 18, 21, 25, 28, 31, 35 };

/// Normalization factor for converting fractional part to f64.
const normalization_factor: f64 = math.pow(f64, 2.0, 53.0);

/// f64 decomposition constants.
const mantissa_bits: u6 = 52; // mantissa bits.
const total_fractional_bits: u6 = 53; // 53 bits for 1.F.
const exponent_bias: i32 = 1023; // Bias for f64 exponent.

/// Fixed-point representation structure. The structure hold the `sign`, `integer_part`,
/// and `fractional_part` of the fixed-point number.
const FixedPointRepresentation = struct {
    sign: u8, // 0 or 1.
    integer_part: u64, // integer bits as an integer.
    fractional_part: u64, // fractional bits as an integer.
};

/// Compresses an array of f64 `uncompressed_values` using the BitPackedBUFF variant of the BUFF
/// method with the specified `method_configuration`. The function parses the configuration
/// to determine the target precision and then decomposes each f64 value into its fixed-point
/// representation. The fractional part is truncated based on the target precision before
/// storing the fixed-point representations.
pub fn compressBitPackedBUFF(
    allocator: Allocator,
    uncompressed_values: []const f64,
    compressed_values: ArrayList(u8),
    method_configuration: []const u8,
) Error!void {
    const parsed_configuration = try configuration.parse(
        allocator,
        configuration.TargetPrecision,
        method_configuration,
    );

    if (parsed_configuration.target_precision > 10) {
        // Limiting to 10 decimal digits of precision based on Table 2.
        // I guess that precision 11 and above do not provide significant
        // compression benefits compared to the overhead of storing them.
        return Error.InvalidConfiguration;
    }

    var fixed_point_representation_array = try ArrayList(FixedPointRepresentation).initCapacity(
        allocator,
        uncompressed_values.len,
    );
    defer fixed_point_representation_array.deinit(allocator);

    // Track minimum integer part for optimizations explained in Section 3.1.2.
    var minimum_integer_part: u64 = std.math.maxInt(u64);

    // Decompose each value into a fixed point representation. Truncate the fractional
    // part based on the target precision and update the minimum integer part.
    for (uncompressed_values) |value| {
        var fixed_point_representation = try decomposeF64ToFixedPointRepresentation(value);
        // Apply the target precision by truncating the fractional part.
        const truncated_fractional_part: u64 =
            (fixed_point_representation.fractional_part >>
                (total_fractional_bits - target_precision_lookout[parsed_configuration.target_precision - 1]));
        fixed_point_representation.fractional_part = truncated_fractional_part;

        try fixed_point_representation_array.append(allocator, fixed_point_representation);

        if (fixed_point_representation.integer_part < minimum_integer_part) {
            minimum_integer_part = fixed_point_representation.integer_part;
        }
    }

    // Apply delta and bitpacking encoding of the interger parts.

}

/// Decomposes a f64 `value` into its fixed-point representation. The function returns a
/// `FixedPointRepresentation` structure containing the sign, integer part, and fractional
/// part of the fixed-point `value`. If the input value is NaN, or infinity, the function
/// returns `Error.UnsupportedInput`.
fn decomposeF64ToFixedPointRepresentation(value: f64) Error!FixedPointRepresentation {
    const exponent_mask: u64 = 0x7FF; // 11 bits.
    const mantissa_mask: u64 = 0x000F_FFFF_FFFF_FFFF; // 52 bits.
    const remove_sign_mask: u64 = 0x7FFF_FFFF_FFFF_FFFF;

    const value_as_u64: u64 = @bitCast(value);

    // Extract the sign of the float.
    const sign: u8 = @intCast(value_as_u64 >> 63);

    // Remove the sign-bit for the decomposition.
    const value_without_sign: u64 = value_as_u64 & remove_sign_mask;

    // Extract exponent and mantissa.
    const exponent: u64 = (value_without_sign >> mantissa_bits) & exponent_mask;
    const mantissa: u64 = value_without_sign & mantissa_mask;

    // Reject NaNs, and infinities (exponent all zeros is also used for subnormals).
    // If exponent is all zeros and mantissa is non-zero, it's a subnormal number.
    // Otherwise, it is zero which we can handle.
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

    // Unbiased exponent.
    const unbiased_exponent: i32 = @as(i32, @intCast(exponent)) - exponent_bias;

    // Build significand equal to 1.F as 53 bits in a u64.
    const significand: u64 = (@as(u64, 1) << mantissa_bits) | mantissa;

    // Handle two cases:
    //   1) unbiased_exponent >= 0: `value` >= 1.0, split into integer and fractional bits.
    //   2) unbiased_exponent < 0 : `value` in (0, 1.0), integer part is 0.
    if (unbiased_exponent >= 0) {
        // Unsupported input if more than 53 bits are needed for the integer part.
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
        // to be represented with our fixed-point representation.
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

    const target_precision: u8 = 5;
    const bits_to_drop: u6 = 53 - target_precision_lookout[target_precision - 1];

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

    const target_precision: u8 = tester.generateBoundRandomInteger(u8, 2, 10, null);
    const bits_to_drop: u6 = 53 - target_precision_lookout[target_precision - 1];

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

        const tolerance: f64 = math.pow(f64, 10.0, -@as(f64, @floatFromInt(target_precision)));

        try testing.expectApproxEqAbs(truncated_reconstructed_value, value, tolerance);
    }
}
