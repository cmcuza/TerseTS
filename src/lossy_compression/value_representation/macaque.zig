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

//! Implementation of two variations of the MACHAQUE algorithm from the paper
//! "Abduvoris Abduvakhobov, Søren Kejser Jensen, Christian Thomsen, Torben Bach Pedersen".
//! Compressing High-Frequency Time Series Through Multiple Models and Stealing from Residuals.
//! ICDE 2026.
//! This module implements MacaqueV as proposed in the original paper and MacaqueS, which is a
//! simplified version which bit-packs the rewritten values without the additional XOR steps.
//! MacaqueS can be useful in scenarios where the consecutive values are not close to each other.
//! MacaqueV can achieve better results when consecutive values are close to each other.

const std = @import("std");
const math = std.math;
const mem = std.mem;
const io = std.io;
const testing = std.testing;
const Writer = std.io.Writer;
const ArrayList = std.ArrayList;
const Allocator = mem.Allocator;

const tersets = @import("../../tersets.zig");
const shared_structs = @import("../../utilities/shared_structs.zig");
const configuration = @import("../../configuration.zig");
const Error = tersets.Error;
const tester = @import("../../tester.zig");

const shared_functions = @import("../../utilities/shared_functions.zig");
const Method = tersets.Method;

const u64_max: u64 = 0xFFFF_FFFF_FFFF_FFFF;
const max_u64_bits: u7 = 64;

/// Helper struct to hold the number of `bits_needed` and the `rewritten_value` after
/// applying the error bound-based rewriting.
const RewrittenValue = struct {
    bits_needed: u6,
    rewritten_value: u64,
};

/// Compress `uncompressed_values` within an error bound using MacaqueS. The function performs
/// the lossy rewriting of the values based on the provided `AbsoluteErrorBound`'s `method_configuration`
/// and bit-packs the rewritten values. The compressed representation is written in `compressed_values`.
/// The `allocator` is used for memory management of intermediates containers and the `method_configuration`
/// parser. If an error occurs it is returned.
pub fn compressMacaqueS(
    allocator: Allocator,
    uncompressed_values: []const f64,
    compressed_values: *ArrayList(u8),
    method_configuration: []const u8,
) Error!void {
    const parsed_configuration = try configuration.parse(
        allocator,
        configuration.AbsoluteErrorBound,
        method_configuration,
    );

    const error_bound: f64 = @floatCast(parsed_configuration.abs_error_bound);

    if (error_bound <= 0.0) return Error.InvalidConfiguration;

    // Since the number of values is stored in 4 bytes in the compressed representation,
    // we need to ensure that the input does not exceed this limit. This is reasonable since
    // compressing more than 4 billion values is uncommon, while saving 4 bytes for most cases.
    if (uncompressed_values.len >= math.maxInt(u32)) {
        return Error.UnsupportedInput;
    }

    try shared_functions.appendValue(
        allocator,
        u32,
        @as(u32, @intCast(uncompressed_values.len)),
        compressed_values,
    );

    const writer = compressed_values.writer(allocator);
    var bit_writer = shared_structs.bitWriter(.little, writer);

    // Step 1: For each value, rewrite bits based on the error bound and calculate the maximum number of bits rewritten.
    for (uncompressed_values) |value| {
        if (!math.isFinite(value) or @abs(value) > tester.max_test_value) return Error.UnsupportedInput;
        const returned_values = rewriteValueBasedOnErrorBound(value, error_bound);

        const bits_needed: u6 = returned_values.bits_needed;
        const rewritten_value: u64 = returned_values.rewritten_value;

        try writeCompressedMacaqueValue(&bit_writer, bits_needed, rewritten_value);
    }

    try bit_writer.flushBits();
}

/// Compress `uncompressed_values` within an error bound using MacaqueV. The function performs the
/// lossy rewriting of the values based on the provided `AbsoluteErrorBound`'s `method_configuration`.
/// The compressed representation is written in `compressed_values` using the XOR-based encoding as described
/// in the original paper. The `allocator` is used for memory management of intermediates containers
/// and the `method_configuration` parser. If an error occurs it is returned.
pub fn compressMacaqueV(
    allocator: Allocator,
    uncompressed_values: []const f64,
    compressed_values: *ArrayList(u8),
    method_configuration: []const u8,
) Error!void {
    const parsed_configuration = try configuration.parse(
        allocator,
        configuration.AbsoluteErrorBound,
        method_configuration,
    );

    const error_bound: f64 = @floatCast(parsed_configuration.abs_error_bound);

    if (error_bound <= 0.0) return Error.InvalidConfiguration;

    // Since the number of values is stored in 4 bytes in the compressed representation,
    // we need to ensure that the input does not exceed this limit. This is reasonable since
    // compressing more than 4 billion values is uncommon, while saving 4 bytes for most cases.
    if (uncompressed_values.len >= math.maxInt(u32)) {
        return Error.UnsupportedInput;
    }

    try shared_functions.appendValue(
        allocator,
        u32,
        @as(u32, @intCast(uncompressed_values.len)),
        compressed_values,
    );

    const writer = compressed_values.writer(allocator);
    var bit_writer = shared_structs.bitWriter(.little, writer);

    const first_value: f64 = uncompressed_values[0];

    // Check if the first value is finite and within the maximum test value.
    // If not, return an error since it cannot be compressed.
    if (!math.isFinite(first_value) or @abs(first_value) > tester.max_test_value)
        return Error.UnsupportedInput;

    // The previous rewritten value is initialized with the first value, which is stored in full.
    var previous_rewritten_value = rewriteValueBasedOnErrorBound(first_value, error_bound);

    try writeCompressedMacaqueValue(
        &bit_writer,
        previous_rewritten_value.bits_needed,
        previous_rewritten_value.rewritten_value,
    );

    var last_leading_zeros: u7 = @clz(previous_rewritten_value.rewritten_value);
    var last_trailing_zeros: u7 = @ctz(previous_rewritten_value.rewritten_value);

    var previous_value: f64 = @bitCast(previous_rewritten_value.rewritten_value);

    for (1..uncompressed_values.len) |i| {
        const value: f64 = uncompressed_values[i];

        if (!math.isFinite(value) or @abs(value) > tester.max_test_value) return Error.UnsupportedInput;

        if (@abs(value - previous_value) <= error_bound) {
            // If the current value is within the error bound of the previous value,
            // we can store a control bit of 0 and skip storing the value.
            try bit_writer.writeBits(@as(u1, 0b0), 1); // Header '0'.
            continue;
        }

        const current_rewritten_value = rewriteValueBasedOnErrorBound(value, error_bound);

        previous_value = @bitCast(current_rewritten_value.rewritten_value);

        // Perform XOR between the previous rewritten value and the current rewritten value to find the bits that differ.
        const xor_value: u64 = previous_rewritten_value.rewritten_value ^ current_rewritten_value.rewritten_value;

        if (xor_value == 0) {
            try bit_writer.writeBits(@as(u1, 0b0), 1); // header '0'.
        } else {
            try bit_writer.writeBits(@as(u1, 0b1), 1); // header '1'.
            const leading_zeros: u7 = @clz(xor_value);
            const trailing_zeros: u7 = @ctz(xor_value);

            if (trailing_zeros >= last_trailing_zeros and leading_zeros >= last_leading_zeros) {
                // Store a control bit of 10.
                try bit_writer.writeBits(@as(u1, 0b0), 1); // second control bit '0'.

                // We only need to store the significant bits that differ from the previous value, which are between the leading and trailing zeros.
                const significant_bits_length: u7 = max_u64_bits - last_leading_zeros - last_trailing_zeros;
                const significant_bits: u64 = xor_value >> @as(u6, @intCast(last_trailing_zeros));

                // Store the significant bits that differ from the previous value.
                try bit_writer.writeBits(significant_bits, significant_bits_length);
            } else {
                // Store a control bit of 11.
                try bit_writer.writeBits(@as(u1, 0b1), 1); // header '1'.

                // We need to store the leading zeros, significant bits length,
                // and the significant bits that differ from the previous value.
                // A u64 XOR can have 0..63 leading zeros, so this field needs 6 bits.
                try bit_writer.writeBits(@as(u6, @intCast(leading_zeros)), 6);

                const meaningful_bits: u7 = max_u64_bits - leading_zeros - trailing_zeros;

                // Store the significant bits length in 6 bits.
                try bit_writer.writeBits(meaningful_bits, 6);
                const significant_bits: u64 = xor_value >> @as(u6, @intCast(trailing_zeros));

                // Store the significant bits that differ from the previous value.
                try bit_writer.writeBits(significant_bits, meaningful_bits);

                last_leading_zeros = leading_zeros;
                last_trailing_zeros = trailing_zeros;
            }
            previous_rewritten_value = current_rewritten_value;
        }
    }

    try bit_writer.flushBits();
}

/// Decompress `compressed_values` produced by MacaqueS. The function writes the result to
/// `decompressed_values`. The `allocator` is used for memory management of intermediates containers.
/// If an error occurs it is returned.
pub fn decompressMacaqueS(
    allocator: Allocator,
    compressed_values: []const u8,
    decompressed_values: *ArrayList(f64),
) Error!void {
    // The first 4 bytes of `compressed_values` represent the number of values.
    // At least those bytes should be present to be able to read the number of values.
    if (compressed_values.len < 4) return Error.UnsupportedInput;

    const number_of_values: usize = @intCast(@as(u32, @bitCast(compressed_values[0..4].*)));

    var stream = io.fixedBufferStream(compressed_values[4..]);
    var bit_reader = shared_structs.bitReader(.little, stream.reader());

    try decompressed_values.ensureTotalCapacity(allocator, number_of_values);

    for (0..number_of_values) |_| {
        const bits_needed: u6 = bit_reader.readBitsNoEof(u6, 6) catch return Error.ByteStreamError;

        const most_significant_bits: u16 = @as(u16, bits_needed) + 12;

        const packed_value: u64 = bit_reader.readBitsNoEof(u64, most_significant_bits) catch return Error.ByteStreamError;

        const shift: u6 = @intCast(64 - most_significant_bits);
        const rewritten_bits: u64 = packed_value << shift;
        const value: f64 = @bitCast(rewritten_bits);

        try decompressed_values.append(allocator, value);
    }
}

/// Decompress `compressed_values` produced by MacaqueV. The function writes the result to
/// `decompressed_values`. The `allocator` is used for memory management of intermediates containers.
/// If an error occurs it is returned.
pub fn decompressMacaqueV(
    allocator: Allocator,
    compressed_values: []const u8,
    decompressed_values: *ArrayList(f64),
) Error!void {
    // The first 4 bytes of `compressed_values` represent the number of values.
    // At least those bytes should be present to be able to read the number of values.
    if (compressed_values.len < 4) return Error.UnsupportedInput;

    const number_of_values: usize = @intCast(@as(u32, @bitCast(compressed_values[0..4].*)));

    var stream = io.fixedBufferStream(compressed_values[4..]);
    var bit_reader = shared_structs.bitReader(.little, stream.reader());

    try decompressed_values.ensureTotalCapacity(allocator, number_of_values);

    const bits_needed: u6 = bit_reader.readBitsNoEof(u6, 6) catch return Error.ByteStreamError;
    const most_significant_bits: u16 = @as(u16, bits_needed) + 12;

    const first_packed_value: u64 = bit_reader.readBitsNoEof(u64, most_significant_bits) catch return Error.ByteStreamError;

    const shift: u6 = @intCast(64 - most_significant_bits);

    // The previous rewritten value is initialized with the first value, which is stored in full.
    var previous_rewritten_value: u64 = first_packed_value << shift;
    var previous_value: f64 = @bitCast(previous_rewritten_value);

    try decompressed_values.append(allocator, previous_value);

    var last_leading_zeros: u7 = @clz(previous_rewritten_value);
    var last_trailing_zeros: u7 = @ctz(previous_rewritten_value);

    for (1..number_of_values) |_| {
        const control_bit: u8 = bit_reader.readBitsNoEof(u8, 1) catch return Error.ByteStreamError;

        if (control_bit == 0) {
            try decompressed_values.append(allocator, previous_value);
        } else {
            const second_control_bit: u8 = bit_reader.readBitsNoEof(u8, 1) catch return Error.ByteStreamError;

            if (second_control_bit == 0) {
                // We only need to read the significant bits that differ from the previous value, which are between the leading and trailing zeros.
                const significant_bits_length: u7 = max_u64_bits - last_leading_zeros - last_trailing_zeros;
                const significant_bits: u64 = bit_reader.readBitsNoEof(u64, significant_bits_length) catch return Error.ByteStreamError;

                const xor_value: u64 = significant_bits << @as(u6, @intCast(last_trailing_zeros));
                const rewritten_value: u64 = previous_rewritten_value ^ xor_value; // Luckily, XOR is its own inverse.
                previous_value = @bitCast(rewritten_value);
                try decompressed_values.append(allocator, previous_value);
                previous_rewritten_value = rewritten_value;
            } else {
                // The previous xor value is not close enough to the current value, so we need to read the leading zeros,
                // significant bits length, and the significant bits that differ from the previous value.
                const leading_zeros: u7 = bit_reader.readBitsNoEof(u7, 6) catch return Error.ByteStreamError;
                const encoded_significant_bits_length: u6 = bit_reader.readBitsNoEof(u6, 6) catch return Error.ByteStreamError;
                const significant_bits_length: u7 = if (encoded_significant_bits_length == 0)
                    max_u64_bits
                else
                    @as(u7, encoded_significant_bits_length);

                if (leading_zeros + significant_bits_length > max_u64_bits) {
                    return Error.ByteStreamError;
                }

                const significant_bits: u64 = bit_reader.readBitsNoEof(u64, significant_bits_length) catch return Error.ByteStreamError;

                // All needed information read from the compressed representation.
                // Now we can reconstruct the rewritten value and then the original value.
                const xor_shift: u6 = @intCast(max_u64_bits - leading_zeros - significant_bits_length);
                const xor_value: u64 = significant_bits << xor_shift;
                const rewritten_value: u64 = previous_rewritten_value ^ xor_value; // Luckily, XOR is its own inverse.
                previous_value = @bitCast(rewritten_value);
                try decompressed_values.append(allocator, previous_value);

                previous_rewritten_value = rewritten_value;
                last_leading_zeros = leading_zeros;
                last_trailing_zeros = max_u64_bits - leading_zeros - significant_bits_length;
            }
        }
    }
}

/// Rewrites the bits of `value` based on the `error_bound`. The function calculates how many bits can be rewritten
/// while keeping the decompressed value within the error bound. It returns the number of bits needed `bits_needed`
/// to be stored and the rewritten value `new_rewritten_value` as a u64.
fn rewriteValueBasedOnErrorBound(
    value: f64,
    error_bound: f64,
) RewrittenValue {

    // Cast the f64 value to its binary representation as a u64 to manipulate the bits directly.
    const value_as_u64: u64 = @bitCast(value);
    // Extract the exponent.
    const exponent: f64 = @floatFromInt(getExponentU64(value_as_u64));
    // The variable `m` tells us how many bits we can rewrite while staying within the error bound.
    // We reuse the variable name `m` from the original paper (Alg. 2 Line 9).
    const m: f64 = error_bound / math.pow(f64, 2, exponent);

    // If m >= 1, we can rewrite all 52 mantissa bits. If m <= 0, we cannot rewrite any bits.
    // For 0 < m < 1, the number of bits we can rewrite is given by -log2(m).
    var bits_needed: u6 = 0;
    if (!math.isFinite(m) or m <= 0.0) {
        // Degenerate case: keep full mantissa precision.
        bits_needed = 52;
    } else if (m >= 1.0) {
        // Large error bound for this magnitude: no mantissa bits are required.
        bits_needed = 0;
    } else {
        // 0 < m < 1 => -log2(m) is positive and defines the required mantissa bits.
        const needed_f64 = @floor(-@log2(m));
        bits_needed = @min(52, @as(u6, @intFromFloat(needed_f64)));
    }

    // The variable `rewrite_position` tells us how many of the least significant bits of the mantissa we can rewrite.
    const rewrite_position: u6 = 52 - bits_needed;
    // We create a mask to zero out the least significant `rewrite_position` bits of the mantissa.
    const mask: u64 = u64_max << rewrite_position;
    const rewritten_value: u64 = value_as_u64 & mask;

    // Cast to f64 to check if the rewritten value is within the error bound.
    const decompressed_value: f64 = @bitCast(rewritten_value);

    // If the decompressed value is within the error bound, we can keep it as is. Otherwise, we need to rewrite less bits.
    if (@abs(decompressed_value - value) <= error_bound) {
        // If the decompressed value is within the error bound, we keep it as is.
        return RewrittenValue{ .bits_needed = bits_needed, .rewritten_value = rewritten_value };
    } else {
        const new_mask: u64 = u64_max << (rewrite_position - 1);
        const new_rewritten_value: u64 = value_as_u64 & new_mask;
        return RewrittenValue{ .bits_needed = bits_needed + 1, .rewritten_value = new_rewritten_value };
    }
}

/// Helper function to extract the exponent from the binary representation of a f64 `value`.
/// The `value` is interpreted as a 64-bit unsigned integer, and the exponent bits are extracted
/// and adjusted by the bias (1023 for f64) to return the actual exponent as an i16.
fn getExponentU64(value: u64) i16 {
    const exponent_bits: i16 = @intCast((value >> 52) & 0x7FF);
    return exponent_bits - 1023;
}

/// Helper function to write a `value` using the `bit_writer`. The function writes the `bits_needed`
/// and the `value` in a bit-packed format. The `bits_needed` is written using 6 bits, and the `value`
/// is written using the number of bits specified by `bits_needed` plus 12 bits for the sign and
/// exponent. If an error occurs during writing, it is returned.
fn writeCompressedMacaqueValue(bit_writer: anytype, bits_needed: u6, value: u64) !void {
    try bit_writer.writeBits(bits_needed, 6);
    const most_significant_bits: u16 = @as(u16, bits_needed) + 12; // 12 bits for sign and exponent.
    const shift: u6 = @intCast(64 - most_significant_bits);
    const packed_value: u64 = value >> shift;
    try bit_writer.writeBits(packed_value, most_significant_bits);
}

test "macaques can compress and decompress bounded values" {
    const allocator = testing.allocator;
    const data_distributions = &[_]tester.DataDistribution{
        .LinearFunctions,
        .BoundedRandomValues,
        .SinusoidalFunction,
        .MixedBoundedValuesFunctions,
    };

    try tester.testErrorBoundedCompressionMethod(
        allocator,
        Method.MacaqueS,
        data_distributions,
    );
}

test "macaquev can compress and decompress bounded values" {
    const allocator = testing.allocator;
    const data_distributions = &[_]tester.DataDistribution{
        .LinearFunctions,
        .BoundedRandomValues,
        .SinusoidalFunction,
        .MixedBoundedValuesFunctions,
    };

    try tester.testErrorBoundedCompressionMethod(
        allocator,
        Method.MacaqueV,
        data_distributions,
    );
}

test "macaquev can compress and decompress values with small differences between consecutive values" {
    const allocator = testing.allocator;
    const uncompressed_values = [6]f64{ 3.12, 4.423, 5.20, 7.9, 8.1, 9.0 };
    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);

    // Set error bound big enough to allow consecutive values to be compressed effectively.
    const method_configuration =
        \\ {"abs_error_bound": 1.5}
    ;

    try compressMacaqueV(
        allocator,
        uncompressed_values[0..],
        &compressed_values,
        method_configuration,
    );

    var decompressed_values = ArrayList(f64).empty;
    defer decompressed_values.deinit(allocator);
    try decompressMacaqueV(
        allocator,
        compressed_values.items,
        &decompressed_values,
    );
    try testing.expectEqual(uncompressed_values.len, decompressed_values.items.len);

    for (0..uncompressed_values.len) |i| {
        const original_value = uncompressed_values[i];
        const decompressed_value = decompressed_values.items[i];
        try testing.expectApproxEqAbs(original_value, decompressed_value, 1.5);
    }
}

test "macaques cannot compress and decompress nan values" {
    const allocator = testing.allocator;
    const uncompressed_values = [3]f64{ 343.0, math.nan(f64), 520.0 };
    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);

    const method_configuration =
        \\ {"abs_error_bound": 0.1}
    ;

    compressMacaqueS(
        allocator,
        uncompressed_values[0..],
        &compressed_values,
        method_configuration,
    ) catch |err| {
        try testing.expectEqual(Error.UnsupportedInput, err);
        return;
    };

    try testing.expectFmt(
        "",
        "The Macaque method cannot compress NaN values",
        .{},
    );
}

test "macaquev cannot compress and decompress nan values" {
    const allocator = testing.allocator;
    const uncompressed_values = [3]f64{ 343.0, math.nan(f64), 520.0 };
    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);

    const method_configuration =
        \\ {"abs_error_bound": 0.1}
    ;

    compressMacaqueV(
        allocator,
        uncompressed_values[0..],
        &compressed_values,
        method_configuration,
    ) catch |err| {
        try testing.expectEqual(Error.UnsupportedInput, err);
        return;
    };

    try testing.expectFmt(
        "",
        "The Macaque method cannot compress NaN values",
        .{},
    );
}

test "macaques cannot compress and decompress values exceeding floating-point precision limits" {
    const allocator = testing.allocator;
    // A value exceeding floating-point precision limits refers to a number that cannot be
    // accurately represented using f64 due to its magnitude, such as 1e20.
    const uncompressed_values = [3]f64{ 343.0, 1e20, 520.0 };
    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);

    const method_configuration =
        \\ {"abs_error_bound": 0.1}
    ;

    compressMacaqueS(
        allocator,
        uncompressed_values[0..],
        &compressed_values,
        method_configuration,
    ) catch |err| {
        try testing.expectEqual(Error.UnsupportedInput, err);
        return;
    };

    try testing.expectFmt(
        "",
        "The Macaque method cannot compress values exceeding floating-point precision limits",
        .{},
    );
}

test "macaquev cannot compress and decompress values exceeding floating-point precision limits" {
    const allocator = testing.allocator;
    // A value exceeding floating-point precision limits refers to a number that cannot be
    // accurately represented using f64 due to its magnitude, such as 1e20.
    const uncompressed_values = [3]f64{ 343.0, 1e20, 520.0 };
    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);

    const method_configuration =
        \\ {"abs_error_bound": 0.1}
    ;

    compressMacaqueV(
        allocator,
        uncompressed_values[0..],
        &compressed_values,
        method_configuration,
    ) catch |err| {
        try testing.expectEqual(Error.UnsupportedInput, err);
        return;
    };

    try testing.expectFmt(
        "",
        "The Macaque method cannot compress values exceeding floating-point precision limits",
        .{},
    );
}

test "macaques can compress and decompress within floating-point precision limits at different scales" {
    const allocator = testing.allocator;
    const error_bound = tester.generateBoundedRandomValue(f32, 0, 1e3, null);

    var uncompressed_values = ArrayList(f64).empty;
    defer uncompressed_values.deinit(allocator);

    try tester.generateBoundedRandomValues(allocator, &uncompressed_values, -1, 1, null);
    try tester.generateBoundedRandomValues(allocator, &uncompressed_values, -1e2, 1e2, null);
    try tester.generateBoundedRandomValues(allocator, &uncompressed_values, -1e4, 1e4, null);
    try tester.generateBoundedRandomValues(allocator, &uncompressed_values, -1e6, 1e6, null);
    try tester.generateBoundedRandomValues(allocator, &uncompressed_values, -1e8, 1e8, null);
    try tester.generateBoundedRandomValues(allocator, &uncompressed_values, -1e14, 1e14, null);

    try tester.testCompressAndDecompress(
        allocator,
        uncompressed_values.items,
        Method.MacaqueS,
        error_bound,
        shared_functions.isWithinErrorBound,
    );
}

test "macaquev can compress and decompress within floating-point precision limits at different scales" {
    const allocator = testing.allocator;
    const error_bound = tester.generateBoundedRandomValue(f32, 0, 1e3, null);

    var uncompressed_values = ArrayList(f64).empty;
    defer uncompressed_values.deinit(allocator);

    try tester.generateBoundedRandomValues(allocator, &uncompressed_values, -1, 1, null);
    try tester.generateBoundedRandomValues(allocator, &uncompressed_values, -1e2, 1e2, null);
    try tester.generateBoundedRandomValues(allocator, &uncompressed_values, -1e4, 1e4, null);
    try tester.generateBoundedRandomValues(allocator, &uncompressed_values, -1e6, 1e6, null);
    try tester.generateBoundedRandomValues(allocator, &uncompressed_values, -1e8, 1e8, null);
    try tester.generateBoundedRandomValues(allocator, &uncompressed_values, -1e14, 1e14, null);

    try tester.testCompressAndDecompress(
        allocator,
        uncompressed_values.items,
        Method.MacaqueV,
        error_bound,
        shared_functions.isWithinErrorBound,
    );
}

test "macaques always reduces size of time series" {
    const allocator = testing.allocator;
    // Generate a random error bound between 10 and 1000, which will be used for quantization.
    const error_bound = @floor(tester.generateBoundedRandomValue(
        f32,
        1e1,
        1e3,
        null,
    )) * 0.1;

    var uncompressed_values = ArrayList(f64).empty;
    defer uncompressed_values.deinit(allocator);

    // Generate 500 random values within different ranges. Even if some values require 8 bytes
    // to be stored, the quantization should reduce the size of the time series since some
    // values require less than 8 bytes to be stored after quantization.
    try tester.generateBoundedRandomValues(allocator, &uncompressed_values, -1, 1, null);
    try tester.generateBoundedRandomValues(allocator, &uncompressed_values, -1e2, 1e2, null);
    try tester.generateBoundedRandomValues(allocator, &uncompressed_values, -1e4, 1e4, null);
    try tester.generateBoundedRandomValues(allocator, &uncompressed_values, -1e6, 1e6, null);
    try tester.generateBoundedRandomValues(allocator, &uncompressed_values, -1e8, 1e8, null);
    try tester.generateBoundedRandomValues(allocator, &uncompressed_values, -1e14, 1e14, null);

    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);

    const method_configuration = try std.fmt.allocPrint(
        allocator,
        "{{\"abs_error_bound\": {d}}}",
        .{error_bound},
    );
    defer allocator.free(method_configuration);

    try compressMacaqueS(
        allocator,
        uncompressed_values.items,
        &compressed_values,
        method_configuration,
    );

    // Considering the range of the input data, the compressed values should always be smaller.
    try testing.expect(uncompressed_values.items.len * 8 > compressed_values.items.len);
}

test "macaquev always reduces size of time series" {
    const allocator = testing.allocator;
    // Generate a random error bound between 10 and 1000, which will be used for quantization.
    const error_bound = @floor(tester.generateBoundedRandomValue(
        f32,
        1e1,
        1e3,
        null,
    )) * 0.1;

    var uncompressed_values = ArrayList(f64).empty;
    defer uncompressed_values.deinit(allocator);

    // Generate 500 random values within different ranges. Even if some values require 8 bytes
    // to be stored, the quantization should reduce the size of the time series since some
    // values require less than 8 bytes to be stored after quantization.
    try tester.generateBoundedRandomValues(allocator, &uncompressed_values, -1, 1, null);
    try tester.generateBoundedRandomValues(allocator, &uncompressed_values, -1e2, 1e2, null);
    try tester.generateBoundedRandomValues(allocator, &uncompressed_values, -1e4, 1e4, null);
    try tester.generateBoundedRandomValues(allocator, &uncompressed_values, -1e6, 1e6, null);
    try tester.generateBoundedRandomValues(allocator, &uncompressed_values, -1e8, 1e8, null);
    try tester.generateBoundedRandomValues(allocator, &uncompressed_values, -1e14, 1e14, null);

    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);

    const method_configuration = try std.fmt.allocPrint(
        allocator,
        "{{\"abs_error_bound\": {d}}}",
        .{error_bound},
    );
    defer allocator.free(method_configuration);

    try compressMacaqueV(
        allocator,
        uncompressed_values.items,
        &compressed_values,
        method_configuration,
    );

    // Considering the range of the input data, the compressed values should always be smaller.
    try testing.expect(uncompressed_values.items.len * 8 > compressed_values.items.len);
}

test "macaquev preserves values when xor has more than 31 leading zeros" {
    const allocator = testing.allocator;
    const uncompressed_values = [5]f64{
        22.7,
        22.70000050919397,
        21.700000077486038,
        22.633333410819766,
        22.616337399924696,
    };
    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);

    const method_configuration =
        \\ {"abs_error_bound": 1e-07}
    ;

    try compressMacaqueV(
        allocator,
        uncompressed_values[0..],
        &compressed_values,
        method_configuration,
    );

    var decompressed_values = ArrayList(f64).empty;
    defer decompressed_values.deinit(allocator);
    try decompressMacaqueV(
        allocator,
        compressed_values.items,
        &decompressed_values,
    );

    try testing.expectEqual(uncompressed_values.len, decompressed_values.items.len);
    for (0..uncompressed_values.len) |i| {
        try testing.expectApproxEqAbs(
            uncompressed_values[i],
            decompressed_values.items[i],
            1e-07,
        );
    }
}

test "check macaques configuration parsing" {
    // Tests the configuration parsing and functionality of the `compress` function.
    // The test verifies that the provided configuration is correctly interpreted and
    // that the `configuration.AbsoluteErrorBound` is expected in the function.
    const allocator = testing.allocator;

    const uncompressed_values = &[4]f64{ 19.0, 48.0, 29.0, 3.0 };

    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);

    const method_configuration =
        \\ {"abs_error_bound": 0.1}
    ;

    // The configuration is properly defined. No error expected.
    try compressMacaqueS(
        allocator,
        uncompressed_values,
        &compressed_values,
        method_configuration,
    );
}

test "check macaquev configuration parsing" {
    // Tests the configuration parsing and functionality of the `compress` function.
    // The test verifies that the provided configuration is correctly interpreted and
    // that the `configuration.AbsoluteErrorBound` is expected in the function.
    const allocator = testing.allocator;

    const uncompressed_values = &[4]f64{ 19.0, 48.0, 29.0, 3.0 };

    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);

    const method_configuration =
        \\ {"abs_error_bound": 0.1}
    ;

    // The configuration is properly defined. No error expected.
    try compressMacaqueV(
        allocator,
        uncompressed_values,
        &compressed_values,
        method_configuration,
    );
}
