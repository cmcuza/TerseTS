// Copyright 2025 TerseTS Contributors
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

//! Contains all shared functions used across TerseTS.

const std = @import("std");
const Allocator = std.mem.Allocator;
const io = std.io;
const math = std.math;
const ArrayList = std.ArrayList;
const tersets = @import("../tersets.zig");
const tester = @import("../tester.zig");
const Error = tersets.Error;
const testing = std.testing;

const shared_structs = @import("shared_structs.zig");
const BitWriter = shared_structs.BitWriter;

/// Computes the Root-Mean-Squared-Errors (RMSE) for a segment of the `uncompressed_values`.
/// This function calculates the error between the actual values and the predicted values
/// based on a linear regression model fitted to the segment defined by `seg_start` and `seg_end`.
pub fn computeRMSE(uncompressed_values: []const f64, seg_start: usize, seg_end: usize) Error!f64 {
    const seg_len: f64 = @floatFromInt(seg_end - seg_start + 1);
    if (seg_len <= 1) return 0.0; // If the segment has one or no points, return zero error.

    // Check if the elements of the segment are within the valid range.
    if (uncompressed_values[seg_start] > tester.max_test_value or
        !math.isFinite(uncompressed_values[seg_start]) or
        uncompressed_values[seg_end] > tester.max_test_value or
        !math.isFinite(uncompressed_values[seg_end]))
    {
        return Error.UnsupportedInput;
    }
    // Calculate the slope and intercept of the line connecting the start and end points.
    const slope: f64 = (uncompressed_values[seg_end] - uncompressed_values[seg_start]) / (seg_len - 1);
    const intercept = uncompressed_values[seg_start];
    var sse: f64 = 0;
    var i: usize = seg_start;
    while (i <= seg_end) : (i += 1) {
        if (uncompressed_values[i] > tester.max_test_value or
            !math.isFinite(uncompressed_values[i])) return Error.UnsupportedInput;

        const scaled_time = @as(f64, @floatFromInt(i - seg_start)); // small numbers: 0,1,2,...
        const pred = intercept + slope * scaled_time;
        const diff = uncompressed_values[i] - pred;
        sse += diff * diff;
    }
    return math.sqrt(sse / seg_len);
}

/// Append `value` of `type` determined at compile time to `compressed_values`.
pub fn appendValue(allocator: Allocator, comptime T: type, value: T, compressed_values: *ArrayList(u8)) !void {
    // Compile-time type check
    switch (@TypeOf(value)) {
        u64, i64, f64, usize => {
            const value_as_bytes: [8]u8 = @bitCast(value);
            try compressed_values.appendSlice(allocator, value_as_bytes[0..]);
        },
        u32, i32, f32 => {
            const value_as_bytes: [4]u8 = @bitCast(value);
            try compressed_values.appendSlice(allocator, value_as_bytes[0..]);
        },
        else => @compileError("Unsupported type for append value function"),
    }
}

/// Append `compressed_value` and `index` to `compressed_values`.
pub fn appendValueAndIndexToArrayList(
    allocator: Allocator,
    compressed_value: f64,
    index: usize,
    compressed_values: *ArrayList(u8),
) !void {
    const valueAsBytes: [8]u8 = @bitCast(compressed_value);
    try compressed_values.appendSlice(allocator, valueAsBytes[0..]);
    const indexAsBytes: [8]u8 = @bitCast(index); // No -1 due to 0 indexing.
    try compressed_values.appendSlice(allocator, indexAsBytes[0..]);
}

/// Read a value of type `T` from `values` starting at `*offset`, advancing `*offset` by `@sizeOf(T)`.
pub fn readOffsetValue(comptime T: type, values: []const u8, offset: *usize) Error!T {
    const offset_delta = @sizeOf(T);
    if (values.len - offset.* < offset_delta) return Error.UnsupportedInput;

    // Read into a fixed-size byte array, then bit-cast.
    var bytes: [@sizeOf(T)]u8 = undefined;
    std.mem.copyForwards(u8, bytes[0..], values[offset.* .. offset.* + offset_delta]);

    offset.* += offset_delta;
    const value: T = @bitCast(bytes);
    return value;
}

/// Test if the RMSE of the linear regression line that fits the points in the segment in `values`
/// is within the `error_bound`.
pub fn testRMSEisWithinErrorBound(
    values: []const f64,
    error_bound: f32,
) !void {
    // At least two points are needed to form a line.
    if (values.len < 2) return;

    const rmse = try computeRMSE(values, 0, values.len - 1);
    try testing.expect(rmse <= error_bound);
}

/// Computes the maximum absolute (Chebyshev, L-inf) error between the actual values and the
/// linear interpolation over a segment of the input array. This function fits a straight
/// line between the values at `seg_start` and `seg_end` in `uncompressed_values`, then
/// calculates the maximum absolute difference between the actual values and the predicted
/// values (from the fitted line) for all indices in the segment `[seg_start, seg_end]`.
pub fn computeMaxAbsoluteError(uncompressed_values: []const f64, seg_start: usize, seg_end: usize) f64 {
    const seg_len: f64 = @floatFromInt(seg_end - seg_start + 1);
    if (seg_len <= 2) return 0.0; // If the segment has less than 3 points, return zero error.

    const slope: f64 = (uncompressed_values[seg_end] - uncompressed_values[seg_start]) / (seg_len - 1);
    const intercept: f64 = uncompressed_values[seg_start] - slope * @as(f64, @floatFromInt(seg_start));

    // Calculate the maximum absolute error of the segment.
    var linf: f64 = 0;
    var i = seg_start;
    while (i <= seg_end) : (i += 1) {
        const pred = slope * @as(f64, @floatFromInt(i)) + intercept; // Predicted value.
        const diff = @abs(uncompressed_values[i] - pred); // Difference between actual and predicted.
        linf = @max(diff, linf);
    }

    // Return max abs.
    return linf;
}

/// Auxiliary function to validate of the decompressed time series is within the error bound of the
/// uncompressed time series. The function returns true if all elements are within the error bound,
/// false otherwise.
pub fn isWithinErrorBound(
    uncompressed_values: []const f64,
    decompressed_values: []const f64,
    error_bound: f32,
) bool {
    if (uncompressed_values.len != decompressed_values.len) {
        return false;
    }

    for (0..uncompressed_values.len) |index| {
        const uncompressed_value = uncompressed_values[index];
        const decompressed_value = decompressed_values[index];
        if (@abs(uncompressed_value - decompressed_value) > error_bound) return false;
    }
    return true;
}

/// Reads a value of compile-time known type `T` from the beginning of the `compressed_values` byte
/// slice. Returns the value if the `compressed_values` contains at least `@sizeOf(T)` bytes. Return
/// an error otherwise.
pub fn readValue(comptime T: type, compressed_values: []const u8) Error!T {
    const size = @sizeOf(T);
    if (size > compressed_values.len) {
        return Error.UnsupportedInput; // Not enough bytes to read the value.
    }
    return @bitCast(compressed_values[0..size].*);
}

/// Returns `true` if two floating-point `value_a` and `value_b` numbers are approximately equal,
/// using both absolute and relative tolerances to account for rounding errors. This function is
/// necessary because direct comparison of floating-point values can fail due to rounding errors
/// and representation limitations inherent in floating-point arithmetic. Absolute tolerance is used
/// for values close to zero, while relative tolerance is used for larger magnitude values to ensure
/// a meaningful comparison. The values are fixed to 1e-12 and 1e-15 for absolute and relative
/// tolerances, respectively, which are suitable for f64 values.
pub fn isApproximatelyEqual(value_a: f64, value_b: f64) bool {
    if (value_a == value_b) return true;
    if (!math.isFinite(value_a) or !math.isFinite(value_b))
        return value_a == value_b; // Handle NaN and infinities.
    const abs_diff = @abs(value_a - value_b);
    const max_abs = @max(@abs(value_a), @abs(value_b));
    return abs_diff <= shared_structs.ABS_EPS or abs_diff <= max_abs * shared_structs.REL_EPS;
}

/// Encodes a signed i64 `value` into an unsigned 64-bit integer (u64) using ZigZag encoding.
/// ZigZag encoding maps small signed integers (both positive and negative) to small
/// unsigned integers, which is useful for variable-length encoding schemes.
pub fn encodeZigZag(value: i64) u64 {
    return @bitCast((value << 1) ^ (value >> 63));
}

/// Decodes an u64 `value` back into a signed 64-bit integer (i64) using ZigZag decoding.
/// This reverses the transformation performed by `zigzagEncode`.
pub fn decodeZigZag(value: u64) i64 {
    // Revert logical shift.
    const shifted_value = value >> 1;
    const last_bit: u64 = value & 1;

    // Make a signed value 0 or -1.
    // last_bit == 0 -> sign = 0.
    // last_bit == 1 -> sign = -1.
    const sign: i64 = -@as(i64, @intCast(last_bit));

    // Reinterpret sign as an unsigned mask: 0 or 0xFFFF...FFFF.
    const sign_mask: u64 = @bitCast(sign);

    return @bitCast(shifted_value ^ sign_mask);
}

/// Encodes an array of u64 `values` using Elias Gamma encoding. Elias Gamma encoding is a
/// universal code that represents integers using a prefix of zero bits followed by the binary
/// representation of the integer. This encoding is efficient for small positive integers and is
/// commonly used in data compression. The memory `allocator` is used to allocate the resulting
/// encoded data. The function returns an `ArrayList(u8)` containing the encoded data, or an error
/// if the input is unsupported or if memory allocation fails.
pub fn encodeEliasGamma(values: []const u64, encoded_values: *ArrayList(u8)) !void {
    var stream = io.fixedBufferStream(encoded_values.items);
    var bit_writer = shared_structs.bitWriter(.little, stream.writer());

    for (values) |value| {
        if (value == 0) {
            // Elias Gamma encoding is not defined for zero.
            return Error.UnsupportedInput;
        }
        // Calculate n = floor(log2(value)).
        const nbits: u16 = math.log2_int(u64, value);

        // Write `nbits` zero bits followed by the value itself.
        for (0..nbits) |_| {
            try bit_writer.writeBits(@as(u1, 0b0), 1);
        }
        try bit_writer.writeBits(value, nbits + 1);
    }
    try bit_writer.flushBits();
}

/// Decodes an array of u8 `compressed_values` previously encoded using Elias Gamma encoding.
/// The memory `allocator` is used to allocate the resulting decoded data. The function returns
/// an `ArrayList(u64)` containing the decoded data, or an error if the input is unsupported,
/// meaning it was not encoded using Elias Gamma encoder or corrupted. Any other memory related
/// error is also returned.
pub fn decodeEliasGamma(
    allocator: Allocator,
    compressed_values: []const u8,
    decoded_values: *ArrayList(u64),
) !void {
    // Create bit reader over full byte slice.
    var stream = std.io.fixedBufferStream(compressed_values);
    var bit_reader = shared_structs.bitReader(.big, stream.reader());

    while (true) {
        // Count leading zeros.
        var leading_bits_number: u16 = 0;
        while (true) {
            const leading_bit: u8 = bit_reader.readBitsNoEof(u8, 1) catch |err| {
                if (err == error.EndOfStream) {
                    // No more values to decode.
                    return;
                }
                // Some other error occurred.
                return Error.ByteStreamError;
            };

            if (leading_bit == 1) break;
            leading_bits_number += 1;
        }

        // Read the value bits.
        if (leading_bits_number == 0) {
            try decoded_values.append(allocator, 1);
        } else {
            const suffix: u64 = bit_reader.readBitsNoEof(u64, leading_bits_number) catch |err| {
                // If we cannot read all k bits, the stream is malformed.
                if (err == error.EndOfStream) {
                    return Error.ByteStreamError;
                }
                return Error.ByteStreamError;
            };
            // Combine leading 1 bit with suffix to form the decoded value.
            const value: u64 = (@as(u64, 1) << @intCast(leading_bits_number)) | suffix;
            try decoded_values.append(allocator, value);
        }
    }
}

/// Convert a floating f64 `value` to its u64 representation, ensuring the sign bit is preserved
/// in the most significant bit. This is useful for comparing floating-point values in a way that
/// respects their ordering, including negative values. The function returns the `u64`, where the
/// sign bit is preserved in the most significant bit.
pub fn floatBitsOrdered(value: f64) u64 {
    const value_bits: u64 = @bitCast(value);
    // If negative: flip all bits (mirror to top range).
    return if ((value_bits >> 63) == 1)
        ~value_bits
    else
        value_bits | (@as(u64, 1) << 63);
}

/// Convert a u64 `value` back to its original `f64` representation, ensuring the sign bit is
/// restored correctly. This is useful for decompressing values that were quantized and
/// bit-packed, preserving the original ordering of the floating-point values.
pub fn orderedBitsToFloat(value: u64) f64 {
    return if ((value >> 63) == 1)
        @bitCast(value & ~(@as(u64, 1) << 63))
    else
        @bitCast(~value);
}

/// Creates a uniform quantization bucket based on the provided `error_bound`. In theory, the bucket
/// size is `2 * error_bound`, since uniform quantization guarantees a maximum reconstruction error
/// of `bucket_size / 2`. However, in practice, floating-point rounding and cancellation can cause
/// the actual error to slightly exceed the theoretical bound. To improve numerical stability, the
/// function shrink the bucket slightly and use `1.998 * error_bound` instead of `2 * error_bound`.
/// This ~0.1% safety margin helps ensure that the maximum decompression error still satisfies the
/// user-provided `error_bound`.
pub fn createQuantizationBucket(error_bound: f32) f64 {
    return @floatCast(1.998 * error_bound);
}

test "zigzag can encode and decode small signed integers correctly" {
    const default_random = tester.getDefaultRandomGenerator();
    const number_of_tests = tester.generateNumberOfValues(default_random);
    // Using math.maxInt(i64) would generate only large integers due to the biases.
    // Thus, limit the range to ensure that small integers are generated.
    for (0..number_of_tests) |_| {
        const original = tester.generateBoundRandomInteger(
            i64,
            -1e5,
            1e5,
            default_random,
        );
        const encoded = encodeZigZag(original);
        const decoded = decodeZigZag(encoded);
        try testing.expectEqual(decoded, original);
    }
}

test "zigzag can encode and decode big signed integers correctly" {
    const default_random = tester.getDefaultRandomGenerator();
    const number_of_tests = tester.generateNumberOfValues(default_random);
    // Using math.maxInt(i64) would generate only large integers due to the biases.
    for (0..number_of_tests) |_| {
        const original = tester.generateBoundRandomInteger(
            i64,
            -math.maxInt(i64),
            math.maxInt(i64),
            default_random,
        );
        const encoded = encodeZigZag(original);
        const decoded = decodeZigZag(encoded);
        try testing.expectEqual(decoded, original);
    }

    for (0..number_of_tests) |_| {
        const original = tester.generateBoundRandomInteger(
            i64,
            -1e5,
            1e5,
            default_random,
        );
        const encoded = encodeZigZag(original);
        const decoded = decodeZigZag(encoded);
        try testing.expectEqual(decoded, original);
    }
}

test "encodeEliasGamma can encode simple values correctly" {
    const allocator = std.testing.allocator;

    const vals = [_]u64{ 15, 7, 3, 1 };
    var encoded_values = ArrayList(u8).init(allocator);
    defer encoded_values.deinit();
    try encodeEliasGamma(&vals, &encoded_values);

    // The expected encoded values for 15, 7, 3, 1 are:
    // at encoded_values.items[0] -> 00011110 == 30
    // at encoded_values.items[1] -> 01110111 == 119
    try testing.expect(encoded_values.items[0] == 30);
    try testing.expect(encoded_values.items[1] == 119);
}

test "encodeEliasGamma cannot encode zero value" {
    const allocator = std.testing.allocator;

    const vals = [_]u64{ 15, 7, 0, 3, 1 };
    var encoded_values = ArrayList(u8).init(allocator);
    defer encoded_values.deinit();

    encodeEliasGamma(&vals, &encoded_values) catch |err| {
        try testing.expect(err == Error.UnsupportedInput);
        return;
    };
}

test "decodeEliasGamma can encode and decode simple values correctly" {
    const allocator = std.testing.allocator;

    const uncompressed_values = [_]u64{ 15, 7, 3, 1 };
    var encoded_values = ArrayList(u8).init(allocator);
    defer encoded_values.deinit();
    try encodeEliasGamma(&uncompressed_values, &encoded_values);

    var decoded_values = ArrayList(u64).init(allocator);
    defer decoded_values.deinit();
    try decodeEliasGamma(encoded_values.items, &decoded_values);

    for (decoded_values.items, 0..) |decoded_value, index| {
        try testing.expect(decoded_value == uncompressed_values[index]);
    }
}

test "decodeEliasGamma can encode and decode complex values correctly" {
    const allocator = std.testing.allocator;
    var uncompressed_values = ArrayList(u64).init(allocator);
    defer uncompressed_values.deinit();

    const default_random = tester.getDefaultRandomGenerator();
    const number_of_tests = tester.generateNumberOfValues(default_random);
    for (0..number_of_tests) |_| {
        const value: u64 = tester.generateBoundRandomInteger(
            u64,
            1, // elias gamma is only defined for positive integers.
            tester.max_test_value,
            default_random,
        );

        try uncompressed_values.append(allocator, value);
    }

    var encoded_values = ArrayList(u8).init(allocator);
    defer encoded_values.deinit();
    try encodeEliasGamma(uncompressed_values.items, &encoded_values);

    var decoded_values = ArrayList(u64).init(allocator);
    defer decoded_values.deinit();
    try decodeEliasGamma(encoded_values.items, &decoded_values);

    for (decoded_values.items, 0..) |decoded_value, index| {
        try testing.expect(decoded_value == uncompressed_values.items[index]);
    }
}
