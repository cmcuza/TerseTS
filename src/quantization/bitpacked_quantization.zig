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

//! Implementation of the "Fixed-width Uniform Quantization Method" based on the description at
//! https://en.wikipedia.org/wiki/Quantization_(signal_processing). Directly after quantization,
//! the values are bit-packed using a "Fixed-Length Bit-Packing Scheme" based on the decription
//! at https://kinematicsoup.com/news/2016/9/6/data-compression-bit-packing-101. The combination
//! of quantization of bit-packing has been shown to be effective in compressing time series data.
//! More information in the paper: "Serf: Streaming Error-Bounded Floating-Point Compression".
//! by Li, Ruiyuan, et al. SIGMOD 2025, 1-27, https://doi.org/10.1145/3725353.

const std = @import("std");
const ArrayList = std.ArrayList;
const math = std.math;
const mem = std.mem;
const testing = std.testing;

const tersets = @import("../tersets.zig");
const Method = tersets.Method;
const Error = tersets.Error;
const tester = @import("../tester.zig");

/// Compress `uncompressed_values` within error_bound using "Bucket Quantization" and a
/// "Fixed-length Bit-Packing". The function writes the result to `compressed_values`. The
/// `compressed_values` includes the bit width, original length and smallest value so that it
/// can be decompressed. The `allocator` is used for memory management of intermediates containers.
/// If an error occurs it is returned.
pub fn compress(
    allocator: mem.Allocator,
    uncompressed_values: []const f64,
    compressed_values: *ArrayList(u8),
    error_bound: f32,
) Error!void {
    // Ensure the compressed values are not empty.
    if (uncompressed_values.len == 0) return Error.UnsupportedInput;

    // Ensure the error bound is non-negative.
    if (error_bound < 0.0) return Error.UnsupportedErrorBound;

    // Find the minimum value.
    var min_val = uncompressed_values[0];
    for (uncompressed_values) |value| {
        if (!math.isFinite(value) or value > 1e15) return Error.UnsupportedInput;
        if (value < min_val) min_val = value;
    }

    // Append the minimum value to the header of the compressed values.
    try appendValue(f64, min_val, compressed_values);

    // All values will map to the closest bucket based on the bucket_size.
    const bucket_size: f32 = 2 * error_bound;

    // Append the minimum value to the header of the compressed values.
    try appendValue(f32, bucket_size, compressed_values);

    // Intermadiate quantized values.
    var quantized_values = ArrayList(usize).init(allocator);
    defer quantized_values.deinit();

    const raw_bits_min_value: usize = floatBitsOrdered(min_val);
    // Quantize and shift values (ensure all quantized values are >= 0).
    var quantized_value: usize = 0;
    for (uncompressed_values) |value| {
        // Bucket quantization: round to nearest multiple.
        if (error_bound == 0.0) {
            // If error bound is zero.
            const raw_bits_value: usize = floatBitsOrdered(value);
            quantized_value = raw_bits_value - raw_bits_min_value;
        } else {
            quantized_value = @intFromFloat(@floor((value - min_val) / bucket_size + 0.5));
        }
        try quantized_values.append(quantized_value);
    }

    // Step 5: Bit-pack quantized values using fixed-length header scheme.
    const small_limit = 0xFF; // Fits in 8 bits.
    const medium_limit = 0xFFFF; // Fits in 16 bits.
    const large_limit = 0xFFFFFFFF; // Fits in 32 bits.

    // Bit-wise packing with fixed-length header.
    var bit_writer = std.io.bitWriter(.little, compressed_values.writer());

    for (quantized_values.items) |val| {
        if (val <= small_limit) {
            try bit_writer.writeBits(@as(u1, 0b0), 1); // header '0'.
            try bit_writer.writeBits(@as(u1, 0b0), 1); // header '0'.
            try bit_writer.writeBits(@as(u8, @intCast(val)), 8); // 8-bit value.
        } else if (val <= medium_limit) {
            try bit_writer.writeBits(@as(u1, 0b0), 1); // header '0'.
            try bit_writer.writeBits(@as(u1, 0b1), 1); // header '1'.
            try bit_writer.writeBits(@as(u16, @intCast(val)), 16); // 16-bit value.
        } else if (val <= large_limit) {
            try bit_writer.writeBits(@as(u1, 0b1), 1); // header '1'.
            try bit_writer.writeBits(@as(u1, 0b0), 1); // header '0'.
            try bit_writer.writeBits(@as(u32, @intCast(val)), 32); // 32-bit value.
        } else {
            try bit_writer.writeBits(@as(u1, 0b1), 1); // header '1'.
            try bit_writer.writeBits(@as(u1, 0b1), 1); // header '1'.
            try bit_writer.writeBits(@as(u64, @intCast(val)), 64); // 64-bit value.
        }
    }

    try bit_writer.flushBits();
}

/// Decompress `compressed_values` produced by "Bucket Quantization" and "Bit-Packing". The function
/// writes the result to `decompressed_values`. If an error occurs it is returned.
pub fn decompress(
    compressed_values: []const u8,
    decompressed_values: *ArrayList(f64),
) Error!void {
    // Ensure the compressed values are not empty, i.e., at least the header is present.
    if (compressed_values.len < 12) return Error.UnsupportedInput;

    // Read min_val and bucket_size from the header.
    const offset: f64 = @bitCast(compressed_values[0..8].*);
    const bucket_size: f32 = @bitCast(compressed_values[8..12].*);

    // Create a bit reader from remaining bytes.
    var stream = std.io.fixedBufferStream(compressed_values[12..]);
    var bit_reader = std.io.bitReader(.little, stream.reader());
    var decompressed_value: f64 = 0.0;

    // Convert the offset to an ordered bit value.
    const bits_ordered_offset = floatBitsOrdered(offset);

    // Read each quantized value based on fixed-length header.
    while (true) {
        // Read the first two bits to determine the length of the quantized value.
        // If the stream ends before reading the bits, we break the loop.
        const header_1: u8 = bit_reader.readBitsNoEof(u8, 1) catch break;
        const header_2: u8 = bit_reader.readBitsNoEof(u8, 1) catch break;
        var quantized_value: usize = 0;

        if (header_1 == 0) {
            if (header_2 == 0) {
                // 1-byte value.
                quantized_value = bit_reader.readBitsNoEof(u8, 8) catch |err| {
                    // The loop termination condition is too optimistic since some padding higher
                    // than 2 bits can be still left. However, if this is the last value, and we
                    // try to read 8 bits, we should get an `EndOfStream`. If we get a different
                    // error, we return a `ByteStreamError`.
                    if (err == error.EndOfStream) break;
                    return Error.ByteStreamError;
                };
            } else {
                // 2-byte value.
                quantized_value = bit_reader.readBitsNoEof(u16, 16) catch return Error.ByteStreamError;
            }
        } else {
            if (header_2 == 0) {
                // 4-byte value.
                quantized_value = bit_reader.readBitsNoEof(u32, 32) catch return Error.ByteStreamError;
            } else {
                // 8-byte value.
                quantized_value = bit_reader.readBitsNoEof(u64, 64) catch return Error.ByteStreamError;
            }
        }

        if (bucket_size == 0.0) {
            // If bucket size is zero, we assume the values were not quantized and are stored as raw bits.
            const raw_bits = quantized_value + bits_ordered_offset;
            decompressed_value = orderedBitsToFloat(raw_bits);
        } else {
            // Reconstruct value from quantized_value and append to decompressed_value.
            decompressed_value = offset + @as(f64, @floatFromInt(quantized_value)) * bucket_size;
        }
        try decompressed_values.append(decompressed_value);
    }
}

/// Append `value` of `type` determined at compile time to `compressed_values`.
fn appendValue(comptime T: type, value: T, compressed_values: *ArrayList(u8)) !void {
    // Compile-time type check.
    switch (@TypeOf(value)) {
        f64, usize => {
            const value_as_bytes: [8]u8 = @bitCast(@as(T, value));
            try compressed_values.appendSlice(value_as_bytes[0..]);
        },
        f32 => {
            const value_as_bytes: [4]u8 = @bitCast(@as(T, value));
            try compressed_values.appendSlice(value_as_bytes[0..]);
        },
        else => @compileError("Unsupported type for append value function"),
    }
}

/// Convert a floating-point `value` to its bit representation, ensuring the sign bit is preserved
/// in the most significant bit. This is useful for comparing floating-point values in a way that
/// respects their ordering, including negative values. The function returns the bit representation
/// as a `u64`, where the sign bit is preserved in the most significant bit.
fn floatBitsOrdered(value: f64) u64 {
    const value_bits: u64 = @bitCast(value);
    // If negative: flip all bits (mirror to top range).
    return if ((value_bits >> 63) == 1)
        ~value_bits
    else
        value_bits | (@as(u64, 1) << 63);
}

/// Convert a bit representation of a floating-point `value_bits` back to its original `f64` value,
/// ensuring the sign bit is restored correctly. This is useful for decompressing values that were
/// quantized and bit-packed, preserving the original ordering of the floating-point values.
fn orderedBitsToFloat(value_bits: u64) f64 {
    return if ((value_bits >> 63) == 1)
        @bitCast(value_bits & ~(@as(u64, 1) << 63))
    else
        @bitCast(~value_bits);
}

test "bitpacked quantization can compress and decompress bounded values" {
    const allocator = testing.allocator;
    const error_bound = tester.generateBoundedRandomValue(f32, 0, 1e6, undefined);

    var uncompressed_values = ArrayList(f64).init(allocator);
    defer uncompressed_values.deinit();

    // Generate 500 random values within the range of -1e7 to 1e7.
    for (0..5) |_| {
        try tester.generateBoundedRandomValues(&uncompressed_values, -1e13, 1e13, undefined);
    }

    try tester.testCompressAndDecompress(
        uncompressed_values.items,
        allocator,
        Method.BitPackedQuantization,
        error_bound,
        tersets.isWithinErrorBound,
    );
}

test "bitpacked quantization can compress and decompress bounded values at different scales" {
    const allocator = testing.allocator;
    const error_bound = tester.generateBoundedRandomValue(f32, 0, 1e3, undefined);

    var uncompressed_values = ArrayList(f64).init(allocator);
    defer uncompressed_values.deinit();

    try tester.generateBoundedRandomValues(&uncompressed_values, -1, 1, undefined);
    try tester.generateBoundedRandomValues(&uncompressed_values, -1e2, 1e2, undefined);
    try tester.generateBoundedRandomValues(&uncompressed_values, -1e4, 1e4, undefined);
    try tester.generateBoundedRandomValues(&uncompressed_values, -1e6, 1e6, undefined);
    try tester.generateBoundedRandomValues(&uncompressed_values, -1e8, 1e8, undefined);

    try tester.testCompressAndDecompress(
        uncompressed_values.items,
        allocator,
        Method.BitPackedQuantization,
        error_bound,
        tersets.isWithinErrorBound,
    );
}

test "bitpacked quantization can compress and decompress with zero error bound at different scales" {
    const allocator = testing.allocator;
    const error_bound = 0;

    var uncompressed_values = ArrayList(f64).init(allocator);
    defer uncompressed_values.deinit();

    try tester.generateBoundedRandomValues(&uncompressed_values, -1, 1, undefined);
    try tester.generateBoundedRandomValues(&uncompressed_values, -1e2, 1e2, undefined);
    try tester.generateBoundedRandomValues(&uncompressed_values, -1e4, 1e4, undefined);
    try tester.generateBoundedRandomValues(&uncompressed_values, -1e6, 1e6, undefined);
    try tester.generateBoundedRandomValues(&uncompressed_values, -1e8, 1e8, undefined);
    try tester.generateBoundedRandomValues(&uncompressed_values, -1e14, 1e14, undefined);

    try tester.testCompressAndDecompress(
        uncompressed_values.items,
        allocator,
        Method.BitPackedQuantization,
        error_bound,
        tersets.isWithinErrorBound,
    );
}

test "bitpacked quantization always reduces size of time series" {
    const allocator = testing.allocator;
    // Generate a random error bound between 10 and 1000, which will be used for quantization.
    const error_bound = @floor(tester.generateBoundedRandomValue(
        f32,
        1e1,
        1e3,
        undefined,
    )) * 0.1;

    var uncompressed_values = ArrayList(f64).init(allocator);
    defer uncompressed_values.deinit();

    // Generate 500 random values within different ranges. Even if some values require 8 bytes
    // to be stored, the quantization should reduce the size of the time series since some
    // values require less than 8 bytes to be stored after quantization.
    try tester.generateBoundedRandomValues(&uncompressed_values, -1, 1, undefined);
    try tester.generateBoundedRandomValues(&uncompressed_values, -1e2, 1e2, undefined);
    try tester.generateBoundedRandomValues(&uncompressed_values, -1e4, 1e4, undefined);
    try tester.generateBoundedRandomValues(&uncompressed_values, -1e6, 1e6, undefined);
    try tester.generateBoundedRandomValues(&uncompressed_values, -1e8, 1e8, undefined);

    var compressed_values = ArrayList(u8).init(allocator);
    defer compressed_values.deinit();

    try compress(
        allocator,
        uncompressed_values.items,
        &compressed_values,
        error_bound,
    );

    // Considering the range of the input data, the compressed values should always be smaller.
    try testing.expect(uncompressed_values.items.len * 8 > compressed_values.items.len);
}
