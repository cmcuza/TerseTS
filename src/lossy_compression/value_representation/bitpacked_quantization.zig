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

//! Implementation of a basic fixed-width uniform quantization scheme followed by fixed-length
//! bit-packing to compress floating-point values. Quantization follows the standard scalar
//! quantization approach described in:
//! Gray & Neuhoff, "Quantization", IEEE Trans. Info. Theory, 1998.
//! https://doi.org/10.1109/18.720541.
//! Bit-packing follows common techniques for efficient integer encoding, as seen in:
//! Lemire et al., "SIMD Compression and the Intersection of Sorted Integers", 2016.
//! https://doi.org/10.1002/spe.2326.

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
const BitWriter = shared_structs.BitWriter;
const configuration = @import("../../configuration.zig");
const Method = tersets.Method;
const Error = tersets.Error;
const tester = @import("../../tester.zig");

const shared_functions = @import("../../utilities/shared_functions.zig");

/// Compress `uncompressed_values` within error_bound using "Bucket Quantization" and a
/// "Fixed-length Bit-Packing". The function writes the result to `compressed_values`. The
/// `compressed_values` includes the bit width, original length and smallest value so that it
/// can be decompressed. The `allocator` is used for memory management of intermediates containers
/// and the `method_configuration` parser. If an error occurs it is returned.
pub fn compress(
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

    const error_bound: f32 = parsed_configuration.abs_error_bound;

    // Find the minimum and maximum value.
    var minimum_value = uncompressed_values[0];
    var maximum_value = uncompressed_values[0];

    for (uncompressed_values) |value| {
        if (!math.isFinite(value) or @abs(value) > tester.max_test_value) return Error.UnsupportedInput;
        if (value < minimum_value) minimum_value = value;
        if (value > maximum_value) maximum_value = value;
    }

    // Append the minimum value to the header of the compressed values.
    try shared_functions.appendValue(allocator, f64, minimum_value, compressed_values);

    // All values will map to the closest bucket based on the bucket_size.
    const bucket_size: f64 = shared_functions.createQuantizationBucket(error_bound);

    if (error_bound != 0.0) {
        // If `bucket_size` is so small that adding it to `minimum_value` does nothing, then the
        // reconstruction grid collapses at `minimum_value` due to f64 precision.
        if (minimum_value + bucket_size == minimum_value) {
            if (@abs(maximum_value - minimum_value) > error_bound) return Error.UnsupportedInput;
        } else {
            // Check whether maximum_value is representable within the error bound under the
            // same quantize+reconstruct. If not, the method cannot be applied to this input since it
            // would violate the error bound.
            const maximum_quantized_value: f64 = @round((maximum_value - minimum_value) / bucket_size);
            const reconstructed_maximum_value: f64 = minimum_value + maximum_quantized_value * bucket_size;

            if (@abs(reconstructed_maximum_value - maximum_value) > error_bound) return Error.UnsupportedInput;
        }
    }

    // Append the minimum value to the header of the compressed values.
    try shared_functions.appendValue(allocator, f64, bucket_size, compressed_values);

    //Intermediate quantized values.
    var quantized_values = ArrayList(u64).empty;
    defer quantized_values.deinit(allocator);

    const u64_minimum_value: u64 = shared_functions.floatBitsOrdered(minimum_value);

    // Quantize each value by mapping it to a discrete bucket index.
    // If the error_bound is zero, we compute the difference between the
    // value and the minimum value, ensuring all resulting integers are >= 0.
    // For non-zero error_bound, we apply fixed-width bucket quantization
    // using the defined bucket size (1.998 × error_bound).
    var quantized_value: u64 = 0;
    for (uncompressed_values) |value| {
        if (error_bound == 0.0) {
            // Bit-diff quantization for the lossless case.
            const u64_value: u64 = shared_functions.floatBitsOrdered(value);
            quantized_value = u64_value - u64_minimum_value;
        } else {
            // Fixed-width bucket quantization with rounding.
            quantized_value = @intFromFloat(@round((value - minimum_value) / bucket_size));
        }
        try quantized_values.append(allocator, quantized_value);
    }

    // Step 5: Bit-pack quantized values using fixed-length header scheme.
    const small_limit = 0xFF; // Fits in 8 bits.
    const medium_limit = 0xFFFF; // Fits in 16 bits.
    const large_limit = 0xFFFFFFFF; // Fits in 32 bits.

    // Bit-wise packing with fixed-length header.
    const writer = compressed_values.writer(allocator);
    var bit_writer = shared_structs.bitWriter(.little, writer);

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
    allocator: Allocator,
    compressed_values: []const u8,
    decompressed_values: *ArrayList(f64),
) Error!void {
    // Ensure the compressed values are not empty, i.e., at least the header is present.
    if (compressed_values.len < 16) return Error.UnsupportedInput;

    // Read minimum_value and bucket_size from the header.
    const minimum_value: f64 = @bitCast(compressed_values[0..8].*);
    const bucket_size: f64 = @bitCast(compressed_values[8..16].*);

    // Create a bit reader from remaining bytes.
    var stream = io.fixedBufferStream(compressed_values[16..]);
    var bit_reader = shared_structs.bitReader(.little, stream.reader());
    var decompressed_value: f64 = 0.0;

    // Convert minimum_value to its ordered bit representation.
    // “Ordered bit representation” means a transformation that makes float bits sortable as integers.
    // This ensures correct decoding when using raw bit differences.
    const bits_ordered_minimum_value = shared_functions.floatBitsOrdered(minimum_value);

    // Read each quantized value based on fixed-length header.
    while (true) {
        // Read two control bits that encode the length of the upcoming value.
        // If the stream ends before reading them, we break the loop.
        const length_prefix_1: u8 = bit_reader.readBitsNoEof(u8, 1) catch break;
        const length_prefix_2: u8 = bit_reader.readBitsNoEof(u8, 1) catch break;
        var quantized_value: u64 = 0;

        if (length_prefix_1 == 0) {
            if (length_prefix_2 == 0) {
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
            if (length_prefix_2 == 0) {
                // 4-byte value.
                quantized_value = bit_reader.readBitsNoEof(u32, 32) catch return Error.ByteStreamError;
            } else {
                // 8-byte value.
                quantized_value = bit_reader.readBitsNoEof(u64, 64) catch return Error.ByteStreamError;
            }
        }

        if (bucket_size == 0.0) {
            // If bucket size is zero, we assume the values were not quantized and are stored as u64 directly.
            const raw_u64 = quantized_value + bits_ordered_minimum_value;
            decompressed_value = shared_functions.orderedBitsToFloat(raw_u64);
        } else {
            // Reconstruct value from quantized_value and append to decompressed_value.
            decompressed_value = minimum_value + @as(f64, @floatFromInt(quantized_value)) * bucket_size;
        }
        try decompressed_values.append(allocator, decompressed_value);
    }
}

test "bitpacked quantization can compress and decompress bounded values" {
    const allocator = testing.allocator;
    // Use only tighted bounded random values for this test.
    // BitPackedQuantization requires bounded values to operate correctly.
    // Other data distributions may generate unbounded values which are not supported.
    const data_distributions = &[_]tester.DataDistribution{.TightlyBoundedRandomValues};

    // This function evaluates BitPackedQuantization using all data distribution stored in
    // `data_distribution`.
    try tester.testErrorBoundedCompressionMethod(
        allocator,
        Method.BitPackedQuantization,
        data_distributions,
    );
}

test "bitpacked quantization cannot compress and decompress nan values" {
    const allocator = testing.allocator;
    const uncompressed_values = [3]f64{ 343.0, math.nan(f64), 520.0 };
    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);

    const method_configuration =
        \\ {"abs_error_bound": 0.1}
    ;

    compress(
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
        "The BitPacked Quantization method cannot compress nan values",
        .{},
    );
}

test "bitpacked quantization cannot compress and decompress unbounded values" {
    const allocator = testing.allocator;
    const uncompressed_values = [3]f64{ 343.0, 1e20, 520.0 };
    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);

    const method_configuration =
        \\ {"abs_error_bound": 0.1}
    ;

    compress(
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
        "The BitPacked Quantization method cannot compress unbounded values",
        .{},
    );
}

test "bitpacked quantization can compress and decompress bounded values at different scales" {
    const allocator = testing.allocator;
    const error_bound = tester.generateBoundedRandomValue(f32, 0, 1e3, null);

    var uncompressed_values = ArrayList(f64).empty;
    defer uncompressed_values.deinit(allocator);

    try tester.generateBoundedRandomValues(allocator, &uncompressed_values, -1, 1, null);
    try tester.generateBoundedRandomValues(allocator, &uncompressed_values, -1e2, 1e2, null);
    try tester.generateBoundedRandomValues(allocator, &uncompressed_values, -1e4, 1e4, null);
    try tester.generateBoundedRandomValues(allocator, &uncompressed_values, -1e6, 1e6, null);
    try tester.generateBoundedRandomValues(allocator, &uncompressed_values, -1e8, 1e8, null);

    try tester.testCompressAndDecompress(
        allocator,
        uncompressed_values.items,
        Method.BitPackedQuantization,
        error_bound,
        shared_functions.isWithinErrorBound,
    );
}

test "bitpacked quantization can compress and decompress with zero error bound at different scales" {
    const allocator = testing.allocator;
    const error_bound = 0;

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
        Method.BitPackedQuantization,
        error_bound,
        shared_functions.isWithinErrorBound,
    );
}

test "bitpacked quantization always reduces size of time series" {
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
    try tester.generateBoundedRandomValues(allocator, &uncompressed_values, 0, 1, null);
    try tester.generateBoundedRandomValues(allocator, &uncompressed_values, 0, 1e2, null);
    try tester.generateBoundedRandomValues(allocator, &uncompressed_values, 1e2, 1e4, null);
    try tester.generateBoundedRandomValues(allocator, &uncompressed_values, 1e4, 1e6, null);
    try tester.generateBoundedRandomValues(allocator, &uncompressed_values, 1e6, 1e8, null);

    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);

    const method_configuration = try std.fmt.allocPrint(
        allocator,
        "{{\"abs_error_bound\": {d}}}",
        .{error_bound},
    );
    defer allocator.free(method_configuration);

    try compress(
        allocator,
        uncompressed_values.items,
        &compressed_values,
        method_configuration,
    );

    // Considering the range of the input data, the compressed values should always be smaller.
    try testing.expect(uncompressed_values.items.len * 8 > compressed_values.items.len);
}

test "check bit-quantization configuration parsing" {
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
    try compress(
        allocator,
        uncompressed_values,
        &compressed_values,
        method_configuration,
    );
}

test "bitpacked quantization detects grid collapse due to precision loss" {
    // Regression test for the precision guard: when bucket_size is so small that
    // minimum_value + bucket_size == minimum_value (grid collapses), and the range exceeds error_bound,
    // compress should return Error.UnsupportedInput.
    const allocator = testing.allocator;

    // Use a very large minimum_value and a tiny error_bound to create a scenario where
    // bucket_size (1.998 * error_bound) is too small relative to minimum_value.
    // This triggers the grid collapse condition: minimum_value + bucket_size == minimum_value
    const minimum_value: f64 = 1e14;
    const maximum_value: f64 = minimum_value + 1.0; // Range of 1.0.
    const error_bound: f32 = 1e-10; // Tiny error bound.

    const uncompressed_values = &[2]f64{ minimum_value, maximum_value };

    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);

    const method_configuration = try std.fmt.allocPrint(
        allocator,
        "{{\"abs_error_bound\": {e}}}",
        .{error_bound},
    );
    defer allocator.free(method_configuration);

    // The grid collapse condition should be detected and return UnsupportedInput.
    compress(
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
        "Expected compression to fail due to grid collapse (minimum_value + bucket_size == minimum_value) and range exceeding error_bound",
        .{},
    );
}
