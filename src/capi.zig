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

//! Provides a C-API for TerseTS.

const std = @import("std");
const math = std.math;
const testing = std.testing;
const mem = std.mem;
const ArrayList = std.ArrayList;

const tersets = @import("tersets.zig");
const Error = tersets.Error;
const Method = tersets.Method;

/// Global memory allocator used by tersets.
var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();

/// A pointer to uncompressed values and the number of values.
pub const UncompressedValues = Array(f64);

/// A pointer to compressed values and the number of bytes.
pub const CompressedValues = Array(u8);

/// A pointer to indices values and the number of values.
pub const IndicesValues = Array(usize);

/// A pointer to coefficient values and the number of values.
pub const CoefficientsValues = Array(f64);

/// Compress `uncompressed_values` to `compressed_values` according to `configuration`.
/// The General Purpose Allocator `allocator` is passed as a parameter to tersets for
/// memory management in the compression methods. On success zero is returned, and the
/// following non-zero values are returned on errors:
/// - 1) Unknown compression method.
/// - 2) Unsupported input.
/// - 3) Invalid configuration.
/// - 4) Corrupted compressed data.
/// - 5) Item not found.
/// - 6) Empty convex hull.
/// - 7) Empty queue.
/// - 8) Byte stream error.
/// - 9) Unsupported method.
/// - 10) Out of memory.
export fn compress(
    uncompressed_values_array: UncompressedValues,
    compressed_values_array: *CompressedValues,
    method_index: u8,
    configuration: [*:0]const u8,
) i32 {
    const uncompressed_values = uncompressed_values_array.data[0..uncompressed_values_array.len];

    const configuration_slice: []const u8 = mem.span(configuration);

    // Returning 1 is equivalent to returning Error.UnknownMethod.
    if (method_index > tersets.getMaxMethodIndex()) return 1;

    const method: Method = @enumFromInt(method_index);

    var compressed_values = tersets.compress(
        allocator,
        uncompressed_values,
        method,
        configuration_slice,
    ) catch |err| return errorToInt(err);

    // Convert the ArrayList into an owned slice with exact length. This call allocates a new buffer
    // sized precisely to `len` and transfers ownership of the data from the ArrayList to the
    // caller. Without this step, freeing later with `len` instead of `capacity` would corrupt the
    // allocator.
    const data_slice = compressed_values.toOwnedSlice(allocator) catch |err| return errorToInt(err);

    compressed_values_array.data = data_slice.ptr;
    compressed_values_array.len = data_slice.len;

    return 0;
}

/// Decompress `compressed_values` to `uncompressed_values`. The method is encoded in the last byte
/// of `compressed_values`. The General Purpose Allocator `allocator` is passed as a parameter to
/// tersets for memory management in the decompression methods. On success zero is returned, and the
/// following non-zero values are returned on errors:
/// - 1) Unknown compression method.
/// - 2) Unsupported input.
/// - 3) Invalid configuration.
/// - 4) Corrupted compressed data.
/// - 5) Item not found.
/// - 6) Empty convex hull.
/// - 7) Empty queue.
/// - 8) Byte stream error.
/// - 9) Unsupported method.
/// - 10) Out of memory.
export fn decompress(
    compressed_values_array: CompressedValues,
    decompressed_values_array: *UncompressedValues,
) i32 {
    const compressed_values = compressed_values_array.data[0..compressed_values_array.len];

    var decompressed_values = tersets.decompress(
        allocator,
        compressed_values,
    ) catch |err| return errorToInt(err);

    // Convert the ArrayList into an owned slice with exact length. This call allocates a new buffer
    // sized precisely to `len` and transfers ownership of the data from the ArrayList to the
    // caller. Without this step, freeing later with `len` instead of `capacity` would corrupt the
    // allocator.
    const data_slice = decompressed_values.toOwnedSlice(allocator) catch |err| return errorToInt(err);

    decompressed_values_array.data = data_slice.ptr;
    decompressed_values_array.len = data_slice.len;

    return 0;
}

/// Extracts indices and coefficients from a compressed buffer using the method encoded in the last byte.
/// On success, fills `indices_values_array` and `coefficients_values_array` with extracted values.
/// Extraction layouts depend on the compression method; see `extractors.zig` for details.
/// On success zero is returned, and the following non-zero values are returned on errors:
/// - 1) Unknown compression method.
/// - 2) Unsupported input.
/// - 3) Invalid configuration.
/// - 4) Corrupted compressed data.
/// - 5) Item not found.
/// - 6) Empty convex hull.
/// - 7) Empty queue.
/// - 8) Byte stream error.
/// - 9) Unsupported method.
/// - 10) Out of memory.
export fn extract(
    compressed_values_array: CompressedValues,
    indices_values_array: *IndicesValues,
    coefficients_values_array: *CoefficientsValues,
) i32 {
    const compressed_values = compressed_values_array.data[0..compressed_values_array.len];
    if (compressed_values.len == 0) return 2;

    var indices = ArrayList(u64).empty;
    var coefficients = ArrayList(f64).empty;

    // Return indices and coefficients together. We need to put the size of indices at front in a
    // u32 to split the values.
    tersets.extract(
        allocator,
        compressed_values,
        &indices,
        &coefficients,
    ) catch |e| return errorToInt(e);

    const indices_slice = indices.toOwnedSlice(allocator) catch |err| return errorToInt(err);
    indices_values_array.data = indices_slice.ptr;
    indices_values_array.len = indices_slice.len;

    const coefficients_slice = coefficients.toOwnedSlice(allocator) catch |err| return errorToInt(err);

    coefficients_values_array.data = coefficients_slice.ptr;
    coefficients_values_array.len = coefficients_slice.len;

    return 0;
}

/// Rebuilds a compressed buffer from provided indices and coefficients, using the specified method.
/// On success, fills `compressed_values_array` with the rebuilt compressed stream (including method byte).
/// Input arrays must match the expected layout for the method; see `extractors.zig` for details.
/// On success zero is returned, and the following non-zero values are returned on errors:
/// - 1) Unknown compression method.
/// - 2) Unsupported input.
/// - 3) Invalid configuration.
/// - 4) Corrupted compressed data.
/// - 5) Item not found.
/// - 6) Empty convex hull.
/// - 7) Empty queue.
/// - 8) Byte stream error.
/// - 9) Unsupported method.
/// - 10) Out of memory.
export fn rebuild(
    indices_values_array: IndicesValues,
    coefficients_values_array: CoefficientsValues,
    compressed_values_array: *CompressedValues,
    method_idx: u8,
) i32 {
    const method: Method = @enumFromInt(method_idx);

    const indices = indices_values_array.data[0..indices_values_array.len];
    const coefficients = coefficients_values_array.data[0..coefficients_values_array.len];

    var components_values = tersets.rebuild(
        allocator,
        indices,
        coefficients,
        method,
    ) catch |e| return errorToInt(e);

    const components_slice = components_values.toOwnedSlice(allocator) catch |err| return errorToInt(err);

    compressed_values_array.data = components_slice.ptr;
    compressed_values_array.len = components_slice.len;

    return 0;
}

/// Frees a `compressed_values` buffer previously produced by `compress`.
/// This function is primarily used by the Python and C bindings.
/// If used independently, ensure that the actual allocated size of
/// `compressed_values.data` matches the value stored in `compressed_values.len`.
/// A mismatch between these two values will corrupt the memory allocator state.
export fn freeCompressedValues(compressed_values: *CompressedValues) void {
    freeValuesMemory(CompressedValues, compressed_values);
}

/// Frees an `uncompressed_values` buffer previously produced by `decompress`.
/// This function is primarily used by the Python and C bindings. If used independently,
/// ensure that the actual allocated size of`uncompressed_values.data` matches the value
/// stored in `uncompressed_values.len`. A mismatch between these two values will corrupt
/// the memory allocator state.
export fn freeUncompressedValues(uncompressed_values: *UncompressedValues) void {
    freeValuesMemory(UncompressedValues, uncompressed_values);
}

/// Frees a `coefficients_values` buffer previously produced by `extract`. This function
/// is primarily used by the Python and C bindings. If used independently, ensure that
/// the actual allocated size of `coefficients_values.data` matches the value stored in
/// `coefficients_values.len`. A mismatch between these two values will corrupt
/// the memory allocator state.
export fn freeCoefficientValues(coefficients_values: *CoefficientsValues) void {
    freeValuesMemory(CoefficientsValues, coefficients_values);
}

/// Frees a `indices_values` buffer previously produced by `extract`.
/// This function is primarily used by the Python and C bindings.
/// If used independently, ensure that the actual allocated size of
/// `indices_values.data` matches the value stored in `indices_values.len`.
/// A mismatch between these two values will corrupt the memory allocator state.
export fn freeIndicesValues(indices_values: *IndicesValues) void {
    freeValuesMemory(IndicesValues, indices_values);
}

/// Frees a `values` struct previously produced by the TerseTS library. This function is an
/// internal helper used by the exported `free*Values` functions for the C and Python bindings.
/// It checks the `ValuesType` at compile time to ensure only supported types are freed.
/// If the `values.len` is zero, it indicates that the memory has already been freed.
fn freeValuesMemory(comptime ValuesType: type, values: *ValuesType) void {
    if (ValuesType != CompressedValues and ValuesType != UncompressedValues and
        ValuesType != CoefficientsValues and ValuesType != IndicesValues)
    {
        @compileError("freeValuesMemory: unsuported ValuesType"); // Invalid type for freeing values.
    }
    if (values.len != 0) {
        allocator.free(values.data[0..values.len]);
        values.len = 0; // Mark it as freed.
    }
}

/// `Array` is a pointer to values of type `data_type` and the number of values.
fn Array(comptime data_type: type) type {
    return extern struct { data: [*]const data_type, len: usize };
}

// Convert `err` to an `i32` as is not guaranteed to be stable `@intFromError`.
fn errorToInt(err: Error) i32 {
    switch (err) {
        Error.UnknownMethod => return 1,
        Error.UnsupportedInput => return 2,
        Error.InvalidConfiguration => return 3,
        Error.CorruptedCompressedData => return 4,
        Error.ItemNotFound => return 5,
        Error.EmptyConvexHull => return 6,
        Error.EmptyQueue => return 7,
        Error.ByteStreamError => return 8,
        Error.UnsupportedMethod => return 9,
        Error.OutOfMemory => return 10,
    }
}

test "method enum must match method constants" {
    try testing.expectEqual(@intFromEnum(tersets.Method.PoorMansCompressionMidrange), 0);
    try testing.expectEqual(@intFromEnum(tersets.Method.PoorMansCompressionMean), 1);
    try testing.expectEqual(@intFromEnum(tersets.Method.SwingFilter), 2);
    try testing.expectEqual(@intFromEnum(tersets.Method.SwingFilterDisconnected), 3);
    try testing.expectEqual(@intFromEnum(tersets.Method.SlideFilter), 4);
    try testing.expectEqual(@intFromEnum(tersets.Method.SimPiece), 5);
    try testing.expectEqual(@intFromEnum(tersets.Method.PiecewiseConstantHistogram), 6);
    try testing.expectEqual(@intFromEnum(tersets.Method.PiecewiseLinearHistogram), 7);
    try testing.expectEqual(@intFromEnum(tersets.Method.ABCLinearApproximation), 8);
    try testing.expectEqual(@intFromEnum(tersets.Method.VisvalingamWhyatt), 9);
    try testing.expectEqual(@intFromEnum(tersets.Method.SlidingWindow), 10);
    try testing.expectEqual(@intFromEnum(tersets.Method.BottomUp), 11);
    try testing.expectEqual(@intFromEnum(tersets.Method.MixPiece), 12);
    try testing.expectEqual(@intFromEnum(tersets.Method.BitPackedQuantization), 13);
    try testing.expectEqual(@intFromEnum(tersets.Method.RunLengthEncoding), 14);
    try testing.expectEqual(@intFromEnum(tersets.Method.NonLinearApproximation), 15);
    try testing.expectEqual(@intFromEnum(tersets.Method.SerfQT), 16);
}

test "error for unknown compression method" {
    const uncompressed_values = UncompressedValues{
        .data = undefined,
        .len = undefined,
    };
    var compressed_values = CompressedValues{
        .data = undefined,
        .len = undefined,
    };

    const method_index: u8 = math.maxInt(u8);
    const configuration = "{ \"abs_error_bound\": 0.0 }";

    const return_code = compress(
        uncompressed_values,
        &compressed_values,
        method_index,
        configuration,
    );

    try testing.expectEqual(1, return_code);
}

test "error for empty input when compressing" {
    const uncompressed_values = UncompressedValues{
        .data = undefined,
        .len = 0,
    };
    var compressed_values = CompressedValues{
        .data = undefined,
        .len = undefined,
    };

    const method_index: u8 = math.maxInt(u8);
    const configuration = "{ \"abs_error_bound\": 0.1 }";

    const return_code = compress(
        uncompressed_values,
        &compressed_values,
        method_index,
        configuration,
    );

    try testing.expectEqual(1, return_code);
}

test "error for negative error bound when compressing" {
    const uncompressed_array = [_]f64{ 0.1, 0.3, 0.4, 0.5 };
    const uncompressed_values = UncompressedValues{
        .data = &uncompressed_array,
        .len = uncompressed_array.len,
    };
    var compressed_values = CompressedValues{ .data = undefined, .len = undefined };

    const method_index: u8 = 0;
    const configuration = "{ \"abs_error_bound\": -1.0 }";

    const return_code = compress(
        uncompressed_values,
        &compressed_values,
        method_index,
        configuration,
    );

    try testing.expectEqual(3, return_code);
}

test "error for unknown decompression method" {
    var single_element_array: [1]u8 = [_]u8{255};

    const compressed_values = CompressedValues{ .data = &single_element_array, .len = 1 };
    var decompressed_values = UncompressedValues{ .data = undefined, .len = undefined };

    const return_code = decompress(compressed_values, &decompressed_values);

    try testing.expectEqual(1, return_code);
}

test "error for empty input when decompressing" {
    const compressed_values = CompressedValues{ .data = undefined, .len = 0 };
    var decompressed_values = UncompressedValues{ .data = undefined, .len = undefined };

    const return_code = decompress(compressed_values, &decompressed_values);

    try testing.expectEqual(4, return_code);
}

test "can compress and decompress" {
    const uncompressed_array = [_]f64{ 0.1, 1.1, 1.9, 2.5, 3.8 };
    const uncompressed_values = UncompressedValues{
        .data = &uncompressed_array,
        .len = uncompressed_array.len,
    };
    var compressed_values = CompressedValues{ .data = undefined, .len = undefined };
    var decompressed_values = UncompressedValues{
        .data = &uncompressed_array,
        .len = uncompressed_array.len,
    };
    const method_index: u8 = 0;
    const configuration = "{ \"abs_error_bound\": 0.0 }";

    const compress_return_code = compress(
        uncompressed_values,
        &compressed_values,
        method_index,
        configuration,
    );
    try testing.expectEqual(0, compress_return_code);

    const decompress_return_code = decompress(
        compressed_values,
        &decompressed_values,
    );

    try testing.expectEqual(0, decompress_return_code);
    try testing.expectEqual(uncompressed_values.len, decompressed_values.len);

    var i: usize = 0;
    while (i < decompressed_values.len) : (i += 1) {
        try testing.expectEqual(uncompressed_values.data[i], decompressed_values.data[i]);
    }
}

test "free allocated compressed values" {
    // Allocate a dummy compressed values array.
    const array = try allocator.alloc(u8, 16);

    // Fill to make it less trivial.
    for (array, 0..) |*value, i|
        value.* = @as(u8, @intCast(i));

    var compressed_values = CompressedValues{
        .data = array.ptr,
        .len = array.len,
    };

    // First free should release the array and zero the length.
    freeValuesMemory(CompressedValues, &compressed_values);
    try testing.expectEqual(@as(usize, 0), compressed_values.len);

    // Second free should no try to deallocate again.
    freeValuesMemory(CompressedValues, &compressed_values);
    try testing.expectEqual(@as(usize, 0), compressed_values.len);
}

test "free allocated uncompressed values" {
    const array = try allocator.alloc(f64, 4);
    for (array, 0..) |*value, i|
        value.* = @as(f64, @floatFromInt(i));

    var uncompressed_values = UncompressedValues{
        .data = array.ptr,
        .len = array.len,
    };

    freeValuesMemory(UncompressedValues, &uncompressed_values);
    try testing.expectEqual(@as(usize, 0), uncompressed_values.len);

    freeValuesMemory(UncompressedValues, &uncompressed_values);
    try testing.expectEqual(@as(usize, 0), uncompressed_values.len);
}
