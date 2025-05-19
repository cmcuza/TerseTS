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
const ArrayList = std.ArrayList;

const tersets = @import("tersets.zig");
const Error = tersets.Error;
const Method = tersets.Method;

const params = @import("params.zig");

/// Global memory allocator used by tersets.
var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();

/// A pointer to uncompressed values and the number of values.
pub const UncompressedValues = Array(f64);

/// A pointer to compressed values and the number of bytes.
pub const CompressedValues = Array(u8);

/// Configuration to use for compression.
pub const Configuration = extern struct { method: u8, parameters: ?*const anyopaque };

/// Compress `uncompressed_values` to `compressed_values` according to `configuration`.
/// The General Purpose Allocator `allocator` is passed as a parameter to tersets for
/// memory management in the compression methods. On success zero is returned, and the
/// following non-zero values are returned on errors:
/// - 1) Unsupported compression method.
/// - 2) No uncompressed values.
/// - 3) Error bound is negative.
/// - 5) Out-of-memory for compression.
export fn compress(
    uncompressed_values_array: UncompressedValues,
    compressed_values_array: *CompressedValues,
    configuration: Configuration,
) i32 {
    const uncompressed_values = uncompressed_values_array.data[0..uncompressed_values_array.len];

    // Returning 1 is equivalent to returning Error.UnknownMethod.
    if (configuration.method > tersets.getMaxMethodIndex()) return 1;

    const method: Method = @enumFromInt(configuration.method);

    const compressed_values = tersets.compress(
        uncompressed_values,
        allocator,
        method,
        configuration.parameters,
    ) catch |err| return errorToInt(err);

    compressed_values_array.data = compressed_values.items.ptr;
    compressed_values_array.len = compressed_values.items.len;

    return 0;
}

/// Decompress `compressed_values` to `uncompressed_values` according to `configuration`.
/// On success zero is returned, and the following non-zero values are returned on errors:
/// - 1) Unsupported decompression method.
/// - 2) No compressed values.
/// - 4) Incorrect compressed values.
/// - 5) Out-of-memory for decompression.
export fn decompress(
    compressed_values_array: CompressedValues,
    decompressed_values_array: *UncompressedValues,
) i32 {
    const compressed_values = compressed_values_array.data[0..compressed_values_array.len];

    const decompressed_values = tersets.decompress(
        compressed_values,
        allocator,
    ) catch |err| return errorToInt(err);

    decompressed_values_array.data = decompressed_values.items.ptr;
    decompressed_values_array.len = decompressed_values.items.len;

    return 0;
}

/// `Array` is a pointer to values of type `data_type` and the number of values.
fn Array(comptime data_type: type) type {
    return extern struct { data: [*]const data_type, len: usize };
}

/// Returns a human-readable description of a TerseTS error code.
export fn tersets_strerror(code: i32) [*:0]const u8 {
    return switch (code) {
        1 => "Unknown method",
        2 => "Unsupported input",
        3 => "Unsupported error bound",
        4 => "Unsupported parameters",
        5 => "Item not found",
        6 => "Out of memory",
        7 => "Empty convex hull",
        8 => "Empty queue",
        else => "Unknown error",
    };
}

// Convert `err` to an `i32` as is not guaranteed to be stable `@intFromError`.
fn errorToInt(err: Error) i32 {
    switch (err) {
        Error.UnknownMethod => return 1,
        Error.UnsupportedInput => return 2,
        Error.UnsupportedErrorBound => return 3,
        Error.UnsupportedParameters => return 4,
        Error.OutOfMemory => return 5,
        Error.ItemNotFound => return 6,
        Error.EmptyConvexHull => return 7,
        Error.EmptyQueue => return 8,
    }
}

/// Force link the functional parameters to ensure they are included in the binary.
export fn _force_link_functional(p: *const params.FunctionalParams) f32 {
    return p.error_bound;
}

/// Force link the basic parameters to ensure they are included in the binary.
export fn _force_link_basic(p: *const params.BasicParams) f32 {
    return p.error_bound;
}

/// Force link the histogram parameters to ensure they are included in the binary.
export fn _force_link_histogram(p: *const params.HistogramParams) usize {
    return p.maximum_buckets;
}

/// Force link the line simplification parameters to ensure they are included in the binary.
export fn _force_link_linesimp(p: *const params.LineSimplificationParams) f32 {
    return p.error_bound;
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
    try testing.expectEqual(@intFromEnum(tersets.Method.IdentityCompression), 10);
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

    var configuration = Configuration{
        .method = 0,
        .parameters = undefined,
    };

    configuration.method = math.maxInt(@TypeOf(configuration.method));

    const return_code = compress(
        uncompressed_values,
        &compressed_values,
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

    const configuration = Configuration{
        .method = 0,
        .parameters = undefined,
    };

    const return_code = compress(
        uncompressed_values,
        &compressed_values,
        configuration,
    );

    try testing.expectEqual(2, return_code);
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

    try testing.expectEqual(2, return_code);
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

    // Calling "Identity" compression with value equal 10, which does not need configuration.
    const configuration = Configuration{ .method = 10, .parameters = undefined };

    const compress_return_code = compress(
        uncompressed_values,
        &compressed_values,
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
