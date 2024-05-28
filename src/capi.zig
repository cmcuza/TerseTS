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

/// Global memory allocator used by tersets.
var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();

/// A pointer to uncompressed values and the number of values.
pub const UncompressedValues = Array(f64);

/// A pointer to compressed values and the number of bytes.
pub const CompressedValues = Array(u8);

/// Configuration to use for compression and/or decompression.
pub const Configuration = extern struct { method: u8, error_bound: f32 };

/// Get the maximun index of the available methods in TerseTS.
fn getMaxMethodIndex(comptime tersets_method: type) usize {
    const type_info = @typeInfo(tersets_method);
    const method_info = switch (type_info) {
        .Enum => |method_info| method_info,
        else => @compileError("Expected a Method enum type"),
    };

    var max_index: usize = 0;
    for (method_info.fields, 0..) |_, i| {
        max_index = if (i > max_index) i else max_index;
    }

    return max_index;
}

/// Compress `uncompressed_values` to `compressed_values` according to `configuration`.
/// On success zero is returned, and the following non-zero values are returned on errors:
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
    var compressed_values = ArrayList(u8).init(allocator);
    // Check if larger than the largest int used by Method.
    if (configuration.method > getMaxMethodIndex(tersets.Method)) return 1;
    const method: tersets.Method = @enumFromInt(configuration.method);

    tersets.compress(
        uncompressed_values,
        &compressed_values,
        method,
        configuration.error_bound,
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
    method_index: u8,
) i32 {
    const compressed_values = compressed_values_array.data[0..compressed_values_array.len];
    var decompressed_values = ArrayList(f64).init(allocator);
    // Check if larger than the largest int used by Method.
    if (method_index > getMaxMethodIndex(tersets.Method)) return 1;
    const method: tersets.Method = @enumFromInt(method_index);

    tersets.decompress(
        compressed_values,
        &decompressed_values,
        method,
    ) catch |err| return errorToInt(err);

    decompressed_values_array.data = decompressed_values.items.ptr;
    decompressed_values_array.len = decompressed_values.items.len;

    return 0;
}

/// `Array` is a pointer to values of type `data_type` and the number of values.
fn Array(comptime data_type: type) type {
    return extern struct { data: [*]const data_type, len: usize };
}

// Convert `err` to an `i32` as is not guaranteed to be stable `@intFromError`.
fn errorToInt(err: Error) i32 {
    switch (err) {
        Error.EmptyInput => return 2,
        Error.NegativeErrorBound => return 3,
        Error.IncorrectInput => return 4,
        Error.OutOfMemory => return 5,
    }
}

test "method enum must match method constants" {
    try testing.expectEqual(@intFromEnum(tersets.Method.PoorMansCompressionMidrange), 0);
    try testing.expectEqual(@intFromEnum(tersets.Method.PoorMansCompressionMean), 1);
    try testing.expectEqual(@intFromEnum(tersets.Method.SwingFilter), 2);
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
    var configuration = Configuration{ .method = 0, .error_bound = 0 };
    configuration.method = math.maxInt(@TypeOf(configuration.method));

    const return_code = compress(
        uncompressed_values,
        &compressed_values,
        configuration,
    );

    try testing.expectEqual(return_code, 1);
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
    const configuration = Configuration{ .method = 0, .error_bound = 0 };

    const return_code = compress(
        uncompressed_values,
        &compressed_values,
        configuration,
    );

    try testing.expectEqual(return_code, 2);
}

test "error for negative error bound when compressing" {
    const uncompressed_values = UncompressedValues{
        .data = undefined,
        .len = 1, // If undefined an empty input error is returned.
    };
    var compressed_values = CompressedValues{
        .data = undefined,
        .len = undefined,
    };
    const configuration = Configuration{ .method = 0, .error_bound = -1 };

    const return_code = compress(
        uncompressed_values,
        &compressed_values,
        configuration,
    );

    try testing.expectEqual(return_code, 3);
}

test "error for unknown decompression method" {
    const compressed_values = CompressedValues{
        .data = undefined,
        .len = undefined,
    };
    var decompressed_values = UncompressedValues{
        .data = undefined,
        .len = undefined,
    };
    var configuration = Configuration{ .method = 0, .error_bound = 0 };
    configuration.method = math.maxInt(@TypeOf(configuration.method));

    const return_code = decompress(
        compressed_values,
        &decompressed_values,
        configuration.method,
    );

    try testing.expectEqual(return_code, 1);
}

test "error for empty input when decompressing" {
    const compressed_values = CompressedValues{
        .data = undefined,
        .len = 0,
    };
    var decompressed_values = UncompressedValues{
        .data = undefined,
        .len = undefined,
    };
    const configuration = Configuration{ .method = 0, .error_bound = 0 };

    const return_code = decompress(
        compressed_values,
        &decompressed_values,
        configuration.method,
    );

    try testing.expectEqual(return_code, 2);
}

test "can compress and decompress" {
    const uncompressed_array = [_]f64{ 0.1, 1.1, 1.9, 2.5, 3.8 };
    const uncompressed_values = UncompressedValues{
        .data = &uncompressed_array,
        .len = uncompressed_array.len,
    };
    var compressed_values = CompressedValues{
        .data = undefined,
        .len = undefined,
    };
    var decompressed_values = UncompressedValues{
        .data = &uncompressed_array,
        .len = uncompressed_array.len,
    };
    const configuration = Configuration{ .method = 0, .error_bound = 0 };

    const compress_code = compress(
        uncompressed_values,
        &compressed_values,
        configuration,
    );
    try testing.expectEqual(compress_code, 0);

    const decompress_code = decompress(
        compressed_values,
        &decompressed_values,
        configuration.method,
    );
    try testing.expectEqual(decompress_code, 0);

    try testing.expectEqual(decompressed_values.len, uncompressed_values.len);

    var i: usize = 0;
    while (i < decompressed_values.len) : (i += 1) {
        try testing.expectEqual(
            decompressed_values.data[i],
            uncompressed_values.data[i],
        );
    }
}
