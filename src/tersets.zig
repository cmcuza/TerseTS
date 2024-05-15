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
const ArrayList = std.ArrayList;
const testing = std.testing;

const pmc = @import("functional/poor_mans_compression.zig");
const swing = @import("functional/swing.zig");

/// Global memory allocator used by tersets.
var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const alloc = gpa.allocator();

/// A pointer to uncompressed values and the number of values.
pub const UncompressedValues = Array(f64);

/// A pointer to compressed values and the number of bytes.
pub const CompressedValues = Array(u8);

/// Configuration to use for compression and/or decompression.
pub const Configuration = extern struct { method: u8, error_bound: f32 };

/// Compress `uncompressed_values` to `compressed_values` according to
/// `configuration`. The following non-zero values are returned on errors:
/// - 1) Unsupported compression method.
export fn compress(
    uncompressed_values: UncompressedValues,
    compressed_values: *CompressedValues,
    configuration: Configuration,
) i32 {
    // TODO: split compress into compress_zig with slice and ArrayList
    // as input and compress_c so C->Zig and Zig-C is implemented once.
    const uncompressed_values_slice = uncompressed_values.data[0..uncompressed_values.len];
    var compressed_values_array_list = ArrayList(u8).init(alloc);
    switch (configuration.method) {
        0 => {
            pmc.poorMansCompressionCompress(
                uncompressed_values_slice,
                &compressed_values_array_list,
                configuration.error_bound,
            ) catch {};
        },
        1 => {
            swing.compress(
                uncompressed_values_slice,
                &compressed_values_array_list,
                configuration.error_bound,
            ) catch {};
        },

        else => return 1,
    }
    // TODO: remove export and return errors to C function
    // https://github.com/Vexu/bog/blob/master/src/lib.zig
    // https://github.com/Vexu/bog/blob/master/include/bog.h
    compressed_values.data = compressed_values_array_list.items.ptr;
    compressed_values.len = compressed_values_array_list.items.len;
    return 0;
}

/// Decompress `compressed_values` to `uncompressed_values` according to
/// `configuration`. The following non-zero values are returned on errors:
/// - 1) Unsupported decompression method.
export fn decompress(
    compressed_values: CompressedValues,
    decompressed_values: *UncompressedValues,
    configuration: Configuration,
) i32 {
    switch (configuration.method) {
        0 => {
            // TODO: split compress into compress_zig with slice and ArrayList
            // as input and compress_c so C->Zig and Zig-C is implemented once.
            const compressed_values_slice = compressed_values.data[0..compressed_values.len];
            var decompressed_values_array_list = ArrayList(f64).init(alloc);
            pmc.poorMansCompressionDecompress(
                compressed_values_slice,
                &decompressed_values_array_list,
            ) catch {}; // TODO: remove export and return errors to C function
            // https://github.com/Vexu/bog/blob/master/src/lib.zig
            // https://github.com/Vexu/bog/blob/master/include/bog.h
            decompressed_values.data = decompressed_values_array_list.items.ptr;
            decompressed_values.len = decompressed_values_array_list.items.len;
        },
        else => return 1,
    }

    return 0;
}

// TODO: Add deinit() function so bindings can deallocate returned array with
// the compressed and decompressed data or assume they can reuse and deallocate?

/// `Array` is a pointer to values of type `data_type` and the number of values.
fn Array(comptime data_type: type) type {
    return extern struct { data: [*]const data_type, len: usize };
}

test "compress and decompress" {
    const uncompressed_array = [_]f64{ 0.1, 1.1, 1.9, 2.5, 3.8 };
    const uncompressed_slice = uncompressed_array[0..5];
    const uncompressed_values = UncompressedValues{
        .data = uncompressed_slice.ptr,
        .len = uncompressed_slice.len,
    };
    var compressed_values = CompressedValues{
        .data = undefined,
        .len = undefined,
    };
    var decompressed_values = UncompressedValues{
        .data = uncompressed_slice.ptr,
        .len = uncompressed_slice.len,
    };
    const configuration = Configuration{ .method = 0, .error_bound = 0 };

    const compress_result = compress(
        uncompressed_values,
        &compressed_values,
        configuration,
    );
    try testing.expect(compress_result == 0);

    const decompress_result = decompress(
        compressed_values,
        &decompressed_values,
        configuration,
    );
    try testing.expect(decompress_result == 0);

    try testing.expectEqual(decompressed_values.len, uncompressed_values.len);

    var i: usize = 0;
    while (i < decompressed_values.len) : (i += 1) {
        try testing.expectEqual(
            decompressed_values.data[i],
            uncompressed_values.data[i],
        );
    }
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

    const result = compress(
        uncompressed_values,
        &compressed_values,
        configuration,
    );

    try testing.expectEqual(result, 1);
}

test "error for unknown decompression method" {
    const compressed_values = CompressedValues{
        .data = undefined,
        .len = undefined,
    };
    var uncompressed_values = UncompressedValues{
        .data = undefined,
        .len = undefined,
    };
    var configuration = Configuration{ .method = 0, .error_bound = 0 };
    configuration.method = math.maxInt(@TypeOf(configuration.method));

    const result = decompress(
        compressed_values,
        &uncompressed_values,
        configuration,
    );

    try testing.expectEqual(result, 1);
}
