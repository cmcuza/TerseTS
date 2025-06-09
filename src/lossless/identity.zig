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

const std = @import("std");
const mem = std.mem;
const ArrayList = std.ArrayList;

const tersets = @import("../tersets.zig");
const Method = tersets.Method;
const Error = tersets.Error;

const tester = @import("../tester.zig");

/// Compresses the `uncompressed_values` using the identity function, i.e., it simply compies
/// all elements to the `compressed_values`. If an error occurs it is returned.
pub fn compress(
    uncompressed_values: []const f64,
    compressed_values: *ArrayList(u8),
) Error!void {
    for (uncompressed_values) |value| {
        const value_as_bytes: [8]u8 = @bitCast(value);
        try compressed_values.appendSlice(value_as_bytes[0..]);
    }
}

/// Decompresses the `compressed_values` by decoding every element and writes the results in
/// `decompressed_values`. If an error occurs it is returned.
pub fn decompress(compressed_values: []const u8, decompressed_values: *ArrayList(f64)) Error!void {
    const compressed_representation = mem.bytesAsSlice(f64, compressed_values);
    for (compressed_representation) |value| {
        try decompressed_values.append(value);
    }
}

test "identity can compress and decompress" {
    const allocator = std.testing.allocator;
    try tester.testGenerateCompressAndDecompress(
        tester.generateRandomValues,
        allocator,
        Method.IdentityCompression,
        0,
        tersets.isWithinErrorBound,
    );
}
