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

//! Provides methods for testing TerseTS.

const std = @import("std");
const Allocator = std.mem.Allocator;
const testing = std.testing;

const tersets = @import("tersets.zig");
const Method = tersets.Method;
const Error = tersets.Error;

/// Test that `uncompressed_values` are within `error_bound` according to
/// `within_error_bound` after it has been compressed and decompressed using
/// `method`.
fn testCompressionAndDecompression(
    uncompressed_values: []const f64,
    allocator: Allocator,
    method: Method,
    error_bound: f32,
    within_error_bound: fn (
        uncompressed_values: []f64,
        decompressed_values: []f64,
        error_bound: f32,
    ) bool,
) Error!void {
    const compressed_values = tersets.compress(uncompressed_values, allocator, method, error_bound);
    const decompressed_values = tersets.decompress(compressed_values, allocator);

    try testing.expectEqual(uncompressed_values.len, decompressed_values.len);
    try testing.expect(within_error_bound(
        uncompressed_values,
        decompressed_values,
        error_bound,
    ));
}
