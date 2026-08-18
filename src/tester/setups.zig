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

//! Provides configurations and assert functions for testing TerseTS.
//! New assert function must use the same interface as the existing
//! so `codegen.zig` can generate integration tests that use them.

const std = @import("std");

const tersets = @import("../tersets.zig");
const Method = tersets.Method;

/// Returns a list of configurations to use for testing `method`.
pub fn getConfiguration(method: Method) []const []const u8 {
    return switch (method) {
        .Uncompressed => &[_][]const u8{""},
        else => &[_][]const u8{},
    };
}

/// Returns the name of the assert to use for testing `method`.
pub fn getAssertFunction(method: Method) []const u8 {
    return switch (method) {
        .Uncompressed => "assertEqual",
        else => "",
    };
}

/// Assert that `uncompressed_values` and `decompressed_values` are equal.
pub fn assertEqual(uncompressed_values: []const f64, decompressed_values: []const f64) !void {
    try std.testing.expectEqualSlices(f64, uncompressed_values, decompressed_values);
}
