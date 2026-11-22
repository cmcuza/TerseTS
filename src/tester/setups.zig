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
const math = std.math;
const json = std.json;
const testing = std.testing;
const Allocator = std.mem.Allocator;

const tersets = @import("../tersets.zig");
const Method = tersets.Method;
const configuration = @import("../configuration.zig");

/// Returns a list of configurations to use for testing `method`.
pub fn getConfiguration(method: Method) []const []const u8 {
    return switch (method) {
        .Uncompressed => &[_][]const u8{""},
        .PoorMansCompressionMidrange => &[_][]const u8{
            "{\\\"abs_error_bound\\\": 0.0}",
            "{\\\"abs_error_bound\\\": 0.1}",
            "{\\\"abs_error_bound\\\": 1.0}",
        },
        .PoorMansCompressionMean => &[_][]const u8{
            "{\\\"abs_error_bound\\\": 0.0}",
            "{\\\"abs_error_bound\\\": 0.1}",
            "{\\\"abs_error_bound\\\": 1.0}",
        },
        else => &[_][]const u8{},
    };
}

/// Returns the name of the assert to use for testing `method`.
pub fn getAssertFunction(method: Method) []const u8 {
    return switch (method) {
        .Uncompressed => "assertEqual",
        .PoorMansCompressionMidrange => "assertWithinAbsoluteErrorBound",
        .PoorMansCompressionMean => "assertWithinAbsoluteErrorBound",
        else => "",
    };
}

/// Assert that `uncompressed_values` and `decompressed_values` are equal.
pub fn assertEqual(
    _: Allocator,
    uncompressed_values: []const f64,
    decompressed_values: []const f64,
    _: []const u8,
) !void {
    try testing.expectEqualSlices(f64, uncompressed_values, decompressed_values);
}

/// Assert that `uncompressed_values` and `decompressed_values` are within an absolute error bound.
pub fn assertWithinAbsoluteErrorBound(
    allocator: Allocator,
    uncompressed_values: []const f64,
    decompressed_values: []const f64,
    method_configuration: []const u8,
) !void {
    try testing.expectEqual(uncompressed_values.len, decompressed_values.len);

    const parsed_configuration = try parse_configuration_for_test(
        allocator,
        configuration.AbsoluteErrorBound,
        method_configuration,
    );

    const error_bound: f64 = parsed_configuration.abs_error_bound;

    for (0..uncompressed_values.len) |index| {
        const uncompressed_value = uncompressed_values[index];
        const decompressed_value = decompressed_values[index];
        if (!equal_or_nan(uncompressed_value, decompressed_value) and
            @abs(uncompressed_value - decompressed_value) > error_bound)
        {
            std.debug.print("{} is not within {} of {}\n", .{ uncompressed_value, error_bound, decompressed_value });
            return error.TestUnexpectedResult;
        }
    }
}

/// This is a small convenience wrapper around json.parseFromSlice that accepts a `ConfigurationType`
/// and parses the the JSON text `method_configuration`. The function returns the parsed value on success,
/// or `null` on failure. The `allocator` is used by the JSON parser. The function purposely ignores
/// unknown fields to allow configurations that contain more field than the field that an assert needs.
fn parse_configuration_for_test(
    allocator: Allocator,
    comptime ConfigurationType: type,
    method_configuration: []const u8,
) !ConfigurationType {
    const parsed = try json.parseFromSlice(
        configuration.AbsoluteErrorBound,
        allocator,
        method_configuration,
        .{ .ignore_unknown_fields = true },
    );
    defer parsed.deinit();
    return parsed.value;
}

/// Returns true if `uncompressed_value` and `decompressed_value` are equivalent or both values are NaN.
fn equal_or_nan(uncompressed_value: f64, decompressed_value: f64) bool {
    return uncompressed_value == decompressed_value or (math.isNan(uncompressed_value) and math.isNan(decompressed_value));
}
