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

//! Provides functions for generating test values for testing TerseTS.
//! New generator function must use the same interface as the existing
//! so `codegen.zig` can generate integration tests that use them.

const std = @import("std");
const math = std.math;
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const Random = std.Random;

/// The names of the generators implemented in this file.
pub const generator_names = &[_][]const u8{
    "generateConstantValues",
    "generateLinearValues",
    "generateRandomValuesInt",
    "generateRandomValuesLog",
};

/// Maximum number of values to generate per test. The exact number of values generators generate is
/// decided by executing `numberOfValuesToGenerate()`. This is a trade-of between test time and coverage.
const value_generation_count = 50;

/// Generate a random sequence of `f64` values using `random` and add them to `uncompressed_values`.
pub fn generateRandomValuesInt(allocator: Allocator, uncompressed_values: *ArrayList(f64), random: Random) !void {
    for (0..numberOfValuesToGenerate(random)) |_| {
        // Generate a random f64 by bit-casting a random u64 to include NaN, Inf, etc.
        const random_value = @as(f64, @bitCast(random.int(u64)));
        try uncompressed_values.append(allocator, random_value);
    }
}

/// Generate a random sequence of `f64` values using `random` and add them to `uncompressed_values`.
/// The value is sampled logarithmically to ensure better coverage across orders of magnitude.
pub fn generateRandomValuesLog(allocator: Allocator, uncompressed_values: *ArrayList(f64), random: Random) !void {
    // Set the range to the min and max of f64 to ensure that we include all possible f64 values.
    const at_least = std.math.floatMin(f64);
    const at_most = std.math.floatMax(f64);
    const at_least_log = toLog(f64, at_least);
    const at_most_log = toLog(f64, at_most);

    for (0..numberOfValuesToGenerate(random)) |_| {
        // Generate a random value in [0, 1) and scale it within the logarithmic range
        // [at_least_log, at_most_log], and then transform back to the real number line.
        const rand_value = random.float(f64);
        const log_sample = at_least_log + (at_most_log - at_least_log) * rand_value;
        const random_value = fromLog(f64, log_sample);
        try uncompressed_values.append(allocator, random_value);
    }
}

/// Generate a constant sequence of `f64` values using `random` and add them to `uncompressed_values`.
pub fn generateConstantValues(allocator: Allocator, uncompressed_values: *ArrayList(f64), random: Random) !void {
    const constant_value = @as(f64, @bitCast(random.int(u64)));
    for (0..numberOfValuesToGenerate(random)) |_| {
        try uncompressed_values.append(allocator, constant_value);
    }
}

/// Generate a linear sequence of `f64` values using `random` and add them to `uncompressed_values`.
pub fn generateLinearValues(allocator: Allocator, uncompressed_values: *ArrayList(f64), random: Random) !void {
    const slope = @as(f64, @bitCast(random.int(u64)));
    const intercept = @as(f64, @bitCast(random.int(u64)));

    for (0..numberOfValuesToGenerate(random)) |index| {
        const linear_value = slope * @as(f64, @bitCast(index + 1)) + intercept;
        try uncompressed_values.append(allocator, linear_value);
    }
}

/// Convert a value to a logarithmic scale using base 10. The function handles both positive
/// and negative values by preserving the sign.
fn toLog(comptime T: type, value: T) T {
    const sign: T = if (value < 0) @as(T, -1.0) else @as(T, 1.0);
    return sign * @log10(@abs(value) + 1.0);
}

/// Convert a value from a logarithmic scale back to its original scale using base 10. The function
/// handles both positive and negative values by preserving the sign.
fn fromLog(comptime T: type, value: T) T {
    const sign: T = if (value < 0) @as(T, -1.0) else @as(T, 1.0);
    return sign * (math.pow(T, 10.0, @abs(value)) - 1.0);
}

/// Generate how many values that should be generated for this test.
fn numberOfValuesToGenerate(random: Random) usize {
    return random.intRangeAtMost(u64, 0, value_generation_count);
}
