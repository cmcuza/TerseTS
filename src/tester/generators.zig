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

const std = @import("std");
const ArrayList = std.ArrayList;
const Allocator = std.mem.Allocator;
const Random = std.Random;

/// Number of time to run each test. This is a trade-of between test time and coverage.
const test_execution_count = 50;

/// Maximum number of values to generate per test. The exact number of values a generators generate is
/// decided by executing `numberOfValuesToGenerate()`. This is a trade-of between test time and coverage.
const value_generation_count = 50;

/// The generators implemented in this file. This is not computed using `comptime` due to the complexity.
//const generators = [
//    generateRandomValues
//];

/// Generate a random number of `f64` values using `random` and add them to `uncompressed_values`.
/// Each value is a random `f64` generated from a random `u64` bit pattern, which may include
/// special values such as NaN or inf. The final number of values is determined by a random
/// generation function that returns an integer value between 100 and 150.
pub fn generateRandomValues(allocator: Allocator, uncompressed_values: *ArrayList(f64), random: Random) !void {
    for (0..numberOfValuesToGenerate(random)) |_| {
        // Generate a random f64 by bit-casting a random u64.
        const random_value = @as(f64, @bitCast(random.int(u64)));
        try uncompressed_values.append(allocator, random_value);
    }
}

/// Generate how many values that should be generated for this test.
fn numberOfValuesToGenerate(random: Random) usize {
    // at_least is two as tersets.zig immediately returns arrays with zero or one values.
    return random.intRangeAtMost(u64, 2, value_generation_count);
}
