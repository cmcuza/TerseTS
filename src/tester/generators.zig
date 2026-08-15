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
const time = std.time;

/// Number of time to run each test. This is a trade-of between test time and coverage.
const test_execution_count = 50;

/// Maximum number of values to generate per test. The exact number of values a generators generate is
/// decided by executing `numberOfValuesToGenerate()`. This is a trade-of between test time and coverage.
const value_generation_count = 50;

/// Default seed used for generating random values. It is initialized by
/// `getDefaultRandomGenerator()` the first time it is called.
var default_random_seed: u64 = 0;

/// Default random number generator used for generating random values.
var default_random_generator: std.Random.DefaultPrng = undefined;

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

/// Generate a random val used for testing. This value needs to be higher than or equal to 2
/// otherwise some of the tests will fail. The value is set to between [100-150] to ensure that the tests are
/// not too slow. The values is generated randomly to obtain a different set of values for each test run.
pub fn numberOfValuesToGenerate(random: Random) usize {
    return random.intRangeAtMost(u64, 0, value_generation_count);
}

/// Returns the default `Random` instance, initializing it with the current millisecond timestamp
/// as the seed if it has not been initialized yet. This ensures that repeated calls return the same
/// pseudo-random number generator unless the seed is reset by setting it to the integer zero.
pub fn getDefaultRandomGenerator() Random {
    if (default_random_seed == 0) {
        default_random_seed = @bitCast(time.milliTimestamp());
        default_random_generator = Random.DefaultPrng.init(default_random_seed);
    }
    return default_random_generator.random();
}
