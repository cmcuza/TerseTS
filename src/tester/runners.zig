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

//! Provides comptime functions for generating test for testing TerseTS.

const std = @import("std");
const ArrayList = std.ArrayList;
const Allocator = std.mem.Allocator;
const Random = std.Random;
const generators = @import("generators.zig");
const Threaded = std.Io.Threaded;
const Clock = std.Io.Clock;

pub fn getGenerators() !void {
    const allocator = std.testing.allocator;
    var uncompressed_values = ArrayList(f64).empty;
    defer uncompressed_values.deinit(allocator);
    const random = generators.getRandomGenerator();
    try generators.generateRandomValues(allocator, &uncompressed_values, random);
    //std.debug.print("hunting", .{});
}

/// Default seed used for generating random values. It is initialized by
/// `getDefaultRandomGenerator()` the first time it is called.
var random_seed: u64 = 0;

/// Default random number generator used for generating random values.
var random_module: std.Random.DefaultPrng = undefined;

/// Returns the a `Random` instance, initializing it with the current millisecond timestamp as
/// the seed if it has not been initialized yet. This ensures that repeated calls return the same
/// pseudo-random number generator unless the seed is reset by setting it to the integer zero.
pub fn getRandomGenerator() Random {
    if (random_seed == 0) {
        // The seed is printed so it can be set to reproduce the same values.
        random_seed = @bitCast(milliTimestamp());
        std.debug.print("Tester Seed: {}", .{random_seed});
        random_module = Random.DefaultPrng.init(random_seed);
    }
    return random_module.random();
}

/// Return a timestamp in milliseconds relative to UTC 1970-01-01.
pub fn milliTimestamp() i64 {
    var threaded: Threaded = .init_single_threaded;
    const timestamp = Clock.real.now(threaded.io());
    threaded.deinit();
    return timestamp.toMilliseconds();
}
