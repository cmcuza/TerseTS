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
const ArrayList = std.ArrayList;
const rand = std.rand;
const Random = rand.Random;
const Allocator = std.mem.Allocator;
const Error = std.mem.Allocator.Error;
const math = std.math;
const time = std.time;
const testing = std.testing;
const debug = std.debug;

const tersets = @import("tersets.zig");
const Method = tersets.Method;

/// Number of values to generate for testing.
const number_of_values = 50;

/// Test that values generated by `uncompressed_values_generator` are within `error_bound` according
/// to `within_error_bound` after they have been compressed and decompressed using `method`. The
/// libraries public interface is used to make it simpler refactor the libraries internals.
pub fn testGenerateCompressAndDecompress(
    uncompressedValuesGenerator: fn (
        uncompressed_values: *ArrayList(f64),
        random: Random,
    ) Error!void,
    allocator: Allocator,
    method: Method,
    error_bound: f32,
    withinErrorBound: fn (
        uncompressed_values: []const f64,
        decompressed_values: []const f64,
        error_bound: f32,
    ) bool,
) !void {
    const seed: u64 = @bitCast(time.milliTimestamp());
    var prng = rand.DefaultPrng.init(seed);
    const random = prng.random();

    var uncompressed_values = ArrayList(f64).init(allocator);
    try uncompressedValuesGenerator(&uncompressed_values, random);
    defer uncompressed_values.deinit();

    // subsequenceStack contains subsequences to run the test for to find shortest failing sequence.
    var subsequenceStack = ArrayList(usize).init(allocator);
    defer subsequenceStack.deinit();

    try subsequenceStack.append(0);
    try subsequenceStack.append(uncompressed_values.items.len);
    var shortestStart: usize = 0;
    var shortestEnd: usize = math.maxInt(usize);

    while (subsequenceStack.items.len != 0) {
        // subsequenceStack contains start and end as separate integers of usize.
        const end = subsequenceStack.pop();
        const start = subsequenceStack.pop();
        const uncompressed_values_subsequence = uncompressed_values.items[start..end];

        testCompressAndDecompress(
            uncompressed_values_subsequence,
            allocator,
            method,
            error_bound,
            withinErrorBound,
        ) catch {
            // To simplify debugging failed tests that use auto generated data, the tests are
            // retried with smaller subsequence to find the smallest subsequence that fails.
            if (start < end - 1) {
                const middle = (start + end) / 2;
                try subsequenceStack.append(start);
                try subsequenceStack.append(middle);
                try subsequenceStack.append(middle);
                try subsequenceStack.append(end);
            }

            if (end - start < shortestEnd - shortestStart) {
                shortestStart = start;
                shortestEnd = end;
            }

            // Subsequences cannot be shorter than one element.
            if (end - start == 1) {
                break;
            }
        };
    }

    // HACK: std.testing has no functionality including a message with an error, thus a string with
    // an informative message is compared to an empty string to ensure it will fail and be printed.
    if (shortestEnd != math.maxInt(usize)) {
        try testing.expectFmt(
            "",
            "Seed: {}, Values: {any}",
            .{ seed, uncompressed_values.items[shortestStart..shortestEnd] },
        );
    }
}

/// Test that `uncompressed_values` are within `error_bound` according to `within_error_bound` after
/// it has been compressed and decompressed using `method`. Assumes that `within_error_bound`
/// returns `false` if the number of uncompressed and decompressed values are different. The
/// libraries public interface is used to make it simpler to refactor the libraries internals.
pub fn testCompressAndDecompress(
    uncompressed_values: []const f64,
    allocator: Allocator,
    method: Method,
    error_bound: f32,
    withinErrorBound: fn (
        uncompressed_values: []const f64,
        decompressed_values: []const f64,
        error_bound: f32,
    ) bool,
) !void {
    const compressed_values = try tersets.compress(
        uncompressed_values,
        allocator,
        method,
        error_bound,
    );
    defer compressed_values.deinit();
    const decompressed_values = try tersets.decompress(compressed_values.items, allocator);
    defer decompressed_values.deinit();

    try testing.expect(withinErrorBound(
        uncompressed_values,
        decompressed_values.items,
        error_bound,
    ));
}

// Replace each normal value in `uncompressed_values` with a positive infinity, negative infinity,
// or NaN with the passed probability. The non-normal values are written to `uncompressed_values`
// in the previously listed order, thus a positive infinity maybe overwritten by a negative infinity
// and so on. The probabilities are asserted to be between zero and one.
pub fn replaceNormalValues(
    uncompressed_values: *ArrayList(f64),
    positive_infinity_probability: f32,
    negative_infinity_probability: f32,
    not_a_number_probability: f32,
    random: Random,
) void {
    debug.assert(0 <= positive_infinity_probability and positive_infinity_probability <= 1);
    debug.assert(0 <= negative_infinity_probability and negative_infinity_probability <= 1);
    debug.assert(0 <= not_a_number_probability and not_a_number_probability <= 1);

    for (0..uncompressed_values.items.len) |index| {
        if (random.float(f32) < positive_infinity_probability) {
            uncompressed_values.items[index] = math.inf(f64);
        }

        if (random.float(f32) < negative_infinity_probability) {
            uncompressed_values.items[index] = -math.inf(f64);
        }

        if (random.float(f32) < not_a_number_probability) {
            uncompressed_values.items[index] = math.nan(f64);
        }
    }
}

/// Generate `number_of_values` of random values for use in testing using `random` and add them to
/// `uncompressed_values`.
pub fn generateRandomValues(uncompressed_values: *ArrayList(f64), random: Random) !void {
    for (0..number_of_values) |_| {
        // math.floatMax(f64) is not used as it is larger than -math.floatMin(f64).
        try uncompressed_values.append(math.floatMin(f64) +
            (-math.floatMin(f64) - math.floatMin(f64)) * rand.float(random, f64));
    }
}
