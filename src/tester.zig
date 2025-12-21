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
//!
//! This file contains several "magic numbers" used in the generation of test data and other
//! operations. These values have been carefully chosen to balance test coverage, numerical
//! stability, and performance. Below is an explanation of the most important magic numbers:
//!
//! - `global_at_least = 10` and `global_at_most = 50`: These values define the default range for
//!   generating the number of random functions or values in tests. They ensure that tests are
//!   neither too small to be meaningful nor too large to be computationally expensive.
//!
//! - `global_replace_probability = 0.05`: This value determines the probability of replacing
//!   normal values with special floating-point values (NaN, +inf, -inf). It is chosen to ensure
//!   that special values appear frequently enough to test edge cases without dominating the data.
//!
//! - `max_test_value = max_test_value`: This is the maximum absolute value used for generating test data.
//!   It is large enough to test high-magnitude numbers while avoiding floating-point overflows
//!   and precision loss in typical `f64` operations.
//!
//! - Exponential function generation:
//!   - `5.0`: This value is used to scale the random logarithmic range for `theta_1` in the interval
//!     `[0, 5.0]`. It ensures that the coefficient `theta_1` spans a wide range of magnitudes
//!     (from `1` to `1e5`), which is critical for testing exponential growth behavior.
//!   - `3.0`: This value is used to scale the random logarithmic range for `theta_2` in the interval
//!     `[-3.0, 0]`. It ensures that the exponent coefficient `theta_2` spans a range of small
//!     values (from `1e-3` to `1`), avoiding overflow or excessively steep exponential growth.
//!     These values are chosen to ensure that the exponential function remains numerically stable
//!     while still testing a variety of growth rates.
//!
//! - Random value ranges:
//!   - `1e-2` to `1e14` for amplitudes, slopes, and intercepts: These ranges ensure that both
//!     small and large values are tested, covering a wide range of magnitudes.
//!   - `1e-4` to `1e1` for frequencies: This range ensures that sinusoidal functions vary from
//!     slow to moderately fast oscillations.
//!   - `[-1e15, 1e15]`: This range is used for bounded random values to ensure numerical stability.
//!
//! - Noise scales and probabilities:
//!   - Noise added to functions is typically in the range `[-0.5%, 0.5%]` of the value's magnitude.
//!     This ensures that noise is small enough to not dominate the data but still tests robustness.
//!   - Probabilities for replacing values with NaN or infinities are asserted to be between `0` and `1`.
//!
//! - Default random value generation:
//!   - `100` to `150` values are generated per test. This range ensures that tests are large enough
//!     to be meaningful but not so large as to slow down execution.
//!
//! These values are documented here to provide clarity and context for their use throughout the file.
//! Nevertheless, they are explained in more detail in the specific functions where they are applied.

const std = @import("std");
const ArrayList = std.ArrayList;
const Random = std.Random;
const Allocator = std.mem.Allocator;
const Error = std.mem.Allocator.Error;
const math = std.math;
const time = std.time;
const testing = std.testing;
const debug = std.debug;

const tersets = @import("tersets.zig");
const Method = tersets.Method;

const shared = @import("utilities/shared_structs.zig");

/// `global_at_least` and `global_at_most` define the default minimum and maximum bounds for
/// generating random integer values in tests. These values are chosen to provide a reasonable
/// range for most scenarios, balancing test coverage and performance.
pub const global_at_least: usize = 10;
pub const global_at_most: usize = 50;

/// Probability used when replacing a value with a special float (NaN, +inf, -inf) in test generators.
/// This value is passed to `replaceNormalValues` and determines the likelihood that any
/// normal value will be replaced by one of the three special types. For example, with
/// `global_replace_probability = 0.05` and 100 values generated, each replacement type has an expected
/// count of ~5. The chance that at least one special value appears is ~0.9999 assuming three
/// independent trials per value. The value should be in the range [0.01, 1] to retain a ~0.99
/// probability of replacement.
pub const global_replace_probability: f32 = 0.05;

/// The maximum absolute value used for generating test data in bounded random value generators.
/// This value is chosen to be 1e15 because it is large enough to test the behavior of compression
/// algorithms with high-magnitude numbers, but still small enough to avoid floating-point overflows
/// and precision loss in typical f64 operations. All test data generated by functions such as
/// `generateDefaultBoundedValues`, `generateRandomSinusoidalFunction`, and `generateRandomLinearFunction`
/// will be clamped within [-max_test_value, max_test_value] to ensure numerical stability.
pub const max_test_value: f64 = 1e15;

/// The maximum value after applying a noise scaling factor to `max_test_value`.
/// This variable is needed to ensure that the test values do not exceed `max_test_value`,
/// accounting for the influence of noise, which helps maintain stability and reliability in tests.
pub const clamped_max_value: f64 = max_test_value - (max_test_value * noise_scale);

/// Default seed and prng to generate random values.
pub var default_seed: u64 = 0;
pub var default_prng: std.Random.DefaultPrng = undefined;

/// Default noise scale used when generating test data with noise.
const noise_scale: f64 = 0.005; // 0.5%

/// Different data distributions used for testing.
pub const DataDistribution = enum {
    LinearFunctions,
    QuadradicFunctions,
    ExponentialFunctions,
    PowerFunctions,
    SqrtFunctions,
    BoundedRandomValues,
    SinusoidalFunction,
    MixedBoundedValuesFunctions,
    FiniteRandomValues,
    RandomValuesWithNansAndInfinities,
    LinearFunctionsWithNansAndInfinities,
    BoundedRandomValuesWithNansAndInfinities,
    SinusoidalFunctionWithNansAndInfinities,
};

/// Run a suite of experiments using different data generators to verify a compression `method`
/// works for a wide range of inputs. The `allocator` is used for memory management, and
/// `data_distributions` specifies which data distributions to test against the compression method.
/// The function generates a random error bound between [1.0e-4, 1.0) per data distribution
/// to guarantee that different error bounds are tested. The error bound is later used
/// to generate an absolute error bound based on the range of the distributed data.
pub fn testErrorBoundedCompressionMethod(
    allocator: Allocator,
    method: Method,
    data_distributions: []const DataDistribution,
) !void {
    const random = getDefaultRandomGenerator();

    for (data_distributions) |dist| {
        const error_bound: f32 = random.float(f32) + 1e-4; // Ensure a non-zero error bound.
        switch (dist) {
            .LinearFunctions => try testGeneratedErrorBoundedCompression(
                allocator,
                generateRandomLinearFunctions,
                method,
                error_bound,
                "Linear Functions",
            ),
            .QuadradicFunctions => try testGeneratedErrorBoundedCompression(
                allocator,
                generateRandomQuadraticFunctions,
                method,
                error_bound,
                "Quadratic Functions",
            ),
            .PowerFunctions => try testGeneratedErrorBoundedCompression(
                allocator,
                generateRandomPowerFunctions,
                method,
                error_bound,
                "Power Functions",
            ),
            .ExponentialFunctions => try testGeneratedErrorBoundedCompression(
                allocator,
                generateRandomExponentialFunctions,
                method,
                error_bound,
                "Exponential Functions",
            ),
            .SqrtFunctions => try testGeneratedErrorBoundedCompression(
                allocator,
                generateRandomSqrtFunctions,
                method,
                error_bound,
                "Square Root Functions",
            ),
            .BoundedRandomValues => try testGeneratedErrorBoundedCompression(
                allocator,
                generateDefaultBoundedValues,
                method,
                error_bound,
                "Bounded Values",
            ),
            .SinusoidalFunction => try testGeneratedErrorBoundedCompression(
                allocator,
                generateRandomSinusoidalFunctions,
                method,
                error_bound,
                "Sinusoidal Function",
            ),
            .MixedBoundedValuesFunctions => try testGeneratedErrorBoundedCompression(
                allocator,
                generateMixedBoundedValuesFunctions,
                method,
                error_bound,
                "Mixed Bounded Values Functions",
            ),
            .FiniteRandomValues => try testGeneratedErrorBoundedCompression(
                allocator,
                generateFiniteRandomValues,
                method,
                error_bound,
                "Finite Values",
            ),
            .RandomValuesWithNansAndInfinities => try testGeneratedErrorBoundedCompression(
                allocator,
                generateRandomValuesWithNaNs,
                method,
                error_bound,
                "Random Values with NaNs and Infinities",
            ),
            .LinearFunctionsWithNansAndInfinities => try testGeneratedErrorBoundedCompression(
                allocator,
                generateRandomLinearFunctionsWithNaNs,
                method,
                error_bound,
                "Linear Functions with NaNs and Infinities",
            ),
            .BoundedRandomValuesWithNansAndInfinities => try testGeneratedErrorBoundedCompression(
                allocator,
                generateDefaultBoundedValuesWithNaNs,
                method,
                error_bound,
                "Bounded Random Values with NaNs and Infinities",
            ),
            .SinusoidalFunctionWithNansAndInfinities => try testGeneratedErrorBoundedCompression(
                allocator,
                generateRandomSinusoidalFunctionWithNaNs,
                method,
                error_bound,
                "Sinusoidal Function with NaNs and Infinities",
            ),
        }
    }
}

/// Run a suite of experiments using different data generators to verify a lossless compression
/// `method` works for a wide range of inputs. The `allocator` is used for memory management, and
/// `data_distributions` specifies which data distributions to test against the compression method.
pub fn testLosslessMethod(
    allocator: Allocator,
    method: Method,
    data_distributions: []const DataDistribution,
) !void {
    for (data_distributions) |dist| {
        switch (dist) {
            .LinearFunctions => try testGeneratedLosslessCompression(
                allocator,
                generateRandomLinearFunctions,
                method,
                "Linear Functions",
            ),
            .QuadradicFunctions => try testGeneratedLosslessCompression(
                allocator,
                generateRandomQuadraticFunctions,
                method,
                "Quadratic Functions",
            ),
            .PowerFunctions => try testGeneratedLosslessCompression(
                allocator,
                generateRandomPowerFunctions,
                method,
                "Power Functions",
            ),
            .ExponentialFunctions => try testGeneratedLosslessCompression(
                allocator,
                generateRandomExponentialFunctions,
                method,
                "Exponential Functions",
            ),
            .SqrtFunctions => try testGeneratedLosslessCompression(
                allocator,
                generateRandomSqrtFunctions,
                method,
                "Square Root Functions",
            ),
            .BoundedRandomValues => try testGeneratedLosslessCompression(
                allocator,
                generateDefaultBoundedValues,
                method,
                "Bounded Values",
            ),
            .SinusoidalFunction => try testGeneratedLosslessCompression(
                allocator,
                generateRandomSinusoidalFunctions,
                method,
                "Sinusoidal Function",
            ),
            .MixedBoundedValuesFunctions => try testGeneratedLosslessCompression(
                allocator,
                generateMixedBoundedValuesFunctions,
                method,
                "Mixed Bounded Values Functions",
            ),
            .FiniteRandomValues => try testGeneratedLosslessCompression(
                allocator,
                generateFiniteRandomValues,
                method,
                "Finite Values",
            ),
            .RandomValuesWithNansAndInfinities => try testGeneratedLosslessCompression(
                allocator,
                generateRandomValuesWithNaNs,
                method,
                "Random Values with NaNs and Infinities",
            ),
            .LinearFunctionsWithNansAndInfinities => try testGeneratedLosslessCompression(
                allocator,
                generateRandomLinearFunctionsWithNaNs,
                method,
                "Linear Functions with NaNs and Infinities",
            ),
            .BoundedRandomValuesWithNansAndInfinities => try testGeneratedLosslessCompression(
                allocator,
                generateDefaultBoundedValuesWithNaNs,
                method,
                "Bounded Random Values with NaNs and Infinities",
            ),
            .SinusoidalFunctionWithNansAndInfinities => try testGeneratedLosslessCompression(
                allocator,
                generateRandomSinusoidalFunctionWithNaNs,
                method,
                "Sinusoidal Function with NaNs and Infinities",
            ),
        }
    }
}

/// Test that values generated by `uncompressedValuesGenerator` are lossless compressed after they
/// have been compressed and decompressed using `method`. The libraries public interface is used to
/// make it simpler refactor the libraries internals. The input `error_bound` is used to computed
/// the an absolute error bound based on the range of the generated values.
pub fn testGeneratedLosslessCompression(
    allocator: Allocator,
    uncompressedValuesGenerator: fn (
        allocator: Allocator,
        uncompressed_values: *ArrayList(f64),
        random: Random,
    ) Error!void,
    method: Method,
    data_distribution_name: []const u8,
) !void {
    const random = getDefaultRandomGenerator();

    var uncompressed_values = ArrayList(f64).empty;
    defer uncompressed_values.deinit(allocator);
    try uncompressedValuesGenerator(allocator, &uncompressed_values, random);

    // Create the configuration json string. Since it is lossless we use (for now),
    // a EmptyConfiguration.
    const method_configuration = "{}";

    var compressed = try tersets.compress(
        allocator,
        uncompressed_values.items,
        method,
        method_configuration,
    );
    defer compressed.deinit(allocator);

    var decompressed = try tersets.decompress(allocator, compressed.items);
    defer decompressed.deinit(allocator);

    if (uncompressed_values.items.len != decompressed.items.len) {
        try testing.expectFmt(
            "",
            "Seed: {}, expected_len {}, found_len {}",
            .{ default_seed, uncompressed_values.items.len, decompressed.items.len },
        );
        return;
    }

    for (uncompressed_values.items, decompressed.items, 0..) |raw_value, decompressed_value, i| {
        // Check if the decompressed value is within the error bound of the expected value.
        // Cast the difference between raw and decompressed values to f32 before comparing
        // to the `f32` error bound, to ignore insignificant differences below f32 precision.
        // This prevents false test failures due to `f64` rounding noise.
        const decompression_error: f32 = @floatCast(@abs(raw_value - decompressed_value));
        if (decompression_error > 0.0) {
            try testing.expectFmt(
                "",
                "Seed: {}, index {}, raw value {}, decompressed value {},\nwith data distribution: {s} \n previous raw value {}, next raw value {}\n",
                .{
                    default_seed,
                    i,
                    raw_value,
                    decompressed_value,
                    data_distribution_name,
                    if (i > 0) uncompressed_values.items[i - 1] else 0.0,
                    if (i + 1 < uncompressed_values.items.len) uncompressed_values.items[i + 1] else 0.0,
                },
            );
            return;
        }
    }
}

/// Test that values generated by `uncompressedValuesGenerator` are within `error_bound` after they
/// have been compressed and decompressed using `method`. The libraries public interface is used to
/// make it simpler refactor the libraries internals. The input `error_bound` is used to computed
/// the an absolute error bound based on the range of the generated values.
pub fn testGeneratedErrorBoundedCompression(
    allocator: Allocator,
    uncompressedValuesGenerator: fn (
        allocator: Allocator,
        uncompressed_values: *ArrayList(f64),
        random: Random,
    ) Error!void,
    method: Method,
    error_bound: f32,
    data_distribution_name: []const u8,
) !void {
    const random = getDefaultRandomGenerator();

    var uncompressed_values = ArrayList(f64).empty;
    defer uncompressed_values.deinit(allocator);
    try uncompressedValuesGenerator(allocator, &uncompressed_values, random);

    // Ensure that the error bound is a percentage of the range of the uncompressed values.
    // To avoid scanning for the minimum and maximum values, the first and last values are used
    // as a proxy for the range. This is a simplification, but it is sufficient for testing purposes.
    const data_range: f32 = @as(f32, @floatCast(@abs(
        uncompressed_values.items[0] - uncompressed_values.items[uncompressed_values.items.len - 1],
    )));
    // Check if the error bound is finite and add 0.5 to avoid zero error bound.
    const ranged_error_bound: f32 = (if (math.isFinite(data_range)) data_range else 1.0) * error_bound + 0.5;

    // Create the configuration json string. This function only checks for methods
    // that support an absolute error bound.
    const method_configuration = try std.fmt.allocPrint(
        allocator,
        "{{\"abs_error_bound\": {d}}}",
        .{ranged_error_bound},
    );
    defer allocator.free(method_configuration);

    var compressed = try tersets.compress(
        allocator,
        uncompressed_values.items,
        method,
        method_configuration,
    );
    defer compressed.deinit(allocator);

    var decompressed = try tersets.decompress(allocator, compressed.items);
    defer decompressed.deinit(allocator);

    if (uncompressed_values.items.len != decompressed.items.len) {
        try testing.expectFmt(
            "",
            "Seed: {}, expected_len {}, found_len {}",
            .{ default_seed, uncompressed_values.items.len, decompressed.items.len },
        );
        return;
    }

    for (uncompressed_values.items, decompressed.items, 0..) |raw_value, decompressed_value, i| {
        // Check if the decompressed value is within the error bound of the expected value.
        // Cast the difference between raw and decompressed values to f32 before comparing
        // to the `f32` error bound, to ignore insignificant differences below f32 precision.
        // This prevents false test failures due to `f64` rounding noise.
        const decompression_error: f32 = @floatCast(@abs(raw_value - decompressed_value));
        if (decompression_error > ranged_error_bound) {
            try testing.expectFmt(
                "",
                "Seed: {}, index {}, raw value {}, compressed value {}, error bound {},\n error bound exceeded by {}({}%), with data distribution: {s} \n previous raw value {}, next raw value {}\n",
                .{
                    default_seed,
                    i,
                    raw_value,
                    decompressed_value,
                    ranged_error_bound,
                    decompression_error - ranged_error_bound,
                    (decompression_error - ranged_error_bound) / ranged_error_bound * 100.0,
                    data_distribution_name,
                    if (i > 0) uncompressed_values.items[i - 1] else 0.0,
                    if (i + 1 < uncompressed_values.items.len) uncompressed_values.items[i + 1] else 0.0,
                },
            );
            return;
        }
    }
}

/// Test that values generated by `uncompressed_values_generator` are within `error_bound` according
/// to `within_error_bound` after they have been compressed and decompressed using `method`. The
/// libraries public interface is used to make it simpler refactor the libraries internals.
pub fn testGenerateCompressAndDecompress(
    allocator: Allocator,
    uncompressedValuesGenerator: fn (
        allocator: Allocator,
        uncompressed_values: *ArrayList(f64),
        random: Random,
    ) Error!void,
    method: Method,
    error_bound: f32,
    withinErrorBound: fn (
        uncompressed_values: []const f64,
        decompressed_values: []const f64,
        error_bound: f32,
    ) bool,
) !void {
    const random = getDefaultRandomGenerator();

    var uncompressed_values = ArrayList(f64).empty;
    defer uncompressed_values.deinit(allocator);
    try uncompressedValuesGenerator(allocator, &uncompressed_values, random);

    // subsequenceStack contains subsequences to run the test for to find shortest failing sequence.
    var subsequenceStack = ArrayList(usize).empty;
    defer subsequenceStack.deinit(allocator);

    try subsequenceStack.append(allocator, 0);
    try subsequenceStack.append(allocator, uncompressed_values.items.len);
    var shortestStart: usize = 0;
    var shortestEnd: usize = math.maxInt(usize);

    while (subsequenceStack.items.len != 0) {
        // subsequenceStack contains start and end as separate integers of usize.
        const end = subsequenceStack.pop().?;
        const start = subsequenceStack.pop().?;
        const uncompressed_values_subsequence = uncompressed_values.items[start..end];

        testCompressAndDecompress(
            allocator,
            uncompressed_values_subsequence,
            method,
            error_bound,
            withinErrorBound,
        ) catch {
            // To simplify debugging failed tests that use auto generated data, the tests are
            // retried with smaller subsequence to find the smallest subsequence that fails.
            if (start < end - 1) {
                const middle = (start + end) / 2;
                try subsequenceStack.append(allocator, start);
                try subsequenceStack.append(allocator, middle);
                try subsequenceStack.append(allocator, middle);
                try subsequenceStack.append(allocator, end);
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
            .{ default_seed, uncompressed_values.items[shortestStart..shortestEnd] },
        );
    }
}

/// Test that `uncompressed_values` are within `error_bound` according to `within_error_bound` after
/// it has been compressed and decompressed using `method`. Assumes that `within_error_bound`
/// returns `false` if the number of uncompressed and decompressed values are different. The
/// libraries public interface is used to make it simpler to refactor the libraries internals.
pub fn testCompressAndDecompress(
    allocator: Allocator,
    uncompressed_values: []const f64,
    method: Method,
    error_bound: f32,
    withinErrorBound: fn (
        uncompressed_values: []const f64,
        decompressed_values: []const f64,
        error_bound: f32,
    ) bool,
) !void {
    const method_configuration = try std.fmt.allocPrint(
        allocator,
        "{{\"abs_error_bound\": {d}}}",
        .{error_bound},
    );
    defer allocator.free(method_configuration);

    var compressed_values = try tersets.compress(
        allocator,
        uncompressed_values,
        method,
        method_configuration,
    );
    defer compressed_values.deinit(allocator);

    var decompressed_values = try tersets.decompress(allocator, compressed_values.items);
    defer decompressed_values.deinit(allocator);

    try testing.expect(withinErrorBound(
        uncompressed_values,
        decompressed_values.items,
        error_bound,
    ));
}

// Replace each normal value in `uncompressed_values` with a positive +inf, -inf, or NaN
// with the passed probability. The non-normal values are written to `uncompressed_values`
// in the previously listed order, thus a +inf maybe overwritten by a -inf
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

/// Generate a random `f64` value using `random_opt`. If `random_opt` is not passed, a random number
/// generator is created.
pub fn generateRandomValue(random_opt: ?Random) f64 {
    var random = resolveRandom(random_opt);

    // rand can only generate f64 values in the range [0, 1).
    const random_value = @as(f64, @bitCast(random.int(u64)));
    return random_value;
}

/// Generate a random number of `f64` values using `random` and add them to `uncompressed_values`.
/// Each value is a random `f64` generated from a random `u64` bit pattern, which may include
/// special values such as NaN or inf. The final number of values is determined by a random
/// generation function that returns an integer value between 100 and 150.
pub fn generateRandomValues(allocator: Allocator, uncompressed_values: *ArrayList(f64), random: Random) !void {
    for (0..generateNumberOfValues(random)) |_| {
        // Generate a random f64 by bit-casting a random u64.
        const random_value = @as(f64, @bitCast(random.int(u64)));
        try uncompressed_values.append(allocator, random_value);
    }
}

/// Generate a random number of `f64` values for use in testing using `random` and add
/// them to `uncompressed_values`. If the value is not finite, it is replaced with zero.
pub fn generateFiniteRandomValues(allocator: Allocator, uncompressed_values: *ArrayList(f64), random: Random) !void {
    var index: usize = 0;
    while (index < generateNumberOfValues(random)) {
        // rand can only generate f64 values in the range [0, 1), thus using u64.
        const random_value = @as(f64, @bitCast(random.int(u64)));
        // Online add finite values.
        if (std.math.isFinite(random_value)) {
            try uncompressed_values.append(allocator, random_value);
            index += 1;
        }
    }
}

/// Generate a random number of sinusoidal functions with random amplitudes, frequencys,
/// and additive noise using `random` and add them to `uncompressed_values`.
/// The number of functions is randomly chosen in the interval [global_at_least, global_at_most].
pub fn generateRandomSinusoidalFunctions(
    allocator: Allocator,
    uncompressed_values: *ArrayList(f64),
    random: Random,
) !void {
    // Generate a random number of functions in the interval `global_at_least` and `global_at_most`.
    const num_functions = random.intRangeAtMost(u32, global_at_least, global_at_most);
    for (0..num_functions) |_| {
        try generateRandomSinusoidalFunction(allocator, uncompressed_values, random);
    }
}

/// Generate a random number of linear functions with random slope and intercept using `random` and
/// add them to `uncompressed_values`. The number of functions is randomly chosen in the interval
/// [global_at_least, global_at_most].
pub fn generateRandomLinearFunctions(allocator: Allocator, uncompressed_values: *ArrayList(f64), random: Random) !void {
    // Generate a random number of functions in the interval `global_at_least` and `global_at_most`.
    const num_lines = random.intRangeAtMost(u32, global_at_least, global_at_most);
    for (0..num_lines) |_| {
        try generateRandomLinearFunction(allocator, uncompressed_values, random);
    }
}

/// Generate a random number of quadratic functions with random coefficients using `random` and
/// add them to `uncompressed_values`.
pub fn generateRandomQuadraticFunctions(allocator: Allocator, uncompressed_values: *ArrayList(f64), random: Random) !void {
    // Generate a random number of functions using `global_at_least` and `global_at_most`.
    const num_functions = random.intRangeAtMost(u32, global_at_least, global_at_most);
    for (0..num_functions) |_| {
        try generateRandomQuadraticFunction(allocator, uncompressed_values, random);
    }
}

/// Generate a random number of power functions with random coefficients using `random` and
/// add them to `uncompressed_values`.
pub fn generateRandomPowerFunctions(allocator: Allocator, uncompressed_values: *ArrayList(f64), random: Random) !void {
    // Generate a random number of functions using `global_at_least` and `global_at_most`.
    const num_functions = random.intRangeAtMost(u32, global_at_least, global_at_most);
    for (0..num_functions) |_| {
        try generateRandomPowerFunction(allocator, uncompressed_values, random);
    }
}

/// Generate a random number of exponential functions with random coefficients using `random` and
/// add them to `uncompressed_values`.
pub fn generateRandomExponentialFunctions(allocator: Allocator, uncompressed_values: *ArrayList(f64), random: Random) !void {
    // Generate a random number of functions using `global_at_least` and `global_at_most`.
    const num_functions = random.intRangeAtMost(u32, global_at_least, global_at_most);
    for (0..num_functions) |_| {
        try generateRandomExponentialFunction(allocator, uncompressed_values, random);
    }
}

/// Generate a random number of square root functions with random coefficients using `random` and
/// add them to `uncompressed_values`.
pub fn generateRandomSqrtFunctions(allocator: Allocator, uncompressed_values: *ArrayList(f64), random: Random) !void {
    // Generate a random number of functions using `global_at_least` and `global_at_most`.
    const num_functions = random.intRangeAtMost(u32, global_at_least, global_at_most);
    for (0..num_functions) |_| {
        try generateRandomSqrtFunction(allocator, uncompressed_values, random);
    }
}

/// Generate a random number of `f64` values for use in testing using `random` and add
/// them to `uncompressed_values`. The function also replaces some of the generated values with NaNs
/// and infinities with a almost probability one.
pub fn generateRandomValuesWithNaNs(allocator: Allocator, values: *ArrayList(f64), random: Random) !void {
    try generateRandomValues(allocator, values, random);
    replaceNormalValues(
        values,
        global_replace_probability,
        global_replace_probability,
        global_replace_probability,
        random,
    );
}

/// Generate a random number of linear functions with random slope and intercept for use
/// in testing using `random` and add them to `uncompressed_values`. The function also replaces some
/// of the generated values with NaNs and infinities with almost probability one.
pub fn generateRandomLinearFunctionsWithNaNs(allocator: Allocator, values: *ArrayList(f64), random: Random) !void {
    try generateRandomLinearFunctions(allocator, values, random);
    replaceNormalValues(
        values,
        global_replace_probability,
        global_replace_probability,
        global_replace_probability,
        random,
    );
}

/// Wrapper around `generateBoundedRandomValues` with a default range. The function generates
/// a random number of `f64` values between [-1e15, 1e15] for use in testing using
/// `random` and adds them to `uncompressed_values`. This range can be represented by a `f64`
/// without losing precision, thus it is used as a default range for testing purposes.
pub fn generateDefaultBoundedValues(allocator: Allocator, values: *ArrayList(f64), random: Random) !void {
    try generateBoundedRandomValues(allocator, values, -1e15, 1e15, random);
}

/// Generate a random number of `f64` values values between -1e15 and 1e15 for use in testing using
/// `random` and add them to `uncompressed_values`. The function also replaces some of the
/// generated values with NaNs and infinities with almost probability one.
pub fn generateDefaultBoundedValuesWithNaNs(allocator: Allocator, values: *ArrayList(f64), random: Random) !void {
    try generateDefaultBoundedValues(allocator, values, random);
    replaceNormalValues(
        values,
        global_replace_probability,
        global_replace_probability,
        global_replace_probability,
        random,
    );
}

/// Generate a random sinusoid (finite values) and then replace some of those values
/// with special values (NaN, +inf, -inf) using the given per‑value probabilities inside
/// `replaceNormalValues`. The values are generated using `random` and return in `values`.
/// If an error is found, it is returned.
pub fn generateRandomSinusoidalFunctionWithNaNs(allocator: Allocator, values: *ArrayList(f64), random: Random) !void {
    try generateRandomSinusoidalFunction(allocator, values, random);
    replaceNormalValues(
        values,
        global_replace_probability,
        global_replace_probability,
        global_replace_probability,
        random,
    );
}

/// Generate a random number of `f64` values between `lower_bound` and `upper_bound` for
/// use in testing using `random` and add them to `uncompressed_values`.
pub fn generateBoundedRandomValues(
    allocator: Allocator,
    uncompressed_values: *ArrayList(f64),
    lower_bound: f64,
    upper_bound: f64,
    random_opt: ?Random,
) !void {
    const random = resolveRandom(random_opt);

    for (0..generateNumberOfValues(random)) |_| {
        // generate f64 values in the range [0, 1).
        const bounded_value = lower_bound + (upper_bound - lower_bound);
        const clamped_value = math.clamp(bounded_value, -clamped_max_value, clamped_max_value);
        try uncompressed_values.append(allocator, clamped_value);
    }
}

/// Generate a random number of `f64` values following a linear function with random slope
/// and intercept, and add them to `uncompressed_values`. The noise added to each value is
/// a random value in the range [-0.5%, 0.5%] times the absolute value. The generated
/// values are bounded within [-1e15, 1e15].
pub fn generateRandomLinearFunction(allocator: Allocator, uncompressed_values: *ArrayList(f64), random: Random) !void {
    // Choose log-uniform magnitude in [1e-2, 1e10]. This allows both small and large values
    // to be equally likely to be sampled. If a uniform distribution was used, it would bias
    // towards larger values. The interval [1e-2, 1e10] for both slope and intercept
    // guarantees final values within [-1e15, 1e15] without bias.
    var log_magnitude = random.float(f64) * 10.0 - 2.0;
    var magnitude = math.pow(f64, 10.0, log_magnitude);
    var sign: f64 = if (random.boolean()) 1.0 else -1.0;
    const slope = sign * magnitude;

    log_magnitude = random.float(f64) * 10.0 - 2.0;
    magnitude = math.pow(f64, 10.0, log_magnitude);
    sign = if (random.boolean()) 1.0 else -1.0;
    const intercept = sign * magnitude;

    for (0..generateNumberOfValues(random)) |x| {
        const linear_function_value = addNoise(slope * @as(f64, @floatFromInt(x)) + intercept);
        const clamped_value = math.clamp(linear_function_value, -clamped_max_value, clamped_max_value);
        try uncompressed_values.append(allocator, clamped_value);
    }
}

/// Generate a random number of `f64` values following a power function with random coefficients
/// theta_1 * x ^ theta_2, and add them to `uncompressed_values`. Small random noise is added to
/// each value. The generated values are bounded within [-1e15, 1e15]. If `random_opt` is
/// not passed, a random number generator is created.
pub fn generateRandomPowerFunction(allocator: Allocator, uncompressed_values: *ArrayList(f64), random: Random) !void {

    // theta_1 in [1e-2, 1e6] (log-uniform).
    const log_theta_1 = random.float(f64) * 7.0;
    const sign: f64 = if (random.boolean()) 1.0 else -1.0;
    const theta_1 = math.pow(f64, 10.0, log_theta_1) * sign;

    // theta_2 in [0.1, 4.0] (uniform). Too high exponents lead to overflow.
    const theta_2 = 0.1 + 4 * random.float(f64);

    for (0..generateNumberOfValues(random)) |x| {
        const power_function_value = addNoise(theta_1 * math.pow(f64, @floatFromInt(x), theta_2));
        const clamped_value = math.clamp(power_function_value, -clamped_max_value, clamped_max_value);
        try uncompressed_values.append(allocator, clamped_value);
    }
}

/// Generate a random number of `f64` values following a sinusoidal function with random amplitude,
/// frequency, and additive noise. The output values are guaranteed to be finite and lie within the
/// range [-1e15, 1e15]. The values are generated using `random_opt` and returned in
/// `uncompressed_values`. If an error occurs, it is returned.
pub fn generateRandomSinusoidalFunction(
    allocator: Allocator,
    uncompressed_values: *ArrayList(f64),
    random: Random,
) !void {
    // Amplitude sampled in [1e-2, 1e14) allows very small to large oscillations.
    const amplitude = math.pow(f64, 10.0, random.float(f64) * 16.0 - 2.0);
    // Frequency sampled in [1e-4, 1e1) covers slowly to moderately fast variation.
    const frequency = math.pow(f64, 10.0, random.float(f64) * 5.0 - 4.0);
    // Phase sampled in [0, 2 * pi) ensures random starting point in the cycle.
    const phase = random.float(f64) * 2.0 * math.pi;

    const n = generateNumberOfValues(random);
    try uncompressed_values.ensureUnusedCapacity(allocator, n);

    var i: usize = 0;
    while (i < n) : (i += 1) {
        const x = @as(f64, @floatFromInt(i));
        var value = addNoise(amplitude * @sin(frequency * x + phase));
        // Clamp to [-1e15, 1e15] to avoid overflows and invalid values.
        value = math.clamp(value, -clamped_max_value, clamped_max_value);
        uncompressed_values.appendAssumeCapacity(value);
    }
}

/// Generate a random number of `f64` values following a quadratic function with random coefficients
/// theta_1 * x ^ 2 + theta_2, and add them to `uncompressed_values`. Small random noise is added to
/// each value. The generated values are bounded within [-1e15, 1e15]. If `random_opt` is
/// not passed, a random number generator is created.
pub fn generateRandomQuadraticFunction(allocator: Allocator, uncompressed_values: *ArrayList(f64), random: Random) !void {
    // theta_1 in [-1e6, 1e6] (log-uniform).
    const log_theta_1 = random.float(f64) * 7.0;
    var sign: f64 = if (random.boolean()) 1.0 else -1.0;
    const theta_1 = math.pow(f64, 10.0, log_theta_1) * sign;

    // theta_2 in [-1e10, 1e10] (log-uniform).
    const log_theta_2 = random.float(f64) * 10.0;
    sign = if (random.boolean()) 1.0 else -1.0;
    const theta_2 = math.pow(f64, 10.0, log_theta_2) * sign;

    for (0..generateNumberOfValues(random)) |x| {
        const xf = @as(f64, @floatFromInt(x));
        // Small random noise in the range [-0.5, 0.5)
        const power_function_value = addNoise(theta_1 * xf * xf + theta_2);
        const clamped_value = math.clamp(power_function_value, -clamped_max_value, clamped_max_value);
        try uncompressed_values.append(allocator, clamped_value);
    }
}

/// Generate a random number of `f64` values following a square root function with random coefficients
/// theta_1 * sqrt(x) + theta_2, and add them to `uncompressed_values`. Small random noise is added to
/// each value. The generated values are bounded within [-1e15, 1e15]. If `random_opt` is
/// not passed, a random number generator is created.
pub fn generateRandomSqrtFunction(allocator: Allocator, uncompressed_values: *ArrayList(f64), random: Random) !void {
    // theta_1 in [-1e10, 1e10] (log-uniform).
    const log_theta_1 = random.float(f64) * 10.0;
    var sign: f64 = if (random.boolean()) 1.0 else -1.0;
    const theta_1 = math.pow(f64, 10.0, log_theta_1) * sign;

    // theta_2 in [-1e10, 1e10] (log-uniform).
    const log_theta_2 = random.float(f64) * 10.0;
    sign = if (random.boolean()) 1.0 else -1.0;
    const theta_2 = math.pow(f64, 10.0, log_theta_2) * sign;

    for (0..generateNumberOfValues(random)) |x| {
        const xf = @as(f64, @floatFromInt(x));
        // Small random noise in the range [-0.5, 0.5)
        const power_function_value = theta_1 * @sqrt(xf) + theta_2;
        const clamped_value = math.clamp(power_function_value, -clamped_max_value, clamped_max_value);
        try uncompressed_values.append(allocator, clamped_value);
    }
}

/// Generate a random number of `f64` values following a exponential function with random coefficients
/// theta_1 * e ^ (x * theta_2), and add them to `uncompressed_values`. Small random noise is added to
/// each value. The generated values are bounded within [-1e15, 1e15]. If `random_opt` is
/// not passed, a random number generator is created.
pub fn generateRandomExponentialFunction(allocator: Allocator, uncompressed_values: *ArrayList(f64), random: Random) !void {

    // This function is prone to overflow, thus using smaller ranges for the coefficients.
    // theta_1 in [-1e5, 1e5] (log-uniform).
    const log_theta_1 = random.float(f64) * 5.0;
    var sign: f64 = if (random.boolean()) 1.0 else -1.0;
    const theta_1 = math.pow(f64, 10.0, log_theta_1) * sign;

    // theta_2 in [0.001, 0.01] (log-uniform).
    const log_theta_2 = random.float(f64) - 3.0;
    sign = if (random.boolean()) 1.0 else -1.0;
    const theta_2 = math.pow(f64, 10.0, log_theta_2) * sign;

    for (0..generateNumberOfValues(random)) |x| {
        const xf = @as(f64, @floatFromInt(x));
        // Small random noise in the range [-0.5, 0.5)
        const rand_value = random.float(f64) - 0.5;
        const power_function_value = theta_1 * @sqrt(xf) + theta_2 + rand_value;
        const clamped_value = math.clamp(power_function_value, -clamped_max_value, clamped_max_value);
        try uncompressed_values.append(allocator, clamped_value);
    }
}

/// Generate a series with a mix of different data distributions using `random` and add them to
/// `uncompressed_values`. The function generates linear, quadratic, power, exponential, sinusoidal,
/// and square root functions. The final time series is a concatenation of these different
/// distributions to create a diverse set of values for testing purposes, with the order randomized.
pub fn generateMixedBoundedValuesFunctions(
    allocator: Allocator,
    uncompressed_values: *ArrayList(f64),
    random: Random,
) Error!void {
    // Type alias for a pointer to a generator function that takes an ArrayList of f64 and a Random,
    // and returns an Error or void.
    const Generator = *const fn (Allocator, *ArrayList(f64), Random) Error!void;

    // Array of generator function pointers, each generating a different type of mathematical function.
    const generators: [6]Generator = .{
        generateRandomLinearFunction,
        generateRandomQuadraticFunction,
        generateRandomPowerFunction,
        generateRandomExponentialFunction,
        generateRandomSqrtFunction,
        generateRandomSinusoidalFunction,
    };

    // Array of indices corresponding to the generator functions, used for shuffling.
    var indices: [generators.len]usize = undefined;

    // Initialize the indices array with sequential values.
    for (indices, 0..) |_, i| indices[i] = i;

    // Shuffle the indices array using the Fisher-Yates algorithm and a random number generator.
    var i = indices.len;
    while (i > 1) : (i -= 1) {
        const j = random.intRangeAtMost(usize, 0, i - 1);
        const tmp = indices[i - 1];
        indices[i - 1] = indices[j];
        indices[j] = tmp;
    }

    // Iterate over the shuffled indices and invoke the corresponding generator function,
    // passing in the uncompressed_values and random number generator.
    for (indices) |idx| {
        try generators[idx](allocator, uncompressed_values, random);
    }
}

/// Generate a random value of type `T` between `at_least` and `at_most` for use in testing using
/// `random_opt`. `T` must be a floating-point type (e.g., `f32`, `f64`). If random_opt is not
/// passed, a random number generator is created using the current time as seed.
pub fn generateBoundedRandomValue(comptime T: type, at_least: T, at_most: T, random_opt: ?Random) T {
    var random = resolveRandom(random_opt);

    const rand_value: T = random.float(T);
    const bounded_value = at_least + (at_most - at_least) * rand_value;
    return bounded_value;
}

/// Generate a random value of type `T` between `at_least` and `at_most` for use in testing using
/// `random_opt`. `T` must be an integer-point type (e.g., `i32`, `usize`). If random_opt is not
/// passed, a random number generator is created using the current time as seed.
pub fn generateBoundRandomInteger(comptime T: type, at_least: T, at_most: T, random_opt: ?Random) T {
    var random = resolveRandom(random_opt);

    const rand_value: T = random.intRangeAtMost(T, at_least, at_most);
    return rand_value;
}

/// Generate a random number of values used for testing. This value needs to be higher than or equal to 2
/// otherwise some of the tests will fail. The value is set to between [100-150] to ensure that the tests are
/// not too slow. The values is generated randomly to obtain a different set of values for each test run.
pub fn generateNumberOfValues(random: Random) usize {
    const number_of_values: usize = random.intRangeAtMost(usize, 100, 150);
    return number_of_values;
}

/// Returns the default `Random` instance, initializing it with the current millisecond timestamp
/// as the seed if it has not been initialized yet. This ensures that repeated calls return the same
/// pseudo-random number generator unless the seed is reset.
pub fn getDefaultRandomGenerator() Random {
    if (default_seed == 0) {
        default_seed = @bitCast(time.milliTimestamp());
        default_prng = std.Random.DefaultPrng.init(default_seed);
    }
    return default_prng.random();
}

/// Returns a `Random` object. If `random_opt` is provided, it is returned directly. Otherwise,
/// this function returns the default `Random` instance.
pub fn resolveRandom(random_opt: ?Random) Random {
    return random_opt orelse getDefaultRandomGenerator();
}

/// Adds noise to a given value based on `noise_scale`. This ensures that the noise is proportional
/// to the magnitude of the input `value`. Returns the noisy value.
fn addNoise(value: f64) f64 {
    // Generate a random value in [-0.5, 0.5).
    const rand_factor = getDefaultRandomGenerator().float(f64) - 0.5;
    const noise = rand_factor * noise_scale * @abs(value);
    return value + noise;
}
