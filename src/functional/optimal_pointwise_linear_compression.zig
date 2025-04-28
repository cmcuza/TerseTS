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

// //! Implementation of Optimal Pointwise Linear Compression under the L-infinity norm
// //! Based on the paper:
// //! "Approximations of One-Dimensional Digital Signals Under the Norm"
// //! by M. Dalai and R. Leonardi, IEEE Transactions on Signal Processing, 2006.

const std = @import("std");
const mem = std.mem;
const time = std.time;
const rand = std.Random;
const ArrayList = std.ArrayList;

const tersets = @import("../tersets.zig");
const tester = @import("../tester.zig");
const Error = tersets.Error;
const Method = tersets.Method;

const shared = @import("../utilities/shared_structs.zig");
const DiscretePoint = shared.DiscretePoint;
const LinearFunction = shared.LinearFunction;
const ConvexHull = @import("../utilities/convex_hull.zig").ConvexHull;
const Segment = shared.Segment;

/// Helper to serialize a value into bytes
fn appendValue(comptime T: type, value: T, compressed: *ArrayList(u8)) !void {
    const bytes: [8]u8 = @bitCast(value);
    try compressed.appendSlice(&bytes);
}

pub fn findOptimalSegment(convex_hull: *ConvexHull) Error!LinearFunction {
    const len = convex_hull.len();

    // Initialize first side l1 = (p0, p1)
    var i: usize = 0;
    var finished = false;

    while (!finished) {
        if (i + 1 >= len) break; // No more sides

        const A = convex_hull.at(i);
        const B = convex_hull.at(i + 1);

        // Compute slope of side l_i
        const side_slope = (B.value - A.value) / (@as(f64, @floatFromInt(B.time)) - @as(f64, @floatFromInt(A.time)));

        // Find v(l_i): the vertex that maximizes deviation from the side l_i
        var max_dev: f64 = -1.0;
        var pivot_idx: usize = i;

        for (i + 1..len) |j| {
            const P = convex_hull.at(j);

            // Deviation of point P from line l_i
            const pred_value = side_slope * (@as(f64, @floatFromInt(P.time)) - @as(f64, @floatFromInt(A.time))) + A.value;
            const deviation = @abs(pred_value - P.value);

            if (deviation > max_dev) {
                max_dev = deviation;
                pivot_idx = j;
            }
        }

        const pivot = convex_hull.at(pivot_idx);

        // Determine relative x-position of pivot wrt l_i
        if (pivot.time > B.time) {
            // x-external to right -> advance to next side
            i += 1;
        } else if (pivot.time < A.time) {
            // x-external to left -> adjust upper hull between v(l_{i-1}) and v(l_i)
            if (i == 0) {
                // If we are already at the first side, we cannot move left.
                // In this case, just accept the current segment as optimal.
                finished = true;
            } else {
                // x-external to left -> adjust search: go back one side
                i -= 1;
            }
        } else {
            // x-internal -> optimal segment found
            finished = true;
        }
    }

    // Once finished, line from A to B is the optimal approximation
    const start = convex_hull.at(i);
    const end = convex_hull.at(i + 1);

    const slope = (end.value - start.value) / (@as(f64, @floatFromInt(end.time)) - @as(f64, @floatFromInt(start.time)));
    const intercept = start.value - slope * @as(f64, @floatFromInt(start.time));

    return LinearFunction{ .slope = slope, .intercept = intercept };
}

/// Compresses the signal using Optimal Pointwise Linear Compression under the L-infinity norm
/// Grows convex hull segments as long as they respect the error bound.
/// Stores end, slope, intercept for each compressed segment.
pub fn compress(
    uncompressed_values: []const f64,
    compressed_values: *ArrayList(u8),
    allocator: mem.Allocator,
    error_bound: f32,
) Error!void {
    if (uncompressed_values.len < 2) return Error.IncorrectInput;
    if (error_bound <= 0.0) return Error.UnsupportedErrorBound;

    const epsilon: f64 = @floatCast(error_bound);

    var convex_hull = try ConvexHull.init(allocator);
    // Free the memory used by the convex hull.
    defer convex_hull.deinit();

    var start: usize = 0;

    while (start < uncompressed_values.len) {
        // Clean convex hull for the new segment
        convex_hull.clean();

        // Start of the new segment (p0)
        var last_valid_end = start;
        var i = start;

        // 'i' walks through the points trying to grow the current segment
        while (i < uncompressed_values.len) : (i += 1) {
            // Section III-A, Step 1: Computing the Convex Hull
            // Add next point to convex hull for current segment
            try convex_hull.add(.{ .time = i, .value = uncompressed_values[i] });

            if (convex_hull.len() < 2) {
                // Need at least two points to define a valid segment
                last_valid_end = i;
                continue;
            }

            // Section III-A, Step 2-3: Find A, B, C and compute the solution line
            // Try to compute the best fitting line using current convex hull points
            // (using slope envelope logic explained in the paper Section III)
            const line = try findOptimalSegment(&convex_hull);

            // Compute maximum error over current segment
            const max_error = try convex_hull.computeMaxError(line);

            if (max_error <= epsilon) {
                // If all points are within error_bound -> segment still valid
                last_valid_end = i;
            } else {
                // Otherwise: error exceeded -> stop extending segment
                break;
            }
        }

        // Finalize the current segment
        const p0 = start;
        const p1 = last_valid_end;
        const p0_val = uncompressed_values[p0];
        const p1_val = uncompressed_values[p1];

        // Calculate slope and intercept for final segment
        const slope = (p1_val - p0_val) / @as(f64, @floatFromInt(p1 - p0));
        const intercept = p0_val - slope * @as(f64, @floatFromInt(p0));

        // Store segment information (end index, slope, intercept)
        try appendValue(usize, p1, compressed_values);
        try appendValue(f64, slope, compressed_values);
        try appendValue(f64, intercept, compressed_values);

        // Start next segment after last_valid_end
        start = p1 + 1;
    }
}

/// Decompress the Optimal Pointwise Linear-compressed stream.
/// Simply rebuilds points from stored slope and intercept.
pub fn decompress(
    compressed_values: []const u8,
    decompressed_values: *ArrayList(f64),
) Error!void {
    // 3 fields x 8 bytes
    if (compressed_values.len % 24 != 0) return Error.IncorrectInput;

    const fields = mem.bytesAsSlice(f64, compressed_values);
    var i: usize = 0;
    var start: usize = 0;

    while (i + 2 < fields.len) {
        const end = @as(usize, @bitCast(fields[i]));
        const slope = fields[i + 1];
        const intercept = fields[i + 2];

        for (start..end + 1) |t| {
            const x = @as(f64, @floatFromInt(t));
            const y = slope * x + intercept;
            try decompressed_values.append(y);
        }

        start = end + 1;
        i += 3;
    }
}

test "compresses and decompresses perfect linear signal" {
    const allocator = std.testing.allocator;
    const error_bound: f32 = 0.01;

    var uncompressed_values = ArrayList(f64).init(allocator);
    defer uncompressed_values.deinit();

    for (0..50) |i| {
        try uncompressed_values.append(2.0 * @as(f64, @floatFromInt(i)) + 5.0);
    }

    try tester.testCompressAndDecompress(uncompressed_values.items, allocator, Method.OptimalPiecewiseLinearApproximation, error_bound, tersets.isWithinErrorBound);
}

test "compresses noisy linear signal within error bound" {
    const allocator = std.testing.allocator;
    const error_bound: f32 = 0.11;

    var uncompressed_values = ArrayList(f64).init(allocator);
    defer uncompressed_values.deinit();

    for (0..50) |i| {
        const noise = if (i % 2 == 0) @as(f64, 0.05) else @as(f64, -0.05);
        try uncompressed_values.append(3.0 * @as(f64, @floatFromInt(i)) + 4.0 + noise);
    }

    try tester.testCompressAndDecompress(uncompressed_values.items, allocator, Method.OptimalPiecewiseLinearApproximation, error_bound, tersets.isWithinErrorBound);
}

test "random lines and error bound compress and decompress" {
    const allocator = std.testing.allocator;
    const error_bound: f32 = 0.01;
    const seed: u64 = @bitCast(time.milliTimestamp());
    var prng = rand.DefaultPrng.init(seed);
    const random = prng.random();

    var uncompressed_values = ArrayList(f64).init(allocator);
    defer uncompressed_values.deinit();

    for (0..20) |_| {
        try tester.generateRandomLinearFunction(&uncompressed_values, random);
    }

    try tester.testCompressAndDecompress(uncompressed_values.items, allocator, Method.OptimalPiecewiseLinearApproximation, error_bound, tersets.isWithinErrorBound);
}

test "odd size compress and decompress" {
    const allocator = std.testing.allocator;
    const error_bound: f32 = 0.01;
    const seed: u64 = @bitCast(time.milliTimestamp());
    var prng = rand.DefaultPrng.init(seed);
    const random = prng.random();

    var uncompressed_values = ArrayList(f64).init(allocator);
    defer uncompressed_values.deinit();
    try tester.generateBoundedRandomValues(&uncompressed_values, 0.0, 1.0, random);

    // Add another element to make the uncompressed values of odd size.
    try uncompressed_values.append(random.float(f64));

    try tester.testCompressAndDecompress(uncompressed_values.items, allocator, Method.OptimalPiecewiseLinearApproximation, error_bound, tersets.isWithinErrorBound);
}

test "compresses and decompresses constant signal" {
    const allocator = std.testing.allocator;
    const error_bound: f32 = 0.001;

    var uncompressed_values = ArrayList(f64).init(allocator);
    defer uncompressed_values.deinit();

    for (0..30) |_| {
        try uncompressed_values.append(7.7);
    }

    try tester.testCompressAndDecompress(uncompressed_values.items, allocator, Method.OptimalPiecewiseLinearApproximation, error_bound, tersets.isWithinErrorBound);
}

test "fails on single point input" {
    const allocator = std.testing.allocator;
    var uncompressed_values = ArrayList(f64).init(allocator);
    defer uncompressed_values.deinit();
    try uncompressed_values.append(42.0);

    var compressed = ArrayList(u8).init(allocator);
    defer compressed.deinit();

    const result = compress(uncompressed_values.items, &compressed, allocator, 0.1);
    try std.testing.expectError(Error.IncorrectInput, result);
}
