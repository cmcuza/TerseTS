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

//! Implementation of ABC Linear Approximation under the L-infinity norm
//! Based on the paper:
//! "Approximations of One-Dimensional Digital Signals Under the Norm"
//! by M. Dalai and R. Leonardi, IEEE Transactions on Signal Processing, 2006.
//
// The name "ABCLinearApproximation" reflects the core geometric concept of the algorithm:
// the optimal segment is determined by evaluating triplets of points A, B,
// and C.

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

/// Compresses the signal using ABCLinearApproximation under the L-infinity norm
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

    var convex_hull = try ConvexHull.init(allocator);
    // Free the memory used by the convex hull.
    defer convex_hull.deinit();

    var start: usize = 0;

    while (start < uncompressed_values.len - 1) {
        // Clean convex hull for the new segment
        convex_hull.clean();

        // Start of the new segment (p0)
        var last_valid_end = start;
        var last_valid_line: ?LinearFunction = null;
        var i = start;

        // 'i' walks through the points trying to grow the current segment
        while (i < uncompressed_values.len) : (i += 1) {
            // Section III-A, Step 1: Computing the Convex Hull
            // Add next point to convex hull for current segment
            try convex_hull.add(.{ .time = i, .value = uncompressed_values[i] });

            if (convex_hull.len() < 2) {
                last_valid_end = i;
                continue;
            }

            // Section III-A, Step 2-3: Find A, B, C and compute the solution line
            // Try to compute the best fitting line using current convex hull points
            const line = try findABCOptimalSegment(&convex_hull, allocator);

            // Compute maximum error over current segment
            const max_error = try convex_hull.computeMaxError(line);

            if (max_error <= error_bound) {

                // If all points are within error_bound -> segment still valid
                last_valid_end = i;
                last_valid_line = line;
            } else {
                // Otherwise: error exceeded -> stop extending segment
                break;
            }
        }

        // Store segment information (end index, slope, intercept)
        if (last_valid_line) |valid_line| {
            try appendValue(usize, last_valid_end, compressed_values);
            try appendValue(f64, @floatCast(valid_line.slope), compressed_values);
            try appendValue(f64, @floatCast(valid_line.intercept), compressed_values);
        }

        // Start next segment after last_valid_end
        start = last_valid_end + 1;
    }
    // Store the last point if left
    if (start == uncompressed_values.len - 1) {
        const value = uncompressed_values[start];
        const slope: f64 = 0.0;
        const intercept = value;

        try appendValue(usize, start, compressed_values);
        try appendValue(f64, slope, compressed_values);
        try appendValue(f64, intercept, compressed_values);
    }
}

/// Decompress the ABC Linear-compressed stream.
/// Simply rebuilds points from stored slope and intercept.
pub fn decompress(
    compressed_values: []const u8,
    decompressed_values: *ArrayList(f64),
) Error!void {
    // 3 fields x 8 bytes
    if (compressed_values.len % 24 != 0) return Error.IncorrectInput;

    const fields = mem.bytesAsSlice(f64, compressed_values);
    var field_index: usize = 0;
    var segment_start: usize = 0;

    while (field_index + 2 < fields.len) {
        const segment_end = @as(usize, @bitCast(fields[field_index]));
        const slope = fields[field_index + 1];
        const intercept = fields[field_index + 2];

        for (segment_start..segment_end + 1) |t| {
            const x = @as(f64, @floatFromInt(t));
            const y = slope * x + intercept;
            try decompressed_values.append(y);
        }

        segment_start = segment_end + 1;
        field_index += 3;
    }
}

/// Helper to serialize a value into bytes
fn appendValue(comptime T: type, value: T, compressed: *ArrayList(u8)) !void {
    const bytes: [8]u8 = @bitCast(value);
    try compressed.appendSlice(&bytes);
}

// Find the optimal segment using ABC structure from convex hulls.
/// A, B = side from either hull
/// C = point from the opposite hull with max deviation, projected vertically into AB
pub fn findABCOptimalSegment(convex_hull: *ConvexHull, allocator: mem.Allocator) Error!LinearFunction {
    const len = convex_hull.len();

    // Initialize first side l1 = (p0, p1)
    var A_index: usize = 0;
    var finished = false;
    var visited = std.AutoHashMap(usize, void).init(allocator);
    defer visited.deinit();

    while (!finished) {
        if (A_index + 1 >= len) break; // No more sides

        const A = convex_hull.at(A_index);
        const B = convex_hull.at(A_index + 1);

        // Find C
        const maybe_pivot_idx = findPivotC(convex_hull, A_index);

        const C_index = maybe_pivot_idx orelse {
            // No valid C found, use AB as the line (in case of just two points in convex hull)
            // Need at least three points to define a valid segment using the ABC method
            const slope = (B.value - A.value) / (@as(f64, @floatFromInt(B.time)) - @as(f64, @floatFromInt(A.time)));
            const intercept = A.value - slope * @as(f64, @floatFromInt(A.time));

            return LinearFunction{ .slope = slope, .intercept = intercept };
        };

        const C = convex_hull.at(C_index);

        if (visited.contains(A_index)) {
            break;
        }
        try visited.put(A_index, {}); // Mark as visited

        // Determine relative x-position of pivot to l_i
        if (C.time > B.time) {
            // x-external to right -> advance to next side
            A_index += 1;
        } else if (C.time < A.time) {
            // x-external to left -> adjust upper hull between v(l_{i-1}) and v(l_i)
            if (A_index == 0) {
                // If we are already at the first side, we cannot move left.
                // In this case, just accept the current segment as optimal.
                finished = true;
            } else {
                // x-external to left -> adjust search: go back one side
                A_index -= 1;
            }
        } else {
            // x-internal -> optimal segment found
            finished = true;
        }
    }

    // Once finished, line from A to B is the optimal approximation
    const start = convex_hull.at(A_index);
    const end = convex_hull.at(A_index + 1);

    const slope = (end.value - start.value) / (@as(f64, @floatFromInt(end.time)) - @as(f64, @floatFromInt(start.time)));
    const intercept = start.value - slope * @as(f64, @floatFromInt(start.time));

    return LinearFunction{ .slope = slope, .intercept = intercept };
}

/// Find the pivot point C.
fn findPivotC(convex_hull: *ConvexHull, A_index: usize) ?usize {
    const A = convex_hull.at(A_index);
    const B = convex_hull.at(A_index + 1);
    var max_dev: f64 = -1.0;
    var pivot_idx: ?usize = null;

    // Check all hull vertices for x-internal points
    // Find v(l_i): the vertex that maximizes deviation from the side l_i
    for (0..convex_hull.len()) |C_index| {
        const C = convex_hull.at(C_index);
        // Explicitly exclude A and B from being C
        if (C_index != A_index and C_index != A_index + 1) {
            const dev = computeDeviation(A, B, C);
            if (dev > max_dev) {
                max_dev = dev;
                pivot_idx = C_index;
            }
        }
    }

    return pivot_idx;
}

fn computeDeviation(A: DiscretePoint, B: DiscretePoint, C: DiscretePoint) f64 {
    // Compute slope of side l_i
    const delta_time = (@as(f64, @floatFromInt(B.time)) - @as(f64, @floatFromInt(A.time)));
    const slope = (B.value - A.value) / delta_time;

    const pred = slope * (@as(f64, @floatFromInt(C.time)) - @as(f64, @floatFromInt(A.time))) + A.value;

    // Deviation of point C from line l_i
    return @abs(pred - C.value);
}

test "compresses and decompresses perfect linear signal" {
    const allocator = std.testing.allocator;
    const error_bound: f32 = 0.01;

    var uncompressed_values = ArrayList(f64).init(allocator);
    defer uncompressed_values.deinit();

    for (0..50) |i| {
        try uncompressed_values.append(2 * @as(f64, @floatFromInt(i)) + 1.0);
    }

    var compressed_values = ArrayList(u8).init(allocator);
    defer compressed_values.deinit();

    try compress(
        uncompressed_values.items,
        &compressed_values,
        allocator,
        error_bound,
    );

    // Only 3 elements are stored in the compressed array.
    try std.testing.expectEqual(compressed_values.items.len / 8, 3);
}

test "random lines and error bound compress and decompress" {
    const allocator = std.testing.allocator;
    const seed: u64 = @bitCast(time.milliTimestamp());
    var prng = rand.DefaultPrng.init(seed);
    const random = prng.random();
    const error_bound: f32 = random.float(f32) * 0.1;

    var uncompressed_values = ArrayList(f64).init(allocator);
    defer uncompressed_values.deinit();

    for (0..20) |_| {
        try tester.generateRandomLinearFunction(&uncompressed_values, random);
    }

    try tester.testCompressAndDecompress(uncompressed_values.items, allocator, Method.ABCLinearApproximation, error_bound, tersets.isWithinErrorBound);
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

    try tester.testCompressAndDecompress(uncompressed_values.items, allocator, Method.ABCLinearApproximation, error_bound, tersets.isWithinErrorBound);
}

test "compresses and decompresses constant signal" {
    const allocator = std.testing.allocator;
    const error_bound: f32 = 0.001;

    var uncompressed_values = ArrayList(f64).init(allocator);
    defer uncompressed_values.deinit();

    for (0..30) |_| {
        try uncompressed_values.append(7.7);
    }

    try tester.testCompressAndDecompress(uncompressed_values.items, allocator, Method.ABCLinearApproximation, error_bound, tersets.isWithinErrorBound);
}
