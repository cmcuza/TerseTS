// //! Implementation of Linear Pointwise Approximation (LPA) under the L-infinity norm
// //! Based on the paper:
// //! "Approximations of One-Dimensional Digital Signals Under the Norm"
// //! by M. Dalai and R. Leonardi, IEEE Transactions on Signal Processing, 2006.

const std = @import("std");
const mem = std.mem;
const ArrayList = std.ArrayList;

const tersets = @import("../tersets.zig");
const tester = @import("../tester.zig");
const Error = tersets.Error;
const Method = tersets.Method;

const shared = @import("../utilities/shared_structs.zig");
const DiscretePoint = shared.DiscretePoint;
const LinearFunction = shared.LinearFunction;
const ConvexHull = @import("../utilities/convex_hull.zig").ConvexHull;

fn appendValue(comptime T: type, value: T, compressed: *ArrayList(u8)) !void {
    const bytes: [8]u8 = @bitCast(value);
    try compressed.appendSlice(&bytes);
}

/// Compress using optimal convex hull based Linear Pointwise Approximation.
pub fn compress(
    uncompressed_values: []const f64,
    compressed_values: *ArrayList(u8),
    error_bound: f32,
) Error!void {
    if (uncompressed_values.len < 2) return Error.IncorrectInput;
    if (error_bound <= 0.0) return Error.UnsupportedErrorBound;

    const epsilon: f64 = @floatCast(error_bound);
    const allocator = compressed_values.allocator;
    var convex_hull = try ConvexHull.init(allocator);
    defer convex_hull.deinit();

    var start: usize = 0;

    while (start < uncompressed_values.len) {
        convex_hull.clean();

        var last_valid_end = start;
        var i = start;

        while (i < uncompressed_values.len) : (i += 1) {
            try convex_hull.add(.{ .time = i, .value = uncompressed_values[i] });
            const line = try convex_hull.computeMABRLinearFunction();
            const max_error = try convex_hull.computeMaxError(line);

            if (max_error <= epsilon) {
                last_valid_end = i;
            } else {
                break;
            }
        }

        // If no valid endpoint was found beyond the start,
        // linear function cannot be defined, return an error for invalid segment
        if (last_valid_end == start) return Error.IncorrectInput;

        const p0 = start;
        const p1 = last_valid_end;
        const p0_val = uncompressed_values[p0];
        const p1_val = uncompressed_values[p1];

        const slope = (p1_val - p0_val) / @as(f64, @floatFromInt(p1 - p0));
        const intercept = p0_val - slope * @as(f64, @floatFromInt(p0));

        try appendValue(usize, p0, compressed_values);
        try appendValue(usize, p1, compressed_values);
        try appendValue(f64, slope, compressed_values);
        try appendValue(f64, intercept, compressed_values);

        start = p1 + 1;
    }
}

/// Decompress the LPA-compressed stream.
pub fn decompress(
    compressed_values: []const u8,
    decompressed_values: *ArrayList(f64),
) Error!void {
    if (compressed_values.len % 32 != 0) return Error.IncorrectInput;

    const fields = mem.bytesAsSlice(f64, compressed_values);
    var i: usize = 0;

    while (i + 3 < fields.len) {
        const start = @as(usize, @bitCast(fields[i]));
        const end = @as(usize, @bitCast(fields[i + 1]));
        const slope = fields[i + 2];
        const intercept = fields[i + 3];

        for (start..end + 1) |t| {
            const x = @as(f64, @floatFromInt(t));
            const y = slope * x + intercept;
            try decompressed_values.append(y);
        }

        i += 4;
    }
}

test "LPA: compresses and decompresses perfect linear signal" {
    const allocator = std.testing.allocator;
    const error_bound: f32 = 0.01;

    var data = ArrayList(f64).init(allocator);
    defer data.deinit();

    for (0..50) |i| {
        try data.append(2.0 * @as(f64, @floatFromInt(i)) + 5.0);
    }

    try tester.testCompressAndDecompress(data.items, allocator, Method.LinearPointwiseApproximation, error_bound, tersets.isWithinErrorBound);
}

test "LPA: compresses noisy linear signal within error bound" {
    const allocator = std.testing.allocator;
    const error_bound: f32 = 0.11;

    var data = ArrayList(f64).init(allocator);
    defer data.deinit();

    for (0..50) |i| {
        const noise = if (i % 2 == 0) @as(f64, 0.05) else @as(f64, -0.05);
        try data.append(3.0 * @as(f64, @floatFromInt(i)) + 4.0 + noise);
    }

    try tester.testCompressAndDecompress(data.items, allocator, Method.LinearPointwiseApproximation, error_bound, tersets.isWithinErrorBound);
}

test "LPA: ompresses and decompresses constant signal" {
    const allocator = std.testing.allocator;
    const error_bound: f32 = 0.001;

    var data = ArrayList(f64).init(allocator);
    defer data.deinit();

    for (0..30) |_| {
        try data.append(7.7);
    }

    try tester.testCompressAndDecompress(data.items, allocator, Method.LinearPointwiseApproximation, error_bound, tersets.isWithinErrorBound);
}

test "LPA: fails on single point input" {
    const allocator = std.testing.allocator;
    var data = ArrayList(f64).init(allocator);
    defer data.deinit();
    try data.append(42.0);

    var compressed = ArrayList(u8).init(allocator);
    defer compressed.deinit();

    const result = compress(data.items, &compressed, 0.1);
    try std.testing.expectError(Error.IncorrectInput, result);
}
