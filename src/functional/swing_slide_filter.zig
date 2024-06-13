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

//! Implementation of the Swing Filter and Slide Filter algorithms from the paper
//! "Hazem Elmeleegy, Ahmed K. Elmagarmid, Emmanuel Cecchet, Walid G. Aref, and Willy Zwaenepoel.
//! Online piece-wise linear approximation of numerical streams with precision guarantees.
//! Proc. VLDB Endow. 2, 1, 2009.
//! https://doi.org/10.14778/1687627.1687645".

const std = @import("std");
const math = std.math;
const mem = std.mem;
const testing = std.testing;
const ArrayList = std.ArrayList;

const tersets = @import("../tersets.zig");
const Error = tersets.Error;
const DiscretePoint = tersets.DiscretePoint;
const DiscreteSegment = tersets.DiscreteSegment;
const ContinousPoint = tersets.ContinousPoint;
const ContinousSegment = tersets.ContinousSegment;

const ConvexHull = @import("geometry/convex_hull.zig").ConvexHull;

/// Linear function of the form y = slope*x+intercept. It uses f80 for numerical stability.
const LinearFunction = struct {
    slope: f80,
    intercept: f80,
};

/// Compress `uncompressed_values` within `error_bound` using "Swing Filter" and write the
/// result to `compressed_values`. If an error occurs it is returned.
pub fn compressSwing(
    uncompressed_values: []const f64,
    compressed_values: *ArrayList(u8),
    error_bound: f32,
) Error!void {
    // Adjust the error bound to avoid exceeding it during decompression.
    const adjusted_error_bound = if (error_bound > 0)
        error_bound - tersets.ErrorBoundMargin
    else
        error_bound;

    var upper_bound: LinearFunction = .{ .slope = undefined, .intercept = undefined };
    var lower_bound: LinearFunction = .{ .slope = undefined, .intercept = undefined };
    var new_upper_bound: LinearFunction = .{ .slope = undefined, .intercept = undefined };
    var new_lower_bound: LinearFunction = .{ .slope = undefined, .intercept = undefined };

    // Initialize the current segment with first two points.
    var current_segment: DiscreteSegment = .{
        .start_point = .{ .time = 0, .value = uncompressed_values[0] },
        .end_point = .{ .time = 1, .value = uncompressed_values[1] },
    };

    // Compute the numerator Eq. (6).
    var slope_derivate: f80 = computeSlopeDerivate(current_segment);

    updateSwingLinearFunction(current_segment, &upper_bound, adjusted_error_bound);
    updateSwingLinearFunction(current_segment, &lower_bound, -adjusted_error_bound);

    // First two points already part of `current_segment`, next point is at index two.
    var current_timestamp: usize = 2;
    while (current_timestamp < uncompressed_values.len) : (current_timestamp += 1) {
        // Evaluate the upper and lower bound linear functions at current timestamp.
        const upper_limit = evaluateLinearFunctionAtTime(upper_bound, usize, current_timestamp);
        const lower_limit = evaluateLinearFunctionAtTime(lower_bound, usize, current_timestamp);

        if ((upper_limit < (uncompressed_values[current_timestamp] - adjusted_error_bound)) or
            (lower_limit > (uncompressed_values[current_timestamp] + adjusted_error_bound)))
        {
            // Recording mechanism (the current points is outside the limits).
            try appendValue(f64, current_segment.start_point.value, compressed_values);
            const segment_size = current_timestamp - current_segment.start_point.time - 1;
            if (segment_size > 1) {
                // Denominator of Eq. (6).
                const sum_square: f80 = @floatFromInt(
                    segment_size * (segment_size + 1) * (2 * segment_size + 1) / 6,
                );

                // Get optimal slope that minimizes the squared error Eq. (5).
                const slope: f80 = @max(
                    @min(slope_derivate / sum_square, upper_bound.slope),
                    lower_bound.slope,
                );

                const linear_approximation: LinearFunction = .{
                    .slope = slope,
                    .intercept = computeInterceptCoefficient(
                        slope,
                        DiscretePoint,
                        current_segment.start_point,
                    ),
                };
                const end_value = evaluateLinearFunctionAtTime(linear_approximation, usize, current_timestamp - 1);

                try appendValue(f64, end_value, compressed_values);
            } else {
                try appendValue(f64, current_segment.end_point.value, compressed_values);
            }

            try appendValue(usize, current_timestamp, compressed_values);

            // Update the current segment.
            current_segment.start_point.time = current_timestamp;
            current_segment.start_point.value = uncompressed_values[current_timestamp];

            // Catch edge case (only one point left). If `current_timestamp+1 >= uncompressed_values.len` then
            // `uncompressed_values[current_timestamp + 1]` will return index out of bound error.
            if (current_timestamp + 1 < uncompressed_values.len) {
                current_segment.end_point.time = current_timestamp + 1;
                current_segment.end_point.value = uncompressed_values[current_timestamp + 1];

                updateSwingLinearFunction(current_segment, &upper_bound, adjusted_error_bound);
                updateSwingLinearFunction(current_segment, &lower_bound, -adjusted_error_bound);

                current_timestamp += 1;
                slope_derivate = computeSlopeDerivate(current_segment);
            } else {
                // Create linear function with slope zero and intercept equal to the current value.
                current_segment.end_point.time = current_timestamp;
                current_segment.end_point.value = uncompressed_values[current_timestamp];
                upper_bound.slope = 0.0;
                upper_bound.intercept = uncompressed_values[current_timestamp];
                lower_bound.slope = 0.0;
                lower_bound.intercept = uncompressed_values[current_timestamp];
            }
        } else {
            //Filtering mechanism (the current point is still inside the limits).
            current_segment.end_point.time = current_timestamp;
            current_segment.end_point.value = uncompressed_values[current_timestamp];

            // Update the potentially new upper and lower bounds with the new current point.
            updateSwingLinearFunction(current_segment, &new_upper_bound, adjusted_error_bound);
            updateSwingLinearFunction(current_segment, &new_lower_bound, -adjusted_error_bound);

            const new_upper_limit: f80 = evaluateLinearFunctionAtTime(
                new_upper_bound,
                usize,
                current_timestamp,
            );
            const new_lower_limit: f80 = evaluateLinearFunctionAtTime(
                new_lower_bound,
                usize,
                current_timestamp,
            );

            // Update the upper and lower bounds if needed.
            if (upper_limit > new_upper_limit) {
                // Swing down.
                upper_bound = new_upper_bound;
            }
            if (lower_limit < new_lower_limit) {
                //Swing up.
                lower_bound = new_lower_bound;
            }

            // Update slope derivate.
            slope_derivate += computeSlopeDerivate(current_segment);
        }
    }

    const segment_size = current_timestamp - current_segment.start_point.time - 1;

    try appendValue(f64, current_segment.start_point.value, compressed_values);
    if (segment_size > 1) {
        // Denominator of Eq. (6).
        const sum_square: f80 = @floatFromInt(
            segment_size * (segment_size + 1) * (2 * segment_size + 1) / 6,
        );

        // Get optimal slope that minimizes the squared error Eq. (5).
        const slope: f80 = @max(
            @min(slope_derivate / sum_square, upper_bound.slope),
            lower_bound.slope,
        );

        const linear_approximation: LinearFunction = .{
            .slope = slope,
            .intercept = computeInterceptCoefficient(
                slope,
                DiscretePoint,
                current_segment.start_point,
            ),
        };

        const end_value: f64 = evaluateLinearFunctionAtTime(
            linear_approximation,
            usize,
            current_timestamp - 1,
        );
        try appendValue(f64, end_value, compressed_values);
    } else {
        try appendValue(f64, current_segment.end_point.value, compressed_values);
    }

    try appendValue(usize, current_timestamp, compressed_values);
}

/// Compress `uncompressed_values` within `error_bound` using "Slide Filter" and write the
/// result to `compressed_values`. The memory `allocator` is used to initialize the ConvexHull.
/// If an error occurs it is returned.
pub fn compressSlide(
    uncompressed_values: []const f64,
    compressed_values: *ArrayList(u8),
    allocator: mem.Allocator,
    error_bound: f32,
) !void {

    // Adjust the error bound to avoid exceeding it during decompression.
    const adjusted_error_bound = if (error_bound > 0)
        error_bound - tersets.ErrorBoundMargin
    else
        error_bound;

    var convex_hull = try ConvexHull.init(allocator);
    defer convex_hull.deinit();

    var current_upper_bound: LinearFunction = .{ .slope = undefined, .intercept = undefined };
    var current_lower_bound: LinearFunction = .{ .slope = undefined, .intercept = undefined };
    var current_segment: DiscreteSegment = .{
        .start_point = .{ .time = 0, .value = uncompressed_values[0] },
        .end_point = .{ .time = 1, .value = uncompressed_values[1] },
    };
    var new_upper_bound: LinearFunction = .{ .slope = undefined, .intercept = undefined };
    var new_lower_bound: LinearFunction = .{ .slope = undefined, .intercept = undefined };

    try convex_hull.addPoint(current_segment.start_point);
    try convex_hull.addPoint(current_segment.end_point);

    updateSlideLinearFunction(
        DiscreteSegment,
        current_segment,
        &current_upper_bound,
        adjusted_error_bound,
    );
    updateSlideLinearFunction(
        DiscreteSegment,
        current_segment,
        &current_lower_bound,
        -adjusted_error_bound,
    );

    var current_timestamp: usize = 2;
    while (current_timestamp < uncompressed_values.len) : (current_timestamp += 1) {
        const upper_limit = evaluateLinearFunctionAtTime(
            current_upper_bound,
            usize,
            current_timestamp,
        );
        const lower_limit = evaluateLinearFunctionAtTime(
            current_lower_bound,
            usize,
            current_timestamp,
        );
        std.debug.print("----------\n", .{});
        std.debug.print("upper bound slope {:.4} and intercept {:.4}\n", .{ current_upper_bound.slope, current_upper_bound.intercept });
        std.debug.print("lower bound slope {:.4} and intercept {:.4}\n", .{ current_lower_bound.slope, current_lower_bound.intercept });
        std.debug.print("----------\n", .{});
        if ((upper_limit < (uncompressed_values[current_timestamp] - adjusted_error_bound)) or
            (lower_limit > (uncompressed_values[current_timestamp] + adjusted_error_bound)))
        {
            // Recording mechanism. Find the linear approximation that passes the interception
            // point of the upper and lower bounds and slope between the slopes of the upper and
            // lower bounds.
            var intercept_point: ContinousPoint = .{ .time = undefined, .value = undefined }; // z^k
            computeInterceptionPoint(current_lower_bound, current_upper_bound, &intercept_point);

            std.debug.print("Recording \n", .{});

            // In the article, the authors suggest to compute the slope using Eq. (6). However, this is
            // not possible since the starting point is unknown until the new linear approximation is
            // determined. In `Swing Filter`, this is possible because the starting point is always the
            // first point of the segment. In `Slide Filter` the starting point depends on alpha and beta
            // as explained in Lines 19, and 20 of Alg 2. A simple and efficient solution is to assing
            // the mean of the upper and lower bounds' slopes. This solution satisfies Eq. (5).
            const current_linear_approximation = .{
                .slope = (current_lower_bound.slope + current_upper_bound.slope) / 2,
                .intercept = computeInterceptCoefficient(
                    (current_lower_bound.slope + current_upper_bound.slope) / 2,
                    ContinousPoint,
                    intercept_point,
                ),
            };

            std.debug.print("Current linear approximation slope {:.4} and intercept {:.4}\n", .{ current_linear_approximation.slope, current_linear_approximation.intercept });

            const init_value = evaluateLinearFunctionAtTime(
                current_linear_approximation,
                usize,
                current_segment.time,
            );

            try appendValue(f64, init_value, compressed_values);

            const end_value = evaluateLinearFunctionAtTime(
                current_linear_approximation,
                usize,
                current_segment.start_point.time - 1,
            );
            try appendValue(f64, end_value, compressed_values);

            try appendValue(usize, current_segment.end_point.time, compressed_values);

            if (current_timestamp + 1 < uncompressed_values.len) {
                // Update the current segment.
                current_segment = .{
                    .start_point = .{
                        .time = current_timestamp,
                        .value = uncompressed_values[current_timestamp],
                    },
                    .end_point = .{
                        .time = current_timestamp + 1,
                        .value = uncompressed_values[current_timestamp + 1],
                    },
                };

                updateSlideLinearFunction(
                    DiscreteSegment,
                    current_segment,
                    &current_upper_bound,
                    adjusted_error_bound,
                );
                updateSlideLinearFunction(
                    DiscreteSegment,
                    current_segment,
                    &current_lower_bound,
                    -adjusted_error_bound,
                );

                std.debug.print("Current point time {}\n", .{current_segment.end_point.time});
                convex_hull.clean();
                try convex_hull.addPoint(current_segment.start_point);
                try convex_hull.addPoint(current_segment.end_point);
                current_timestamp += 1;
            } else { // Create linear function with slope zero and intercept equal to the current value.
                current_segment.end_point.time = current_timestamp;
                current_segment.end_point.value = uncompressed_values[current_timestamp];
                current_upper_bound.slope = 0.0;
                current_upper_bound.intercept = uncompressed_values[current_timestamp];
                current_lower_bound.slope = 0.0;
                current_lower_bound.intercept = uncompressed_values[current_timestamp];
            }
        } else {
            //Filtering mechanism
            current_segment.end_point = .{
                .time = current_timestamp,
                .value = uncompressed_values[current_timestamp],
            };

            try convex_hull.addPoint(current_segment.end_point);

            std.debug.print("Filtering with point ({}, {})\n", .{ current_segment.end_point.time, current_segment.end_point.value });
            // Compute the potentially new upper and lower bounds
            updateSlideLinearFunction(
                DiscreteSegment,
                current_segment,
                &new_upper_bound,
                adjusted_error_bound,
            );
            updateSlideLinearFunction(
                DiscreteSegment,
                current_segment,
                &new_lower_bound,
                -adjusted_error_bound,
            );

            const new_upper_limit = evaluateLinearFunctionAtTime(
                new_upper_bound,
                usize,
                current_timestamp,
            );
            const new_lower_limit = evaluateLinearFunctionAtTime(
                new_lower_bound,
                usize,
                current_timestamp,
            );

            if (upper_limit > new_upper_limit) {
                // Slide down
                std.debug.print("Slide down\n", .{});
                for (convex_hull.upper_hull.items[0 .. convex_hull.upper_hull.items.len - 1]) |hull_point| {
                    updateSlideLinearFunction(
                        DiscreteSegment,
                        .{ .start_point = hull_point, .end_point = current_segment.end_point },
                        &new_upper_bound,
                        adjusted_error_bound,
                    );

                    if (new_upper_bound.slope < current_upper_bound.slope) {
                        current_upper_bound = new_upper_bound;
                    }
                }
            }
            if (lower_limit < new_lower_limit) {
                //Slide up
                std.debug.print("Slide up\n", .{});
                for (convex_hull.lower_hull.items[0 .. convex_hull.lower_hull.items.len - 1]) |hull_point| {
                    updateSlideLinearFunction(
                        DiscreteSegment,
                        .{ .start_point = hull_point, .end_point = current_segment.end_point },
                        &new_lower_bound,
                        -adjusted_error_bound,
                    );

                    if (new_lower_bound.slope > current_lower_bound.slope) {
                        current_lower_bound = new_lower_bound;
                    }
                }
            }
        }
    }

    // const segment_size = current_timestamp - current_segment.start_point.time - 1;

    // if (segment_size > 1) {} else {
    //     try appendValue(f64, current_segment.end_point.value, compressed_values);
    // }

    // try appendValue(usize, current_timestamp, compressed_values);
}

/// Decompress `compressed_values` produced by "Swing Filter" and "Slide Filter" and write the
/// result to `decompressed_values`. If an error occurs it is returned.
pub fn decompressSwing(
    compressed_values: []const u8,
    decompressed_values: *ArrayList(f64),
) Error!void {
    // The compressed representation is composed of three values: (start_value, end_value, end_time)
    // all of type 64-bit float.
    if (compressed_values.len % 24 != 0) return Error.IncorrectInput;

    const compressed_lines_and_index = mem.bytesAsSlice(f64, compressed_values);

    var current_linear_function: LinearFunction = .{ .slope = undefined, .intercept = undefined };

    var first_timestamp: usize = 0;
    var index: usize = 0;
    while (index < compressed_lines_and_index.len) : (index += 3) {
        const current_segment: DiscreteSegment = .{
            .start_point = .{ .time = first_timestamp, .value = compressed_lines_and_index[index] },
            .end_point = .{
                .time = @as(usize, @bitCast(compressed_lines_and_index[index + 2])) - 1,
                .value = compressed_lines_and_index[index + 1],
            },
        };

        if (current_segment.start_point.time < current_segment.end_point.time) {
            updateSwingLinearFunction(current_segment, &current_linear_function, 0.0);
            try decompressed_values.append(current_segment.start_point.value);
            var current_timestamp: usize = current_segment.start_point.time + 1;
            while (current_timestamp < current_segment.end_point.time) : (current_timestamp += 1) {
                const y: f64 = evaluateLinearFunctionAtTime(
                    current_linear_function,
                    usize,
                    current_timestamp,
                );
                try decompressed_values.append(y);
            }
            try decompressed_values.append(current_segment.end_point.value);
            first_timestamp = current_timestamp + 1;
        } else {
            try decompressed_values.append(current_segment.start_point.value);
            first_timestamp += 1;
        }
    }
}

/// Computes the numerator of the slope derivate as in Eq. (6).
fn computeSlopeDerivate(segment: DiscreteSegment) f80 {
    return (segment.end_point.value - segment.start_point.value) *
        usizeToF80(segment.end_point.time - segment.start_point.time);
}

/// Updates the linear function coeficients in `linear_function` that passes throught the two
/// points of the `segment`. The linear function is swinged down or up based on the `error_bound`.
/// If `error_bound` is negative, `linear_function` is swing down. It is swing up otherwise.
fn updateSwingLinearFunction(
    segment: DiscreteSegment,
    linear_function: *LinearFunction,
    error_bound: f32,
) void {
    if (segment.end_point.time != segment.start_point.time) {
        const duration: f80 = @floatFromInt(segment.end_point.time - segment.start_point.time);
        linear_function.slope = (segment.end_point.value + error_bound -
            segment.start_point.value) / duration;
        linear_function.intercept = segment.start_point.value - linear_function.slope *
            usizeToF80(segment.start_point.time);
    } else {
        linear_function.slope = 0.0;
        linear_function.intercept = segment.start_point.value;
    }
}

/// Evaluate `linear_function` at `time`.
fn evaluateLinearFunctionAtTime(
    linear_function: LinearFunction,
    comptime time_type: type,
    time: time_type,
) f64 {
    if (time_type == usize) {
        return @floatCast(linear_function.slope * usizeToF80(time) + linear_function.intercept);
    } else {
        return @floatCast(linear_function.slope * time + linear_function.intercept);
    }
}

/// Append `value` of `type` determined at compile time to `compressed_values`.
fn appendValue(comptime T: type, value: T, compressed_values: *std.ArrayList(u8)) !void {
    // Compile-time type check
    switch (@TypeOf(value)) {
        f64, usize => {
            const value_as_bytes: [8]u8 = @bitCast(value);
            try compressed_values.appendSlice(value_as_bytes[0..]);
        },
        else => @compileError("Unsupported type for append value function"),
    }
}

/// Computes the intercept coefficient of a linear function that passes through the `DiscretePoint`
/// `point` with the given `slope` coefficient.
fn computeInterceptCoefficient(slope: f80, comptime point_type: type, point: point_type) f80 {
    if (point_type == DiscretePoint) {
        return point.value - slope * usizeToF80(point.time);
    } else {
        return point.value - slope * point.time;
    }
}

/// Updates the linear function coeficients in `linear_function` that passes throught the two
/// points of the `segment` slided up/down at the start point and down/up at the end point based
/// on the `error_bound`. Specifically, the `linear_function` will pass through the points
/// (`segment.start_point.time`, `segment.start_point.value - error_bound`) and
/// (`segment.end_point.time`, `segment.end_point.value + error_bound`).
fn updateSlideLinearFunction(
    comptime segment_type: type,
    segment: segment_type,
    linear_function: *LinearFunction,
    error_bound: f32,
) void {
    if (segment.end_point.time != segment.start_point.time) {
        var duration: f80 = undefined;
        var starting_point: f80 = undefined;
        if (segment_type == DiscreteSegment) {
            duration = @floatFromInt(segment.end_point.time - segment.start_point.time);
            starting_point = usizeToF80(segment.start_point.time);
        } else {
            duration = segment.end_point.time - segment.start_point.time;
            starting_point = segment.start_point.time;
        }

        linear_function.slope = (segment.end_point.value + 2 * error_bound -
            segment.start_point.value) / duration;

        linear_function.intercept = segment.start_point.value - error_bound -
            linear_function.slope * starting_point;
    } else {
        linear_function.slope = 0.0;
        linear_function.intercept = segment.start_point.value;
    }
}

/// Computes the interception point between `linear_function_1` and `linear_function_2` and
/// returns it in `point`. If the lines are parallel, the interception is `undefined`.
fn computeInterceptionPoint(
    linear_function_1: LinearFunction,
    linear_function_2: LinearFunction,
    point: *ContinousPoint,
) void {
    if (linear_function_1.slope != linear_function_2.slope) {
        point.time = @floatCast((linear_function_2.intercept - linear_function_1.intercept) /
            (linear_function_1.slope - linear_function_2.slope));
        point.value = @floatCast(linear_function_1.slope * point.time + linear_function_1.intercept);
    } else {
        // There is no interception, the linear functions are parallel.
        point.time = undefined;
        point.value = undefined;
    }
}

/// Converts `value` of `usize` to `f80`.
fn usizeToF80(value: usize) f80 {
    return @as(f80, @floatFromInt(value));
}

test "swing filter zero error bound and even size compress and decompress" {
    const allocator = testing.allocator;
    const linear_function = LinearFunction{ .slope = 1, .intercept = 0.0 };

    var list_values = ArrayList(f64).init(allocator);
    defer list_values.deinit();
    var compressed_values = ArrayList(u8).init(allocator);
    defer compressed_values.deinit();
    var decompressed_values = ArrayList(f64).init(allocator);
    defer decompressed_values.deinit();
    const error_bound: f32 = 0.0;

    var rnd = std.rand.DefaultPrng.init(0);

    var i: usize = 0;
    while (i < 100) : (i += 1) {
        const noise = rnd.random().float(f64) * 0.1 - 0.05;
        try list_values.append(evaluateLinearFunctionAtTime(linear_function, usize, i) + noise);
    }

    const uncompressed_values = list_values.items;

    try compressSwing(uncompressed_values[0..], &compressed_values, error_bound);
    try decompressSwing(compressed_values.items, &decompressed_values);

    try testing.expect(tersets.isWithinErrorBound(
        uncompressed_values,
        decompressed_values.items,
        error_bound,
    ));
}

test "swing filter zero error bound and odd size compress and decompress" {
    const allocator = testing.allocator;

    const linear_function = LinearFunction{ .slope = 1, .intercept = 0.0 };

    var list_values = ArrayList(f64).init(allocator);
    defer list_values.deinit();
    var compressed_values = ArrayList(u8).init(allocator);
    defer compressed_values.deinit();
    var decompressed_values = ArrayList(f64).init(allocator);
    defer decompressed_values.deinit();
    const error_bound: f32 = 0.0;

    var rnd = std.rand.DefaultPrng.init(0);

    var i: usize = 0;
    while (i < 101) : (i += 1) {
        const noise = rnd.random().float(f64) * 0.1 - 0.05;
        try list_values.append(evaluateLinearFunctionAtTime(linear_function, usize, i) + noise);
    }

    const uncompressed_values = list_values.items;

    try compressSwing(uncompressed_values[0..], &compressed_values, error_bound);
    try decompressSwing(compressed_values.items, &decompressed_values);

    try testing.expect(tersets.isWithinErrorBound(
        uncompressed_values,
        decompressed_values.items,
        error_bound,
    ));
}

test "swing filter four random lines and random error bound compress and decompress" {
    const allocator = testing.allocator;
    var rnd = std.rand.DefaultPrng.init(@as(u64, @bitCast(std.time.milliTimestamp())));

    const linear_functions = [_]LinearFunction{
        LinearFunction{
            .slope = 2 * (rnd.random().float(f64) - 0.5),
            .intercept = 2 * (rnd.random().float(f64) - 0.5),
        },
        LinearFunction{
            .slope = 2 * (rnd.random().float(f64) - 0.5),
            .intercept = 2 * (rnd.random().float(f64) - 0.5),
        },
        LinearFunction{
            .slope = 2 * (rnd.random().float(f64) - 0.5),
            .intercept = 2 * (rnd.random().float(f64) - 0.5),
        },
        LinearFunction{
            .slope = 2 * (rnd.random().float(f64) - 0.5),
            .intercept = 2 * (rnd.random().float(f64) - 0.5),
        },
    };

    var list_values = ArrayList(f64).init(allocator);
    defer list_values.deinit();
    var compressed_values = ArrayList(u8).init(allocator);
    defer compressed_values.deinit();
    var decompressed_values = ArrayList(f64).init(allocator);
    defer decompressed_values.deinit();
    const error_bound: f32 = rnd.random().float(f32) * 0.1;

    var i: usize = 0;
    var lineIndex: usize = 0;
    while (i < 1000) : (i += 1) {
        lineIndex = i / 250;
        const noise = rnd.random().float(f64) * 0.1 - 0.05;
        try list_values.append(evaluateLinearFunctionAtTime(linear_functions[lineIndex], usize, i) + noise);
    }

    const uncompressed_values = list_values.items;

    try compressSwing(uncompressed_values[0..], &compressed_values, error_bound);
    try decompressSwing(compressed_values.items, &decompressed_values);

    try testing.expect(tersets.isWithinErrorBound(
        uncompressed_values,
        decompressed_values.items,
        error_bound,
    ));
}

test "slide filter zero error bound and even size compress and decompress" {
    const allocator = testing.allocator;
    var list_values = ArrayList(f64).init(allocator);
    defer list_values.deinit();
    var compressed_values = ArrayList(u8).init(allocator);
    defer compressed_values.deinit();
    // var decompressed_values = ArrayList(f64).init(allocator);
    // defer decompressed_values.deinit();
    const error_bound: f32 = 0.5;

    try list_values.append(2.3);
    try list_values.append(3.0);
    try list_values.append(3.5);
    try list_values.append(4.0);
    try list_values.append(2.5);
    try list_values.append(1.6);
    try list_values.append(1);
    try list_values.append(0.9);
    try list_values.append(4.0);
    try list_values.append(3.6);
    try list_values.append(3.2);
    try list_values.append(2.7);

    const uncompressed_values = list_values.items;

    std.debug.print("Items\n", .{});
    for (uncompressed_values) |item| {
        std.debug.print("{} ", .{item});
    }
    std.debug.print("\n", .{});

    try compressSlide(uncompressed_values[0..], &compressed_values, allocator, error_bound);
    // try decompressSwing(compressed_values.items, &decompressed_values);

    // try testing.expect(tersets.isWithinErrorBound(
    //     uncompressed_values,
    //     decompressed_values.items,
    //     error_bound,
    // ));
}
