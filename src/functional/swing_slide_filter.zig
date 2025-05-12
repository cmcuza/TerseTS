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
//! The implementation of the Slide Filter differs from the proposed method in two key aspects.
//! First, the slope of the linear approximation does not minimize the squared error due to the
//! unknown initial point until the end of the segment being approximated. Additionally, there
//! is a contradiction in Lemma (4.4) where f^{k}{i} (f^{k'}{i}) should be less than t_{j_{k-1}},
//! but Fig. 5(b) shows f^{k}{i} > t{j_{k-1}}. Given these errors, the following changes have been
//! made to the implementation:
//! 1) The slope for the linear approximation averages the slopes of the upper and lower bounds.
//! This method aligns with Eq. (5) and adheres the linear approximation to the given error bound.
//! 2) All consecutive linear approximations remain disjoint. For each approximation g_k(x), we
//! record the initial and final segment times on g_k, separately. Normally, consecutive linear
//! approximations could share the final and initial segment time of g_{k-1} and g_k, respectively,
//! if meeting three conditions given in Lemma (4.4). However, due to the inconsistencies and
//! unclear proof of Lemma (4.4), it is uncertain if Lemma (4.4) can ensure that the derived bounds
//! will consistently yield a linear approximation within the error bound.

const std = @import("std");
const math = std.math;
const mem = std.mem;
const testing = std.testing;
const ArrayList = std.ArrayList;

const tersets = @import("../tersets.zig");
const Method = tersets.Method;
const Error = tersets.Error;

const tester = @import("../tester.zig");

const shared = @import("../utilities/shared_structs.zig");
const DiscretePoint = shared.DiscretePoint;
const ContinousPoint = shared.ContinousPoint;
const Segment = shared.Segment;
const LinearFunction = shared.LinearFunction;

const ConvexHull = @import("../utilities/convex_hull.zig").ConvexHull;

/// Compress `uncompressed_values` within `error_bound` using "Swing Filter". The function writes
/// the result to `compressed_values`. If an error occurs it is returned.
pub fn compressSwingFilter(
    uncompressed_values: []const f64,
    compressed_values: *ArrayList(u8),
    error_bound: f32,
) Error!void {
    // Adjust the error bound to avoid exceeding it during decompression due to numerical
    // inestabilities. This can happen if the linear approximation is equal to one of the
    // upper or lower bounds.
    const adjusted_error_bound = if (error_bound > 0)
        error_bound - shared.ErrorBoundMargin
    else
        error_bound;

    // Create the upper lower and upper bounds used throughout the function. They are uninitialized
    // as they will be modified throughout the function.
    var upper_bound: LinearFunction = .{ .slope = undefined, .intercept = undefined };
    var lower_bound: LinearFunction = .{ .slope = undefined, .intercept = undefined };
    var new_upper_bound: LinearFunction = .{ .slope = undefined, .intercept = undefined };
    var new_lower_bound: LinearFunction = .{ .slope = undefined, .intercept = undefined };

    // Check if the first two points are NaN or infinite. If so, return an error.
    if (!(math.isFinite(uncompressed_values[0]) and math.isFinite(uncompressed_values[1]))) return Error.UnsupportedInput;

    // Initialize the current segment with first two points.
    var current_segment: Segment = .{
        .start_point = .{ .time = 0, .value = uncompressed_values[0] },
        .end_point = .{ .time = 1, .value = uncompressed_values[1] },
    };

    // Compute the numerator Eq. (6).
    var slope_derivate: f80 = computeSlopeDerivate(current_segment);

    updateSwingLinearFunction(current_segment, &upper_bound, adjusted_error_bound);
    updateSwingLinearFunction(current_segment, &lower_bound, -adjusted_error_bound);

    // Add the first point to the compressed values. From this point on, the algorithm will find all
    // connected segments and add them to the compressed values.
    try appendValue(f64, current_segment.start_point.value, compressed_values);

    // The first two points are already part of `current_segment`, the next point is at index two.
    var current_timestamp: usize = 2;
    while (current_timestamp < uncompressed_values.len) : (current_timestamp += 1) {
        // Evaluate the upper and lower bound linear functions at the current timestamp.
        const upper_limit = evaluateLinearFunctionAtTime(upper_bound, usize, current_timestamp);
        const lower_limit = evaluateLinearFunctionAtTime(lower_bound, usize, current_timestamp);
        var end_value: f64 = 0;

        // Check if the current point is NaN or infinite. If so, return an error.
        if (!math.isFinite(uncompressed_values[current_timestamp])) return Error.UnsupportedInput;

        if ((upper_limit < (uncompressed_values[current_timestamp] - adjusted_error_bound)) or
            (lower_limit > (uncompressed_values[current_timestamp] + adjusted_error_bound)))
        {
            // Recording mechanism (the current point is outside the limits).
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

                end_value = evaluateLinearFunctionAtTime(linear_approximation, usize, current_timestamp - 1);

                try appendValue(f64, end_value, compressed_values);
            } else {
                // Storing uncompressed values instead of those from the linear approximation is crucial
                // for numerical stability, particularly when the error bound is zero. In such cases,
                // decompression must be lossless, and even minimal approximation errors are unacceptable.
                end_value = current_segment.end_point.value;
                try appendValue(f64, current_segment.end_point.value, compressed_values);
            }

            try appendValue(usize, current_timestamp, compressed_values);

            // Update the current segment.
            current_segment.start_point.time = current_timestamp - 1;
            current_segment.start_point.value = end_value;

            // Check if there is only one point left. If so, update only the `end_point`.
            // Otherwise, update the `end_point`, the upper and lower bounds and the `current_timestamp`.
            if (current_timestamp < uncompressed_values.len) {
                current_segment.end_point.time = current_timestamp;
                current_segment.end_point.value = uncompressed_values[current_timestamp];

                updateSwingLinearFunction(current_segment, &upper_bound, adjusted_error_bound);
                updateSwingLinearFunction(current_segment, &lower_bound, -adjusted_error_bound);

                slope_derivate = computeSlopeDerivate(current_segment);
            } else {
                // Only one point left. The `end_point` is at the `current_timestamp`.
                current_segment.end_point.time = current_timestamp;
                current_segment.end_point.value = uncompressed_values[current_timestamp];
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

    // If the last segment is not empty, it means that the recording mechanism was not triggered.
    // Thus, the current semgent has the last line segment which needs to be recorded.
    // Given the way the for loop is structured, the last segment will always have at least one point.
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

/// Compress `uncompressed_values` within `error_bound` using "Slide Filter". The function writes
/// the result to `compressed_values`. The `allocator` is used to allocate memory for the convex hull.
/// If an error occurs it is returned.
pub fn compressSlideFilter(
    uncompressed_values: []const f64,
    compressed_values: *ArrayList(u8),
    allocator: mem.Allocator,
    error_bound: f32,
) Error!void {

    // Adjust the error bound to avoid exceeding it during decompression due to numerical
    // inestabilities. This can happen if the linear approximation is equal to one of the
    // upper or lower bounds.
    const adjusted_error_bound = if (error_bound > 0)
        error_bound - shared.ErrorBoundMargin
    else
        error_bound;

    var convex_hull = try ConvexHull.init(allocator);
    defer convex_hull.deinit();

    // Initialize the interception point between the upper and lower bounds. The point will be
    // defined as part of the method's logic, thus now it is undefined.
    var intercept_point: ContinousPoint = .{ .time = undefined, .value = undefined };

    // Initialize the linear function used across the method. Their values will be defined as part
    // of the method's logic, thus now are undefined.
    var upper_bound: LinearFunction = .{ .slope = undefined, .intercept = undefined };
    var lower_bound: LinearFunction = .{ .slope = undefined, .intercept = undefined };
    var new_upper_bound: LinearFunction = .{ .slope = undefined, .intercept = undefined };
    var new_lower_bound: LinearFunction = .{ .slope = undefined, .intercept = undefined };

    // Check if the first two points are NaN or infinite. If so, return an error.
    if (!(math.isFinite(uncompressed_values[0]) and math.isFinite(uncompressed_values[1]))) return Error.UnsupportedInput;

    // Initialize the current segment with first two points.
    var current_segment: Segment = .{
        .start_point = .{ .time = 0, .value = uncompressed_values[0] },
        .end_point = .{ .time = 1, .value = uncompressed_values[1] },
    };

    try convex_hull.add(current_segment.start_point);
    try convex_hull.add(current_segment.end_point);

    updateSlideLinearFunction(
        current_segment,
        &upper_bound,
        adjusted_error_bound,
    );
    updateSlideLinearFunction(
        current_segment,
        &lower_bound,
        -adjusted_error_bound,
    );

    // The first two points are already part of `current_segment`, the next point is at index two.
    var current_timestamp: usize = 2;
    while (current_timestamp < uncompressed_values.len) : (current_timestamp += 1) {
        // Evaluate the upper and lower bound linear functions at the current timestamp.
        const upper_limit = evaluateLinearFunctionAtTime(
            upper_bound,
            usize,
            current_timestamp,
        );
        const lower_limit = evaluateLinearFunctionAtTime(
            lower_bound,
            usize,
            current_timestamp,
        );

        // Check if the point at the current timestamp is NaN or infinite. If so, return an error.
        if (!math.isFinite(uncompressed_values[current_timestamp])) return Error.UnsupportedInput;

        if ((upper_limit < (uncompressed_values[current_timestamp] - adjusted_error_bound)) or
            (lower_limit > (uncompressed_values[current_timestamp] + adjusted_error_bound)))
        {
            // Recording mechanism. The current points is outside the limits. The linear approximation
            // crosses the interception point of the upper and lower bounds.
            computeInterceptionPoint(lower_bound, upper_bound, &intercept_point);

            const current_linear_approximation = LinearFunction{
                .slope = (lower_bound.slope + upper_bound.slope) / 2,
                .intercept = computeInterceptCoefficient(
                    (lower_bound.slope + upper_bound.slope) / 2,
                    ContinousPoint,
                    intercept_point,
                ),
            };

            const segment_size = current_segment.end_point.time - current_segment.start_point.time;

            if (segment_size > 1) {
                const init_value = evaluateLinearFunctionAtTime(
                    current_linear_approximation,
                    usize,
                    current_segment.start_point.time,
                );

                try appendValue(f64, init_value, compressed_values);

                const end_value = evaluateLinearFunctionAtTime(
                    current_linear_approximation,
                    usize,
                    current_segment.end_point.time,
                );

                try appendValue(f64, end_value, compressed_values);
            } else {
                // Storing uncompressed values instead of those from the linear approximation is crucial
                // for numerical stability, particularly when the error bound is zero. In such cases,
                // decompression must be lossless, and even minimal approximation errors are unacceptable.
                try appendValue(f64, current_segment.start_point.value, compressed_values);
                try appendValue(f64, current_segment.end_point.value, compressed_values);
            }
            try appendValue(usize, current_timestamp, compressed_values);

            // Update the current segment.
            current_segment.start_point.time = current_timestamp;
            current_segment.start_point.value = uncompressed_values[current_timestamp];

            // Edge case as only one point is left.
            if (current_timestamp + 1 < uncompressed_values.len) {

                // Check if the point at the current timestamp is NaN or infinite. If so, return an error.
                if (!math.isFinite(uncompressed_values[current_timestamp + 1])) return Error.UnsupportedInput;

                // Update the current segment.
                current_segment.end_point = .{
                    .time = current_timestamp + 1,
                    .value = uncompressed_values[current_timestamp + 1],
                };

                updateSlideLinearFunction(current_segment, &upper_bound, adjusted_error_bound);
                updateSlideLinearFunction(current_segment, &lower_bound, -adjusted_error_bound);

                convex_hull.clean();
                try convex_hull.add(current_segment.start_point);
                try convex_hull.add(current_segment.end_point);

                current_timestamp += 1;
            } else {
                // Only one point left. The `end_point` is at the `current_timestamp`.
                current_segment.end_point.time = current_timestamp;
                current_segment.end_point.value = uncompressed_values[current_timestamp];
            }
        } else {
            // Filtering mechanism. The current point is still inside the limits.
            current_segment.end_point.time = current_timestamp;
            current_segment.end_point.value = uncompressed_values[current_timestamp];

            try convex_hull.add(current_segment.end_point);

            // The new upper bound can be found on the upper hull. Lemma (4.3).
            for (convex_hull.getUpperHullExceptLast()) |hull_point| {
                updateSlideLinearFunction(
                    .{ .start_point = hull_point, .end_point = current_segment.end_point },
                    &new_upper_bound,
                    adjusted_error_bound,
                );

                if (new_upper_bound.slope < upper_bound.slope) {
                    // Slide down.
                    upper_bound = new_upper_bound;
                }
            }

            // The new lower bound can be found on the lower hull. Lemma (4.3).
            for (convex_hull.getLowerHullExceptLast()) |hull_point| {
                updateSlideLinearFunction(
                    .{ .start_point = hull_point, .end_point = current_segment.end_point },
                    &new_lower_bound,
                    -adjusted_error_bound,
                );

                if (new_lower_bound.slope > lower_bound.slope) {
                    // Slide up.
                    lower_bound = new_lower_bound;
                }
            }
        }
    }

    const segment_size = current_timestamp - current_segment.start_point.time - 1;

    if (segment_size > 1) {
        computeInterceptionPoint(lower_bound, upper_bound, &intercept_point);
        const linear_approximation = LinearFunction{
            .slope = (lower_bound.slope + upper_bound.slope) / 2,
            .intercept = computeInterceptCoefficient(
                (lower_bound.slope + upper_bound.slope) / 2,
                ContinousPoint,
                intercept_point,
            ),
        };

        const init_value = evaluateLinearFunctionAtTime(
            linear_approximation,
            usize,
            current_segment.start_point.time,
        );

        try appendValue(f64, init_value, compressed_values);

        const end_value = evaluateLinearFunctionAtTime(
            linear_approximation,
            usize,
            current_timestamp - 1,
        );
        try appendValue(f64, end_value, compressed_values);
    } else {
        try appendValue(f64, current_segment.start_point.value, compressed_values);
        try appendValue(f64, current_segment.end_point.value, compressed_values);
    }

    try appendValue(usize, current_timestamp, compressed_values);
}

/// Compress `uncompressed_values` within `error_bound` using "Swing Filter"'s filtering mechanism.
/// Different from the proposed papper, this implementation allows a extra degree of freedom by
/// disconnecting adjacent segments in the recording mechanism. The algorithm writes the result to
/// `compressed_values`. If an error occurs it is returned.
pub fn compressSwingFilterDisconnected(
    uncompressed_values: []const f64,
    compressed_values: *ArrayList(u8),
    error_bound: f32,
) Error!void {
    // Adjust the error bound to avoid exceeding it during decompression due to numerical
    // inestabilities. This can happen if the linear approximation is equal to one of the
    // upper or lower bounds.
    const adjusted_error_bound = if (error_bound > 0)
        error_bound - shared.ErrorBoundMargin
    else
        error_bound;

    // Check if the first two points are NaN or infinite. If so, return an error.
    if (!(math.isFinite(uncompressed_values[0]) and math.isFinite(uncompressed_values[1]))) return Error.UnsupportedInput;

    // Create the upper lower and upper bounds used throughout the function. They are uninitialized
    // as they will be modified throughout the function.
    var upper_bound: LinearFunction = .{ .slope = undefined, .intercept = undefined };
    var lower_bound: LinearFunction = .{ .slope = undefined, .intercept = undefined };
    var new_upper_bound: LinearFunction = .{ .slope = undefined, .intercept = undefined };
    var new_lower_bound: LinearFunction = .{ .slope = undefined, .intercept = undefined };

    // Initialize the current segment with the first two points.
    var current_segment: Segment = .{
        .start_point = .{ .time = 0, .value = uncompressed_values[0] },
        .end_point = .{ .time = 1, .value = uncompressed_values[1] },
    };

    // Compute the numerator Eq. (6).
    var slope_derivate: f80 = computeSlopeDerivate(current_segment);

    updateSwingLinearFunction(current_segment, &upper_bound, adjusted_error_bound);
    updateSwingLinearFunction(current_segment, &lower_bound, -adjusted_error_bound);

    // The first two points are already part of `current_segment`, the next point is at index two.
    var current_timestamp: usize = 2;
    while (current_timestamp < uncompressed_values.len) : (current_timestamp += 1) {
        // Calculate the upper and lower bound linear functions at the current timestamp.
        const upper_limit = evaluateLinearFunctionAtTime(upper_bound, usize, current_timestamp);
        const lower_limit = evaluateLinearFunctionAtTime(lower_bound, usize, current_timestamp);

        // Check if the current point is NaN or infinite. If so, return an error.
        if (!math.isFinite(uncompressed_values[current_timestamp])) return Error.UnsupportedInput;

        if ((upper_limit < (uncompressed_values[current_timestamp] - adjusted_error_bound)) or
            (lower_limit > (uncompressed_values[current_timestamp] + adjusted_error_bound)))
        {
            // Recording mechanism (the current point is outside the limits).
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
                // Storing uncompressed values instead of those from the linear approximation is crucial
                // for numerical stability, particularly when the error bound is zero. In such cases,
                // decompression must be lossless, and even minimal approximation errors are unacceptable.
                try appendValue(f64, current_segment.end_point.value, compressed_values);
            }

            try appendValue(usize, current_timestamp, compressed_values);

            // Update the current segment.
            current_segment.start_point.time = current_timestamp;
            current_segment.start_point.value = uncompressed_values[current_timestamp];

            // Check if there is only one point left. If so, update only the `end_point`.
            // Otherwise, update the `end_point`, the upper and lower bounds and the `current_timestamp`.
            if (current_timestamp + 1 < uncompressed_values.len) {

                // Check if the current point is NaN or infinite. If so, return an error.
                if (!math.isFinite(uncompressed_values[current_timestamp + 1])) return Error.UnsupportedInput;

                current_segment.end_point.time = current_timestamp + 1;
                current_segment.end_point.value = uncompressed_values[current_timestamp + 1];

                updateSwingLinearFunction(current_segment, &upper_bound, adjusted_error_bound);
                updateSwingLinearFunction(current_segment, &lower_bound, -adjusted_error_bound);

                current_timestamp += 1;
                slope_derivate = computeSlopeDerivate(current_segment);
            } else {
                // Only one point left. The `end_point` is at the `current_timestamp`.
                current_segment.end_point.time = current_timestamp;
                current_segment.end_point.value = uncompressed_values[current_timestamp];
            }
        } else {
            //Filtering mechanism (the current point is still inside the limits).
            current_segment.end_point.time = current_timestamp;
            current_segment.end_point.value = uncompressed_values[current_timestamp];

            // Update the potentially new upper and lower bounds with the new current point.
            // If the new lines create tighter bounds, update the upper and lower bounds.
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

    // If the last segment is not empty, it means that the recording mechanism was not triggered.
    // Thus, the current semgent has the last line segment which needs to be recorded.
    // Given the way the for loop is structured, the last segment will always have at least one point.
    const segment_size = current_timestamp - current_segment.start_point.time - 1;

    try appendValue(f64, current_segment.start_point.value, compressed_values);
    // Check if the last segment has more than one point. If so, the recording mechanism is triggered.
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
        // Only point left. The `end_point` is at the `current_timestamp`.
        try appendValue(f64, current_segment.end_point.value, compressed_values);
    }
    // The `current_timestamp` indicate the final timestamp.
    try appendValue(usize, current_timestamp, compressed_values);
}

/// Decompress `compressed_values` produced by "Swing Filter". The algorithm writes the result to
/// `decompressed_values`. If an error occurs it is returned.
pub fn decompressSwingFilter(
    compressed_values: []const u8,
    decompressed_values: *ArrayList(f64),
) Error!void {
    // The compressed representation is composed of two values after extracting the first since all
    // segments are connected. Therefore, the condition checks that after the first value, the rest
    // of the values are in pairs (end_value, end_time) and that they are all of type 64-bit float.
    if ((compressed_values.len - 8) % 16 != 0) return Error.UnsupportedInput;

    const compressed_lines_and_index = mem.bytesAsSlice(f64, compressed_values);

    var linear_approximation: LinearFunction = .{ .slope = undefined, .intercept = undefined };

    var index: usize = 0;

    // Extract the start point from the compressed representation.
    var start_point: DiscretePoint = .{ .time = 0, .value = compressed_lines_and_index[0] };
    try decompressed_values.append(start_point.value);

    // Iterate over the compressed representation to reconstruct the time series
    while (index < compressed_lines_and_index.len - 1) : (index += 2) {
        // index + 1 is the end value and index + 2 is the end time.
        const current_segment: Segment = .{
            .start_point = start_point,
            .end_point = .{
                .time = @as(usize, @bitCast(compressed_lines_and_index[index + 2])) - 1,
                .value = compressed_lines_and_index[index + 1],
            },
        };

        // Check if the start and end points of the current segment are different. If they are the same,
        // it means that the segment is a single point, and we can directly append the value.
        if (current_segment.start_point.time != current_segment.end_point.time) {
            // Create the linear approximation for the current segment.
            updateSwingLinearFunction(current_segment, &linear_approximation, 0.0);
            var current_timestamp: usize = current_segment.start_point.time + 1;
            // Interpolate the values between the start and end points of the current segment.
            while (current_timestamp < current_segment.end_point.time) : (current_timestamp += 1) {
                const y: f64 = evaluateLinearFunctionAtTime(
                    linear_approximation,
                    usize,
                    current_timestamp,
                );
                try decompressed_values.append(y);
            }
            try decompressed_values.append(current_segment.end_point.value);
        } else {
            // If the start and end points are the same, append the start point value directly.
            try decompressed_values.append(current_segment.start_point.value);
        }

        // The start point of the next segment is the end point of the current segment.
        start_point = current_segment.end_point;
    }
}

/// Decompress `compressed_values` produced by "Slide Filter". This implementation is reused to
/// decompress the "Swing Filter Disconnected" algorithm since the `compressed_values` are
/// expected to have the same structure. The algorithm writes the result to `decompressed_values`.
/// If an error occurs it is returned.
pub fn decompressSlideFilter(
    compressed_values: []const u8,
    decompressed_values: *ArrayList(f64),
) Error!void {
    // The compressed representation is composed of three values: (start_value, end_value, end_time)
    // all of type 64-bit float.
    if (compressed_values.len % 24 != 0) return Error.UnsupportedInput;

    const compressed_lines_and_index = mem.bytesAsSlice(f64, compressed_values);

    var linear_approximation: LinearFunction = .{ .slope = undefined, .intercept = undefined };

    var first_timestamp: usize = 0;
    var index: usize = 0;

    while (index < compressed_lines_and_index.len) : (index += 3) {
        const current_segment: Segment = .{
            .start_point = .{ .time = first_timestamp, .value = compressed_lines_and_index[index] },
            .end_point = .{
                .time = @as(usize, @bitCast(compressed_lines_and_index[index + 2])) - 1,
                .value = compressed_lines_and_index[index + 1],
            },
        };

        // Check if the start and end points of the current segment are different. If they are the same,
        // it means that the segment is a single point, and we can directly append the value.
        if (current_segment.start_point.time != current_segment.end_point.time) {
            updateSwingLinearFunction(current_segment, &linear_approximation, 0.0);
            try decompressed_values.append(current_segment.start_point.value);
            var current_timestamp: usize = current_segment.start_point.time + 1;

            // Interpolate the values between the start and end points of the current segment.
            while (current_timestamp < current_segment.end_point.time) : (current_timestamp += 1) {
                const y: f64 = evaluateLinearFunctionAtTime(
                    linear_approximation,
                    usize,
                    current_timestamp,
                );
                try decompressed_values.append(y);
            }
            try decompressed_values.append(current_segment.end_point.value);
            first_timestamp = current_timestamp + 1;
        } else {
            // If the start and end points are the same, append the start point value directly.
            try decompressed_values.append(current_segment.start_point.value);
            first_timestamp += 1;
        }
    }
}

/// Computes the numerator of the slope derivate as in Eq. (6).
fn computeSlopeDerivate(segment: Segment) f80 {
    return (segment.end_point.value - segment.start_point.value) *
        usizeToF80(segment.end_point.time - segment.start_point.time);
}

/// Updates the linear function coeficients in `linear_function` that passes throught the two
/// points of the `segment`. The linear function is swinged down or up based on the `error_bound`.
/// If `error_bound` is negative, `linear_function` is swing down. It is swing up otherwise.
fn updateSwingLinearFunction(
    segment: Segment,
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
    segment: Segment,
    linear_function: *LinearFunction,
    error_bound: f32,
) void {
    if (segment.end_point.time != segment.start_point.time) {
        const duration: f80 = @floatFromInt(segment.end_point.time - segment.start_point.time);
        linear_function.slope = (segment.end_point.value + 2 * error_bound -
            segment.start_point.value) / duration;
        linear_function.intercept = segment.start_point.value - error_bound -
            linear_function.slope * usizeToF80(segment.start_point.time);
    } else {
        linear_function.slope = 0.0;
        linear_function.intercept = segment.start_point.value;
    }
}

/// Computes the interception point between `linear_function_one` and `linear_function_two` and
/// returns it in `point`. If the lines are parallel, the interception with the y-axis is returned.
fn computeInterceptionPoint(
    linear_function_one: LinearFunction,
    linear_function_two: LinearFunction,
    point: *ContinousPoint,
) void {
    if (linear_function_one.slope != linear_function_two.slope) {
        point.time = @floatCast((linear_function_two.intercept - linear_function_one.intercept) /
            (linear_function_one.slope - linear_function_two.slope));
        point.value = @floatCast(linear_function_one.slope * point.time + linear_function_one.intercept);
    } else {
        // There is no interception, the linear functions are parallel. Any point is part of the
        // interception. Return the interception with the y-axis as the interception.
        point.time = 0;
        point.value = @floatCast(linear_function_one.intercept);
    }
}

/// Converts `value` of `usize` to `f80`.
fn usizeToF80(value: usize) f80 {
    return @as(f80, @floatFromInt(value));
}

test "swing filter can always compress and decompress" {
    const allocator = testing.allocator;
    try tester.testGenerateCompressAndDecompress(
        tester.generateFiniteRandomValues,
        allocator,
        Method.SwingFilter,
        0,
        tersets.isWithinErrorBound,
    );
}

test "swing filter disconnected can always compress and decompress" {
    const allocator = testing.allocator;
    try tester.testGenerateCompressAndDecompress(
        tester.generateFiniteRandomValues,
        allocator,
        Method.SwingFilterDisconnected,
        0,
        tersets.isWithinErrorBound,
    );
}

test "slide filter disconnected can always compress and decompress" {
    const allocator = testing.allocator;
    try tester.testGenerateCompressAndDecompress(
        tester.generateFiniteRandomValues,
        allocator,
        Method.SlideFilter,
        0,
        tersets.isWithinErrorBound,
    );
}

test "swing and slide filter fail if NaN or infinite values are present" {
    const allocator = testing.allocator;
    var uncompressed_values = ArrayList(f64).init(allocator);
    defer uncompressed_values.deinit();
    var compressed_values = ArrayList(u8).init(allocator);
    defer compressed_values.deinit();

    try tester.generateBoundedRandomValues(&uncompressed_values, 0.0, 1.0, undefined);
    try uncompressed_values.append(math.nan(f64));
    try tester.generateBoundedRandomValues(&uncompressed_values, 0.0, 1.0, undefined);

    compressSwingFilter(uncompressed_values.items, &compressed_values, 0.0) catch |err| {
        try testing.expectEqual(err, Error.UnsupportedInput);
    };
    compressSwingFilterDisconnected(uncompressed_values.items, &compressed_values, 0.0) catch |err| {
        try testing.expectEqual(err, Error.UnsupportedInput);
    };
    compressSlideFilter(uncompressed_values.items, &compressed_values, allocator, 0.0) catch |err| {
        try testing.expectEqual(err, Error.UnsupportedInput);
    };
}

test "swing filter zero error bound and even size compress and decompress" {
    const allocator = testing.allocator;

    var uncompressed_values = ArrayList(f64).init(allocator);
    defer uncompressed_values.deinit();
    var compressed_values = ArrayList(u8).init(allocator);
    defer compressed_values.deinit();
    var decompressed_values = ArrayList(f64).init(allocator);
    defer decompressed_values.deinit();

    const error_bound: f32 = 0.0;

    try tester.generateBoundedRandomValues(&uncompressed_values, 0.0, 1.0, undefined);

    try compressSwingFilter(uncompressed_values.items, &compressed_values, error_bound);
    try decompressSwingFilter(compressed_values.items, &decompressed_values);

    try testing.expect(tersets.isWithinErrorBound(
        uncompressed_values.items,
        decompressed_values.items,
        error_bound,
    ));
}

test "swing filter disconnected zero error bound and even size compress and decompress" {
    const allocator = testing.allocator;

    var uncompressed_values = ArrayList(f64).init(allocator);
    defer uncompressed_values.deinit();
    var compressed_values = ArrayList(u8).init(allocator);
    defer compressed_values.deinit();
    var decompressed_values = ArrayList(f64).init(allocator);
    defer decompressed_values.deinit();

    const error_bound: f32 = 0.0;

    try tester.generateBoundedRandomValues(&uncompressed_values, 0.0, 1.0, undefined);

    try compressSwingFilterDisconnected(uncompressed_values.items, &compressed_values, error_bound);
    // Uses the same decompress function as the "Slide Filter" since the compressed values are
    // expected to have the same structure.
    try decompressSlideFilter(compressed_values.items, &decompressed_values);

    try testing.expect(tersets.isWithinErrorBound(
        uncompressed_values.items,
        decompressed_values.items,
        error_bound,
    ));
}

test "swing filter zero error bound and odd size compress and decompress" {
    const allocator = testing.allocator;

    var uncompressed_values = ArrayList(f64).init(allocator);
    defer uncompressed_values.deinit();
    var compressed_values = ArrayList(u8).init(allocator);
    defer compressed_values.deinit();
    var decompressed_values = ArrayList(f64).init(allocator);
    defer decompressed_values.deinit();

    const error_bound: f32 = 0.0;

    try tester.generateBoundedRandomValues(&uncompressed_values, 0.0, 1.0, undefined);

    //  Extra element to make the size odd.
    try uncompressed_values.append(0.1);

    try compressSwingFilter(uncompressed_values.items, &compressed_values, error_bound);
    try decompressSwingFilter(compressed_values.items, &decompressed_values);

    try testing.expect(tersets.isWithinErrorBound(
        uncompressed_values.items,
        decompressed_values.items,
        error_bound,
    ));
}

test "swing filter random lines and random error bound compress and decompress" {
    const allocator = testing.allocator;

    var uncompressed_values = ArrayList(f64).init(allocator);
    defer uncompressed_values.deinit();
    var compressed_values = ArrayList(u8).init(allocator);
    defer compressed_values.deinit();
    var decompressed_values = ArrayList(f64).init(allocator);
    defer decompressed_values.deinit();

    const error_bound: f32 = tester.generateBoundedRandomValue(f32, 1, 10, undefined);

    const max_lines: usize = @intFromFloat(@round(tester.generateBoundedRandomValue(f64, 4, 25, undefined)));
    for (0..max_lines) |_| {
        // Generate a random linear function and add it to the uncompressed values.
        try tester.generateRandomLinearFunction(&uncompressed_values, undefined);
    }

    try compressSwingFilter(uncompressed_values.items, &compressed_values, error_bound);
    try decompressSwingFilter(compressed_values.items, &decompressed_values);

    try testing.expect(tersets.isWithinErrorBound(
        uncompressed_values.items,
        decompressed_values.items,
        error_bound,
    ));
}

test "swing filter disconnected random lines and random error bound compress and decompress" {
    const allocator = testing.allocator;

    var uncompressed_values = ArrayList(f64).init(allocator);
    defer uncompressed_values.deinit();
    var compressed_values = ArrayList(u8).init(allocator);
    defer compressed_values.deinit();
    var decompressed_values = ArrayList(f64).init(allocator);
    defer decompressed_values.deinit();

    const error_bound: f32 = tester.generateBoundedRandomValue(f32, 1, 10, undefined);

    const max_lines: usize = @intFromFloat(@round(tester.generateBoundedRandomValue(f64, 4, 25, undefined)));

    for (0..max_lines) |_| {
        // Generate a random linear function and add it to the uncompressed values.
        try tester.generateRandomLinearFunction(&uncompressed_values, undefined);
    }

    try compressSwingFilterDisconnected(uncompressed_values.items, &compressed_values, error_bound);
    try decompressSlideFilter(compressed_values.items, &decompressed_values);

    try testing.expect(tersets.isWithinErrorBound(
        uncompressed_values.items,
        decompressed_values.items,
        error_bound,
    ));
}

test "slide filter zero error bound and even size compress and decompress" {
    const allocator = testing.allocator;

    var uncompressed_values = ArrayList(f64).init(allocator);
    defer uncompressed_values.deinit();
    var compressed_values = ArrayList(u8).init(allocator);
    defer compressed_values.deinit();
    var decompressed_values = ArrayList(f64).init(allocator);
    defer decompressed_values.deinit();

    // Zero error bound is curently failing due to numerical instabilities at very high precision levels.
    // The error occurs
    const error_bound: f32 = 0.0;

    try tester.generateBoundedRandomValues(&uncompressed_values, 0.0, 1.0, undefined);

    try compressSlideFilter(
        uncompressed_values.items,
        &compressed_values,
        allocator,
        error_bound,
    );
    try decompressSlideFilter(compressed_values.items, &decompressed_values);

    try testing.expect(tersets.isWithinErrorBound(
        uncompressed_values.items,
        decompressed_values.items,
        error_bound,
    ));
}

test "slide filter zero error bound and odd size compress and decompress" {
    const allocator = testing.allocator;

    var uncompressed_values = ArrayList(f64).init(allocator);
    defer uncompressed_values.deinit();
    var compressed_values = ArrayList(u8).init(allocator);
    defer compressed_values.deinit();
    var decompressed_values = ArrayList(f64).init(allocator);
    defer decompressed_values.deinit();

    const error_bound: f32 = 0.0;

    try tester.generateBoundedRandomValues(&uncompressed_values, 0.0, 1.0, undefined);
    //  Extra element to make the size odd.
    try uncompressed_values.append(0.1);

    try compressSlideFilter(
        uncompressed_values.items,
        &compressed_values,
        allocator,
        error_bound,
    );
    try decompressSlideFilter(compressed_values.items, &decompressed_values);

    try testing.expect(tersets.isWithinErrorBound(
        uncompressed_values.items,
        decompressed_values.items,
        error_bound,
    ));
}

test "slide filter random lines and random error bound compress and decompress" {
    const allocator = testing.allocator;

    var uncompressed_values = ArrayList(f64).init(allocator);
    defer uncompressed_values.deinit();
    var compressed_values = ArrayList(u8).init(allocator);
    defer compressed_values.deinit();
    var decompressed_values = ArrayList(f64).init(allocator);
    defer decompressed_values.deinit();

    const error_bound: f32 = tester.generateBoundedRandomValue(f32, 1, 10, undefined);

    const max_lines: usize = @intFromFloat(@round(tester.generateBoundedRandomValue(f64, 4, 25, undefined)));

    for (0..max_lines) |_| {
        // Generate a random linear function and add it to the uncompressed values.
        try tester.generateRandomLinearFunction(&uncompressed_values, undefined);
    }

    try compressSlideFilter(
        uncompressed_values.items,
        &compressed_values,
        allocator,
        error_bound,
    );
    try decompressSlideFilter(compressed_values.items, &decompressed_values);

    try testing.expect(tersets.isWithinErrorBound(
        uncompressed_values.items,
        decompressed_values.items,
        error_bound,
    ));
}
