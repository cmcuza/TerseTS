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

//! Implementation of The Swing Filter and Slide Filter algorithms from the paper
//! "Hazem Elmeleegy, Ahmed K. Elmagarmid, Emmanuel Cecchet, Walid G. Aref, and Willy Zwaenepoel.
//! Online piece-wise linear approximation of numerical streams with precision guarantees.
//! Proc. VLDB Endow. 2, 1, 2009.
//! https://doi.org/10.14778/1687627.1687645".

const std = @import("std");
const ts = @import("../tersets.zig");

const mem = std.mem;
const testing = std.testing;

const ArrayList = std.ArrayList;
const Error = ts.Error;

/// Segment represented by the points: (start_time, start_value) and (end_time, end_value).
const Segment = struct {
    start_time: usize,
    start_value: f64,
    end_time: usize,
    end_value: f64,
};

/// Line structure uses f80 to avoid overflows with f64.
const LinearFunction = struct {
    slope: f80,
    intercept: f80,
};

fn usizeToF80(value: usize) f80 {
    return @as(f80, @floatFromInt(value));
}

/// Computes the intercept coefficient of a line that passes through the point
/// (x_time, y_value) and with the given slope coefficient.
fn computeInterceptCoefficient(slope: f80, x_time: usize, y_value: f64) f80 {
    return y_value - slope * usizeToF80(x_time);
}

/// Computes the line's coeficients that passes throught the two points
/// of the segment and "swings" down or up depending of the error bound.
fn updateSwingLine(segment: Segment, line: *LinearFunction, error_bound: f32) void {
    if (segment.end_time != segment.start_time) {
        const duration: f80 = @floatFromInt(segment.end_time - segment.start_time);
        line.slope = (segment.end_value + error_bound - segment.start_value) / duration;
        line.intercept = segment.start_value - line.slope * usizeToF80(segment.start_time);
    } else {
        line.slope = 0.0;
        line.intercept = segment.start_value;
    }
}

fn evaluateLineAtTime(line: LinearFunction, x_time: usize) f64 {
    return @floatCast(line.slope * usizeToF80(x_time) + line.intercept);
}

fn appendValue(value: f64, compressed_values: *ArrayList(u8)) !void {
    const valueAsBytes: [8]u8 = @bitCast(value);
    try compressed_values.appendSlice(valueAsBytes[0..]);
}

fn appendIndex(index: usize, compressed_values: *ArrayList(u8)) !void {
    const indexAsFloat: [8]u8 = @bitCast(@as(f64, @floatFromInt(index)));
    try compressed_values.appendSlice(indexAsFloat[0..]);
}

/// Computes the numerator of the slope derivate as in Eq. 6.
fn computeSlopeDerivate(segment: Segment) f80 {
    return (segment.end_value - segment.start_value) * usizeToF80(segment.end_time - segment.start_time);
}

/// Compress `uncompressed_values` within `error_bound` using "Swing-Filter"
/// and write the result to `compressed_values`. If an error occurs it is returned.
pub fn compress_swing(
    uncompressed_values: []const f64,
    compressed_values: *ArrayList(u8),
    error_bound: f32,
) Error!void {
    // Initiallization.
    var upper_limit: f80 = 0;
    var lower_limit: f80 = 0;

    const adjusted_error_bound = if (error_bound > 0) error_bound - ts.ErrorBoundMargin else error_bound;

    var upper_bound_line: LinearFunction = LinearFunction{ .slope = 0, .intercept = 0 };
    var lower_bound_line: LinearFunction = LinearFunction{ .slope = 0, .intercept = 0 };
    var new_upper_bound_line: LinearFunction = LinearFunction{ .slope = 0, .intercept = 0 };
    var new_lower_bound_line: LinearFunction = LinearFunction{ .slope = 0, .intercept = 0 };
    var current_line: LinearFunction = LinearFunction{ .slope = 0, .intercept = 0 };

    // `first_timestamp` is the beginging of the segment.
    var first_timestamp: usize = 0;

    // Initialize the current segment with first two points.
    var current_segment: Segment = Segment{
        .start_time = first_timestamp,
        .start_value = uncompressed_values[first_timestamp],
        .end_time = first_timestamp + 1,
        .end_value = uncompressed_values[first_timestamp + 1],
    };
    var slope_derivate: f80 = computeSlopeDerivate(current_segment);

    // Get the upper and lower bounds.
    updateSwingLine(current_segment, &upper_bound_line, adjusted_error_bound);
    updateSwingLine(current_segment, &lower_bound_line, -adjusted_error_bound);

    var current_timestamp: usize = 2;
    while (current_timestamp < uncompressed_values.len) : (current_timestamp += 1) {
        upper_limit = evaluateLineAtTime(upper_bound_line, current_timestamp);
        lower_limit = evaluateLineAtTime(lower_bound_line, current_timestamp);

        if ((upper_limit < (uncompressed_values[current_timestamp] - adjusted_error_bound)) or
            (lower_limit > (uncompressed_values[current_timestamp] + adjusted_error_bound)))
        { // Recording mechanism.

            try appendValue(current_segment.start_value, compressed_values);
            const segment_size = current_timestamp - first_timestamp - 1;
            if (segment_size > 1) {
                const sum_square: f80 = @floatFromInt(segment_size * (segment_size + 1) * (2 * segment_size + 1) / 6);

                // Get optimal slope that minimizes the squared error.
                const slope: f80 = @max(
                    @min(slope_derivate / sum_square, upper_bound_line.slope),
                    lower_bound_line.slope,
                );

                current_line.slope = slope;
                current_line.intercept = computeInterceptCoefficient(
                    slope,
                    first_timestamp,
                    uncompressed_values[first_timestamp],
                );
                const end_value: f64 = evaluateLineAtTime(current_line, current_timestamp - 1);
                try appendValue(end_value, compressed_values);
            } else {
                try appendValue(current_segment.end_value, compressed_values);
            }

            try appendIndex(current_timestamp, compressed_values);

            // Update the current segment.
            first_timestamp = current_timestamp;
            current_segment.start_time = current_timestamp;
            current_segment.start_value = uncompressed_values[current_timestamp];
            // Catch edge case (only one point left).
            // If `current_timestamp+1 >= uncompressed_values.len`
            // then `uncompressed_values[current_timestamp + 1]` will return index out of bound error
            if (current_timestamp + 1 < uncompressed_values.len) {
                current_segment.end_time = current_timestamp + 1;
                current_segment.end_value = uncompressed_values[current_timestamp + 1];

                updateSwingLine(current_segment, &upper_bound_line, adjusted_error_bound);
                updateSwingLine(current_segment, &lower_bound_line, -adjusted_error_bound);

                current_timestamp += 1;
                slope_derivate = computeSlopeDerivate(current_segment);
            } else { // Create line with slope 0 and intercept current value
                current_segment.end_time = current_timestamp;
                current_segment.end_value = uncompressed_values[current_timestamp];
                upper_bound_line.slope = 0.0;
                upper_bound_line.intercept = uncompressed_values[current_timestamp];
                lower_bound_line.slope = 0.0;
                lower_bound_line.intercept = uncompressed_values[current_timestamp];
            }
        } else { //filtering mechanism

            current_segment.end_time = current_timestamp;
            current_segment.end_value = uncompressed_values[current_timestamp];

            // Update the potentially new upper and lower bounds with the new current point.
            updateSwingLine(current_segment, &new_upper_bound_line, adjusted_error_bound);
            updateSwingLine(current_segment, &new_lower_bound_line, -adjusted_error_bound);

            const new_upper_limit: f80 = evaluateLineAtTime(new_upper_bound_line, current_timestamp);
            const new_lower_limit: f80 = evaluateLineAtTime(new_lower_bound_line, current_timestamp);

            // Update the upper and lower bounds if needed
            if (upper_limit > new_upper_limit) { // Swing down
                upper_bound_line.slope = new_upper_bound_line.slope;
                upper_bound_line.intercept = new_upper_bound_line.intercept;
                upper_limit = new_upper_limit;
            }
            if (lower_limit < new_lower_limit) { //Swing up
                lower_bound_line.slope = new_lower_bound_line.slope;
                lower_bound_line.intercept = new_lower_bound_line.intercept;
                lower_limit = new_lower_limit;
            }

            // Update slope derivate
            slope_derivate += computeSlopeDerivate(current_segment);
        }
    }

    const segment_size = current_timestamp - first_timestamp - 1;

    try appendValue(current_segment.start_value, compressed_values);
    if (segment_size > 1) {
        // Denominator of Eq. 6.
        const sum_square: f80 = @floatFromInt(segment_size * (segment_size + 1) * (2 * segment_size + 1) / 6);

        const slope: f80 = @max(
            @min(slope_derivate / sum_square, upper_bound_line.slope),
            lower_bound_line.slope,
        );

        current_line.slope = slope;
        current_line.intercept = computeInterceptCoefficient(
            slope,
            first_timestamp,
            uncompressed_values[first_timestamp],
        );

        const end_value: f64 = evaluateLineAtTime(current_line, current_timestamp - 1);
        try appendValue(end_value, compressed_values);
    } else {
        try appendValue(current_segment.end_value, compressed_values);
    }

    try appendIndex(current_timestamp, compressed_values);
}

/// Decompress `compressed_values` produced by "Swing-Filter" and
/// "Slide-Filter" and write the result to `decompressed_values`. If an error
/// occurs it is returned.
pub fn decompress(compressed_values: []const u8, decompressed_values: *ArrayList(f64)) Error!void {
    // The compressed representation is composed of three values:
    // (start_value, end_value, end_time) all of type 64-bit float.
    if (compressed_values.len % 24 != 0) return Error.IncorrectInput;

    const compressed_lines_and_index = mem.bytesAsSlice(
        f64,
        compressed_values,
    );

    var current_line: LinearFunction = LinearFunction{ .slope = 0, .intercept = 0 };
    var current_segment: Segment = Segment{
        .start_time = 0,
        .start_value = 0,
        .end_time = 0,
        .end_value = 0,
    };

    var first_timestamp: usize = 0;
    var index: usize = 0;
    while (index < compressed_lines_and_index.len) : (index += 3) {
        current_segment.start_time = first_timestamp;
        current_segment.start_value = compressed_lines_and_index[index];
        current_segment.end_value = compressed_lines_and_index[index + 1];
        current_segment.end_time = @intFromFloat(compressed_lines_and_index[index + 2] - 1);

        if (current_segment.start_time < current_segment.end_time) {
            updateSwingLine(current_segment, &current_line, 0.0);
            try decompressed_values.append(current_segment.start_value);
            var current_timestamp: usize = current_segment.start_time + 1;
            while (current_timestamp < current_segment.end_time) : (current_timestamp += 1) {
                const y: f64 = evaluateLineAtTime(current_line, current_timestamp);
                try decompressed_values.append(y);
            }
            try decompressed_values.append(current_segment.end_value);
            first_timestamp = current_timestamp + 1;
        } else {
            try decompressed_values.append(current_segment.start_value);
            first_timestamp += 1;
        }
    }
}

test "swing filter 0 error bound, even size compress and decompress" {
    const allocator = testing.allocator;
    const line = LinearFunction{ .slope = 1, .intercept = 0.0 };

    var list_values = std.ArrayList(f64).init(allocator);
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
        try list_values.append(evaluateLineAtTime(line, i) + noise);
    }

    const uncompressed_values = list_values.items;

    try compress_swing(uncompressed_values[0..], &compressed_values, error_bound);
    try decompress(compressed_values.items, &decompressed_values);

    try testing.expect(ts.isWithinErrorBound(
        uncompressed_values,
        decompressed_values.items,
        error_bound,
    ));
}

test "swing filter 0 error bound, odd size compress and decompress" {
    const allocator = testing.allocator;

    const line = LinearFunction{ .slope = 1, .intercept = 0.0 };

    var list_values = std.ArrayList(f64).init(allocator);
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
        try list_values.append(evaluateLineAtTime(line, i) + noise);
    }

    const uncompressed_values = list_values.items;

    try compress_swing(uncompressed_values[0..], &compressed_values, error_bound);
    try decompress(compressed_values.items, &decompressed_values);

    try testing.expect(ts.isWithinErrorBound(
        uncompressed_values,
        decompressed_values.items,
        error_bound,
    ));
}

test "swing filter 4 random lines, random error bound compress and decompress" {
    const allocator = testing.allocator;
    var rnd = std.rand.DefaultPrng.init(@as(u64, @bitCast(std.time.milliTimestamp())));

    const lines = [_]LinearFunction{
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

    var list_values = std.ArrayList(f64).init(allocator);
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
        try list_values.append(evaluateLineAtTime(lines[lineIndex], i) + noise);
    }

    const uncompressed_values = list_values.items;

    try compress_swing(uncompressed_values[0..], &compressed_values, error_bound);
    try decompress(compressed_values.items, &decompressed_values);

    try testing.expect(ts.isWithinErrorBound(
        uncompressed_values,
        decompressed_values.items,
        error_bound,
    ));
}
