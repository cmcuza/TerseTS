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

//! Implementation of Swing and Slide Filter algorithms from the paper:
//! Hazem Elmeleegy, Ahmed K. Elmagarmid, Emmanuel Cecchet, Walid G. Aref, and Willy Zwaenepoel.
//! Online piece-wise linear approximation of numerical streams with precision guarantees.
//! Proc. VLDB Endow. 2, 1, 2009.
//! https://doi.org/10.14778/1687627.1687645.

const std = @import("std");
const ts = @import("../tersets.zig");

const mem = std.mem;
const testing = std.testing;

const ArrayList = std.ArrayList;

const Segment = struct {
    start_time: usize,
    start_value: f64,
    end_time: usize,
    end_value: f64,
};

/// Line structure uses f80 to avoid overflows with f64
const Line = struct {
    slope: f80,
    intercept: f80,
};

pub fn isWithinErrorBound(
    uncompressed_values: []f64,
    decompressed_values: []f64,
    error_bound: f32,
) bool {
    for (decompressed_values, 0..) |item, i| {
        if (@abs(uncompressed_values[i] - item) > error_bound + 1e-7) return false;
    }
    return true;
}

fn usizeToF64(value: usize) f64 {
    return @as(f64, @floatFromInt(value));
}

fn f64ToUsize(value: f64) usize {
    return @as(usize, @intFromFloat(value));
}

fn usizeToF80(value: usize) f80 {
    return @as(f80, @floatFromInt(value));
}

fn f80Tof64(value: f80) f64 {
    return @as(f64, @floatCast(value));
}

fn f64Tof80(value: f64) f80 {
    return @as(f80, @floatCast(value));
}

fn getIntercept(slope: f80, x: usize, y: f64, line: *Line) !void {
    line.slope = slope;
    line.intercept = y - slope * usizeToF64(x);
}

fn getSwingLine(segment: *Segment, line: *Line, error_bound: f32) !void {
    if (segment.end_time != segment.start_time) {
        const duration = usizeToF80(segment.end_time - segment.start_time);
        line.slope = (segment.end_value + error_bound - segment.start_value) / duration;
        line.intercept = segment.start_value - line.slope * usizeToF80(segment.start_time);
    } else {
        line.slope = 0.0;
        line.intercept = segment.start_value;
    }
}

fn evaluate(line: *Line, timestamp: usize) f80 {
    return line.slope * usizeToF80(timestamp) + line.intercept;
}

fn appendLine(
    line: *Line,
    compressed_values: *ArrayList(u8),
) !void {
    const slopeAsBytes: [8]u8 = @bitCast(f80Tof64(line.slope));
    try compressed_values.appendSlice(slopeAsBytes[0..]);
    const interceptAsBytes: [8]u8 = @bitCast(f80Tof64(line.intercept));
    try compressed_values.appendSlice(interceptAsBytes[0..]);
}

fn appendValue(
    value: f64,
    compressed_values: *ArrayList(u8),
) !void {
    const valueAsBytes: [8]u8 = @bitCast(value);
    try compressed_values.appendSlice(valueAsBytes[0..]);
}

fn appendSegment(
    segment: *Segment,
    compressed_values: *ArrayList(u8),
) !void {
    // const stAsBytes: [8]u8 = @bitCast(usizeToF64(segment.start_time));
    // try compressed_values.appendSlice(stAsBytes[0..]);
    const svAsBytes: [8]u8 = @bitCast(segment.start_value);
    try compressed_values.appendSlice(svAsBytes[0..]);
    const evAsBytes: [8]u8 = @bitCast(segment.end_value);
    try compressed_values.appendSlice(evAsBytes[0..]);
    const etAsBytes: [8]u8 = @bitCast(usizeToF64(segment.end_time + 1));
    try compressed_values.appendSlice(etAsBytes[0..]);
}

fn appendIndex(index: usize, compressed_values: *ArrayList(u8)) !void {
    const indexAsFloat: [8]u8 = @bitCast(usizeToF64(index));
    try compressed_values.appendSlice(indexAsFloat[0..]);
}

fn getDerivate(segment: *Segment) f80 {
    return (segment.end_value - segment.start_value) * usizeToF80(segment.end_time - segment.start_time);
}

/// @param uncompressed_values: the input time series and length
/// @param compressed_values: the compressed representation
/// @param error_bound: the error bound information
pub fn compress_swing(
    uncompressed_values: []const f64,
    compressed_values: *ArrayList(u8),
    error_bound: f32,
) !void {
    // Initiallization
    var upper_y: f80 = 0;
    var lower_y: f80 = 0;

    var upper_bound_line: Line = Line{ .slope = 0, .intercept = 0 };
    var lower_bound_line: Line = Line{ .slope = 0, .intercept = 0 };
    var new_upper_bound_line: Line = Line{ .slope = 0, .intercept = 0 };
    var new_lower_bound_line: Line = Line{ .slope = 0, .intercept = 0 };
    var current_line: Line = Line{ .slope = 0, .intercept = 0 };

    var first_timestamp: usize = 0;
    var current_timestamp: usize = 1;

    var current_segment: Segment = Segment{
        .start_time = first_timestamp,
        .start_value = uncompressed_values[first_timestamp],
        .end_time = current_timestamp,
        .end_value = uncompressed_values[current_timestamp],
    };

    var slope_derivate: f80 = getDerivate(&current_segment); // Numerator of slope derivate to optimize Eq. (6)

    // Get the upper and lower bounds
    try getSwingLine(&current_segment, &upper_bound_line, error_bound);
    try getSwingLine(&current_segment, &lower_bound_line, -error_bound);

    current_timestamp += 1;
    while (current_timestamp < uncompressed_values.len) : (current_timestamp += 1) {
        upper_y = evaluate(&upper_bound_line, current_timestamp);
        lower_y = evaluate(&lower_bound_line, current_timestamp);

        // Is the new point more than \epsilon above upper_y or bellow lower_y?
        if ((upper_y < (uncompressed_values[current_timestamp] - error_bound)) or
            (lower_y > (uncompressed_values[current_timestamp] + error_bound)))
        { // Recording mechanism
            const n = current_timestamp - first_timestamp - 1;
            const den = n * (n + 1) * (2 * n + 1) / 6; // Denominator of Eq. 6
            const slope: f80 = @max(
                @min(
                    slope_derivate / usizeToF80(den),
                    upper_bound_line.slope,
                ),
                lower_bound_line.slope,
            ); // Get optimal slope that minimizes the squared error

            try getIntercept(slope, first_timestamp, uncompressed_values[first_timestamp], &current_line);

            const end_value: f64 = f80Tof64(evaluate(&current_line, current_timestamp - 1));

            // Record the current segment
            try appendValue(current_segment.start_value, compressed_values);
            try appendValue(end_value, compressed_values);
            try appendIndex(current_timestamp, compressed_values);

            // Update the current segment
            first_timestamp = current_timestamp;
            current_segment.start_time = current_timestamp;
            current_segment.start_value = uncompressed_values[current_timestamp];
            if (current_timestamp + 1 < uncompressed_values.len) { // Catch edge case (only one point left)
                // Update the current segment
                current_segment.end_time = current_timestamp + 1;
                current_segment.end_value = uncompressed_values[current_timestamp + 1];

                // Recompute the upper and lower bounds
                try getSwingLine(&current_segment, &upper_bound_line, error_bound);
                try getSwingLine(&current_segment, &lower_bound_line, -error_bound);

                current_timestamp += 1; // advance the current_timestamp
                slope_derivate = getDerivate(&current_segment);
            } else { // Only one element left, create line with slope 0 to create const line
                current_segment.end_time = current_timestamp;
                current_segment.end_value = uncompressed_values[current_timestamp];
                upper_bound_line.slope = 0.0;
                upper_bound_line.intercept = uncompressed_values[current_timestamp];
                lower_bound_line.slope = 0.0;
                lower_bound_line.intercept = uncompressed_values[current_timestamp];
            }
        } else { //filtering mechanism
            // Update the current segment
            current_segment.end_time = current_timestamp;
            current_segment.end_value = uncompressed_values[current_timestamp];

            // Compute the potentially new upper and lower bounds
            try getSwingLine(&current_segment, &new_upper_bound_line, error_bound);
            try getSwingLine(&current_segment, &new_lower_bound_line, -error_bound);

            const new_upper_y: f80 = evaluate(&new_upper_bound_line, current_timestamp);
            const new_lower_y: f80 = evaluate(&new_lower_bound_line, current_timestamp);

            // Outdate the upper and lower bounds if needed
            if (upper_y > new_upper_y) { // Swing down
                upper_bound_line.slope = new_upper_bound_line.slope;
                upper_bound_line.intercept = new_upper_bound_line.intercept;
                upper_y = new_upper_y;
            }
            if (lower_y < new_lower_y) { //Swing up
                lower_bound_line.slope = new_lower_bound_line.slope;
                lower_bound_line.intercept = new_lower_bound_line.intercept;
                lower_y = new_lower_y;
            }

            // Update slope derivate
            slope_derivate += getDerivate(&current_segment);
        }
    }

    const n = current_timestamp - first_timestamp - 1;
    const den = n * (n + 1) * (2 * n + 1) / 6; // Denominator of Eq. 6
    const slope: f80 = @max(
        @min(slope_derivate / usizeToF80(den), upper_bound_line.slope),
        lower_bound_line.slope,
    ); // Get optimal slope that minimizes the squared error

    try getIntercept(
        slope,
        first_timestamp,
        uncompressed_values[first_timestamp],
        &current_line,
    );

    const end_value: f64 = f80Tof64(evaluate(&current_line, current_timestamp - 1));

    // Record the current segment
    try appendValue(current_segment.start_value, compressed_values);
    try appendValue(end_value, compressed_values);
    try appendIndex(current_timestamp, compressed_values);
}

pub fn decompress(
    compressed_values: []const u8,
    decompressed_values: *ArrayList(f64),
) error{OutOfMemory}!void {
    // TODO: Check length of uncompressed_values is modulo 32.

    const compressed_lines_and_index = mem.bytesAsSlice(
        f64,
        compressed_values,
    );

    var current_line: Line = Line{ .slope = 0, .intercept = 0 };
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
        current_segment.end_time = f64ToUsize(compressed_lines_and_index[index + 2] - 1);

        try getSwingLine(&current_segment, &current_line, 0.0);
        var current_timestamp: usize = current_segment.start_time;
        while (current_timestamp < current_segment.end_time + 1) : (current_timestamp += 1) {
            const y: f64 = f80Tof64(evaluate(&current_line, current_timestamp));
            try decompressed_values.append(y);
        }
        first_timestamp = current_timestamp;
    }
}

test "swing-filter single line compress and decompress" {
    const alloc = testing.allocator;
    const uncompressed_values = [_]f64{ 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5 };
    var compressed_values = ArrayList(u8).init(alloc);
    defer compressed_values.deinit();
    var decompressed_values = ArrayList(f64).init(alloc);
    defer decompressed_values.deinit();

    const error_bound: f32 = 0.0;

    try compress_swing(uncompressed_values[0..], &compressed_values, error_bound);

    try decompress(compressed_values.items, &decompressed_values);
    try testing.expect(mem.eql(f64, uncompressed_values[0..], decompressed_values.items));
}

test "swing-filter single line and single point (odd size) compress and decompress" {
    const alloc = testing.allocator;
    const uncompressed_values = [_]f64{ 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 1.5 };
    var compressed_values = ArrayList(u8).init(alloc);
    defer compressed_values.deinit();
    var decompressed_values = ArrayList(f64).init(alloc);
    defer decompressed_values.deinit();

    const error_bound: f32 = 0.0;

    try compress_swing(uncompressed_values[0..], &compressed_values, error_bound);

    try decompress(compressed_values.items, &decompressed_values);
    try testing.expect(mem.eql(f64, uncompressed_values[0..], decompressed_values.items));
}

test "swing-filter single line and single point (even size) compress and decompress" {
    const alloc = testing.allocator;
    const uncompressed_values = [_]f64{ 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 1.5 };
    var compressed_values = ArrayList(u8).init(alloc);
    defer compressed_values.deinit();
    var decompressed_values = ArrayList(f64).init(alloc);
    defer decompressed_values.deinit();

    const error_bound: f32 = 0.0;

    try compress_swing(uncompressed_values[0..], &compressed_values, error_bound);

    try decompress(compressed_values.items, &decompressed_values);
    try testing.expect(mem.eql(f64, uncompressed_values[0..], decompressed_values.items));
}

test "swing-filter two parallel lines compress and decompress" {
    const alloc = testing.allocator;
    const uncompressed_values = [_]f64{ 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5 };

    var compressed_values = ArrayList(u8).init(alloc);
    defer compressed_values.deinit();
    var decompressed_values = ArrayList(f64).init(alloc);
    defer decompressed_values.deinit();
    const error_bound: f32 = 0.0;

    try compress_swing(uncompressed_values[0..], &compressed_values, error_bound);
    try decompress(compressed_values.items, &decompressed_values);

    try testing.expect(mem.eql(f64, uncompressed_values[0..], decompressed_values.items));
}

test "swing-filter 4 v-shaped lines compress and decompress" {
    const alloc = testing.allocator;

    var lines = [_]Line{
        Line{ .slope = 1, .intercept = 0.0 },
        Line{ .slope = -1, .intercept = 50 },
        Line{ .slope = 1, .intercept = -50 },
        Line{ .slope = -1, .intercept = 100 },
    };

    var list_values = std.ArrayList(f64).init(alloc);
    defer list_values.deinit();
    var compressed_values = ArrayList(u8).init(alloc);
    defer compressed_values.deinit();
    var decompressed_values = ArrayList(f64).init(alloc);
    defer decompressed_values.deinit();
    const error_bound: f32 = 0.0;

    var i: usize = 0;
    var lineIndex: usize = 0;
    while (i < 100) : (i += 1) {
        lineIndex = i / 25;
        try list_values.append(f80Tof64(evaluate(&lines[lineIndex], i)));
    }

    const uncompressed_values = list_values.items;

    try compress_swing(uncompressed_values[0..], &compressed_values, error_bound);
    try decompress(compressed_values.items, &decompressed_values);

    try testing.expect(mem.eql(f64, uncompressed_values[0..], decompressed_values.items));
}

test "swing-filter noisy line with 0 error bound and (even size) compress and decompress" {
    const alloc = testing.allocator;
    var line = Line{ .slope = 1, .intercept = 0.0 };

    var list_values = std.ArrayList(f64).init(alloc);
    defer list_values.deinit();
    var compressed_values = ArrayList(u8).init(alloc);
    defer compressed_values.deinit();
    var decompressed_values = ArrayList(f64).init(alloc);
    defer decompressed_values.deinit();
    const error_bound: f32 = 0.0;

    var rnd = std.rand.DefaultPrng.init(0);

    var i: usize = 0;
    while (i < 10) : (i += 1) {
        const noise = rnd.random().float(f64) * 0.1 - 0.05;
        try list_values.append(f80Tof64(evaluate(&line, i) + noise));
    }

    const uncompressed_values = list_values.items;

    try compress_swing(uncompressed_values[0..], &compressed_values, error_bound);
    try decompress(compressed_values.items, &decompressed_values);

    try testing.expect(isWithinErrorBound(uncompressed_values, decompressed_values.items, error_bound));
}

test "swing-filter noisy line with 0 error bound and (odd size) compress and decompress" {
    const alloc = testing.allocator;

    var line = Line{ .slope = 1, .intercept = 0.0 };

    var list_values = std.ArrayList(f64).init(alloc);
    defer list_values.deinit();
    var compressed_values = ArrayList(u8).init(alloc);
    defer compressed_values.deinit();
    var decompressed_values = ArrayList(f64).init(alloc);
    defer decompressed_values.deinit();
    const error_bound: f32 = 0.0;

    var rnd = std.rand.DefaultPrng.init(0);

    var i: usize = 0;
    while (i < 11) : (i += 1) {
        const noise = rnd.random().float(f64) * 0.1 - 0.05;
        try list_values.append(f80Tof64(evaluate(&line, i) + noise));
    }

    const uncompressed_values = list_values.items;

    try compress_swing(uncompressed_values[0..], &compressed_values, error_bound);
    try decompress(compressed_values.items, &decompressed_values);

    try testing.expect(isWithinErrorBound(uncompressed_values, decompressed_values.items, error_bound));
}

test "swing-filter noisy lines with 0.1 error bound and (even size) compress and decompress" {
    const alloc = testing.allocator;

    var lines = [_]Line{
        Line{ .slope = 1, .intercept = 0.0 },
        Line{ .slope = -1, .intercept = 50 },
        Line{ .slope = 1, .intercept = -50 },
        Line{ .slope = -1, .intercept = 100 },
    };

    var list_values = std.ArrayList(f64).init(alloc);
    defer list_values.deinit();
    var compressed_values = ArrayList(u8).init(alloc);
    defer compressed_values.deinit();
    var decompressed_values = ArrayList(f64).init(alloc);
    defer decompressed_values.deinit();
    const error_bound: f32 = 0.1;

    var rnd = std.rand.DefaultPrng.init(0);

    var i: usize = 0;
    var lineIndex: usize = 0;
    while (i < 100) : (i += 1) {
        lineIndex = i / 25;
        const noise = rnd.random().float(f64) * 0.1 - 0.05;
        try list_values.append(f80Tof64(evaluate(&lines[lineIndex], i) + noise));
    }

    const uncompressed_values = list_values.items;

    try compress_swing(uncompressed_values[0..], &compressed_values, error_bound);
    try decompress(compressed_values.items, &decompressed_values);

    try testing.expect(isWithinErrorBound(uncompressed_values, decompressed_values.items, error_bound));
}

test "swing-filter with noisy lines, and random error bound compress and decompress" {
    const alloc = testing.allocator;
    var rnd = std.rand.DefaultPrng.init(@as(u64, @bitCast(std.time.milliTimestamp())));

    var lines = [_]Line{
        Line{ .slope = 2 * (rnd.random().float(f64) - 0.5), .intercept = 2 * (rnd.random().float(f64) - 0.5) },
        Line{ .slope = 2 * (rnd.random().float(f64) - 0.5), .intercept = 2 * (rnd.random().float(f64) - 0.5) },
        Line{ .slope = 2 * (rnd.random().float(f64) - 0.5), .intercept = 2 * (rnd.random().float(f64) - 0.5) },
        Line{ .slope = 2 * (rnd.random().float(f64) - 0.5), .intercept = 2 * (rnd.random().float(f64) - 0.5) },
    };

    var list_values = std.ArrayList(f64).init(alloc);
    defer list_values.deinit();
    var compressed_values = ArrayList(u8).init(alloc);
    defer compressed_values.deinit();
    var decompressed_values = ArrayList(f64).init(alloc);
    defer decompressed_values.deinit();
    const error_bound: f32 = rnd.random().float(f32) * 0.1;

    var i: usize = 0;
    var lineIndex: usize = 0;
    while (i < 1000) : (i += 1) {
        lineIndex = i / 250;
        const noise = rnd.random().float(f64) * 0.1 - 0.05;
        try list_values.append(f80Tof64(evaluate(&lines[lineIndex], i) + noise));
    }

    const uncompressed_values = list_values.items;

    try compress_swing(uncompressed_values[0..], &compressed_values, error_bound);
    try decompress(compressed_values.items, &decompressed_values);

    try testing.expect(isWithinErrorBound(uncompressed_values, decompressed_values.items, error_bound));
}
