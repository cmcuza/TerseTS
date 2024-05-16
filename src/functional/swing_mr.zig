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

//! Implementation a variation of Swing Filter algorithm from the paper:
//! Hazem Elmeleegy, Ahmed K. Elmagarmid, Emmanuel Cecchet, Walid G. Aref, and Willy Zwaenepoel.
//! Online piece-wise linear approximation of numerical streams with precision guarantees.
//! Proc. VLDB Endow. 2, 1, 2009.
//! https://doi.org/10.14778/1687627.1687645.
//! In the paper the approximation line minimizes the Square Error of the points in the segment.
//! This implementation finds the Line that cross the Middle of the Upper and Lower Bounds.

const std = @import("std");
const ts = @import("../tersets.zig");
const utils = @import("../utils.zig");
const print = std.debug.print;

const Line = utils.Line;
const Segment = utils.Segment;

const mem = std.mem;
const testing = std.testing;

const ArrayList = std.ArrayList;

fn appendLine(
    line: *Line,
    compressed_values: *ArrayList(u8),
) !void {
    const valueAsBytes: [8]u8 = @bitCast(line.slope);
    try compressed_values.appendSlice(valueAsBytes[0..]);
    const indexAsBytes: [8]u8 = @bitCast(line.intercept);
    try compressed_values.appendSlice(indexAsBytes[0..]);
}

fn appendIndex(index: usize, compressed_values: *ArrayList(u8)) !void {
    const indexAsFloat: [8]u8 = @bitCast(@as(f64, @floatFromInt(index)));
    try compressed_values.appendSlice(indexAsFloat[0..]);
}

/// @param uncompressed_values: the input time series and length
/// @param compressed_values: the compressed representation
/// @param error_bound: the error bound information
pub fn compress(
    uncompressed_values: []const f64,
    compressed_values: *ArrayList(u8),
    error_bound: f32,
) !void {
    // Initiallization
    var upper_y: f64 = 0;
    var lower_y: f64 = 0;

    var upper_bound_line: Line = Line{ .slope = 0, .intercept = 0 };
    var lower_bound_line: Line = Line{ .slope = 0, .intercept = 0 };
    var new_upper_bound_line: Line = Line{ .slope = 0, .intercept = 0 };
    var new_lower_bound_line: Line = Line{ .slope = 0, .intercept = 0 };
    var current_line: Line = Line{ .slope = 0, .intercept = 0 };
    var first_timestamp: usize = 0;
    var current_timestamp: usize = 1;
    var current_segment: Segment = Segment{ .start_time = first_timestamp, .start_value = uncompressed_values[first_timestamp], .end_time = current_timestamp, .end_value = uncompressed_values[current_timestamp] };

    try utils.getBoundLine(&current_segment, &upper_bound_line, error_bound);
    try utils.getBoundLine(&current_segment, &lower_bound_line, -error_bound);

    current_timestamp += 1;
    while (current_timestamp < uncompressed_values.len) : (current_timestamp += 1) {
        upper_y = utils.evaluate(&upper_bound_line, current_timestamp);
        lower_y = utils.evaluate(&lower_bound_line, current_timestamp);
        // Is the new point more than \epsilon above upper_y or bellow lower_y?
        if ((upper_y < (uncompressed_values[current_timestamp] - error_bound)) or
            (lower_y > (uncompressed_values[current_timestamp] + error_bound)))
        { // Recording mechanism

            upper_y = utils.evaluate(&upper_bound_line, current_timestamp - 1);
            lower_y = utils.evaluate(&lower_bound_line, current_timestamp - 1);

            current_segment.end_value = (upper_y + lower_y) / 2; // Mid-Range of the upper and lower bound

            try utils.getLine(&current_segment, &current_line);
            try appendLine(&current_line, compressed_values);
            try appendIndex(current_timestamp, compressed_values);

            // Update the current segment
            first_timestamp = current_timestamp;
            current_segment.start_time = current_timestamp;
            current_segment.start_value = uncompressed_values[current_timestamp];
            if (current_timestamp + 1 < uncompressed_values.len) { // (edge case) only one point left
                // Update the current segment
                current_segment.end_time = current_timestamp + 1;
                current_segment.end_value = uncompressed_values[current_timestamp + 1];

                // Recompute the upper and lower bounds
                try utils.getBoundLine(&current_segment, &upper_bound_line, error_bound);
                try utils.getBoundLine(&current_segment, &lower_bound_line, -error_bound);

                current_timestamp += 1; // advance the current_timestamp
            } else {
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
            try utils.getBoundLine(&current_segment, &new_upper_bound_line, error_bound);
            try utils.getBoundLine(&current_segment, &new_lower_bound_line, -error_bound);

            const new_upper_y: f64 = utils.evaluate(&new_upper_bound_line, current_timestamp);
            const new_lower_y: f64 = utils.evaluate(&new_lower_bound_line, current_timestamp);

            // Outdate the upper and lower bounds if needed
            if (upper_y > new_upper_y) { // Swing down
                upper_bound_line.slope = new_upper_bound_line.slope;
                upper_bound_line.intercept = new_upper_bound_line.intercept;
            }
            if (lower_y < new_lower_y) { //Swing up
                lower_bound_line.slope = new_lower_bound_line.slope;
                lower_bound_line.intercept = new_lower_bound_line.intercept;
            }
        }
    }

    upper_y = utils.evaluate(&upper_bound_line, current_timestamp - 1);
    lower_y = utils.evaluate(&lower_bound_line, current_timestamp - 1);

    current_segment.end_value = (upper_y + lower_y) / 2;

    try utils.getLine(&current_segment, &current_line);
    try appendLine(&current_line, compressed_values);
    try appendIndex(current_timestamp, compressed_values);
}

pub fn decompress(
    compressed_values: []const u8,
    decompressed_values: *ArrayList(f64),
) error{OutOfMemory}!void {
    // TODO: Check length of uncompressed_values is modulo 24.

    const compressed_lines_and_index = mem.bytesAsSlice(
        f64,
        compressed_values,
    );

    var previous_timestamp: usize = 0;
    var current_line: Line = Line{ .slope = 0, .intercept = 0 };

    var index: usize = 0;
    while (index < compressed_lines_and_index.len) : (index += 3) {
        const next_timestamp = @as(usize, @intFromFloat(compressed_lines_and_index[index + 2]));
        var current_timestamp = previous_timestamp;
        current_line.slope = compressed_lines_and_index[index];
        current_line.intercept = compressed_lines_and_index[index + 1];
        while (current_timestamp < next_timestamp) : (current_timestamp += 1) {
            const y: f64 = utils.evaluate(&current_line, current_timestamp);
            try decompressed_values.append(y);
        }
        previous_timestamp = current_timestamp;
    }
}

test "swing-mr single line compress and decompress" {
    const alloc = testing.allocator;
    const uncompressed_values = [_]f64{ 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5 };
    var compressed_values = ArrayList(u8).init(alloc);
    defer compressed_values.deinit();
    var decompressed_values = ArrayList(f64).init(alloc);
    defer decompressed_values.deinit();

    const error_bound: f32 = 0.0;

    try compress(uncompressed_values[0..], &compressed_values, error_bound);

    try decompress(compressed_values.items, &decompressed_values);

    try testing.expect(mem.eql(f64, uncompressed_values[0..], decompressed_values.items));
}

test "swing-mr single line and single point (odd size) compress and decompress" {
    const alloc = testing.allocator;
    const uncompressed_values = [_]f64{ 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 1.5 };
    var compressed_values = ArrayList(u8).init(alloc);
    defer compressed_values.deinit();
    var decompressed_values = ArrayList(f64).init(alloc);
    defer decompressed_values.deinit();

    const error_bound: f32 = 0.0;

    try compress(uncompressed_values[0..], &compressed_values, error_bound);

    try decompress(compressed_values.items, &decompressed_values);
    try testing.expect(mem.eql(f64, uncompressed_values[0..], decompressed_values.items));
}

test "swing-mr single line and single point (even size) compress and decompress" {
    const alloc = testing.allocator;
    const uncompressed_values = [_]f64{ 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 1.5 };
    var compressed_values = ArrayList(u8).init(alloc);
    defer compressed_values.deinit();
    var decompressed_values = ArrayList(f64).init(alloc);
    defer decompressed_values.deinit();

    const error_bound: f32 = 0.0;

    try compress(uncompressed_values[0..], &compressed_values, error_bound);

    try decompress(compressed_values.items, &decompressed_values);
    try testing.expect(mem.eql(f64, uncompressed_values[0..], decompressed_values.items));
}

test "swing-mr two parallel lines compress and decompress" {
    const alloc = testing.allocator;
    const uncompressed_values = [_]f64{ 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5 };

    var compressed_values = ArrayList(u8).init(alloc);
    defer compressed_values.deinit();
    var decompressed_values = ArrayList(f64).init(alloc);
    defer decompressed_values.deinit();
    const error_bound: f32 = 0.0;

    try compress(uncompressed_values[0..], &compressed_values, error_bound);
    try decompress(compressed_values.items, &decompressed_values);

    try testing.expect(mem.eql(f64, uncompressed_values[0..], decompressed_values.items));
}

test "swing-mr 4 v-shaped lines compress and decompress" {
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
        try list_values.append(utils.evaluate(&lines[lineIndex], i));
    }

    const uncompressed_values = list_values.items;

    try compress(uncompressed_values[0..], &compressed_values, error_bound);
    try decompress(compressed_values.items, &decompressed_values);

    try testing.expect(mem.eql(f64, uncompressed_values[0..], decompressed_values.items));
}

test "swing-mr noisy line with 0 error bound and (even size) compress and decompress" {
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
        try list_values.append(utils.evaluate(&line, i) + noise);
    }

    const uncompressed_values = list_values.items;

    try compress(uncompressed_values[0..], &compressed_values, error_bound);
    try decompress(compressed_values.items, &decompressed_values);

    try testing.expect(utils.isWithinErrorBound(uncompressed_values, decompressed_values.items, error_bound));
}

test "swing-mr noisy line with 0 error bound and (odd size) compress and decompress" {
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
        try list_values.append(utils.evaluate(&line, i) + noise);
    }

    const uncompressed_values = list_values.items;

    try compress(uncompressed_values[0..], &compressed_values, error_bound);
    try decompress(compressed_values.items, &decompressed_values);

    try testing.expect(utils.isWithinErrorBound(uncompressed_values, decompressed_values.items, error_bound));
}

test "swing-mr noisy lines with 0.1 error bound and (even size) compress and decompress" {
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
    while (i < 1000) : (i += 1) {
        lineIndex = i / 250;
        const noise = rnd.random().float(f64) * 0.2 - 0.05;
        try list_values.append(utils.evaluate(&lines[lineIndex], i) + noise);
    }

    const uncompressed_values = list_values.items;

    try compress(uncompressed_values[0..], &compressed_values, error_bound);
    try decompress(compressed_values.items, &decompressed_values);

    try testing.expect(utils.isWithinErrorBound(uncompressed_values, decompressed_values.items, error_bound));
}

test "swing-mr noisy lines with 0.2 error bound and (even size) compress and decompress" {
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
    const error_bound: f32 = 0.2;

    var rnd = std.rand.DefaultPrng.init(0);

    var i: usize = 0;
    var lineIndex: usize = 0;
    while (i < 1000) : (i += 1) {
        lineIndex = i / 250;
        const noise = rnd.random().float(f64) * 0.2 - 0.05;
        try list_values.append(utils.evaluate(&lines[lineIndex], i) + noise);
    }

    const uncompressed_values = list_values.items;

    try compress(uncompressed_values[0..], &compressed_values, error_bound);
    try decompress(compressed_values.items, &decompressed_values);

    try testing.expect(utils.isWithinErrorBound(uncompressed_values, decompressed_values.items, error_bound));
}

test "swing-mr with noisy lines, and random error bound compress and decompress" {
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
        try list_values.append(utils.evaluate(&lines[lineIndex], i) + noise);
    }

    const uncompressed_values = list_values.items;

    try compress(uncompressed_values[0..], &compressed_values, error_bound);
    try decompress(compressed_values.items, &decompressed_values);

    try testing.expect(utils.isWithinErrorBound(uncompressed_values, decompressed_values.items, error_bound));
}
