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

// This file implements the Swing Filter algorithm.
// Swing compresses a time series by removing points that
// do not deviate significantly from a linear interpolation of their
// neighboring points, within the error bound.

const std = @import("std");
const ts = @import("../tersets.zig");
const testing = std.testing;

pub const Segment = struct {
    start_time: usize,
    start_value: f64,
    end_time: usize,
    end_value: f64,
};

pub const Line = struct {
    slope: f64,
    intercept: f64,
};

fn get_line(segment: *Segment, line: *Line) !void {
    std.debug.assert(segment.end_time != segment.start_time);
    const duration = @as(f64, @floatFromInt(segment.end_time - segment.start_time));
    line.slope = (segment.end_value - segment.start_value) / duration;
    line.intercept = segment.start_value - line.slope * @as(f64, @floatFromInt(segment.start_time));
}

fn get_bound_line(segment: *Segment, line: *Line, error_bound: f32) !void {
    std.debug.assert(segment.end_time != segment.start_time);
    const duration = @as(f64, @floatFromInt(segment.end_time - segment.start_time));
    line.slope = (segment.end_value + error_bound - segment.start_value) / duration;
    line.intercept = segment.start_value - line.slope * @as(f64, @floatFromInt(segment.start_time));
}

fn printBytesAsHex(data: [*]const u8, length: usize) void {
    std.debug.print("Hexadecimal byte values: ", .{});
    var i: usize = 0;
    while (i < length) : (i += 1) {
        std.debug.print("{x} ", .{data[i]});
    }
    std.debug.print("\n", .{});
}

fn evaluate(line: *Line, timestamp: usize) f64 {
    return line.slope * @as(f64, @floatFromInt(timestamp)) + line.intercept;
}

fn printArrayList(list: []const f64) void {
    std.debug.print("\nPrint array: \n", .{});
    for (list) |item| {
        std.debug.print("{} ", .{item});
    }
    std.debug.print("\n", .{});
}

fn printLine(line: *Line, islower: bool) void {
    if (islower) {
        std.debug.print("lower", .{});
    } else {
        std.debug.print("upper", .{});
    }
    std.debug.print("--->line slope {} intercept {}\n", .{ line.slope, line.intercept });
}

fn printSegment(segment: *Segment) void {
    std.debug.print("current segment: st {}, sv {}, et {}, ev {} \n", .{ segment.start_time, segment.start_value, segment.end_time, segment.end_value });
}
/// Applies the Swing Filter to compress a time series.
/// @param uncompressed_values: the input time series and length
/// @param compressed_values: the compressed representation
/// @param error_bound: the error bound information
/// @returns an ArrayList of compressed data points
pub fn compress(uncompressed_values: ts.UncompressedValues, compressed_values: *ts.CompressedValues, error_bound: f32) !void {
    const gpa = std.heap.page_allocator;

    var list_values = std.ArrayList(f64).init(gpa);
    // defer list_values.deinit();

    // Initiallization
    var upper_y: f64 = 0;
    var lower_y: f64 = 0;
    var tmp_upper_y: f64 = 0;
    var tmp_lower_y: f64 = 0;
    var upper_bound_line: Line = Line{ .slope = 0, .intercept = 0 };
    var lower_bound_line: Line = Line{ .slope = 0, .intercept = 0 };
    var tmp_upper_bound_line: Line = Line{ .slope = 0, .intercept = 0 };
    var tmp_lower_bound_line: Line = Line{ .slope = 0, .intercept = 0 };
    var current_line: Line = Line{ .slope = 0, .intercept = 0 };
    var first_timestamp: usize = 0;
    var current_timestamp: usize = 1;
    var current_segment: Segment = Segment{ .start_time = first_timestamp, .start_value = uncompressed_values.data[first_timestamp], .end_time = current_timestamp, .end_value = uncompressed_values.data[current_timestamp] };

    try get_bound_line(&current_segment, &upper_bound_line, error_bound);
    try get_bound_line(&current_segment, &lower_bound_line, -error_bound);

    current_timestamp += 1;
    while (current_timestamp < uncompressed_values.len) : (current_timestamp += 1) {
        upper_y = evaluate(&upper_bound_line, current_timestamp);
        lower_y = evaluate(&lower_bound_line, current_timestamp);

        if ((upper_y < (uncompressed_values.data[current_timestamp] - error_bound)) or
            (lower_y > (uncompressed_values.data[current_timestamp] + error_bound)))
        {
            upper_y = evaluate(&upper_bound_line, current_timestamp - 1);
            lower_y = evaluate(&lower_bound_line, current_timestamp - 1);
            current_segment.end_value = (upper_y + lower_y) / 2;
            try get_line(&current_segment, &current_line);
            try list_values.append(@as(f64, @floatFromInt(current_segment.start_time)));
            try list_values.append(current_line.slope);
            try list_values.append(current_line.intercept);
            first_timestamp = current_timestamp;
            current_segment.start_time = current_timestamp;
            current_segment.start_value = uncompressed_values.data[current_timestamp];
            current_segment.end_time = current_timestamp + 1;
            current_segment.end_value = uncompressed_values.data[current_timestamp + 1];
            current_timestamp += 1;
            try get_bound_line(&current_segment, &upper_bound_line, error_bound);
            try get_bound_line(&current_segment, &lower_bound_line, -error_bound);
        } else {
            current_segment.end_time = current_timestamp;
            current_segment.end_value = uncompressed_values.data[current_timestamp];
            try get_bound_line(&current_segment, &tmp_upper_bound_line, error_bound);
            try get_bound_line(&current_segment, &tmp_lower_bound_line, -error_bound);

            tmp_upper_y = evaluate(&tmp_upper_bound_line, current_timestamp);
            tmp_lower_y = evaluate(&tmp_lower_bound_line, current_timestamp);
            if (upper_y > tmp_upper_y) {
                upper_bound_line.slope = tmp_upper_bound_line.slope;
                upper_bound_line.intercept = tmp_upper_bound_line.intercept;
                upper_y = tmp_upper_y;
            }
            if (lower_y < tmp_lower_y) {
                lower_bound_line.slope = tmp_lower_bound_line.slope;
                lower_bound_line.intercept = tmp_lower_bound_line.intercept;
                lower_y = tmp_lower_y;
            }
        }
    }
    current_segment.end_value = (upper_y + lower_y) / 2;
    try get_line(&current_segment, &current_line);
    try list_values.append(@as(f64, @floatFromInt(current_segment.start_time)));
    try list_values.append(current_line.slope);
    try list_values.append(current_line.intercept);
    try list_values.append(@as(f64, @floatFromInt(current_timestamp)));
    const f64_slice = list_values.items;
    compressed_values.data = @alignCast(@ptrCast(f64_slice));
    compressed_values.len = f64_slice.len * 8;
}

pub fn decompress(compressed_values: ts.CompressedValues, uncompressed_values: *ts.UncompressedValues) !void {
    const gpa = std.heap.page_allocator;

    var list_values = std.ArrayList(f64).init(gpa);
    // defer list_values.deinit();

    const serialized_values: [*]const f64 = @alignCast(@ptrCast(compressed_values.data));
    const length = compressed_values.len / 8;

    // const size: i32 = @as(i32, @intFromFloat(values[values.len - 1]));
    var previous_timestamp: usize = 0;
    var current_timestamp: usize = 0;
    var next_timestamp: usize = 0;
    var current_line: Line = Line{ .slope = 0, .intercept = 0 };

    var index: usize = 0;
    while (index < length - 1) : (index += 3) {
        next_timestamp = @as(usize, @intFromFloat(serialized_values[index + 3]));
        current_timestamp = previous_timestamp;
        current_line.slope = serialized_values[index + 1];
        current_line.intercept = serialized_values[index + 2];
        while (current_timestamp < next_timestamp) : (current_timestamp += 1) {
            const y: f64 = evaluate(&current_line, current_timestamp);
            try list_values.append(y);
        }
        previous_timestamp = current_timestamp;
    }

    const f64_slice = list_values.items;
    uncompressed_values.data = @alignCast(@ptrCast(f64_slice));
    uncompressed_values.len = next_timestamp; //TODO: fix this

}

test "swing a line compress and decompress" {
    const uncompressed_array = [_]f64{ 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5 };
    const uncompressed_slice = uncompressed_array[0..8];
    const uncompressed_values = ts.UncompressedValues{
        .data = uncompressed_slice.ptr,
        .len = uncompressed_slice.len,
    };
    var compressed_values = ts.CompressedValues{
        .data = undefined,
        .len = undefined,
    };
    var decompressed_values = ts.UncompressedValues{
        .data = uncompressed_slice.ptr,
        .len = uncompressed_slice.len,
    };
    const error_bound: f32 = 0.5;
    try compress(
        uncompressed_values,
        &compressed_values,
        error_bound,
    );

    // try testing.expect(compress_result == 0);

    try decompress(compressed_values, &decompressed_values);
    try testing.expectEqual(decompressed_values.len, uncompressed_values.len);

    var i: usize = 0;
    while (i < decompressed_values.len) : (i += 1) {
        try testing.expect((uncompressed_values.data[i] + error_bound > decompressed_values.data[i]) and (uncompressed_values.data[i] - error_bound < decompressed_values.data[i]));
    }
}

test "swing two lines compress and decompress" {
    const uncompressed_array = [_]f64{ 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5 };

    const uncompressed_slice = uncompressed_array[0..16];
    const uncompressed_values = ts.UncompressedValues{
        .data = uncompressed_slice.ptr,
        .len = uncompressed_slice.len,
    };
    var compressed_values = ts.CompressedValues{
        .data = undefined,
        .len = undefined,
    };
    var decompressed_values = ts.UncompressedValues{
        .data = uncompressed_slice.ptr,
        .len = uncompressed_slice.len,
    };
    const error_bound: f32 = 0.5;
    try compress(
        uncompressed_values,
        &compressed_values,
        error_bound,
    );

    // try testing.expect(compress_result == 0);

    try decompress(compressed_values, &decompressed_values);
    try testing.expectEqual(uncompressed_values.len, decompressed_values.len);
    var i: usize = 0;

    i = 0;
    while (i < decompressed_values.len) : (i += 1) {
        try testing.expect((uncompressed_values.data[i] + error_bound > decompressed_values.data[i]) and (uncompressed_values.data[i] - error_bound < decompressed_values.data[i]));
    }
}

test "swing multiple lines compress and decompress" {
    var lines = [_]Line{
        Line{ .slope = 1.0, .intercept = 2.0 },
        Line{ .slope = -0.5, .intercept = 1.0 },
        Line{ .slope = 0.2, .intercept = -1.5 },
        Line{ .slope = -0.1, .intercept = 1.5 },
    };
    const gpa = std.heap.page_allocator;

    var list_values = std.ArrayList(f64).init(gpa);
    var i: usize = 0;
    var index: usize = 0;
    while (i < 1000) : (i += 1) {
        index = i / 250;
        try list_values.append(evaluate(&lines[index], i));
    }

    const uncompressed_slice = list_values.items;
    const uncompressed_values = ts.UncompressedValues{
        .data = uncompressed_slice.ptr,
        .len = uncompressed_slice.len,
    };
    var compressed_values = ts.CompressedValues{
        .data = undefined,
        .len = undefined,
    };
    var decompressed_values = ts.UncompressedValues{
        .data = uncompressed_slice.ptr,
        .len = uncompressed_slice.len,
    };
    const error_bound: f32 = 0.5;
    try compress(
        uncompressed_values,
        &compressed_values,
        error_bound,
    );

    // try testing.expect(compress_result == 0);

    try decompress(compressed_values, &decompressed_values);
    try testing.expectEqual(uncompressed_values.len, decompressed_values.len);
    i = 0;
    while (i < decompressed_values.len) : (i += 1) {
        try testing.expect((uncompressed_values.data[i] + error_bound > decompressed_values.data[i]) and (uncompressed_values.data[i] - error_bound < decompressed_values.data[i]));
    }
}
