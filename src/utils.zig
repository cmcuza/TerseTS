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

const std = @import("std");
const ArrayList = std.ArrayList;

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

pub fn getIntercept(slope: f64, x: usize, y: f64, line: *Line) !void {
    line.slope = slope;
    line.intercept = y - slope * usizeToF64(x);
}

pub fn getLine(segment: *Segment, line: *Line) !void {
    if (segment.end_time != segment.start_time) {
        const duration = @as(f64, @floatFromInt(segment.end_time - segment.start_time));
        line.slope = (segment.end_value - segment.start_value) / duration;
        line.intercept = segment.start_value - line.slope * @as(f64, @floatFromInt(segment.start_time));
    } else {
        line.slope = 0;
        line.intercept = segment.end_value;
    }
}

pub fn getBoundLine(segment: *Segment, line: *Line, error_bound: f32) !void {
    std.debug.assert(segment.end_time != segment.start_time);
    const duration = @as(f64, @floatFromInt(segment.end_time - segment.start_time));
    line.slope = (segment.end_value + error_bound - segment.start_value) / duration;
    line.intercept = segment.start_value - line.slope * @as(f64, @floatFromInt(segment.start_time));
}

pub fn evaluate(line: *Line, timestamp: usize) f64 {
    return line.slope * @as(f64, @floatFromInt(timestamp)) + line.intercept;
}

pub fn appendLine(
    line: *Line,
    compressed_values: *ArrayList(u8),
) !void {
    const valueAsBytes: [8]u8 = @bitCast(line.slope);
    try compressed_values.appendSlice(valueAsBytes[0..]);
    const indexAsBytes: [8]u8 = @bitCast(line.intercept);
    try compressed_values.appendSlice(indexAsBytes[0..]);
}

pub fn appendIndex(index: usize, compressed_values: *ArrayList(u8)) !void {
    const indexAsFloat: [8]u8 = @bitCast(@as(f64, @floatFromInt(index)));
    try compressed_values.appendSlice(indexAsFloat[0..]);
}

pub fn usizeToF64(value: usize) f64 {
    return @as(f64, @floatFromInt(value));
}

pub fn getDerivate(segment: *Segment) f64 {
    return (segment.end_value - segment.start_value) * @as(f64, @floatFromInt(segment.end_time - segment.start_time));
}

pub fn printLine(line: *Line) void {
    std.debug.print("\nLine: slope {} intercept {}\n", .{ line.slope, line.intercept });
}

pub fn printSegment(segment: *Segment) void {
    std.debug.print("Segment: st {}, sv {}, et {}, ev {} \n", .{ segment.start_time, segment.start_value, segment.end_time, segment.end_value });
}
