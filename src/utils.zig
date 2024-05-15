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
        if (!((uncompressed_values[i] < item + error_bound) and (uncompressed_values[i] > item - error_bound))) return false;
    }
    return true;
}
pub fn getLine(segment: *Segment, line: *Line) !void {
    std.debug.assert(segment.end_time != segment.start_time);
    const duration = @as(f64, @floatFromInt(segment.end_time - segment.start_time));
    line.slope = (segment.end_value - segment.start_value) / duration;
    line.intercept = segment.start_value - line.slope * @as(f64, @floatFromInt(segment.start_time));
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

pub fn printLine(line: *Line) void {
    std.debug.print("Line: slope {} intercept {}\n", .{ line.slope, line.intercept });
}

pub fn printSegment(segment: *Segment) void {
    std.debug.print("Segment: st {}, sv {}, et {}, ev {} \n", .{ segment.start_time, segment.start_value, segment.end_time, segment.end_value });
}
