// Copyright 2025 TerseTS Contributors
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
const mem = std.mem;
const testing = std.testing;
const time = std.time;

const Encoding = @import("../utilities/encoding.zig");
const tersets = @import("../tersets.zig");
const Tester = @import("../tester.zig");

const ArrayList = std.ArrayList;
const Random = std.Random;

const Error = tersets.Error;

pub fn compress(values: []const f64, out: *ArrayList(u8), error_bound: f32) Error!void {
    const output_size = try compressSerfQtResultSize(f64, values, @floatCast(error_bound));
    try out.ensureTotalCapacity(output_size + @sizeOf(f32));
    try out.appendNTimes(0, output_size + @sizeOf(f32));
    // write error bound to start of stream
    // NOTE: zig appears not to support any of the more straightforward ways of doing this
    const err_bound_u32 = @as(u32, @bitCast(error_bound));
    out.items[0] = @as(u8, @truncate(err_bound_u32 >> 24));
    out.items[1] = @as(u8, @truncate(err_bound_u32 >> 16));
    out.items[2] = @as(u8, @truncate(err_bound_u32 >> 8));
    out.items[3] = @as(u8, @truncate(err_bound_u32));
    try compressSerfQtUnmanaged(f64, values, @floatCast(error_bound), out.items[@sizeOf(f32)..out.items.len]);
}

pub fn decompress(stream_raw: []const u8, out: *ArrayList(f64)) Error!void {
    // extract error bound as f32 from start of stream
    // NOTE: zig appears not to support any of the more straightforward ways of doing this
    const err_bound_u32 =
        (@as(u32, stream_raw[0]) << 24)
        | (@as(u32, stream_raw[1]) << 16)
        | (@as(u32, stream_raw[2]) << 8)
        | @as(u32, stream_raw[3]);
    const error_bound = @as(f32, @bitCast(err_bound_u32));

    const stream = stream_raw[@sizeOf(f32)..stream_raw.len];

    const out_elems = decompressSerfQtElemCount(stream);
    try out.ensureTotalCapacity(out_elems);
    try out.appendNTimes(0, out_elems);

    decompressSerfQtUnmanaged(
        f64,
        stream,
        error_bound,
        out.items
    );
}

fn errorBoundToQuantum(error_bound: f64) f64 {
    // using error_bound * 2 can and does cause occasional issues with staying
    // within error bound
    return error_bound * 1.9;
}

fn compressSerfQtResultSize(comptime FP_TYPE: type, values: []const FP_TYPE, error_bound: f64) Error!u64 {
    const quantum = errorBoundToQuantum(error_bound);
    var predicted: f64 = 0.0;
    var result: u64 = 0;

    for (values) |v| {
        const quantized_delta = try quantize(v, predicted, quantum);
        predicted = dequantize(quantized_delta, predicted, quantum);
        const zigzagd = Encoding.zigzagEncode(quantized_delta);
        const eg_res = Encoding.gammaEncodeSingle(zigzagd + 1);
        result += 1 + (eg_res.leading_zeros * 2);
    }

    // convert to bytes
    result += 7;
    result /= 8;
    return result;
}

fn compressSerfQtUnmanaged(comptime FP_TYPE: type, values: []const FP_TYPE, error_bound: f64, out: []u8) Error!void {
    const quantum = errorBoundToQuantum(error_bound);
    var predicted: f64 = 0.0;
    var gamma_index: u64 = 0;

    for (values) |v| {
        const quantized_delta = try quantize(v, predicted, quantum);
        predicted = dequantize(quantized_delta, predicted, quantum);
        const zigzagd = Encoding.zigzagEncode(quantized_delta);
        Encoding.gammaEncodeIter(zigzagd + 1, out, &gamma_index);
    }
}


pub fn decompressSerfQtElemCount(stream: []const u8) u64 {
    return Encoding.gammaDecodeElemCount(stream);
}

pub fn decompressSerfQtUnmanaged(comptime FP_TYPE: type, stream: []const u8, error_bound: f64, values: []FP_TYPE) void {
    const quantum = errorBoundToQuantum(error_bound);

    var predicted: f64 = 0.0;
    var degamma_idx: u64 = 0;

    const num_bits = 8 * stream.len;
    var i: u64 = 0;
    while ((i < values.len) and (degamma_idx < num_bits)) {
        if (Encoding.onlyZerosFollow(stream, degamma_idx)) {
            break;
        }
        const degamma_v = Encoding.gammaDecodeIter(stream, &degamma_idx);
        const dezigzagd = Encoding.zigzagDecode(degamma_v - 1);
        const v = dequantize(dezigzagd, predicted, quantum);
        predicted = v;
        values[i] = @as(FP_TYPE, @floatCast(v));

        i += 1;
    }
}

const I64_MAX = std.math.maxInt(i64);
const I64_MIN = std.math.minInt(i64);

fn quantize(value: f64, base: f64, quantum: f64) Error!i64 {
    const delta = value - base;
    const fres = @floor((delta / quantum) + 0.5);

    if ((fres < @as(f64, @floatFromInt(I64_MAX))) and (fres > @as(f64, @floatFromInt(I64_MIN)))) {
        return @intFromFloat(fres);
    }
    return error.UnsupportedInput;
}

fn dequantize(value: i64, base: f64, quantum: f64) f64 {
    const delta = @as(f64, @floatFromInt(value)) * quantum;
    return delta + base;
}

// ~1M iterations; if the compiler isn't horrid, this should run quick but still give us some usable indication
test "quantization -> dequantization is accurate when using quantum = error_bound * 1.9 with error bounds between 0.0001 and 0.1" {
    // NOTE: using time to seed a PRNG - while common practice - makes for a somewhat poorly seeded PRNG
    // TODO: use /dev/urandom instead
    var prng = Random.DefaultPrng.init(@bitCast(time.milliTimestamp()));
    const random = prng.random();
    for (1..1000) |err_b_int| {
        const error_bound = 0.0001 * @as(f64, @floatFromInt(err_b_int));
        const quantum = error_bound * 1.9;

        const lower_bound = -1000.0;
        const upper_bound = 1000.0;

        for (0..1000) |_| {
            const num = Tester.generateBoundedRandomValue(f64, lower_bound, upper_bound, random);
            const quantized = try quantize(num, 0, quantum);
            const dequantized = dequantize(quantized, 0, quantum);

            try testing.expectApproxEqAbs(num, dequantized, error_bound);
        }
    }
}

test "quantization fails if input delta is unrepresentable as i64" {
    try testing.expectError(error.UnsupportedInput, quantize(@as(f64, @floatFromInt(I64_MAX)) + 10000.0, 0, 1.0));
    try testing.expectError(error.UnsupportedInput, quantize(@as(f64, @floatFromInt(I64_MIN)) - 10000.0, 0, 1.0));
}


test "serf-qt fails when delta can't be quantized using quantization.quantize" {
    const v = [1]f64 { 8 * @as(f64, @floatFromInt(@as(u64, 1) << 63)) };
    var buffer: [128]u8 = .{
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    };
    try testing.expectError(error.UnsupportedInput, compressSerfQtUnmanaged(f64, &v, 1.0, &buffer));
}

test "serf-qt can compress and decompress constant data" {
    const input: [128]f64 = .{
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    };
    var buffer = ArrayList(u8).init(testing.allocator);
    defer buffer.deinit();

    // should compress down to 128 bits
    const count = try compressSerfQtResultSize(f64, &input, 1.0);
    try testing.expectEqual(count, 128 / 8);

    try compress(&input, &buffer, 1.0);

    var out = ArrayList(f64).init(testing.allocator);
    defer out.deinit();

    try decompress(buffer.items, &out);

    for (input, out.items) |i, o| {
        try testing.expectEqual(i, o);
    }
}

test "serf-qt can compress and decompress random quantizable data" {
    var prng = Random.DefaultPrng.init(@bitCast(std.time.milliTimestamp()));
    const random = prng.random();
    for (1..1000) |inputs_length| {
        const error_bound = Tester.generateBoundedRandomValue(f64, 0.00001, 30.0, random);

        var inputs = try std.ArrayList(f64).initCapacity(testing.allocator, inputs_length);
        defer inputs.deinit();
        var prev: f64 = 0.0;
        for (0..inputs_length) |_| {
            const val = Tester.generateBoundedRandomValue(f64, prev - (error_bound * 10000), prev + (error_bound * 10000), random);
            try inputs.append(val);
            prev = val;
        }

        var outputs = std.ArrayList(u8).init(testing.allocator);
        defer outputs.deinit();

        try compress(inputs.items, &outputs, @floatCast(error_bound));

        var decompressed = std.ArrayList(f64).init(testing.allocator);
        defer decompressed.deinit();

        try decompress(outputs.items, &decompressed);

        for (inputs.items, decompressed.items) |expected, actual| {
            try testing.expectApproxEqAbs(expected, actual, error_bound);
        }
    }
}

// TODO: more tests (one for each dataset, with some randomization around precision)
