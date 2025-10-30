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
const assert = std.debug.assert;
const testing = std.testing;

const Tester = @import("../tester.zig");

// signed in, unsigned of same size out
pub fn zigzagEncode(v: i64) u64 {
    return @bitCast((v << 1) ^ (v >> 63));
}

pub fn zigzagDecode(u: u64) i64 {
    const v = @as(i64, @bitCast(u));
    return (v >> 1) ^ (-(v & 1));
}

pub fn bitIdxRead(s: []const u8, idx: u64) u8 {
    return (s[idx >> 3] >> @truncate(idx & 0x7)) & 1;
}

pub fn bitIdxWrite(s: []u8, idx: u64, v: u8) void {
    const byte_idx = idx >> 3;
    const bit_idx = @as(u3, @truncate(idx & 0x7));
    
    const tmp = s[byte_idx];
    const clear = tmp & ~(@as(u8, 1) << bit_idx);
    const set = clear | ((v & 1) << bit_idx);
    s[byte_idx] = set;
}

pub fn onlyZerosFollow(stream: []const u8, stream_index: u64) bool {
    for (stream_index..(@bitSizeOf(u8) * stream.len)) |i| {
        if (0 != bitIdxRead(stream, i)) {
            return false;
        }
    }
    return true;
}

// necessary because Zig doesn't support multiple return values properly
pub const GammaResult = struct {
    leading_zeros: u64,
    trailing_part: u64,
};

// trailing bit length is same as number of leading zeroes
pub fn gammaEncodeSingle(value: u64) GammaResult {
    assert(value > 0);
    const leading_zeros = 63 - @as(u64, @intCast(@clz(value)));
    return GammaResult {
        .leading_zeros = leading_zeros, // largest power of 2 in value
        .trailing_part = value & ~(@as(u64, 1) << @truncate(leading_zeros)),
    };
}

pub fn gammaEncodeIter(value: u64, stream: []u8, stream_index: *u64) void {
    const start = stream_index.*;
    const gres = gammaEncodeSingle(value);
    const end = start + 1 + (gres.leading_zeros * 2);
    
    // TODO (sio): this can be done more efficiently I'm sure
    for (start..(start + gres.leading_zeros)) |i| {
        bitIdxWrite(stream, i, 0);
    }
    bitIdxWrite(stream, start + gres.leading_zeros, 1);

    if (gres.leading_zeros > 0) {
        // write trailing part
        for (0..gres.leading_zeros) |i| {
            const v = (gres.trailing_part >> @truncate(i)) & 1;
            bitIdxWrite(stream, start + 1 + gres.leading_zeros + i, @as(u8, @truncate(v)));
        }
    }
    
    stream_index.* = end;
}

pub fn gammaEncodeSlice(values: []const u64, stream: []u8) void {
    var stream_index: u64 = 0;
    for (values) |v| {
        gammaEncodeIter(v, stream, &stream_index);
    }
}

// returns how many bytes the elias gamma encoded stream would occupy
pub fn gammaEncodeResultSize(values: []const u64) u64 {
    var encoded_size: u64 = 0;
    
    for (values) |v| {
        const gres = gammaEncodeSingle(v);
        encoded_size += 1 + (gres.leading_zeros * 2);
    }
    
    // convert from bit count to byte count
    encoded_size += 7; // ensure that iff encoded_size % 8 > 0 we add another byte
    encoded_size /= 8;
    
    return encoded_size;
}

pub fn gammaDecodeIter(stream: []const u8, stream_index: *u64) u64 {
    var value: u64 = 0;
    
    const start = stream_index.*;
    var zero_counter: u64 = 0;
    // count 0s until first 1
    while (0 == bitIdxRead(stream, start + zero_counter)) {
        zero_counter += 1;
    }
    
    // eat the 1 and as many bits after it as we've seen zeros before
    value |= @as(u64, bitIdxRead(stream, start + zero_counter)) << @truncate(zero_counter);
    for (0..zero_counter) |i| {
        const idx = i + 1 + zero_counter;
        const v = bitIdxRead(stream, start + idx);
        value |= @as(u64, @intCast(v)) << @truncate(i);
    }
    
    stream_index.* = start + zero_counter + 1 + zero_counter;
    return value;
}

pub fn gammaDecodeSlice(stream: []const u8, out: []u64) void {
    var stream_index: u64 = 0;
    const end: u64 = stream.len * @bitSizeOf(u8);
    var i: u64 = 0;
    while (i < out.len and (stream_index < end)) {
        if (onlyZerosFollow(stream, stream_index)) {
            break;
        }
        out[i] = gammaDecodeIter(stream, &stream_index);
        i += 1;
    }
}

// returns how many elements are encoded into stream
pub fn gammaDecodeElemCount(stream: []const u8) u64 {
    var stream_index: u64 = 0;
    const end: u64 = stream.len * @bitSizeOf(u8);
    var i: u64 = 0;
    while (stream_index < end) {
        if (onlyZerosFollow(stream, stream_index)) {
            break;
        }
        _ = gammaDecodeIter(stream, &stream_index);
        i += 1;
    }
    return i;
}

test "zigzag encode -> decode works" {
    var prng = std.Random.DefaultPrng.init(@bitCast(std.time.milliTimestamp()));
    const random = prng.random();
    const maximum = (@as(i64, 1) << 62) - 1;
    const minimum = -maximum;
    for (0..100000) |_| {
        const v = random.intRangeAtMost(i64, minimum, maximum);
        try testing.expectEqual(v, zigzagDecode(zigzagEncode(v)));
    }
}

test "zigzag encoding of trivial values produces correct result" {
    const Tst = struct { in: i64, out: u64 };
    const result_tbl = [_]Tst {
        .{ .in = 0, .out = 0 },
        .{ .in = -1, .out = 1 },
        .{ .in = 1, .out = 2 },
        .{ .in = -2, .out = 3 },
        .{ .in = 2, .out = 4 },
        .{ .in = -3, .out = 5 },
        .{ .in = 3, .out = 6 },
    };
    for (result_tbl) |tst| {
        try testing.expectEqual(tst.out, zigzagEncode(tst.in));
    }
}

test "elias gamma encode of trivial numbers works correctly" {
    const result_tbl = [_]GammaResult {
    .{ .leading_zeros = 0, .trailing_part = 0 }, // 1
    .{ .leading_zeros = 1, .trailing_part = 0 }, // 2
    .{ .leading_zeros = 1, .trailing_part = 1 }, // 3
    .{ .leading_zeros = 2, .trailing_part = 0 }, // 4
    .{ .leading_zeros = 2, .trailing_part = 1 }, // 5
    .{ .leading_zeros = 2, .trailing_part = 2 }, // 6
    .{ .leading_zeros = 2, .trailing_part = 3 }, // 7
    .{ .leading_zeros = 3, .trailing_part = 0 }, // 8
    .{ .leading_zeros = 3, .trailing_part = 1 }, // 9
    .{ .leading_zeros = 3, .trailing_part = 2 }, // 10
    .{ .leading_zeros = 3, .trailing_part = 3 }, // 11
    .{ .leading_zeros = 3, .trailing_part = 4 }, // 12
    .{ .leading_zeros = 3, .trailing_part = 5 }, // 13
    .{ .leading_zeros = 3, .trailing_part = 6 }, // 14
    .{ .leading_zeros = 3, .trailing_part = 7 }, // 15
    .{ .leading_zeros = 4, .trailing_part = 0 }, // 16
    };
    for (result_tbl, 0..) |expected, i| {
        const gres = gammaEncodeSingle(i + 1);
        try testing.expectEqual(expected, gres);
    }
}

test "elias gamma result size for trivial numbers is correct" {
    for (1..32) |i| {
        const v = [1]u64 { i };
        if (i >= 1 and i < 15) {
            try testing.expectEqual(1, gammaEncodeResultSize(&v));
        } else if (i >= 16) {
            try testing.expectEqual(2, gammaEncodeResultSize(&v));
        }
    }
}

test "elias gamma encode/decode iter of trivial numbers works correctly" {
    const data = [_][2]u8 {
        // { in, out }
        [2]u8 {  1, 0b1 },
        [2]u8 {  2, 0b010 },
        [2]u8 {  3, 0b110 },
        [2]u8 {  4, 0b00100 },
        [2]u8 {  5, 0b01100 },
        [2]u8 {  6, 0b10100 },
        [2]u8 {  7, 0b11100 },
        [2]u8 {  8, 0b0001000 },
        [2]u8 {  9, 0b0011000 },
        [2]u8 { 10, 0b0101000 },
        [2]u8 { 11, 0b0111000 },
        [2]u8 { 12, 0b1001000 },
        [2]u8 { 13, 0b1011000 },
        [2]u8 { 14, 0b1101000 },
        [2]u8 { 15, 0b1111000 },
    };
    for (data) |d| {
        const n: u64 = @intCast(d[0]);
        var buffer = [8]u8 { 0, 0, 0, 0, 0, 0, 0, 0 };
        var index: u64 = 0;
        gammaEncodeIter(n, &buffer, &index);
        try testing.expectEqual(d[1], buffer[0]);
        index = 0;
        const res = gammaDecodeIter(&buffer, &index);
        try testing.expectEqual(d[0], @as(u8, @truncate(res)));
    }
}

test "elias gamma encode into decode" {
    var prng = std.Random.DefaultPrng.init(@bitCast(std.time.milliTimestamp()));
    const random = prng.random();
    for (1..1000) |inputs_length| {
        var inputs = try std.ArrayList(u64).initCapacity(testing.allocator, inputs_length);
        defer inputs.deinit();
        try inputs.appendNTimes(0, inputs_length);
        for (0..inputs_length) |i| {
            inputs.items[i] = random.intRangeAtMost(u64, 1, @as(u64, 1) << 63);
        }

        const encode_result_length = gammaEncodeResultSize(inputs.items[0..inputs_length]);
        var encode_result = try std.ArrayList(u8).initCapacity(testing.allocator, encode_result_length);
        defer encode_result.deinit();
        try encode_result.appendNTimes(0, encode_result_length);

        gammaEncodeSlice(inputs.items[0..inputs_length], encode_result.items[0..encode_result_length]);

        const encoded_count = gammaDecodeElemCount(encode_result.items[0..encode_result_length]);
        try testing.expectEqual(inputs_length, encoded_count);

        var decode_result = try std.ArrayList(u64).initCapacity(testing.allocator, encoded_count);
        defer decode_result.deinit();
        try decode_result.appendNTimes(0, encoded_count);

        gammaDecodeSlice(encode_result.items[0..encode_result_length], decode_result.items[0..encoded_count]);

        for (0..encoded_count) |i| {
            try testing.expectEqual(inputs.items[i], decode_result.items[i]);
        }
    }
}

