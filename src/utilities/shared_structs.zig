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

//! Contains all shared variables and structures used across TerseTS.

const std = @import("std");
const mem = std.mem;
const testing = std.testing;

const ArrayListUnmanaged = std.ArrayListUnmanaged;
const HashMap = std.HashMap;

/// Margin to adjust the error bound for numerical stability. Reducing the error bound by this
/// margin ensures that all the elements of the decompressed time series are within the error bound
/// with respect to the uncompressed time series.
pub const ErrorBoundMargin: f32 = 1e-7;

/// `Point` with discrete `time` axis.
pub const DiscretePoint = Point(usize);

/// `Point` with continous `time` axis.
pub const ContinousPoint = Point(f64);

/// ParameterSpacePoint represents a point in a generic 2D parameter space.
pub const ParameterSpacePoint = struct {
    x_axis: f64,
    y_axis: f64,
};

/// Absolute and relative tolerances for floating-point comparisons.
pub const ABS_EPS: f64 = 1e-12;
pub const REL_EPS: f64 = 1e-15;

/// `Segment` models a straight line segment from `start_point` to `end_point`. All segments
/// have discrete points.
pub const Segment = struct {
    start_point: DiscretePoint,
    end_point: DiscretePoint,
};

/// Linear function of the form y = slope*x+intercept.
pub const LinearFunction = struct {
    slope: f64,
    intercept: f64,
};

pub const BitStream = struct {
    buffer: ArrayListUnmanaged(u8),
    allocator: mem.Allocator,
    fill: u64,

    pub fn make(allocator: mem.Allocator) !BitStream {
        return BitStream {
            .buffer = try ArrayListUnmanaged(u8).initCapacity(allocator, 1),
            .allocator = allocator,
            .fill = 0,
        };
    }

    // "takes ownership" of the array list
    pub fn makeFromArrayList(al: *std.ArrayList(u8), fill: u64) BitStream {
        return BitStream {
            .buffer = al.moveToUnmanaged(),
            .allocator = al.allocator,
            .fill = fill,
        };
    }

    pub fn makeFromArrayUnmanaged(arr: []u8) BitStream {
        return BitStream {
            .buffer = ArrayListUnmanaged(u8).initBuffer(arr),
            // NOTE (sio): Zig does not have a panic allocator, and does not allow easy construction of an allocator that fails. So this is the only viable way to actually _maybe_ fail.
            .allocator = std.mem.Allocator { .ptr = &arr[0], .vtable = std.heap.page_allocator.vtable },
            .fill = 0,
        };
    }

    pub fn deinit(self: *BitStream) void {
        self.buffer.deinit(self.allocator);
    }

    pub fn empty(self: *const BitStream) bool {
        return self.fill == 0;
    }

    // for usage as an output stream:
    // initialize with an arraylist to be overwritten
    // set fill to 0
    // fill will be advanced and arraylist size will be increased with zeroes as necessary to fit contents

    // write up to 64 bits, incrementing fill
    pub fn write(self: *BitStream, content: u64, len: u32) !void {
        var j: usize = len;
        while (j > 0) {
            const i = j - 1;
            const v = content >> @truncate(i);
            try self.writeBit(@truncate(v));
            j -= 1;
        }
    }

    pub fn writeBit(self: *BitStream, content: u1) !void {
        if (self.buffer.items.len <= ((self.fill + 1) * 8)) {
            try self.buffer.append(self.allocator, 0);
        }
        self.set(self.fill, content);
        self.fill += 1;
    }

    // for usage as an input stream:
    // initialize with an arraylist to be read from
    // set fill to 0
    // fill will be advanced for every bit read until end-of-array is hit

    // read up to 64 bits, incrementing fill
    pub fn read(self: *BitStream, len: u32) u64 {
        var res: u64 = 0;
        var j: usize = len;
        while (j > 0) {
            const i = j - 1;
            res |= (@as(u64, @intCast(self.readBit())) << @truncate(i));
            j -= 1;
        }
        return res;
    }

    pub fn canRead(self: *BitStream, len: u32) bool {
        return self.fill + len < self.buffer.items.len;
    }

    pub fn readBit(self: *BitStream) u1 {
        const t = self.get(self.fill);
        self.fill += 1;
        return t;
    }

    // read len bits off the end and decrement fill
    pub fn pop(self: *BitStream, len: u32) u64 {
        std.debug.assert(self.fill >= len);
        self.fill -= len;
        const res = self.read(len);
        self.fill -= len;
        return res;
    }


    // for usage as a plain bit array

    pub fn get(self: *BitStream, index: u64) u1 {
        return @truncate((self.buffer.items[index >> 3] >> @truncate(index & 0x7)) & 1);
    }

    pub fn set(self: *BitStream, index: u64, value: u1) void {
        const byte_idx = index >> 3;
        const bit_idx = index & 0x7;

        const tmp = self.buffer.items[byte_idx];
        const clear = tmp & ~(@as(u8, 1) << @truncate(bit_idx));
        const set_bit = clear | (@as(u8, @intCast(value)) << @truncate(bit_idx));
        self.buffer.items[byte_idx] = set_bit;
    }
};

/// `Point` is a point represented by `time` and `value`. `time` is of datatype `time_type`.
fn Point(comptime time_type: type) type {
    return struct { time: time_type, value: f64 };
}

/// `SegmentMetadata` stores the information about an approximated segment during the execution
/// of Sim-Piece and Mix-Piece. It stores the starting time of the segment in `start_time`, the
/// `interception` point used to create the linear function approximation, and the slopes of
/// the upper and lower bounds that constraint the linear approximation in that segment.
pub const SegmentMetadata = struct {
    start_time: usize,
    intercept: f64,
    upper_bound_slope: f64,
    lower_bound_slope: f64,
};

/// `HashF64Context` provides context for hashing and comparing `f64` values for use in `HashMap`.
/// This context is essential when using `f64` as keys in a `HashMap`. It defines how the keys are
/// hashed and compared for equality.
pub const HashF64Context = struct {
    /// Hashes an `f64` `value` by bitcasting it to `u64`.
    pub fn hash(_: HashF64Context, value: f64) u64 {
        return @as(u64, @bitCast(value));
    }
    /// Compares two `f64` values for equality.
    pub fn eql(_: HashF64Context, value_one: f64, value_two: f64) bool {
        return value_one == value_two;
    }
};

/// Returns a HashMap with key type f64 and `value_type` given by the user.
pub fn HashMapf64(comptime value_type: type) type {
    return HashMap(f64, value_type, HashF64Context, std.hash_map.default_max_load_percentage);
}
test "empty BitStream is recognized as empty" {
    var d = [1]u8 { 0 };
    var bs = BitStream.makeFromArrayUnmanaged(&d);
    bs.fill = 0;
    try testing.expect(bs.empty());
}

test "can read and write integers of various sizes to/from bitstream" {
    var bs = try BitStream.make(testing.allocator);
    defer bs.deinit();
    var prng = std.Random.DefaultPrng.init(@bitCast(std.time.milliTimestamp()));
    const random = prng.random();
    for (1..65) |v1size| {
        var v1sz: u64 = std.math.maxInt(u64);
        if (v1size < 64) {
            v1sz = 1;
            v1sz <<= @truncate(v1size);
            v1sz -= 1;
        }
        const v1 = random.uintAtMost(u64, v1sz);
        try bs.write(v1, @truncate(v1size));
        const v1r = bs.pop(@truncate(v1size));
        try testing.expectEqual(v1, v1r);
    }
}


test "can read and write 2 successive integers of various sizes to/from bitstream" {
    var bs = try BitStream.make(testing.allocator);
    defer bs.deinit();
    var prng = std.Random.DefaultPrng.init(@bitCast(std.time.milliTimestamp()));
    const random = prng.random();
    for (1..65) |v1size| {
        var v1sz: u64 = std.math.maxInt(u64);
        if (v1size < 64) {
            v1sz = 1;
            v1sz <<= @truncate(v1size);
            v1sz -= 1;
        }
        const v1 = random.uintAtMost(u64, v1sz);
        try bs.write(v1, @truncate(v1size));
        for (1..65) |v2size| {
            var v2sz: u64 = std.math.maxInt(u64);
            if (v2size < 64) {
                v2sz = 1;
                v2sz <<= @truncate(v2size);
                v2sz -= 1;
            }
            const v2 = random.uintAtMost(u64, v2sz);
            try bs.write(v2, @truncate(v2size));
            try testing.expectEqual(bs.pop(@truncate(v2size)), v2);
        }
        try testing.expectEqual(bs.pop(@truncate(v1size)), v1);
    }
}
