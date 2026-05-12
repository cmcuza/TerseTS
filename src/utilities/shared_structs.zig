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
const assert = std.debug.assert;
const Allocator = mem.Allocator;
const ArrayList = std.ArrayList;
const HashMap = std.HashMap;
const Reader = std.Io.Reader;

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

/// `Point` is a point represented by `index` and `value`. `index` is of datatype `index_type`.
fn Point(comptime index_type: type) type {
    return struct { index: index_type, value: f64 };
}

/// `SegmentMetadata` stores the information about an approximated segment during the execution
/// of Sim-Piece and Mix-Piece. It stores the starting index of the segment in `start_index`, the
/// `interception` point used to create the linear function approximation, and the slopes of
/// the upper and lower bounds that constraint the linear approximation in that segment.
pub const SegmentMetadata = struct {
    start_index: usize,
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

/// Creates a `BitWriter` which allows for writing bits with Big Endian. The code for `BitWriter` was
/// originally copied from Zig's MIT licensed standard library as suggested in GitHub PR 24614 since
/// it was removed in Zig 0.15.1. The code has since been modified to simplify its use in TerseTS.
pub const BitWriter = struct {
    allocator: Allocator,
    bytes: *ArrayList(u8),
    bits: u8 = 0,
    count: u4 = 0,

    const Self = @This();

    const low_bit_mask = [9]u8{
        0b00000000,
        0b00000001,
        0b00000011,
        0b00000111,
        0b00001111,
        0b00011111,
        0b00111111,
        0b01111111,
        0b11111111,
    };

    /// Initialize an empty `BitWriter`.
    pub fn init(allocator: Allocator, bytes: *ArrayList(u8)) !Self {
        return .{ .allocator = allocator, .bytes = bytes, .bits = 0, .count = 0 };
    }

    /// Write the specified number of bits to the writer from the least significant bits of the
    /// input `value`. Bits will only be written to the writer when there are enough to fill a byte.
    pub fn writeBits(self: *Self, value: anytype, num: u16) !void {
        const T = @TypeOf(value);
        const UT = std.meta.Int(.unsigned, @bitSizeOf(T));
        const U = if (@bitSizeOf(T) < 8) u8 else UT;

        const in: U = @as(UT, @bitCast(value));
        var in_count: u16 = num;

        if (self.count > 0) {
            // if we can't fill the buffer, add what we have.
            const bits_free = 8 - self.count;
            if (num < bits_free) {
                self.addBits(@truncate(in), @intCast(num));
                return;
            }

            // Finish filling the buffer and flush it.
            if (num == bits_free) {
                self.addBits(@truncate(in), @intCast(num));
                return self.flushBits();
            }

            const bits = in >> @intCast(in_count - bits_free);
            self.addBits(@truncate(bits), bits_free);

            in_count -= bits_free;
            try self.flushBits();
        }

        // Write full bytes while we can.
        const full_bytes_left = in_count / 8;
        for (0..full_bytes_left) |_| {
            const bits = in >> @intCast(in_count - 8);
            try self.bytes.append(self.allocator, @truncate(bits));
            in_count -= 8;
        }

        // Save the remaining bits in the buffer.
        self.addBits(@truncate(in), @intCast(in_count));
    }

    /// Convenience function for adding bits to the buffer.
    fn addBits(self: *Self, bits: u8, num: u4) void {
        if (num == 8) self.bits = bits else {
            self.bits <<= @intCast(num);
            self.bits |= bits & low_bit_mask[num];
        }
        self.count += num;
    }

    /// Flush any remaining bits to the writer, filling
    /// unused bits with 0s.
    pub fn flushBits(self: *Self) !void {
        if (self.count == 0) return;
        self.bits <<= @intCast(8 - self.count);
        try self.bytes.append(self.allocator, self.bits);
        self.bits = 0;
        self.count = 0;
    }
};

/// Creates a `BitReader` which allows for reading bits in Big Endian. The code for `BitReader` was
/// originally copied from Zig's MIT licensed standard library as suggested in GitHub PR 24614 since
/// it was removed in Zig 0.15.1. The code has since been modified to simplify its use in TerseTS.
pub const BitReader = struct {
    bytes: Reader,
    bits: u8 = 0,
    count: u4 = 0,

    const Self = @This();

    const low_bit_mask = [9]u8{
        0b00000000,
        0b00000001,
        0b00000011,
        0b00000111,
        0b00001111,
        0b00011111,
        0b00111111,
        0b01111111,
        0b11111111,
    };

    fn Bits(comptime T: type) type {
        return struct {
            T,
            u16,
        };
    }

    fn initBits(comptime T: type, out: anytype, num: u16) Bits(T) {
        const UT = std.meta.Int(.unsigned, @bitSizeOf(T));
        return .{
            @bitCast(@as(UT, @intCast(out))),
            num,
        };
    }

    /// Initialize an empty `BitReader`.
    pub fn init(bytes: Reader) Self {
        return .{ .bytes = bytes };
    }

    /// Reads `bits` bits from the reader and returns a specified type
    ///  containing them in the least significant end, returning an error if the
    ///  specified number of bits could not be read.
    pub fn readBitsNoEof(self: *Self, comptime T: type, num: u16) !T {
        const b, const c = try self.readBitsTuple(T, num);
        if (c < num) return error.EndOfStream;
        return b;
    }

    /// Reads `bits` bits from the reader and returns a specified type
    /// containing them in the least significant end. The number of bits successfully
    /// read is placed in `out_bits`, as reaching the end of the stream is not an error.
    pub fn readBits(self: *Self, comptime T: type, num: u16, out_bits: *u16) !T {
        const b, const c = try self.readBitsTuple(T, num);
        out_bits.* = c;
        return b;
    }

    /// Reads `bits` bits from the reader and returns a tuple of the specified type
    /// containing them in the least significant end, and the number of bits successfully
    /// read. Reaching the end of the stream is not an error.
    pub fn readBitsTuple(self: *Self, comptime T: type, num: u16) !Bits(T) {
        const UT = std.meta.Int(.unsigned, @bitSizeOf(T));
        const U = if (@bitSizeOf(T) < 8) u8 else UT; //it is a pain to work with <u8

        // Dump any bits in our buffer first.
        if (num <= self.count) return initBits(T, self.removeBits(@intCast(num)), num);

        var out_count: u16 = self.count;
        var out: U = self.removeBits(self.count);

        // Grab all the full bytes we need and put their bits where they belong.
        const full_bytes_left = (num - out_count) / 8;

        for (0..full_bytes_left) |_| {
            const byte = self.bytes.takeByte() catch |err| switch (err) {
                error.EndOfStream => return initBits(T, out, out_count),
                else => |e| return e,
            };

            if (U == u8) out = 0 else out <<= 8; //shifting u8 by 8 is illegal in Zig
            out |= byte;
            out_count += 8;
        }

        const bits_left = num - out_count;
        const keep = 8 - bits_left;

        if (bits_left == 0) return initBits(T, out, out_count);

        const final_byte = self.bytes.takeByte() catch |err| switch (err) {
            error.EndOfStream => return initBits(T, out, out_count),
            else => |e| return e,
        };

        out <<= @intCast(bits_left);
        out |= final_byte >> @intCast(keep);
        self.bits = final_byte & low_bit_mask[keep];

        self.count = @intCast(keep);
        return initBits(T, out, num);
    }

    // Convenience function for removing bits.
    fn removeBits(self: *Self, num: u4) u8 {
        if (num == 8) {
            self.count = 0;
            return self.bits;
        }

        const keep = self.count - num;
        const bits = self.bits >> @intCast(keep);
        self.bits &= low_bit_mask[keep];

        self.count = keep;
        return bits;
    }

    pub fn alignToByte(self: *Self) void {
        self.bits = 0;
        self.count = 0;
    }
};
