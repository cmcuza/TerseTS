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

/// Creates a `BitWriter` which allows for writing bits to a `Writer`. `BitWriter` was removed in
/// Zig 0.15.1. Thus, it was copied from Zig's standard library as suggested in GitHub PR 24614
/// "Sorry, you will have to copy the old code into your application, or use a third party package."
/// Zig's standard library is released under the MIT license. A new `BitWriter` was added for flate
/// after Zig 0.15.1 was released, however, it currently only supports `u56` and not `u64`. Thus,
/// the old `BitWriter` is used despite its use of the old `Writer` interface. To make it as
/// explicit as possible that this code is copied from Zig's standard library, no attempt to make it
/// consistent with TerseTS has been made.
pub fn BitWriter(comptime endian: std.builtin.Endian, comptime Writer: type) type {
    return struct {
        writer: Writer,
        bits: u8 = 0,
        count: u4 = 0,

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

        /// Write the specified number of bits to the writer from the least significant bits of
        ///  the specified value. Bits will only be written to the writer when there
        ///  are enough to fill a byte.
        pub fn writeBits(self: *@This(), value: anytype, num: u16) !void {
            const T = @TypeOf(value);
            const UT = std.meta.Int(.unsigned, @bitSizeOf(T));
            const U = if (@bitSizeOf(T) < 8) u8 else UT; //<u8 is a pain to work with

            var in: U = @as(UT, @bitCast(value));
            var in_count: u16 = num;

            if (self.count > 0) {
                //if we can't fill the buffer, add what we have
                const bits_free = 8 - self.count;
                if (num < bits_free) {
                    self.addBits(@truncate(in), @intCast(num));
                    return;
                }

                //finish filling the buffer and flush it
                if (num == bits_free) {
                    self.addBits(@truncate(in), @intCast(num));
                    return self.flushBits();
                }

                switch (endian) {
                    .big => {
                        const bits = in >> @intCast(in_count - bits_free);
                        self.addBits(@truncate(bits), bits_free);
                    },
                    .little => {
                        self.addBits(@truncate(in), bits_free);
                        in >>= @intCast(bits_free);
                    },
                }
                in_count -= bits_free;
                try self.flushBits();
            }

            //write full bytes while we can
            const full_bytes_left = in_count / 8;
            for (0..full_bytes_left) |_| {
                switch (endian) {
                    .big => {
                        const bits = in >> @intCast(in_count - 8);
                        try self.writer.writeByte(@truncate(bits));
                    },
                    .little => {
                        try self.writer.writeByte(@truncate(in));
                        if (U == u8) in = 0 else in >>= 8;
                    },
                }
                in_count -= 8;
            }

            //save the remaining bits in the buffer
            self.addBits(@truncate(in), @intCast(in_count));
        }

        //convenience funciton for adding bits to the buffer
        //in the appropriate position based on endianess
        fn addBits(self: *@This(), bits: u8, num: u4) void {
            if (num == 8) self.bits = bits else switch (endian) {
                .big => {
                    self.bits <<= @intCast(num);
                    self.bits |= bits & low_bit_mask[num];
                },
                .little => {
                    const pos = bits << @intCast(self.count);
                    self.bits |= pos;
                },
            }
            self.count += num;
        }

        /// Flush any remaining bits to the writer, filling
        /// unused bits with 0s.
        pub fn flushBits(self: *@This()) !void {
            if (self.count == 0) return;
            if (endian == .big) self.bits <<= @intCast(8 - self.count);
            try self.writer.writeByte(self.bits);
            self.bits = 0;
            self.count = 0;
        }
    };
}

// Helper function to create a `BitWriter` with a specific type. `BitWriter` was removed in Zig
// 0.15.1. Thus, it was copied from Zig's standard library as suggested in GitHub PR 24614 "Sorry,
// you will have to copy the old code into your application, or use a third party package." Zig's
// standard library is released under the MIT license. To make it as explicit as possible that this
// code is copied from Zig's standard library, no attempt to make it consistent with TerseTS has
// been made.
pub fn bitWriter(comptime endian: std.builtin.Endian, writer: anytype) BitWriter(endian, @TypeOf(writer)) {
    return .{ .writer = writer };
}

/// Creates a `BitReader` which allows for reading bits from a `Reader`. `BitReader` was removed in
/// Zig 0.15.1. Thus, it was copied from Zig's standard library as suggested in GitHub PR 24614
/// "Sorry, you will have to copy the old code into your application, or use a third party package."
/// Zig's standard library is released under the MIT license. A new `BitReader` has not yet been
/// added to master. To make it as explicit as possible that this code is copied from Zig's standard
/// library, no attempt to make it consistent with TerseTS has been made.
fn BitReader(comptime endian: std.builtin.Endian, comptime Reader: type) type {
    return struct {
        reader: Reader,
        bits: u8 = 0,
        count: u4 = 0,

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

        /// Reads `bits` bits from the reader and returns a specified type
        ///  containing them in the least significant end, returning an error if the
        ///  specified number of bits could not be read.
        pub fn readBitsNoEof(self: *@This(), comptime T: type, num: u16) !T {
            const b, const c = try self.readBitsTuple(T, num);
            if (c < num) return error.EndOfStream;
            return b;
        }

        /// Reads `bits` bits from the reader and returns a specified type
        ///  containing them in the least significant end. The number of bits successfully
        ///  read is placed in `out_bits`, as reaching the end of the stream is not an error.
        pub fn readBits(self: *@This(), comptime T: type, num: u16, out_bits: *u16) !T {
            const b, const c = try self.readBitsTuple(T, num);
            out_bits.* = c;
            return b;
        }

        /// Reads `bits` bits from the reader and returns a tuple of the specified type
        ///  containing them in the least significant end, and the number of bits successfully
        ///  read. Reaching the end of the stream is not an error.
        pub fn readBitsTuple(self: *@This(), comptime T: type, num: u16) !Bits(T) {
            const UT = std.meta.Int(.unsigned, @bitSizeOf(T));
            const U = if (@bitSizeOf(T) < 8) u8 else UT; //it is a pain to work with <u8

            //dump any bits in our buffer first
            if (num <= self.count) return initBits(T, self.removeBits(@intCast(num)), num);

            var out_count: u16 = self.count;
            var out: U = self.removeBits(self.count);

            //grab all the full bytes we need and put their
            //bits where they belong
            const full_bytes_left = (num - out_count) / 8;

            for (0..full_bytes_left) |_| {
                const byte = self.reader.readByte() catch |err| switch (err) {
                    error.EndOfStream => return initBits(T, out, out_count),
                    else => |e| return e,
                };

                switch (endian) {
                    .big => {
                        if (U == u8) out = 0 else out <<= 8; //shifting u8 by 8 is illegal in Zig
                        out |= byte;
                    },
                    .little => {
                        const pos = @as(U, byte) << @intCast(out_count);
                        out |= pos;
                    },
                }
                out_count += 8;
            }

            const bits_left = num - out_count;
            const keep = 8 - bits_left;

            if (bits_left == 0) return initBits(T, out, out_count);

            const final_byte = self.reader.readByte() catch |err| switch (err) {
                error.EndOfStream => return initBits(T, out, out_count),
                else => |e| return e,
            };

            switch (endian) {
                .big => {
                    out <<= @intCast(bits_left);
                    out |= final_byte >> @intCast(keep);
                    self.bits = final_byte & low_bit_mask[keep];
                },
                .little => {
                    const pos = @as(U, final_byte & low_bit_mask[bits_left]) << @intCast(out_count);
                    out |= pos;
                    self.bits = final_byte >> @intCast(bits_left);
                },
            }

            self.count = @intCast(keep);
            return initBits(T, out, num);
        }

        //convenience function for removing bits from
        //the appropriate part of the buffer based on
        //endianess.
        fn removeBits(self: *@This(), num: u4) u8 {
            if (num == 8) {
                self.count = 0;
                return self.bits;
            }

            const keep = self.count - num;
            const bits = switch (endian) {
                .big => self.bits >> @intCast(keep),
                .little => self.bits & low_bit_mask[num],
            };
            switch (endian) {
                .big => self.bits &= low_bit_mask[keep],
                .little => self.bits >>= @intCast(num),
            }

            self.count = keep;
            return bits;
        }

        pub fn alignToByte(self: *@This()) void {
            self.bits = 0;
            self.count = 0;
        }
    };
}

// Helper function to create a `BitReader` with a specific type. `BitReader` was removed in Zig
// 0.15.1. Thus, it was copied from Zig's standard library as suggested in GitHub PR 24614 "Sorry,
// you will have to copy the old code into your application, or use a third party package." Zig's
// standard library is released under the MIT license. To make it as explicit as possible that this
// code is copied from Zig's standard library, no attempt to make it consistent with TerseTS has
// been made.
pub fn bitReader(comptime endian: std.builtin.Endian, reader: anytype) BitReader(endian, @TypeOf(reader)) {
    return .{ .reader = reader };
}
