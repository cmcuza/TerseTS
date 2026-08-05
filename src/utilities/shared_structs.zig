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

/// Number of bits in an IEEE-754 `f64`, the value type of every TerseTS time series.
pub const bits_per_value = @bitSizeOf(f64);

/// Quantized leading-zero counts from the Chimp paper (`leadingRound` in the authors' Java
/// implementation). `@clz(xor)` is rounded down to one of these eight boundaries so the chosen
/// bucket index fits in three bits. Shared by the Chimp and Elf family of codecs.
pub const leading_zero_buckets = [_]u6{ 0, 8, 12, 16, 18, 20, 22, 24 };

/// Bit width of a `leading_zero_buckets` index in Chimp-family streams.
pub const leading_zero_bucket_bits = std.math.log2_int(usize, leading_zero_buckets.len);

/// Maps an exact leading-zero count, as returned by `@clz` on a 64-bit XOR, to the index of the
/// largest `leading_zero_buckets` entry that does not exceed it. Read through
/// `shared_functions.leadingZeroBucketIndex`.
pub const leading_zero_bucket_index = [_]u3{
    0, 0, 0, 0, 0, 0, 0, 0, // 0..7   -> bucket 0 (0 leading zeros).
    1, 1, 1, 1, //             8..11  -> bucket 1 (8).
    2, 2, 2, 2, //             12..15 -> bucket 2 (12).
    3, 3, //                   16..17 -> bucket 3 (16).
    4, 4, //                   18..19 -> bucket 4 (18).
    5, 5, //                   20..21 -> bucket 5 (20).
    6, 6, //                   22..23 -> bucket 6 (22).
    7, 7, 7, 7, 7, 7, 7, 7, // 24..63 -> bucket 7 (24).
    7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7,
};

/// Chimp-family encoders place this where a leading-zero bucket would go when there is no bucket
/// the next value may reuse, forcing the next value onto the "store a new bucket" marker. Any
/// value no real bucket can equal works; 65 matches the reference implementation.
pub const no_reusable_leading_bucket: u7 = 65;

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

/// A `BitWriter` variant for hot encoders: it buffers bits in a 64-bit accumulator and flushes
/// eight bytes at a time with a single big-endian store, rather than appending one byte at a time.
/// Output is byte-identical to `BitWriter` (Big Endian, MSB first) and the API is the same.
pub const BulkBitWriter = struct {
    allocator: Allocator,
    bytes: *ArrayList(u8),
    // Buffered bits in MSB-first order, occupying the low `count` positions of `acc`.
    acc: u64 = 0,
    count: u6 = 0,

    const Self = @This();

    /// Initialize an empty `BulkBitWriter`.
    pub fn init(allocator: Allocator, bytes: *ArrayList(u8)) !Self {
        return .{ .allocator = allocator, .bytes = bytes };
    }

    /// Write the `num` least-significant bits of `value`, most-significant first.
    pub fn writeBits(self: *Self, value: anytype, num: u16) !void {
        const unsigned = std.meta.Int(.unsigned, @bitSizeOf(@TypeOf(value)));
        const n: u32 = num;
        const bits: u64 = @as(u64, @as(unsigned, @bitCast(value))) & lowMask(n);
        const room = 64 - @as(u32, self.count); // free positions before `acc` is full
        if (n < room) {
            self.acc = (self.acc << @intCast(n)) | bits;
            self.count += @intCast(n);
            return;
        }
        // Fill `acc` with the top `room` bits of `value`, flush 8 bytes, keep the remainder.
        const rest = n - room;
        self.acc = if (room == 64) bits else (self.acc << @intCast(room)) | (bits >> @intCast(rest));
        var buffer: [8]u8 = undefined;
        std.mem.writeInt(u64, &buffer, self.acc, .big);
        try self.bytes.appendSlice(self.allocator, &buffer);
        self.acc = if (rest == 0) 0 else bits & lowMask(rest);
        self.count = @intCast(rest);
    }

    /// Flush any buffered bits, padding the final byte with trailing zeros.
    pub fn flushBits(self: *Self) !void {
        if (self.count == 0) return;
        const num_bytes = (@as(u32, self.count) + 7) / 8;
        const aligned = self.acc << @intCast(num_bytes * 8 - self.count);
        var index = num_bytes;
        while (index > 0) {
            index -= 1;
            try self.bytes.append(self.allocator, @truncate(aligned >> @intCast(index * 8)));
        }
        self.acc = 0;
        self.count = 0;
    }

    /// Mask selecting the low `n` bits, for `n` in 0..64.
    fn lowMask(n: u32) u64 {
        return if (n >= 64) ~@as(u64, 0) else (@as(u64, 1) << @intCast(n)) - 1;
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

/// A `BitReader` variant for hot decoders: it reads straight from a byte slice (no stream wrapper)
/// and buffers up to 64 bits in an accumulator, serving many reads between byte loads instead of
/// taking one byte at a time. Big Endian / MSB first, matching `BitWriter` and `BulkBitWriter`.
pub const BulkBitReader = struct {
    bytes: []const u8,
    pos: usize = 0,
    // The `count` valid bits live right-aligned in the low bits of `acc`, in stream order: the next
    // bit to serve is the most significant of the valid region.
    acc: u64 = 0,
    count: u16 = 0,

    const Self = @This();

    /// Wrap `bytes` (the remaining compressed stream) for reading.
    pub fn init(bytes: []const u8) Self {
        return .{ .bytes = bytes };
    }

    /// Read `num` bits (0..64), most-significant first, into the low bits of `T`. Returns
    /// `error.EndOfStream` if the slice runs out before `num` bits are available.
    pub fn readBitsNoEof(self: *Self, comptime T: type, num: u16) error{EndOfStream}!T {
        if (num == 0) return 0;
        // Refill while there is room for another byte (count <= 56) and we still need bits.
        while (self.count < num and self.count <= 56) {
            if (self.pos >= self.bytes.len) return error.EndOfStream;
            self.acc = (self.acc << 8) | self.bytes[self.pos];
            self.pos += 1;
            self.count += 8;
        }
        if (self.count >= num) {
            const shift = self.count - num;
            const result = (self.acc >> @intCast(shift)) & lowMask(num);
            self.count = shift;
            self.acc &= lowMask(shift);
            return @intCast(result);
        }
        // Boundary: a >56-bit read with the accumulator just short and no room to refill a full
        // byte. Combine the buffered bits with the top `needed` bits of the next byte.
        const needed = num - self.count;
        if (self.pos >= self.bytes.len) return error.EndOfStream;
        const next: u64 = self.bytes[self.pos];
        self.pos += 1;
        const result = (self.acc << @intCast(needed)) | (next >> @intCast(8 - needed));
        self.count = 8 - needed;
        self.acc = next & lowMask(self.count);
        return @intCast(result);
    }

    /// Mask selecting the low `n` bits, for `n` in 0..64.
    fn lowMask(n: u16) u64 {
        return if (n >= 64) ~@as(u64, 0) else (@as(u64, 1) << @intCast(n)) - 1;
    }
};
