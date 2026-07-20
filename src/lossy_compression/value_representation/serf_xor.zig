// Copyright 2026 TerseTS Contributors
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

//! Implementation of the SERF-XOR algorithm from the paper
//! "Li, Ruiyuan, Zechao Chen, Ruyun Lu, Xiaolong Xu, Guangchao Yang, Chao Chen, Jie Bao, and Yu Zheng.
//! Serf: Streaming Error-Bounded Floating-Point Compression.
//! ACM SIGMOD 2025.
//! https://doi.org/10.1145/3725353.
//! SerfXOR combines a Shifter (adds an integer offset so all values share one exponent and the
//! XORs gain leading zeros), an Approximator (picks bits within the error bound that share the
//! longest suffix with the previous value, maximizing trailing zeros), a three-case XOR encoder,
//! and adaptive leading/trailing-zero rules recomputed per window by the post-office solver.
//! The bit-level layout and the approximator follow the paper authors' reference C++
//! implementation (`serf_xor_compressor.cc`, `serf_xor_decompressor.cc`, `serf_utils_64.cc`).
//! Like the streaming reference, the stream is terminated with a NaN sentinel: this adaptation
//! stores a `[shift: f64]` header followed by one continuous bit stream and a final XOR-encoded
//! NaN value that marks the end, and both sides run the window rule update after every
//! `window_size`-th value.

const std = @import("std");
const math = std.math;
const mem = std.mem;
const testing = std.testing;
const ArrayList = std.ArrayList;
const Allocator = mem.Allocator;

const tersets = @import("../../tersets.zig");
const configuration = @import("../../configuration.zig");
const shared_functions = @import("../../utilities/shared_functions.zig");
const shared_structs = @import("../../utilities/shared_structs.zig");
const post_office_solver = @import("../../utilities/post_office_solver.zig");
const tester = @import("../../tester.zig");

const Error = tersets.Error;
const Method = tersets.Method;

/// Number of values per adaptive-rule window. After every `window_size`-th encoded value the
/// encoder decides whether to recompute the rounding rules and the decoder mirrors that decision
/// by reading the update flag.
const window_size = 1000;

/// Number of bits in an IEEE-754 `f64`, the width of every XOR-encoded value.
const bits_per_value = 64;

/// Bit pattern of `f64` 2.0, the initial "previous value" both codec sides start from, so the
/// very first value is XOR-encoded against a known constant like in the reference.
const initial_stored_value: u64 = @bitCast(@as(f64, 2.0));

/// Sentinel for `stored_leading_zeros`/`stored_trailing_zeros` before the first case `00` value.
/// Any value larger than 63 forces the first non-zero XOR into case `00`; the reference uses
/// `std::numeric_limits<int>::max()` for the same purpose.
const zeros_sentinel: u32 = 255;

comptime {
    // Real bounds are `u6` and never reach the sentinel, so case 1 stays unreachable until a
    // case 00 stores real bounds.
    std.debug.assert(zeros_sentinel > math.maxInt(u6));
}

/// Mask of the IEEE-754 `f64` sign bit.
const sign_bit_mask: u64 = 0x8000000000000000;

/// Mask selecting everything but the sign bit, i.e. the magnitude of an `f64` bit pattern.
const magnitude_mask: u64 = 0x7fffffffffffffff;

/// Initial encoder table rounding an exact leading-zero count down to its bucket boundary.
const initial_leading_round = [post_office_solver.table_size]u6{
    0,  0,  0,  0,  0,  0,  0,  0,
    8,  8,  8,  8,  12, 12, 12, 12,
    16, 16, 18, 18, 20, 20, 22, 22,
    24, 24, 24, 24, 24, 24, 24, 24,
    24, 24, 24, 24, 24, 24, 24, 24,
    24, 24, 24, 24, 24, 24, 24, 24,
    24, 24, 24, 24, 24, 24, 24, 24,
    24, 24, 24, 24, 24, 24, 24, 24,
};

/// Initial encoder table mapping an exact leading-zero count to the code written to the stream.
const initial_leading_representation = [post_office_solver.table_size]u6{
    0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 2, 2, 2, 2,
    3, 3, 4, 4, 5, 5, 6, 6,
    7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7,
};

/// Initial encoder table rounding an exact trailing-zero count down to its bucket boundary.
const initial_trailing_round = [post_office_solver.table_size]u6{
    0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  22, 22,
    22, 22, 22, 22, 28, 28, 28, 28,
    32, 32, 32, 32, 36, 36, 36, 36,
    40, 40, 42, 42, 42, 42, 46, 46,
    46, 46, 46, 46, 46, 46, 46, 46,
    46, 46, 46, 46, 46, 46, 46, 46,
};

/// Initial encoder table mapping an exact trailing-zero count to the code written to the stream.
const initial_trailing_representation = [post_office_solver.table_size]u6{
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 1, 1,
    1, 1, 1, 1, 2, 2, 2, 2,
    3, 3, 3, 3, 4, 4, 4, 4,
    5, 5, 6, 6, 6, 6, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7,
};

/// Initial decoder table mapping a leading-zero code back to the zero count. Only the first
/// eight entries are meaningful initially; rule updates may rewrite up to `max_positions`.
const initial_leading_decode: [post_office_solver.table_size]u6 = buildDecodeTable(
    &[_]u6{ 0, 8, 12, 16, 18, 20, 22, 24 },
);

/// Initial decoder table mapping a trailing-zero code back to the zero count.
const initial_trailing_decode: [post_office_solver.table_size]u6 = buildDecodeTable(
    &[_]u6{ 0, 22, 28, 32, 36, 40, 42, 46 },
);

/// Build a zero-padded decoder table of `table_size` entries from the `entries` prefix. The
/// table is oversized so that any code read from a (possibly corrupted) stream indexes safely.
fn buildDecodeTable(entries: []const u6) [post_office_solver.table_size]u6 {
    var table: [post_office_solver.table_size]u6 = @splat(0);
    for (entries, 0..) |entry, index| table[index] = entry;
    return table;
}

/// Compress `uncompressed_values` within `error_bound` using "Serf-XOR". The function writes the
/// result to `compressed_values` as a `[shift: f64]` header followed by the XOR-encoded bit
/// stream, which is terminated by one XOR-encoded NaN sentinel value. The `allocator` is used for
/// memory management of intermediate containers and the `method_configuration` parser. The
/// function expects an `AbsoluteErrorBound` configuration; an `error_bound` of zero selects the
/// lossless mode in which values are XOR-encoded bit-exactly without shifting or approximation.
/// Inputs that are not finite, exceed `tester.max_test_value` in magnitude, or defeat the
/// error-bound guarantee through floating-point rounding are rejected with
/// `Error.UnsupportedInput`. If an error occurs it is returned.
pub fn compress(
    allocator: Allocator,
    uncompressed_values: []const f64,
    compressed_values: *ArrayList(u8),
    method_configuration: []const u8,
) Error!void {
    const parsed_configuration = try configuration.parse(
        allocator,
        configuration.AbsoluteErrorBound,
        method_configuration,
    );

    const error_bound: f32 = parsed_configuration.abs_error_bound;
    // Widen once: all error computations run in f64 like the reference.
    const max_diff: f64 = error_bound;

    for (uncompressed_values) |value| {
        if (!math.isFinite(value) or @abs(value) > tester.max_test_value)
            return Error.UnsupportedInput;
    }

    // Shifter (paper Theorem 1): add an integer offset so all shifted values share one exponent,
    // maximizing XOR leading zeros. Lossless mode keeps shift 0 (the +/-shift round trip is not
    // exact in floating point).
    var adjust_digit: f64 = 0.0;
    if (error_bound != 0.0 and uncompressed_values.len > 0) {
        var min_value = uncompressed_values[0];
        var max_value = uncompressed_values[0];
        for (uncompressed_values[1..]) |value| {
            min_value = @min(min_value, value);
            max_value = @max(max_value, value);
        }
        const range = @floor(max_value) - @floor(min_value) + 1.0;
        // Defensive: validated finite inputs already guarantee `range >= 1.0`; this only keeps
        // `@log2` safe if that ever changes.
        const exponent: f64 = if (range < 1.0) 0.0 else @ceil(@log2(range));
        adjust_digit = @exp2(exponent) - @floor(min_value);
    }

    // Encode into a scratch buffer so `compressed_values` stays untouched if a value defeats the
    // error-bound guarantee mid-stream.
    var payload = ArrayList(u8).empty;
    defer payload.deinit(allocator);

    var compressor = Compressor{};
    var bit_writer = try shared_structs.BulkBitWriter.init(allocator, &payload);

    for (uncompressed_values) |value| {
        var approximated_bits: u64 = undefined;
        if (error_bound == 0.0) {
            // Lossless mode: the XOR codec itself is exact, skip the approximator entirely.
            approximated_bits = @bitCast(value);
        } else if (@abs(@as(f64, @bitCast(compressor.stored_value)) - adjust_digit - value) >
            max_diff)
        {
            const adjusted_value = value + adjust_digit;
            approximated_bits = findApproximation(
                adjusted_value - max_diff,
                adjusted_value + max_diff,
                value,
                compressor.stored_value,
                max_diff,
                adjust_digit,
            );
        } else {
            // The previous bits are still within the bound: reuse them for an XOR of zero.
            approximated_bits = compressor.stored_value;
        }

        // Verify every approximation before writing: the fallback path and shift rounding can
        // exceed the bound. The negated comparison also rejects non-finite differences.
        const reconstructed = @as(f64, @bitCast(approximated_bits)) - adjust_digit;
        if (!(@abs(reconstructed - value) <= max_diff)) return Error.UnsupportedInput;

        compressor.bits_this_window += try compressor.compressValue(
            &bit_writer,
            approximated_bits,
        );
        compressor.values_this_window += 1;

        // Window rule update at every window boundary; the trailing NaN sentinel guarantees
        // the decoder always reads this flag before stopping.
        if (compressor.values_this_window == window_size) {
            try compressor.updateRules(&bit_writer);
        }
    }

    // Terminate the stream with a NaN sentinel, XOR-encoded through the normal path. NaN never
    // occurs among the finite shifted approximations, so it uniquely marks the end.
    _ = try compressor.compressValue(&bit_writer, @bitCast(math.nan(f64)));
    try bit_writer.flushBits();

    try shared_functions.appendValue(allocator, f64, adjust_digit, compressed_values);
    try compressed_values.appendSlice(allocator, payload.items);
}

/// Decompress `compressed_values` produced by "Serf-XOR". The function writes the result to
/// `decompressed_values`. The `allocator` is used to manage the memory of intermediate results.
/// If an error occurs it is returned.
pub fn decompress(
    allocator: Allocator,
    compressed_values: []const u8,
    decompressed_values: *ArrayList(f64),
) Error!void {
    // Ensure at least the `[shift: f64]` header is present. A shorter stream is malformed.
    if (compressed_values.len < @sizeOf(f64)) return Error.CorruptedCompressedData;

    var offset: usize = 0;
    const adjust_digit = try shared_functions.readOffsetValue(f64, compressed_values, &offset);

    var decompressor = Decompressor{};
    var bit_reader = shared_structs.BulkBitReader.init(compressed_values[offset..]);

    var values_this_window: u32 = 0;
    while (true) {
        const value_bits = try decompressor.readValue(&bit_reader);
        // The trailing NaN sentinel marks the end of the stream; stop without appending it.
        if (math.isNan(@as(f64, @bitCast(value_bits)))) break;
        try decompressed_values.append(allocator, @as(f64, @bitCast(value_bits)) - adjust_digit);

        values_this_window += 1;
        // Mirror the encoder: read the update flag at every window boundary.
        if (values_this_window == window_size) {
            try decompressor.updateRulesIfFlagged(&bit_reader);
            values_this_window = 0;
        }
    }
}

/// Encoder state: the previous stored bits, the active leading/trailing bounds and code widths,
/// the rounding tables, and the per-window bookkeeping driving the adaptive rule updates.
const Compressor = struct {
    stored_value: u64 = initial_stored_value,
    stored_leading_zeros: u32 = zeros_sentinel,
    stored_trailing_zeros: u32 = zeros_sentinel,
    leading_bits_per_value: u16 = 3,
    trailing_bits_per_value: u16 = 3,
    leading_round: [post_office_solver.table_size]u6 = initial_leading_round,
    leading_representation: [post_office_solver.table_size]u6 = initial_leading_representation,
    trailing_round: [post_office_solver.table_size]u6 = initial_trailing_round,
    trailing_representation: [post_office_solver.table_size]u6 = initial_trailing_representation,
    lead_distribution: [post_office_solver.table_size]u32 = @splat(0),
    trail_distribution: [post_office_solver.table_size]u32 = @splat(0),
    bits_this_window: u64 = 0,
    values_this_window: u32 = 0,
    compression_ratio_last_window: f64 = 0.0,

    /// XOR-encode `value` against the previous stored bits into `bit_writer` using the three-case
    /// layout of the reference (`01` identical, `1` reuse stored bounds, `00` new bounds), update
    /// the zero-count distributions, and store `value` as the new previous bits. Returns the
    /// number of bits written. If an error occurs it is returned.
    fn compressValue(
        self: *Compressor,
        bit_writer: *shared_structs.BulkBitWriter,
        value: u64,
    ) Error!u32 {
        var this_size: u32 = 0;
        const xor_result = self.stored_value ^ value;

        if (xor_result == 0) {
            // Case 01: the value repeats the previous bits exactly.
            try bit_writer.writeBits(@as(u2, 0b01), 2);
            this_size = 2;
        } else {
            const leading_count: u32 = @clz(xor_result);
            const trailing_count: u32 = @ctz(xor_result);
            const leading_zeros: u32 = self.leading_round[leading_count];
            const trailing_zeros: u32 = self.trailing_round[trailing_count];
            // The initial tables and the solver both guarantee `round[j] <= j`, so `center_bits`
            // stays positive in case 00 (`clz + ctz <= 63` for a non-zero XOR).
            std.debug.assert(leading_zeros <= leading_count and trailing_zeros <= trailing_count);
            self.lead_distribution[leading_count] += 1;
            self.trail_distribution[trailing_count] += 1;

            if (leading_zeros >= self.stored_leading_zeros and
                trailing_zeros >= self.stored_trailing_zeros and
                (leading_zeros - self.stored_leading_zeros) +
                    (trailing_zeros - self.stored_trailing_zeros) <
                    1 + self.leading_bits_per_value + self.trailing_bits_per_value)
            {
                // Case 1: reuse the stored bounds; only the center bits follow the control bit.
                // Reachable only after a case 00 set real bounds, so the subtraction cannot wrap.
                std.debug.assert(
                    self.stored_leading_zeros + self.stored_trailing_zeros < bits_per_value,
                );
                const center_bits: u16 =
                    @intCast(bits_per_value - self.stored_leading_zeros - self.stored_trailing_zeros);
                try bit_writer.writeBits(@as(u1, 0b1), 1);
                try bit_writer.writeBits(
                    xor_result >> @intCast(self.stored_trailing_zeros),
                    center_bits,
                );
                this_size = 1 + center_bits;
            } else {
                // Case 00: store new bounds as codes, then the center bits.
                self.stored_leading_zeros = leading_zeros;
                self.stored_trailing_zeros = trailing_zeros;
                const center_bits: u16 = @intCast(bits_per_value - leading_zeros - trailing_zeros);
                try bit_writer.writeBits(@as(u2, 0b00), 2);
                try bit_writer.writeBits(
                    self.leading_representation[leading_zeros],
                    self.leading_bits_per_value,
                );
                try bit_writer.writeBits(
                    self.trailing_representation[trailing_zeros],
                    self.trailing_bits_per_value,
                );
                try bit_writer.writeBits(xor_result >> @intCast(trailing_zeros), center_bits);
                this_size = 2 + self.leading_bits_per_value + self.trailing_bits_per_value +
                    center_bits;
            }
        }

        self.stored_value = value;
        return this_size;
    }

    /// Run the window rule update: if the compression ratio of the closing window got worse,
    /// recompute the rounding rules from the observed distributions with the post-office solver,
    /// write flag bit 1 and the new positions; otherwise write flag bit 0. Either way remember
    /// the ratio and reset the window bookkeeping. If an error occurs it is returned.
    fn updateRules(self: *Compressor, bit_writer: *shared_structs.BulkBitWriter) Error!void {
        const compression_ratio_this_window = @as(f64, @floatFromInt(self.bits_this_window)) /
            @as(f64, @floatFromInt(@as(u64, self.values_this_window) * bits_per_value));

        if (self.compression_ratio_last_window < compression_ratio_this_window) {
            // The ratio got worse: re-optimize the rules and transmit the new positions.
            const lead_positions = post_office_solver.initRoundAndRepresentation(
                &self.lead_distribution,
                &self.leading_representation,
                &self.leading_round,
            );
            self.leading_bits_per_value =
                post_office_solver.position_length_to_bits[lead_positions.len];
            const trail_positions = post_office_solver.initRoundAndRepresentation(
                &self.trail_distribution,
                &self.trailing_representation,
                &self.trailing_round,
            );
            self.trailing_bits_per_value =
                post_office_solver.position_length_to_bits[trail_positions.len];
            try bit_writer.writeBits(@as(u1, 0b1), 1);
            _ = try post_office_solver.writePositions(lead_positions, bit_writer);
            _ = try post_office_solver.writePositions(trail_positions, bit_writer);
        } else {
            try bit_writer.writeBits(@as(u1, 0b0), 1);
        }

        self.compression_ratio_last_window = compression_ratio_this_window;
        self.lead_distribution = @splat(0);
        self.trail_distribution = @splat(0);
        self.bits_this_window = 0;
        self.values_this_window = 0;
    }
};

/// Decoder state mirroring `Compressor`: the previous stored bits, the active bounds and code
/// widths, and the code-to-zero-count tables updated in-stream by the window rule updates.
const Decompressor = struct {
    stored_value: u64 = initial_stored_value,
    stored_leading_zeros: u32 = zeros_sentinel,
    stored_trailing_zeros: u32 = zeros_sentinel,
    leading_bits_per_value: u16 = 3,
    trailing_bits_per_value: u16 = 3,
    leading_decode: [post_office_solver.table_size]u6 = initial_leading_decode,
    trailing_decode: [post_office_solver.table_size]u6 = initial_trailing_decode,

    /// Read one XOR-encoded value from `bit_reader`, reversing the three-case layout written by
    /// `Compressor.compressValue`, and return the reconstructed bits (also stored as the new
    /// previous bits). If an error occurs it is returned.
    fn readValue(self: *Decompressor, bit_reader: *shared_structs.BulkBitReader) Error!u64 {
        const first_bit = bit_reader.readBitsNoEof(u1, 1) catch return Error.ByteStreamError;
        if (first_bit == 1) {
            // Case 1: center bits with the stored bounds.
            try self.readCenterBits(bit_reader);
            return self.stored_value;
        }

        const second_bit = bit_reader.readBitsNoEof(u1, 1) catch return Error.ByteStreamError;
        if (second_bit == 1) {
            // Case 01: the previous bits repeat unchanged.
            return self.stored_value;
        }

        // Case 00: read new bounds as codes, then the center bits.
        const leading_code = bit_reader.readBitsNoEof(
            u8,
            self.leading_bits_per_value,
        ) catch return Error.ByteStreamError;
        const trailing_code = bit_reader.readBitsNoEof(
            u8,
            self.trailing_bits_per_value,
        ) catch return Error.ByteStreamError;
        self.stored_leading_zeros = self.leading_decode[leading_code];
        self.stored_trailing_zeros = self.trailing_decode[trailing_code];
        try self.readCenterBits(bit_reader);
        return self.stored_value;
    }

    /// Read `64 - stored_leading_zeros - stored_trailing_zeros` center bits and XOR them, shifted
    /// back in place, into the stored bits. A valid stream always keeps the bound sum below 64;
    /// larger sums (including the initial sentinels) only occur for corrupted streams and are
    /// rejected. If an error occurs it is returned.
    fn readCenterBits(self: *Decompressor, bit_reader: *shared_structs.BulkBitReader) Error!void {
        const bound_sum = self.stored_leading_zeros + self.stored_trailing_zeros;
        if (bound_sum >= bits_per_value) return Error.CorruptedCompressedData;
        const center_bits: u16 = @intCast(bits_per_value - bound_sum);
        const center = bit_reader.readBitsNoEof(u64, center_bits) catch
            return Error.ByteStreamError;
        self.stored_value ^= center << @intCast(self.stored_trailing_zeros);
    }

    /// Mirror of the encoder's window rule update: read the flag bit and, when it is set, replace
    /// both decode tables and code widths with the positions transmitted in-stream. If an error
    /// occurs it is returned.
    fn updateRulesIfFlagged(
        self: *Decompressor,
        bit_reader: *shared_structs.BulkBitReader,
    ) Error!void {
        const flag = bit_reader.readBitsNoEof(u1, 1) catch return Error.ByteStreamError;
        if (flag == 1) {
            self.leading_bits_per_value = try readDecodeTable(bit_reader, &self.leading_decode);
            self.trailing_bits_per_value = try readDecodeTable(bit_reader, &self.trailing_decode);
        }
    }

    /// Read one transmitted position table into `decode`: a 5-bit count (0 encodes 32) followed
    /// by one 6-bit zero count per code. Returns the new code width. If an error occurs it is
    /// returned.
    fn readDecodeTable(
        bit_reader: *shared_structs.BulkBitReader,
        decode: *[post_office_solver.table_size]u6,
    ) Error!u16 {
        const raw_count = bit_reader.readBitsNoEof(u8, 5) catch return Error.ByteStreamError;
        const count: usize = if (raw_count == 0) 32 else raw_count;
        decode.* = @splat(0);
        for (0..count) |index| {
            decode[index] = bit_reader.readBitsNoEof(u6, 6) catch return Error.ByteStreamError;
        }
        return post_office_solver.position_length_to_bits[count];
    }
};

/// Approximator (`FindAppLong` of the reference): return the `u64` bits of an approximation of
/// `original` whose `f64` value `w` satisfies `|w - adjust_digit - original| <= max_diff`,
/// sharing the longest possible suffix with `last_bits` to maximize the XOR trailing zeros.
/// `min` and `max` delimit the acceptable shifted range `original + adjust_digit ± max_diff`.
/// The dispatch reduces every case to non-negative magnitudes handled by
/// `findApproximationCore`, picking the sign that matches `last_bits` when the range straddles
/// zero.
fn findApproximation(
    min: f64,
    max: f64,
    original: f64,
    last_bits: u64,
    max_diff: f64,
    adjust_digit: f64,
) u64 {
    if (min >= 0) {
        // Both bounds positive.
        return findApproximationCore(min, max, 0, original, last_bits, max_diff, adjust_digit);
    } else if (max <= 0) {
        // Both bounds negative: search the mirrored magnitudes with the sign bit set.
        return findApproximationCore(
            -max,
            -min,
            sign_bit_mask,
            original,
            last_bits,
            max_diff,
            adjust_digit,
        );
    } else if (last_bits >> 63 == 0) {
        // Range straddles zero, previous bits positive: search the positive part only.
        return findApproximationCore(0, max, 0, original, last_bits, max_diff, adjust_digit);
    } else {
        // Range straddles zero, previous bits negative: search the negative part only.
        return findApproximationCore(
            0,
            -min,
            sign_bit_mask,
            original,
            last_bits,
            max_diff,
            adjust_digit,
        );
    }
}

/// Core of the approximator, operating on non-negative magnitudes `min`/`max` with the target
/// `sign` bit applied to candidates on return. Starting from the most significant differing bit
/// of the two bounds, it tries progressively shorter prefixes of `min`'s bits combined with the
/// suffix of `last_bits` (plus the same candidate incremented at the split position) and returns
/// the first candidate inside `[min, max]` whose value is within `max_diff` of
/// `original + adjust_digit`. Falls back to the exact bits of `original + adjust_digit`.
fn findApproximationCore(
    min: f64,
    max: f64,
    sign: u64,
    original: f64,
    last_bits: u64,
    max_diff: f64,
    adjust_digit: f64,
) u64 {
    // Clear the sign bit: `min` may be negative zero after mirroring.
    const min_bits = @as(u64, @bitCast(min)) & magnitude_mask;
    const max_bits: u64 = @bitCast(max);
    const leading_zeros: u32 = @clz(min_bits ^ max_bits);
    var shift: i32 = bits_per_value - @as(i32, @intCast(leading_zeros));
    // Both bounds are non-negative here, so the XOR has a leading zero and `shift <= 63`.
    std.debug.assert(shift <= 63);

    while (shift >= 0) : (shift -= 1) {
        const front_mask = math.shl(u64, ~@as(u64, 0), shift);
        const append = (front_mask & min_bits) | (~front_mask & last_bits);
        if (append >= min_bits and append <= max_bits) {
            const candidate = append ^ sign;
            const diff = @as(f64, @bitCast(candidate)) - adjust_digit - original;
            if (diff >= -max_diff and diff <= max_diff) return candidate;
        }

        // Also try the next representable prefix; mask the sign bit to avoid overflowing into it.
        const incremented = (append +% (@as(u64, 1) << @intCast(shift))) & magnitude_mask;
        if (incremented <= max_bits) {
            const candidate = incremented ^ sign;
            const diff = @as(f64, @bitCast(candidate)) - adjust_digit - original;
            if (diff >= -max_diff and diff <= max_diff) return candidate;
        }
    }

    // No candidate satisfied the bound: fall back to the exact shifted bits.
    return @bitCast(original + adjust_digit);
}

test "serf-xor can compress and decompress bounded values" {
    const allocator = testing.allocator;
    const data_distributions = &[_]tester.DataDistribution{
        .LinearFunctions,
        .BoundedRandomValues,
        .SinusoidalFunction,
        .MixedBoundedValuesFunctions,
    };

    // Floating-point precision can make the error bound unachievable (e.g. shift rounding), so
    // `UnsupportedInput` is documented behavior here and such cases are skipped.
    tester.testErrorBoundedCompressionMethod(
        allocator,
        Method.SerfXOR,
        data_distributions,
    ) catch |err| switch (err) {
        error.UnsupportedInput => {},
        else => return err,
    };
}

test "serf-xor cannot compress and decompress nan values" {
    const allocator = testing.allocator;
    const uncompressed_values = [3]f64{ 343.0, math.nan(f64), 520.0 };
    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);

    const method_configuration =
        \\ {"abs_error_bound": 0.1}
    ;

    compress(
        allocator,
        uncompressed_values[0..],
        &compressed_values,
        method_configuration,
    ) catch |err| {
        try testing.expectEqual(Error.UnsupportedInput, err);
        return;
    };

    try testing.expectFmt(
        "",
        "The Serf-XOR method cannot compress NaN values",
        .{},
    );
}

test "serf-xor cannot compress and decompress values exceeding floating-point precision limits" {
    const allocator = testing.allocator;
    // A magnitude too large to represent accurately in f64, such as 1e20.
    const uncompressed_values = [3]f64{ 343.0, 1e20, 520.0 };
    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);

    const method_configuration =
        \\ {"abs_error_bound": 0.1}
    ;

    compress(
        allocator,
        uncompressed_values[0..],
        &compressed_values,
        method_configuration,
    ) catch |err| {
        try testing.expectEqual(Error.UnsupportedInput, err);
        return;
    };

    try testing.expectFmt(
        "",
        "The Serf-XOR method cannot compress values exceeding floating-point precision limits",
        .{},
    );
}

test "serf-xor can compress and decompress within floating-point precision limits at different scales" {
    const allocator = testing.allocator;
    const error_bound = tester.generateBoundedRandomValue(f32, 1e2, 1e3, null);

    var uncompressed_values = ArrayList(f64).empty;
    defer uncompressed_values.deinit(allocator);

    try tester.generateBoundedRandomValues(allocator, &uncompressed_values, -1, 1, null);
    try tester.generateBoundedRandomValues(allocator, &uncompressed_values, -1e2, 1e2, null);
    try tester.generateBoundedRandomValues(allocator, &uncompressed_values, -1e4, 1e4, null);
    try tester.generateBoundedRandomValues(allocator, &uncompressed_values, -1e6, 1e6, null);
    try tester.generateBoundedRandomValues(allocator, &uncompressed_values, -1e8, 1e8, null);
    try tester.generateBoundedRandomValues(allocator, &uncompressed_values, -1e14, 1e14, null);

    try tester.testCompressAndDecompress(
        allocator,
        uncompressed_values.items,
        Method.SerfXOR,
        error_bound,
        shared_functions.isWithinErrorBound,
    );
}

test "serf-xor is bit-exact with zero error bound at different scales" {
    const allocator = testing.allocator;

    var uncompressed_values = ArrayList(f64).empty;
    defer uncompressed_values.deinit(allocator);

    try tester.generateBoundedRandomValues(allocator, &uncompressed_values, -1, 1, null);
    try tester.generateBoundedRandomValues(allocator, &uncompressed_values, -1e2, 1e2, null);
    try tester.generateBoundedRandomValues(allocator, &uncompressed_values, -1e4, 1e4, null);
    try tester.generateBoundedRandomValues(allocator, &uncompressed_values, -1e6, 1e6, null);
    try tester.generateBoundedRandomValues(allocator, &uncompressed_values, -1e8, 1e8, null);
    try tester.generateBoundedRandomValues(allocator, &uncompressed_values, -1e14, 1e14, null);

    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);
    var decompressed_values = ArrayList(f64).empty;
    defer decompressed_values.deinit(allocator);

    const method_configuration =
        \\ {"abs_error_bound": 0}
    ;

    try compress(
        allocator,
        uncompressed_values.items,
        &compressed_values,
        method_configuration,
    );
    try decompress(allocator, compressed_values.items, &decompressed_values);

    // Zero-error-bound mode must round trip bit-exactly, stronger than the error bound.
    try testing.expectEqual(uncompressed_values.items.len, decompressed_values.items.len);
    for (uncompressed_values.items, decompressed_values.items) |expected, actual| {
        try testing.expectEqual(
            @as(u64, @bitCast(expected)),
            @as(u64, @bitCast(actual)),
        );
    }
}

test "serf-xor stays within error bound across adaptive rule window boundaries" {
    const allocator = testing.allocator;
    const error_bound: f32 = 0.5;

    // >2500 mixed/noisy values cross two window boundaries: the noisy prefix worsens the ratio
    // so rules are recomputed and transmitted, while the smooth tail keeps its tables (flag 0).
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    var uncompressed_values = ArrayList(f64).empty;
    defer uncompressed_values.deinit(allocator);
    for (0..2500) |index| {
        const base = 250.0 * @sin(0.01 * @as(f64, @floatFromInt(index)));
        const noise: f64 = if (index < 1200) 100.0 * (random.float(f64) - 0.5) else 0.0;
        try uncompressed_values.append(allocator, base + noise);
    }

    // Window-boundary edge cases: exactly one window, one value past a boundary, exactly two
    // windows, and the full series spanning two boundaries.
    for ([_]usize{ 1000, 1001, 2000, 2001, 2500 }) |value_count| {
        const values = uncompressed_values.items[0..value_count];

        var compressed_values = ArrayList(u8).empty;
        defer compressed_values.deinit(allocator);
        var decompressed_values = ArrayList(f64).empty;
        defer decompressed_values.deinit(allocator);

        const method_configuration = try std.fmt.allocPrint(
            allocator,
            "{{\"abs_error_bound\": {d}}}",
            .{error_bound},
        );
        defer allocator.free(method_configuration);

        try compress(allocator, values, &compressed_values, method_configuration);
        try decompress(allocator, compressed_values.items, &decompressed_values);

        try testing.expectEqual(value_count, decompressed_values.items.len);
        try testing.expect(shared_functions.isWithinErrorBound(
            values,
            decompressed_values.items,
            error_bound,
        ));
    }
}

test "serf-xor compresses a constant series far below raw size" {
    const allocator = testing.allocator;
    const error_bound: f32 = 0.1;

    // A constant series exercises the case 01 path for every value after the first and crosses
    // a window boundary.
    var uncompressed_values: [1500]f64 = undefined;
    @memset(&uncompressed_values, 42.75);

    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);
    var decompressed_values = ArrayList(f64).empty;
    defer decompressed_values.deinit(allocator);

    const method_configuration = try std.fmt.allocPrint(
        allocator,
        "{{\"abs_error_bound\": {d}}}",
        .{error_bound},
    );
    defer allocator.free(method_configuration);

    try compress(allocator, &uncompressed_values, &compressed_values, method_configuration);
    try decompress(allocator, compressed_values.items, &decompressed_values);

    try testing.expect(shared_functions.isWithinErrorBound(
        &uncompressed_values,
        decompressed_values.items,
        error_bound,
    ));
    try testing.expect(compressed_values.items.len < uncompressed_values.len * @sizeOf(f64));
}

test "serf-xor compresses a smooth sinusoid below eight bytes per value" {
    const allocator = testing.allocator;
    const error_bound: f32 = 1.0;

    var uncompressed_values: [500]f64 = undefined;
    for (&uncompressed_values, 0..) |*value, index| {
        value.* = 100.0 * @sin(0.05 * @as(f64, @floatFromInt(index)));
    }

    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);

    const method_configuration = try std.fmt.allocPrint(
        allocator,
        "{{\"abs_error_bound\": {d}}}",
        .{error_bound},
    );
    defer allocator.free(method_configuration);

    try compress(allocator, &uncompressed_values, &compressed_values, method_configuration);

    // With a generous bound the XOR cases must beat storing raw eight-byte values.
    try testing.expect(compressed_values.items.len < uncompressed_values.len * @sizeOf(f64));
}

test "serf-xor handles adversarial value ranges" {
    const allocator = testing.allocator;

    // Negative-only, mixed-sign with a bound larger than the range, and values at the +-1e15
    // input limit must all round trip within the bound.
    const AdversarialCase = struct { values: []const f64, error_bound: f32 };
    const cases = [_]AdversarialCase{
        .{
            .values = &[_]f64{ -5.5, -3.25, -100.75, -99.5, -0.125, -42.0 },
            .error_bound = 0.5,
        },
        .{
            .values = &[_]f64{ 10.0, 10.5, 10.25, 10.75, 10.125, 10.625 },
            .error_bound = 50.0,
        },
        .{
            .values = &[_]f64{ -1e15, 1e15, -1e15 + 0.5, 1e15 - 0.5, 0.0 },
            .error_bound = 1.0,
        },
    };

    for (cases) |adversarial_case| {
        var compressed_values = ArrayList(u8).empty;
        defer compressed_values.deinit(allocator);
        var decompressed_values = ArrayList(f64).empty;
        defer decompressed_values.deinit(allocator);

        const method_configuration = try std.fmt.allocPrint(
            allocator,
            "{{\"abs_error_bound\": {d}}}",
            .{adversarial_case.error_bound},
        );
        defer allocator.free(method_configuration);

        try compress(
            allocator,
            adversarial_case.values,
            &compressed_values,
            method_configuration,
        );
        try decompress(allocator, compressed_values.items, &decompressed_values);

        try testing.expect(shared_functions.isWithinErrorBound(
            adversarial_case.values,
            decompressed_values.items,
            adversarial_case.error_bound,
        ));
    }
}

test "serf-xor rejects error bounds defeated by the shift rounding" {
    const allocator = testing.allocator;

    // A wide range forces a large shift, so `0.1 + shift` can't be represented within the tiny
    // bound: the guarantee check must reject the input and write nothing.
    const uncompressed_values = &[_]f64{ 0.1, 1e15 };

    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);

    const method_configuration =
        \\ {"abs_error_bound": 1e-7}
    ;

    try testing.expectError(Error.UnsupportedInput, compress(
        allocator,
        uncompressed_values,
        &compressed_values,
        method_configuration,
    ));
    try testing.expectEqual(@as(usize, 0), compressed_values.items.len);
}

test "serf-xor decompress rejects truncated and corrupted input" {
    const allocator = testing.allocator;

    var uncompressed_values: [100]f64 = undefined;
    for (&uncompressed_values, 0..) |*value, index| {
        value.* = @floatFromInt(index * 7 % 23);
    }

    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);

    const method_configuration =
        \\ {"abs_error_bound": 0.1}
    ;

    try compress(
        allocator,
        &uncompressed_values,
        &compressed_values,
        method_configuration,
    );

    // A stream shorter than the 8-byte header is rejected up front as corrupted data.
    var short_decompressed_values = ArrayList(f64).empty;
    defer short_decompressed_values.deinit(allocator);
    try testing.expectError(
        Error.CorruptedCompressedData,
        decompress(allocator, compressed_values.items[0..4], &short_decompressed_values),
    );

    // Truncating mid-payload removes the NaN sentinel, so the decoder runs off the end of the
    // stream: it must return a documented error, never panic.
    var truncated_decompressed_values = ArrayList(f64).empty;
    defer truncated_decompressed_values.deinit(allocator);
    try testing.expect(compressed_values.items.len > @sizeOf(f64) + 4);
    decompress(
        allocator,
        compressed_values.items[0 .. @sizeOf(f64) + 4],
        &truncated_decompressed_values,
    ) catch |err| switch (err) {
        Error.CorruptedCompressedData, Error.ByteStreamError => return,
        else => return err,
    };
    try testing.expectFmt("", "truncated Serf-XOR stream must be rejected", .{});
}

test "check serf-xor configuration parsing" {
    // Verifies `compress` parses the configuration and expects `AbsoluteErrorBound`.
    const allocator = testing.allocator;

    const uncompressed_values = &[4]f64{ 19.0, 48.0, 29.0, 3.0 };

    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);

    const method_configuration =
        \\ {"abs_error_bound": 0.1}
    ;

    // The configuration is properly defined. No error expected.
    try compress(
        allocator,
        uncompressed_values,
        &compressed_values,
        method_configuration,
    );

    // A configuration with an unexpected field is rejected.
    const invalid_configuration =
        \\ {"histogram_bins_number": 3}
    ;
    try testing.expectError(Error.InvalidConfiguration, compress(
        allocator,
        uncompressed_values,
        &compressed_values,
        invalid_configuration,
    ));
}

test "serf-xor round trips fuzzed inputs within the error bound" {
    const allocator = testing.allocator;

    // Deterministic property fuzzer: fixed base seed for reproducibility; the per-iteration seed
    // is printed on failure so a failing case can be replayed.
    const base_seed: u64 = 0x5345_5246_584f_5231;
    const iteration_count = 2000;

    for (0..iteration_count) |iteration| {
        const seed = base_seed +% iteration;
        errdefer std.debug.print(
            "serf-xor round-trip fuzzer failed for seed {d} (iteration {d})\n",
            .{ seed, iteration },
        );

        var prng = std.Random.DefaultPrng.init(seed);
        const random = prng.random();

        // Mostly short inputs for speed; every tenth draws up to 3000 values to cross the
        // `window_size` = 1000 boundary.
        const max_count: usize = if (iteration % 10 == 0) 3001 else 129;
        const value_count = random.uintLessThan(usize, max_count);

        // Mix zero and positive bounds across magnitudes; draw values up to the
        // `tester.max_test_value` input limit.
        const error_bound: f32 = if (random.uintLessThan(u8, 4) == 0)
            0.0
        else
            tester.generateBoundedRandomValue(f32, 1e-6, 1e3, random);
        const magnitude: f64 = switch (random.uintLessThan(u8, 4)) {
            0 => 1.0,
            1 => 1e4,
            2 => 1e8,
            else => tester.max_test_value,
        };

        var uncompressed_values = ArrayList(f64).empty;
        defer uncompressed_values.deinit(allocator);
        for (0..value_count) |_| {
            try uncompressed_values.append(
                allocator,
                tester.generateBoundedRandomValue(f64, -magnitude, magnitude, random),
            );
        }

        var compressed_values = ArrayList(u8).empty;
        defer compressed_values.deinit(allocator);
        var decompressed_values = ArrayList(f64).empty;
        defer decompressed_values.deinit(allocator);

        const method_configuration = try std.fmt.allocPrint(
            allocator,
            "{{\"abs_error_bound\": {d}}}",
            .{error_bound},
        );
        defer allocator.free(method_configuration);

        compress(
            allocator,
            uncompressed_values.items,
            &compressed_values,
            method_configuration,
        ) catch |err| switch (err) {
            // Floating-point precision can defeat the error bound (e.g. shift rounding), so
            // these inputs are rejected.
            Error.UnsupportedInput => continue,
            else => return err,
        };
        try decompress(allocator, compressed_values.items, &decompressed_values);

        try testing.expectEqual(value_count, decompressed_values.items.len);
        if (error_bound == 0.0) {
            // Lossless mode must round trip bit-exactly, stronger than the error bound.
            for (uncompressed_values.items, decompressed_values.items) |expected, actual| {
                try testing.expectEqual(
                    @as(u64, @bitCast(expected)),
                    @as(u64, @bitCast(actual)),
                );
            }
        } else {
            try testing.expect(shared_functions.isWithinErrorBound(
                uncompressed_values.items,
                decompressed_values.items,
                error_bound,
            ));
        }
    }
}

test "serf-xor decompress survives fuzzed corrupted input without crashing" {
    const allocator = testing.allocator;

    // Deterministic decoder-robustness fuzzer: `decompress` must succeed or return a documented
    // error for arbitrary bytes, never panic or hit UB (run under `-Doptimize=ReleaseFast` to
    // catch safety-checked UB). Fixed base seed for reproducibility; seed printed on failure.
    const base_seed: u64 = 0x5345_5246_584f_5232;
    const iteration_count = 2000;

    // Build a valid stream from noisy data crossing a window boundary, so corruption can hit the
    // case codes and rule-update tables, not just the header.
    var data_prng = std.Random.DefaultPrng.init(base_seed);
    const data_random = data_prng.random();
    var uncompressed_values: [1500]f64 = undefined;
    for (&uncompressed_values, 0..) |*value, index| {
        value.* = 100.0 * @sin(0.02 * @as(f64, @floatFromInt(index))) +
            50.0 * (data_random.float(f64) - 0.5);
    }

    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);

    const method_configuration =
        \\ {"abs_error_bound": 0.25}
    ;

    try compress(
        allocator,
        &uncompressed_values,
        &compressed_values,
        method_configuration,
    );

    for (0..iteration_count) |iteration| {
        const seed = base_seed +% (iteration + 1);
        errdefer std.debug.print(
            "serf-xor corrupted-input fuzzer failed for seed {d} (iteration {d})\n",
            .{ seed, iteration },
        );

        var prng = std.Random.DefaultPrng.init(seed);
        const random = prng.random();

        var corrupted_values = ArrayList(u8).empty;
        defer corrupted_values.deinit(allocator);

        switch (random.uintLessThan(u8, 3)) {
            0 => {
                // Flip 1-8 random bytes anywhere (including the header) with a non-zero XOR mask.
                try corrupted_values.appendSlice(allocator, compressed_values.items);
                const flip_count = 1 + random.uintLessThan(usize, 8);
                for (0..flip_count) |_| {
                    const byte_index = random.uintLessThan(usize, corrupted_values.items.len);
                    corrupted_values.items[byte_index] ^= 1 + random.uintLessThan(u8, 255);
                }
            },
            1 => {
                // Truncate the valid stream to a random prefix, from empty up to full length.
                const truncated_len = random.uintLessThan(
                    usize,
                    compressed_values.items.len + 1,
                );
                try corrupted_values.appendSlice(
                    allocator,
                    compressed_values.items[0..truncated_len],
                );
            },
            else => {
                // A fully random buffer of random length: no valid structure at all.
                try corrupted_values.resize(allocator, random.uintLessThan(usize, 513));
                random.bytes(corrupted_values.items);
            },
        }

        var decompressed_values = ArrayList(f64).empty;
        defer decompressed_values.deinit(allocator);

        // Corrupted input may decode into garbage values (acceptable); only an unexpected
        // error, panic, or UB is a failure.
        decompress(allocator, corrupted_values.items, &decompressed_values) catch |err|
            switch (err) {
                Error.CorruptedCompressedData,
                Error.ByteStreamError,
                => {},
                else => return err,
            };
    }
}
