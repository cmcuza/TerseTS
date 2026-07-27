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

//! Selects which leading-zero or trailing-zero counts SerfXOR transmits as short codes, and
//! derives the codec's two lookup tables from that selection.
//! Sending an exact zero count needs six bits, so only a small set of counts is transmitted and
//! every other count is rounded down to the nearest one in that set, at the cost of encoding the
//! difference explicitly. Selecting more counts shortens the rounding gaps but lengthens every
//! code, so `initRoundAndRepresentation` picks the set minimizing both by dynamic programming
//! over the observed distribution, then fills in `round` (which count each zero count rounds
//! down to) and `representation` (the code sent for it). SerfXOR recomputes both per window.
//! This file is a semantically faithful port of `post_office_solver.{h,cc}` from the
//! reference C++ implementation of the paper "Li, Ruiyuan, Zechao Chen, Ruyun Lu, Xiaolong Xu,
//! Guangchao Yang, Chao Chen, Jie Bao, and Yu Zheng. Serf: Streaming Error-Bounded Floating-Point
//! Compression. ACM SIGMOD 2025. https://doi.org/10.1145/3725353".
//! The implementation is partially based on the code released at
//! https://github.com/Spatio-Temporal-Lab/Serf (accessed on 27-07-26).

const std = @import("std");
const math = std.math;
const testing = std.testing;
const ArrayList = std.ArrayList;

const tersets = @import("../tersets.zig");
const shared_structs = @import("shared_structs.zig");

const Error = tersets.Error;

/// Number of distinct zero counts, i.e. the length of the distributions and of the
/// `round`/`representation` tables, and the first dimension of the dynamic-programming cache. A
/// 64-bit XOR result has between 0 and 63 leading (or trailing) zeros, one entry each.
pub const table_size = 64;

/// Maximum number of zero counts that can be transmitted, and the second dimension of the
/// dynamic-programming cache. How many are transmitted is itself sent in 5 bits, so at most
/// `2^5 = 32` fit on the wire (a count of 32 wraps to 0).
pub const max_positions = 32;

/// Maps a number of transmitted zero counts (0 to 64) to the number of bits one code then needs,
/// i.e. `ceil(log2(count))`. Mirrors `kPositionLength2Bits` in the reference implementation; only
/// counts up to `max_positions` occur in practice.
pub const position_length_to_bits = [65]u8{
    0, 0, 1, 2, 2, 3, 3, 3, 3,
    4, 4, 4, 4, 4, 4, 4, 4, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6,
};

/// Marks dynamic-programming cells that hold no valid cost yet. The reference C++ implementation
/// uses `std::numeric_limits<int>::max()` for the same purpose. Real costs must stay below this
/// value for the "unreached cell" comparisons to be unambiguous, which holds as long as callers
/// pass distributions whose total count times `table_size` is below `maxInt(i32)` — SerfXOR's
/// windows of 1000 values are far below that limit.
const no_valid_cost_value: i64 = math.maxInt(i32);

/// The set of zero counts selected for transmission: the first `len` entries of `values`, in
/// increasing order and always starting at zero. The encoder writes the set into the stream at
/// every window boundary and the decoder rebuilds its own tables from it.
pub const Positions = struct {
    values: [max_positions]u6 = @splat(0),
    len: usize = 0,
};

/// The selected zero counts and the total number of bits wasted by rounding `distribution` down
/// to them.
const BuildResult = struct {
    positions: Positions,
    total_app_cost: i64,
};

/// Select the zero counts to transmit for the observed `distribution` and rebuild the two lookup
/// tables from them. All three are indexed by exact zero count: `distribution[j]` is how many
/// values had exactly `j` zeros, `round[j]` is the greatest selected count less than or equal to
/// `j`, and `representation[j]` is the code transmitted for it, i.e. the position of `round[j]`
/// within the selected set.
/// How many counts to select is decided by trying each power of two `2^bits_per_code` in
/// increasing order with `bits_per_code <= min(position_length_to_bits[non_zeros_count], 5)`,
/// keeping the one that minimizes the total cost `total_count * bits_per_code + wasted_bits` and
/// stopping early once the code cost alone reaches the best total found so far. Mirrors
/// `InitRoundAndRepresentation` of the reference implementation.
pub fn initRoundAndRepresentation(
    distribution: *const [table_size]u32,
    representation: *[table_size]u6,
    round: *[table_size]u6,
) Positions {
    // Number of observed zero counts up to and including each index (index 0 always counts).
    var pre_non_zeros: [table_size]u32 = undefined;
    // Number of observed zero counts strictly after each index.
    var post_non_zeros: [table_size]u32 = undefined;

    var total_count: i64 = distribution[0];
    var non_zeros_count: usize = table_size;
    pre_non_zeros[0] = 1; // Index 0 is treated as observed like in the reference.
    for (1..table_size) |index| {
        total_count += distribution[index];
        if (distribution[index] == 0) {
            non_zeros_count -= 1;
            pre_non_zeros[index] = pre_non_zeros[index - 1];
        } else {
            pre_non_zeros[index] = pre_non_zeros[index - 1] + 1;
        }
    }
    for (0..table_size) |index| {
        post_non_zeros[index] = @as(u32, @intCast(non_zeros_count)) - pre_non_zeros[index];
    }

    // At most 5 bits are used to represent one code, hence at most 2^5 positions.
    const max_bits_per_code: usize = @min(position_length_to_bits[non_zeros_count], 5);
    var total_cost: i64 = no_valid_cost_value;
    var positions = Positions{};

    var bits_per_code: usize = 0;
    while (bits_per_code <= max_bits_per_code) : (bits_per_code += 1) {
        // Cost of transmitting one code per counted value with `bits_per_code` bits each.
        const present_cost: i64 = total_count * @as(i64, @intCast(bits_per_code));
        if (present_cost >= total_cost) break;
        const num = @as(usize, 1) << @intCast(bits_per_code);
        const result = selectZeroCounts(
            distribution,
            num,
            non_zeros_count,
            &pre_non_zeros,
            &post_non_zeros,
        );
        const temp_total_cost = result.total_app_cost + present_cost;
        if (temp_total_cost < total_cost) {
            total_cost = temp_total_cost;
            positions = result.positions;
        }
    }

    representation[0] = 0;
    round[0] = 0;
    var position_index: usize = 1;
    for (1..table_size) |j| {
        if (position_index < positions.len and j == positions.values[position_index]) {
            representation[j] = representation[j - 1] + 1;
            round[j] = @intCast(j);
            position_index += 1;
        } else {
            representation[j] = representation[j - 1];
            round[j] = round[j - 1];
        }
    }

    return positions;
}

/// Write `positions` to `bit_writer` as a 5-bit count (a count of 32 wraps to 0) followed by each
/// selected zero count as a 6-bit value, exactly like `WritePositions` of the reference
/// implementation. Returns the number of bits written, or `Error.WriteFailed`/
/// `Error.ByteStreamError` if a write fails.
pub fn writePositions(
    positions: Positions,
    bit_writer: *shared_structs.BulkBitWriter,
) Error!u32 {
    try bit_writer.writeBits(@as(u5, @truncate(positions.len)), 5);
    for (positions.values[0..positions.len]) |position| {
        try bit_writer.writeBits(position, 6);
    }
    return @intCast(5 + 6 * positions.len);
}

/// Select exactly `original_num` zero counts from `distribution` so that rounding every observed
/// count down to the nearest selected one wastes as few bits as possible. Solved by dynamic
/// programming over the zero counts: `dp[i][j]` is the fewest bits wasted by counts `0..i` when
/// count `i` is the `j`-th selected one, and `pre[i][j]` remembers which count was selected
/// before it in that optimum, so the selection can be recovered by walking the chain backwards.
/// When `original_num` exceeds `non_zeros_count`, the reference clamps the number selected by the
/// dynamic program and afterwards pads the result back to `original_num`; that block is ported as
/// well. `pre_non_zeros` and `post_non_zeros` are the prefix/suffix non-zero counts computed by
/// `initRoundAndRepresentation`.
fn selectZeroCounts(
    distribution: *const [table_size]u32,
    original_num: usize,
    non_zeros_count: usize,
    pre_non_zeros: *const [table_size]u32,
    post_non_zeros: *const [table_size]u32,
) BuildResult {
    const num = @min(original_num, non_zeros_count);

    var dp: [table_size][max_positions]i64 = @splat(@splat(no_valid_cost_value));
    // `pre[i][j]` holds the previously selected count; -1 terminates the chain. Only `pre[0][0]`
    // is read with `j == 0`, so initializing all to -1 is safe.
    var pre: [table_size][max_positions]i8 = @splat(@splat(-1));

    // Zero counts always start the selection and waste nothing.
    dp[0][0] = 0;
    pre[0][0] = -1;

    for (1..table_size) |i| {
        if (distribution[i] == 0) continue;
        // Counts after `i` must still accommodate the remaining `num - j` selections.
        var j: usize = if (num + i > table_size) @max(1, num + i - table_size) else 1;
        while (j <= i and j < num) : (j += 1) {
            if (i > 1 and j == 1) {
                var sum: i64 = 0;
                for (1..i) |k| {
                    sum += @as(i64, distribution[k]) * @as(i64, @intCast(k));
                }
                dp[i][1] = sum;
                pre[i][1] = 0;
            } else {
                if (pre_non_zeros[i] < j + 1 or post_non_zeros[i] < num - 1 - j) continue;
                var app_cost: i64 = no_valid_cost_value;
                var best_previous_count: usize = 0;
                var k: usize = j - 1;
                while (k <= i - 1) : (k += 1) {
                    if ((distribution[k] == 0 and k > 0) or
                        pre_non_zeros[k] < j or
                        post_non_zeros[k] < num - j) continue;
                    // Explicit check: skip cells the reference's guards leave undefined.
                    if (dp[k][j - 1] == no_valid_cost_value) continue;
                    var sum: i64 = dp[k][j - 1];
                    for (k + 1..i) |p| {
                        sum += @as(i64, distribution[p]) * @as(i64, @intCast(p - k));
                    }
                    if (app_cost > sum) {
                        app_cost = sum;
                        best_previous_count = k;
                        // A zero cost cannot be improved, terminate early.
                        if (sum == 0) break;
                    }
                }
                if (app_cost != no_valid_cost_value) {
                    dp[i][j] = app_cost;
                    pre[i][j] = @intCast(best_previous_count);
                }
            }
        }
    }

    // Pick the last selected count minimizing the total waste, including all counts after it.
    var temp_total_app_cost: i64 = no_valid_cost_value;
    var temp_best_last: isize = -1;
    for (num - 1..table_size) |i| {
        // A single selected count must be zero.
        if (num - 1 == 0 and i > 0) break;
        if ((distribution[i] == 0 and i > 0) or pre_non_zeros[i] < num) continue;
        if (dp[i][num - 1] == no_valid_cost_value) continue;
        var sum: i64 = dp[i][num - 1];
        for (i + 1..table_size) |j| {
            sum += @as(i64, distribution[j]) * @as(i64, @intCast(j - i));
        }
        if (temp_total_app_cost > sum) {
            temp_total_app_cost = sum;
            temp_best_last = @intCast(i);
        }
    }

    // Walk the `pre` chain backwards to recover the whole selection: exactly `num` links, since
    // every `dp[i][j>=1]` was set with a `pre` entry and only `pre[0][0]` terminates with -1.
    var positions = Positions{ .len = num };
    var count: usize = 1;
    var best_last = temp_best_last;
    while (best_last != -1) {
        positions.values[num - count] = @intCast(best_last);
        best_last = pre[@intCast(best_last)][num - count];
        count += 1;
    }

    if (original_num > non_zeros_count) {
        // Fewer observed zero counts than requested selections: pad back to `original_num` by
        // inserting unobserved counts in ascending order.
        var modified = Positions{ .len = original_num };
        var j: usize = 0;
        var k: usize = 0;
        while (j < original_num and k < num) {
            if (j - k < original_num - num and j < positions.values[k]) {
                modified.values[j] = @intCast(j);
                j += 1;
            } else {
                modified.values[j] = positions.values[k];
                j += 1;
                k += 1;
            }
        }
        positions = modified;
    }

    return .{ .positions = positions, .total_app_cost = temp_total_app_cost };
}

/// Test helper verifying the invariants that the SerfXOR codec relies on: `round[j] <= j`,
/// `representation[j] < positions.len`, and `round[j]` equals the greatest position `<= j`.
fn expectConsistentTables(
    positions: Positions,
    representation: *const [table_size]u6,
    round: *const [table_size]u6,
) !void {
    try testing.expect(positions.len >= 1);
    try testing.expectEqual(@as(u6, 0), positions.values[0]);
    for (0..table_size) |j| {
        try testing.expect(round[j] <= j);
        try testing.expect(representation[j] < positions.len);

        var greatest: u6 = 0;
        for (positions.values[0..positions.len]) |position| {
            if (position <= j and position >= greatest) greatest = position;
        }
        try testing.expectEqual(greatest, round[j]);
    }
}

test "zero count rounding spreads positions over a uniform distribution" {
    var distribution: [table_size]u32 = @splat(0);
    const expected_positions = [_]u6{ 0, 8, 16, 24, 32, 40, 48, 56 };
    for (expected_positions) |position| {
        distribution[position] = 100;
    }

    var representation: [table_size]u6 = @splat(0);
    var round: [table_size]u6 = @splat(0);
    const positions = initRoundAndRepresentation(&distribution, &representation, &round);

    // Eight equally weighted zero counts are selected exactly, wasting no bits.
    try testing.expectEqual(@as(usize, 8), positions.len);
    try testing.expectEqualSlices(u6, &expected_positions, positions.values[0..positions.len]);
    try expectConsistentTables(positions, &representation, &round);
}

test "zero count rounding handles a distribution concentrated at one index" {
    var distribution: [table_size]u32 = @splat(0);
    distribution[10] = 500;

    var representation: [table_size]u6 = @splat(0);
    var round: [table_size]u6 = @splat(0);
    const positions = initRoundAndRepresentation(&distribution, &representation, &round);

    // Zero is always selected; a second selection at index 10 absorbs the whole mass.
    try testing.expectEqual(@as(usize, 2), positions.len);
    try testing.expectEqual(@as(u6, 0), positions.values[0]);
    try testing.expectEqual(@as(u6, 10), positions.values[1]);
    try expectConsistentTables(positions, &representation, &round);

    // `round` must be a non-decreasing step function jumping exactly at the positions, and
    // `representation` must increment exactly at each position.
    for (1..table_size) |j| {
        try testing.expect(round[j] >= round[j - 1]);
        if (j == 10) {
            try testing.expectEqual(representation[j - 1] + 1, representation[j]);
        } else {
            try testing.expectEqual(representation[j - 1], representation[j]);
        }
    }
}

test "zero count rounding positions survive a write and read round trip" {
    const allocator = testing.allocator;

    var distribution: [table_size]u32 = @splat(0);
    distribution[0] = 10;
    distribution[5] = 300;
    distribution[22] = 40;
    distribution[46] = 7;

    var representation: [table_size]u6 = @splat(0);
    var round: [table_size]u6 = @splat(0);
    const positions = initRoundAndRepresentation(&distribution, &representation, &round);

    var bytes = ArrayList(u8).empty;
    defer bytes.deinit(allocator);
    var bit_writer = try shared_structs.BulkBitWriter.init(allocator, &bytes);
    const written_bits = try writePositions(positions, &bit_writer);
    try bit_writer.flushBits();

    try testing.expectEqual(@as(u32, @intCast(5 + 6 * positions.len)), written_bits);

    var bit_reader = shared_structs.BulkBitReader.init(bytes.items);
    const raw_count = try bit_reader.readBitsNoEof(u8, 5);
    const count: usize = if (raw_count == 0) 32 else raw_count;
    try testing.expectEqual(positions.len, count);
    for (positions.values[0..positions.len]) |position| {
        const decoded = try bit_reader.readBitsNoEof(u6, 6);
        try testing.expectEqual(position, decoded);
    }
}

test "zero count rounding handles gaps between observed zero counts" {
    var distribution: [table_size]u32 = @splat(0);
    // Irregular gaps between observed zero counts exercise the pre/post non-zero-count guards.
    distribution[1] = 3;
    distribution[2] = 900;
    distribution[13] = 17;
    distribution[14] = 1;
    distribution[40] = 250;
    distribution[63] = 60;

    var representation: [table_size]u6 = @splat(0);
    var round: [table_size]u6 = @splat(0);
    const positions = initRoundAndRepresentation(&distribution, &representation, &round);

    try expectConsistentTables(positions, &representation, &round);
}

test "zero count rounding pads back when fewer counts observed than requested" {
    // Only three observed zero counts {0, 1, 63} give `non_zeros_count == 3`, but
    // `initRoundAndRepresentation` still probes `original_num == 4`, triggering the pad-back
    // branch (`original_num > non_zeros_count`) that other tests hit only incidentally. Call
    // `selectZeroCounts` directly to pin the exact padded ordering.
    var distribution: [table_size]u32 = @splat(0);
    distribution[0] = 100;
    distribution[1] = 100;
    distribution[63] = 100;

    // Reproduce the prefix/suffix non-zero bookkeeping `initRoundAndRepresentation` computes
    // before it calls `selectZeroCounts`.
    var pre_non_zeros: [table_size]u32 = undefined;
    var post_non_zeros: [table_size]u32 = undefined;
    var non_zeros_count: usize = table_size;
    pre_non_zeros[0] = 1;
    for (1..table_size) |index| {
        if (distribution[index] == 0) {
            non_zeros_count -= 1;
            pre_non_zeros[index] = pre_non_zeros[index - 1];
        } else {
            pre_non_zeros[index] = pre_non_zeros[index - 1] + 1;
        }
    }
    for (0..table_size) |index| {
        post_non_zeros[index] = @as(u32, @intCast(non_zeros_count)) - pre_non_zeros[index];
    }
    try testing.expectEqual(@as(usize, 3), non_zeros_count);

    const result = selectZeroCounts(
        &distribution,
        4,
        non_zeros_count,
        &pre_non_zeros,
        &post_non_zeros,
    );

    // The three selections sit exactly on the observed counts (no waste); the pad-back inserts
    // the unobserved count 2 in ascending order, preserving the real ones.
    const expected_positions = [_]u6{ 0, 1, 2, 63 };
    try testing.expectEqual(@as(usize, 4), result.positions.len);
    try testing.expectEqualSlices(
        u6,
        &expected_positions,
        result.positions.values[0..result.positions.len],
    );
    try testing.expectEqual(@as(i64, 0), result.total_app_cost);
}

test "zero count rounding handles all-zero distribution except index zero" {
    var distribution: [table_size]u32 = @splat(0);
    distribution[0] = 100;

    var representation: [table_size]u6 = @splat(0);
    var round: [table_size]u6 = @splat(0);
    const positions = initRoundAndRepresentation(&distribution, &representation, &round);

    // A single observed zero count needs a single selection at zero and zero-bit codes.
    try testing.expectEqual(@as(usize, 1), positions.len);
    try testing.expectEqual(@as(u6, 0), positions.values[0]);
    try testing.expectEqual(@as(u8, 0), position_length_to_bits[positions.len]);
    for (0..table_size) |j| {
        try testing.expectEqual(@as(u6, 0), round[j]);
        try testing.expectEqual(@as(u6, 0), representation[j]);
    }
    try expectConsistentTables(positions, &representation, &round);
}
