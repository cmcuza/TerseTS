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
const ArrayList = std.ArrayList;
const assert = std.debug.assert;

const shared_structs = @import("../utilities/shared_structs.zig");
const BitStream = shared_structs.BitStream;

const Tester = @import("../tester.zig");

const U32_MAX: u32 = 0xFFFF_FFFF;

pub fn compress(values: []const f64, out: *ArrayList(u8), error_bound: f32) !void {
    // ensure there's at least the minimum amount of space we need
    try out.ensureTotalCapacity((values.len / 2) + (2 * @sizeOf(u64)));

    const adjust_digit = adjustDigitFromInput(values);
    try writeU64ToArrayList(out, adjust_digit);

    // TODO: make window size adjustable or just use the default that is being used in the serf paper
    const window_size = default_window_size;
    try writeU64ToArrayList(out, window_size);

    const max_diff = maxDiffFromErrorBound(error_bound);

    var state = try SerfXORCompress.start(max_diff, adjust_digit, window_size, out);
    state.buffer.fill = 2 * @bitSizeOf(u64);
    for (values, 0..) |v, v_idx| {
        std.log.warn("compress {} (idx {})", .{v, v_idx});
        try state.compress(v);
    }
    out.* = try state.close();
}

pub fn decompress(input: []const u8, out: *ArrayList(f64)) !void {
    // read adjust digit from input
    const adjust_digit = readU64FromArray(input);
    const window_size = readU64FromArray(input[@sizeOf(u64)..input.len]);

    var state = SerfXORDecompress.start(adjust_digit, window_size, input[(2 * @sizeOf(u64))..input.len]);
    var value_counter: usize = 0;
    while (state.haveNextValue()) {
        const v = state.readNextValue();
        std.log.warn("value {} counter {}", .{ v, value_counter });
        try out.append(v);
        value_counter += 1;
    }
}

pub const PostOfficeSolver = struct {
    pub const PostOfficeResult = struct {
        office_positions: SmallUnmanagedArrayList,
        total_app_cost: u32,

        pub fn init() PostOfficeResult {
            return PostOfficeResult {
                .office_positions = SmallUnmanagedArrayList.init(),
                .total_app_cost = 0,
            };
        }
    };

    const position_length_2bits = [65]u32 {
     0, 0, 1, 2, 2, 3, 3, 3, 3,
        4, 4, 4, 4, 4, 4, 4, 4,
        5, 5, 5, 5, 5, 5, 5, 5,
        5, 5, 5, 5, 5, 5, 5, 5,
        6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6,
    };

    pub fn initRoundAndRepresentation(distribution: *[64]u32, representation: *[64]u32, round: *[64]u32) SmallUnmanagedArrayList {
        var pre_non_zeros_count = [64]u32 {
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
        };
        var post_non_zeros_count = [64]u32 {
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
        };
        const total_count_and_non_zeros_counts = calTotalCountAndNonZerosCounts(distribution, &pre_non_zeros_count, &post_non_zeros_count);

        const max_z = @min(position_length_2bits[total_count_and_non_zeros_counts[1]], 5);
        var total_cost: u32 = U32_MAX;

        var positions = SmallUnmanagedArrayList.init();
        var present_cost: u32 = 0;
        var por = PostOfficeResult.init();
        for (0..(max_z + 1)) |z| {
            present_cost = total_count_and_non_zeros_counts[0] * @as(u32, @truncate(z));
            if (present_cost >= total_cost) {
                break;
            }

            const num = pow2z(@truncate(z));
            por = buildPostOffice(distribution, num, total_count_and_non_zeros_counts[1], &pre_non_zeros_count, &post_non_zeros_count);
            const temp_total_cost = por.total_app_cost + present_cost;
            if (temp_total_cost < total_cost) {
                total_cost = temp_total_cost;
                positions = por.office_positions;
            }
        }

        representation[0] = 0;
        round[0] = 0;
        var i: u32 = 1;
        for (1..distribution.len) |j| {
            const magic_code = @as(u32, @intFromBool((i < positions.len) and (j == positions.items[i])));
            representation[j] = representation[j - 1] + magic_code;
            // original implementation suggests a cmov is desired here
            if (magic_code != 0) {
                round[j] = @truncate(j);
            } else {
                round[j] = round[j - 1];
            }
            i += magic_code;

            // equivalent code to the above:
            // if (i < positions.len && j == positions[i]) {
            //     representation[j] = representation[j - 1] + 1;
            //     round[j] = j;
            //     i += 1;
            // } else {
            //     representation[j] = representation[j - 1];
            //     round[j] = round[j - 1];
            // }
        }

        return positions;
    }

    pub fn writePositions(positions: SmallUnmanagedArrayList, out: *BitStream) !u64 {
        var bit_count: u64 = 5;
        try out.write(positions.len, 5);
        for (0..positions.len) |i| {
            const pos = positions.items[i];
            try out.write(pos, 6);
            bit_count += 6;
        }
        return bit_count;
    }

    pub fn pow2z(i: u32) u32 {
        std.debug.assert(i < 6);
        return @as(u32, 1) << @truncate(i);
    }

    pub fn calTotalCountAndNonZerosCounts(arr: *[64]u32, out_pre_non_zeros_count: *[64]u32, out_post_non_zeros_count: *[64]u32) [2]u32 {
        std.debug.assert(arr.len == out_pre_non_zeros_count.len);
        std.debug.assert(arr.len == out_post_non_zeros_count.len);

        var non_zeros_count = arr.len;
        var total_count = arr[0];
        out_pre_non_zeros_count[0] = 1; // initial value
        for (1..arr.len) |i| {
            total_count += arr[i];
            // less comprehensible but faster code
            const magic_code = arr[i] == 0;
            non_zeros_count -= @as(usize, @intFromBool(magic_code));
            out_pre_non_zeros_count[i] = out_pre_non_zeros_count[i - 1] + @as(u32, @intFromBool(!magic_code));

            // equivalent code follows
            // if (arr[i] == 0) {
            //     non_zeros_count -= 1;
            //     out_pre_non_zeros_count[i] = out_pre_non_zeros_count[i - 1];
            // } else {
            //     out_pre_non_zeros_count[i] = out_pre_non_zeros_count[i - 1] + 1;
            // }
        }
        for (0..arr.len) |i| {
            out_post_non_zeros_count[i] = @as(u32, @truncate(non_zeros_count)) - out_pre_non_zeros_count[i];
        }
        return [2]u32 { total_count, @truncate(non_zeros_count) };
    }

    pub fn buildPostOffice(arr: *[64]u32, original_num: u32, non_zeros_count: u32, pre_non_zeros_count: []u32, post_non_zeros_count: []u32) PostOfficeResult {
        const num = @min(original_num, non_zeros_count);

        var dp = [64]SmallUnmanagedArrayList {
            SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num),
            SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num),
            SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num),
            SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num),
            SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num),
            SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num),
            SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num),
            SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num),
        };
        var pre = [64]SmallUnmanagedArrayList {
            SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num),
            SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num),
            SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num),
            SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num),
            SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num),
            SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num),
            SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num),
            SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num), SmallUnmanagedArrayList.initLen(num),
        };

        dp[0].items[0] = 0;
        pre[0].items[0] = 0; // WAS: -1

        for (1..arr.len) |iu| {
            const i = @as(u32, @truncate(iu));
            if (arr[i] == 0) {
                continue;
            }
            for (@max(1, @as(i64, @intCast(num)) + @as(i64, @intCast(i)) - @as(i64, @intCast(arr.len)))..@min(i + 1, num)) |ju| {
                const j = @as(u32, @truncate(ju));
                if ((i > 1) and (j == 1)) {
                    dp[i].items[j] = 0;
                    for (1..i) |ku| {
                        const k = @as(u32, @truncate(ku));
                        dp[i].items[j] += arr[k] * k;
                    }
                    pre[i].items[j] = 0;
                } else {
                    if ((pre_non_zeros_count[i] < j + 1) or (post_non_zeros_count[i] < num - 1 - j)) {
                        continue;
                    }
                    var app_cost: u32 = 0xFFFF_FFFF;
                    var pre_k: u32 = 0;
                    for ((j - 1)..i) |ku| {
                        const k = @as(u32, @truncate(ku));
                        if ((arr[k] == 0 and k > 0) or (pre_non_zeros_count[k] < j) or (post_non_zeros_count[k] < num - j)) {
                            continue;
                        }
                        var sum = dp[k].items[j - 1];
                        for ((k + 1)..i) |pu| {
                            const p = @as(u32, @truncate(pu));
                            sum += arr[p] * (p - k);
                        }
                        if (app_cost > sum) {
                            app_cost = sum;
                            pre_k = k;
                            if (sum == 0) {
                                break;
                            }
                        }
                    }
                    if (app_cost != 0xFFFF_FFFF) {
                        dp[i].items[j] = app_cost;
                        pre[i].items[j] = pre_k;
                    }
                }
            }
        }
        var temp_total_app_cost: u32 = 0xFFFF_FFFF;
        var temp_best_last: u32 = 0x7FFF_FFFF;
        for ((num - 1)..arr.len) |iu| {
            const i = @as(u32, @truncate(iu));
            if ((num - 1 == 0) and (i > 0)) {
                break;
            }
            if (((arr[i] == 0) and (i > 0)) or (pre_non_zeros_count[i] < num)) {
                continue;
            }
            var sum = dp[i].items[num - 1];
            for ((i + 1)..arr.len) |ju| {
                const j = @as(u32, @truncate(ju));
                sum += arr[j] * (j - 1);
            }
            if (temp_total_app_cost > sum) {
                temp_total_app_cost = sum;
                temp_best_last = i;
            }
        }

        var office_positions = SmallUnmanagedArrayList.initLen(num);
        var i: u32 = 1;
        while (temp_best_last != -1 and i <= num) {
            office_positions.items[num - i] = temp_best_last;
            temp_best_last = pre[temp_best_last].items[num - i];
            i += 1;
        }

        if (original_num > non_zeros_count) {
            var modifying_office_positions = SmallUnmanagedArrayList.initLen(original_num);
            var j: u32 = 0;
            var k: u32 = 0;
            while ((j < original_num) and (k < num)) {
                if ((j - k < original_num - num) and (j < office_positions.items[k])) {
                    modifying_office_positions.items[j] = j;
                    j += 1;
                } else {
                    modifying_office_positions.items[j] = office_positions.items[k];
                    j += 1;
                    k += 1;
                }
            }
            office_positions = modifying_office_positions;
        }

        return PostOfficeResult {
            .office_positions = office_positions,
            .total_app_cost = temp_total_app_cost,
        };
    }
};

pub const SerfXORCompress = struct {
    buffer: BitStream,
    max_diff: f64,
    adjust_digit: u64,
    
    stored_val: u64,
    
    leading_representation: [64]u32,
    leading_round: [64]u32,
    trailing_representation: [64]u32,
    trailing_round: [64]u32,
    leading_bits_per_value: u32,
    trailing_bits_per_value: u32,
    lead_distribution: [64]u32,
    trail_distribution: [64]u32,
    stored_leading_zeros: u32,
    stored_trailing_zeros: u32,
    
    window_size: u32,
    number_of_values_this_window: u32,
    compressed_size_this_window: u64,
    compression_ratio_last_window: f64,
    
    pub fn start(max_diff: f64, adjust_digit: u64, window_size: u32, out: *ArrayList(u8)) !SerfXORCompress {
        const buffer = BitStream.makeFromArrayList(out, 0);
        return SerfXORCompress {
            .buffer = buffer,
            .max_diff = max_diff,
            .adjust_digit = adjust_digit,
            
            .stored_val = @as(f64, 2.0),
            .leading_representation = [64]u32 {
                0, 0, 0, 0, 0, 0, 0, 0,
                1, 1, 1, 1, 2, 2, 2, 2,
                3, 3, 4, 4, 5, 5, 6, 6,
                7, 7, 7, 7, 7, 7, 7, 7,
                7, 7, 7, 7, 7, 7, 7, 7,
                7, 7, 7, 7, 7, 7, 7, 7,
                7, 7, 7, 7, 7, 7, 7, 7,
                7, 7, 7, 7, 7, 7, 7, 7,
            },
            .leading_round = [64]u32 {
                0, 0, 0, 0, 0, 0, 0, 0,
                8, 8, 8, 8, 12, 12, 12, 12,
                16, 16, 18, 18, 20, 20, 22, 22,
                24, 24, 24, 24, 24, 24, 24, 24,
                24, 24, 24, 24, 24, 24, 24, 24,
                24, 24, 24, 24, 24, 24, 24, 24,
                24, 24, 24, 24, 24, 24, 24, 24,
                24, 24, 24, 24, 24, 24, 24, 24,
            },
            .trailing_representation = [64]u32 {
                0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 1, 1,
                1, 1, 1, 1, 2, 2, 2, 2,
                3, 3, 3, 3, 4, 4, 4, 4,
                5, 5, 6, 6, 6, 6, 7, 7,
                7, 7, 7, 7, 7, 7, 7, 7,
                7, 7, 7, 7, 7, 7, 7, 7,
            },
            .trailing_round = [64]u32 {
                0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 22, 22,
                22, 22, 22, 22, 28, 28, 28, 28,
                32, 32, 32, 32, 36, 36, 36, 36,
                40, 40, 42, 42, 42, 42, 46, 46,
                46, 46, 46, 46, 46, 46, 46, 46,
                46, 46, 46, 46, 46, 46, 46, 46,
            },
            .leading_bits_per_value = 3,
            .trailing_bits_per_value = 3,
            .lead_distribution = [64]u32 {
                0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
            },
            .trail_distribution = [64]u32 {
                0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
            },
            .stored_leading_zeros = 0xFFFF_FFFF,
            .stored_trailing_zeros = 0xFFFF_FFFF,
            .window_size = window_size,
            .number_of_values_this_window = 0,
            .compressed_size_this_window = 0,
            .compression_ratio_last_window = 0,
        };
    }
    
    // NetSerfXORCompressor::Compress()
    pub fn compress(state: *SerfXORCompress, v: f64) !void {
        var this_val: u64 = state.stored_val;
        if (@abs(@as(f64, @bitCast(state.stored_val)) - @as(f64, @floatFromInt(state.adjust_digit)) - v) > state.max_diff) {
            // much like the original implementation, we ignore any sorts of special cases that might occur here
            const adjust_value = v + @as(f64, @floatFromInt(state.adjust_digit));
            this_val = state.findAppLong(adjust_value - state.max_diff, adjust_value + state.max_diff, v);
        }
        try state.addValue(this_val);
        state.stored_val = this_val;
    }
    
    pub fn close(state: *SerfXORCompress) !ArrayList(u8) {
        // std.math.nan() produces a quiet NaN
        state.compressed_size_this_window += try state.compressValueInternal(@as(u64, @bitCast(std.math.nan(f64))));
        
        try state.updatePositionsIfNeeded();

        return state.buffer.buffer.toManaged(state.buffer.allocator);
    }
    
    pub fn delete(state: *SerfXORCompress) void {
        state.buffer.destroy();
        mem.zero(state);
    }
    
    pub fn addValue(state: *SerfXORCompress, v: u64) !void {
        // NOTE: transmission header shenanigans removed
        if (state.number_of_values_this_window >= state.window_size) {
            try state.updatePositionsIfNeeded();
        }
        state.compressed_size_this_window += try state.compressValueInternal(v);
        state.number_of_values_this_window += 1;
    }

    pub fn updatePositionsIfNeeded(state: *SerfXORCompress) !void {
        const compression_ratio_this_window = @as(f64, @floatFromInt(state.buffer.fill)) / (@as(f64, @floatFromInt(state.number_of_values_this_window)) * 64);
        if (state.compression_ratio_last_window < compression_ratio_this_window) {
            // update positions

            const lead_positions = PostOfficeSolver.initRoundAndRepresentation(&state.lead_distribution, &state.leading_representation, &state.leading_round);
            //const leading_bits_per_value = PostOfficeSolver.position_length_2bits[lead_positions.len];
            const trail_positions = PostOfficeSolver.initRoundAndRepresentation(&state.trail_distribution, &state.trailing_representation, &state.trailing_round);
            //const trailing_bits_per_value = PostOfficeSolver.position_length_2bits[trail_positions.len];

            try state.buffer.writeBit(1);

            _ = try PostOfficeSolver.writePositions(lead_positions, &state.buffer);
            _ = try PostOfficeSolver.writePositions(trail_positions, &state.buffer);
        } else {
            try state.buffer.writeBit(0);
        }
        state.compression_ratio_last_window = compression_ratio_this_window;
        state.compressed_size_this_window = 0;
        state.number_of_values_this_window = 0;
        state.lead_distribution = [64]u32 {
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
        };
        state.trail_distribution = state.lead_distribution;
    }
    
    pub fn compressValueInternal(state: *SerfXORCompress, value: u64) !u64 {
        const starting_fill = state.buffer.fill;
        const xor_result = state.stored_val ^ value;
        if (xor_result == 0) {
            // case 01
            try state.buffer.writeBit(0);
            try state.buffer.writeBit(1);
            return state.buffer.fill - starting_fill;
        }
        
        const leading_count = @clz(xor_result);
        const trailing_count = @ctz(xor_result);
        const leading_zeros = state.leading_round[leading_count];
        const trailing_zeros = state.trailing_round[trailing_count];
        state.lead_distribution[leading_count] += 1;
        state.trail_distribution[trailing_count] += 1;
        
        if ((leading_zeros >= state.stored_leading_zeros)
            and (trailing_zeros >= state.stored_trailing_zeros)
            and ((leading_zeros - state.stored_leading_zeros) + (trailing_zeros - state.stored_trailing_zeros)
                < (1 + state.leading_bits_per_value + state.trailing_bits_per_value))) {
            // case 1
            const center_bits = 64 - state.stored_leading_zeros - state.stored_trailing_zeros;
            if (1 + center_bits > 64) {
                try state.buffer.writeBit(1);
                try state.buffer.write(xor_result >> @truncate(state.stored_trailing_zeros), @truncate(center_bits));
            } else {
                try state.buffer.write((@as(u64, 1) << @truncate(center_bits)) | (xor_result >> @truncate(state.stored_trailing_zeros)), @truncate(1 + center_bits));
            }
            return state.buffer.fill - starting_fill;
        }
        
        // case 00
        state.stored_leading_zeros = leading_zeros;
        state.stored_trailing_zeros = trailing_zeros;
        const center_bits = 64 - leading_zeros - trailing_zeros;
        const len = 2 + state.leading_bits_per_value + state.trailing_bits_per_value + center_bits;
        if (len > 64) {
            var v = state.leading_representation[leading_zeros] << @truncate(state.trailing_bits_per_value);
            v |= state.trailing_representation[trailing_zeros];
            try state.buffer.write(v, @truncate(len - center_bits));
            try state.buffer.write(xor_result >> @truncate(trailing_zeros), @truncate(center_bits));
        } else {
            var v = @as(u64, @intCast(state.leading_representation[leading_zeros])) << @truncate(state.trailing_bits_per_value + center_bits);
            v |= @as(u64, @intCast(state.trailing_representation[trailing_zeros])) << @truncate(center_bits);
            v |= xor_result >> @truncate(trailing_zeros);
            try state.buffer.write(v, @truncate(len));
        }
        return state.buffer.fill - starting_fill;
    }
    
    pub fn bitWeight(x: u8) u64 {
        std.debug.assert(x < 64);
        return @as(u64, 1) << @truncate(x);
    }
    
    pub fn findAppLong(state: *SerfXORCompress, lo: f64, hi: f64, v: f64) u64 {
        var v_lo = lo;
        var v_hi = hi;
        var sign: u64 = 0;
        if (lo >= 0) {
            // both positive
            // nothing to do
        } else if (hi <= 0) {
            // both negative
            v_lo = -hi;
            v_hi = -lo;
            sign = 0x8000000000000000;
        } else if (state.stored_val >> 63 == 0) {
            // consider only positive side to make more leading zeros
            v_lo = 0;
        } else {
            // consider only negative side to make more leading zeros
            v_lo = 0;
            v_hi = -lo;
            sign = 0x8000000000000000;
        }
        return findAppLongSign(v_lo, v_hi, sign, v, state.stored_val, state.max_diff, @floatFromInt(state.adjust_digit));
    }
    
    pub fn findAppLongSign(min_d: f64, max_d: f64, sign: u64, original: f64, last_long: u64, max_diff: f64, adjust_digit: f64) u64 {
        // mask off sign bit
        const min = @as(u64, @bitCast(min_d)) & 0x7fffffffffffffff;
        const max: u64 = @bitCast(max_d);
        const leading_zeros = @clz(min ^ max);
        
        var front_mask = @as(u64, 0xffffffffffffffff) << @truncate(64 - leading_zeros);
        var shift = 64 - @as(i64, @intCast(leading_zeros));
        
        var result: i64 = 0;
        var diff: f64 = 0;
        var append: u64 = 0;
        
        while (shift >= 0) {
            const front = front_mask & min;
            const rear = (~front_mask) & last_long;
            
            append = rear | front;
            
            const cond1: bool = ((append >= min) and (append <= max));
            
            // the way this is written in the original implementation suggests
            // that they want this to compile to a CMOV instruction
            if (cond1) {
                result = @as(i64, @bitCast(append ^ @as(u64, @bitCast(sign))));
            } else {
                result = 0;
            }
            
            diff = @as(f64, @bitCast(result)) - adjust_digit - original;
            var diff_satisfied = (diff >= -max_diff) and (diff <= max_diff);
            
            if (cond1 and diff_satisfied) {
                return @bitCast(result);
            }
            
            append = (append + bitWeight(@intCast(shift))) & 0x7fffffffffffffff;
            
            const cond2: bool = append <= max;
            // again, way this is written in the original impl suggests that they want this to compile to a CMOV
            if (cond2) {
                result = @as(i64, @bitCast(append ^ sign));
            } else {
                result = 0;
            }
            
            diff = @as(f64, @bitCast(result)) - adjust_digit - original;
            diff_satisfied = (diff >= -max_diff) and (diff <= max_diff);
            
            if (cond2 and diff_satisfied) {
                return @bitCast(result);
            }
            
            front_mask >>= 1;
            shift -= 1;
        }
        
        // can't find a value that satifies constraints, return original value
        return @as(u64, @bitCast(original + adjust_digit));
    }
};

pub const SerfXORDecompress = struct {
    input: []const u8,
    buffer: BitStream,
    adjust_digit: u64,
    window_size: u64,
    leading_representation: SmallUnmanagedArrayList,
    trailing_representation: SmallUnmanagedArrayList,
    leading_bits_per_value: u32,
    trailing_bits_per_value: u32,
    stored_leading_zeros: u32,
    stored_trailing_zeros: u32,
    number_of_values: u64,
    stored_val: u64,

    stored_next_value: f64,
    have_stored_next_value: bool,
    
    pub fn start(adjust_digit: u64, window_size: u64, input: []const u8) SerfXORDecompress {
        var state = SerfXORDecompress {
            .input = input,
            .buffer = BitStream.makeFromArrayUnmanaged(@constCast(input)),
            .adjust_digit = adjust_digit,
            .window_size = window_size,
            .leading_representation = SmallUnmanagedArrayList.init(),
            .trailing_representation = SmallUnmanagedArrayList.init(),
            .leading_bits_per_value = 3,
            .trailing_bits_per_value = 3,
            .stored_leading_zeros = 0,
            .stored_trailing_zeros = 0,
            
            .number_of_values = 0,
            .stored_val = 0,
            .stored_next_value = 0,
            .have_stored_next_value = false,
        };
        state.buffer.buffer.items.len = input.len;

        state.leading_representation.len = 8;
        state.leading_representation.items[0] = 0;
        state.leading_representation.items[1] = 8;
        state.leading_representation.items[2] = 12;
        state.leading_representation.items[3] = 16;
        state.leading_representation.items[4] = 18;
        state.leading_representation.items[5] = 20;
        state.leading_representation.items[6] = 22;
        state.leading_representation.items[7] = 24;

        state.trailing_representation.len = 8;
        state.trailing_representation.items[0] = 0;
        state.trailing_representation.items[1] = 22;
        state.trailing_representation.items[2] = 28;
        state.trailing_representation.items[3] = 32;
        state.trailing_representation.items[4] = 36;
        state.trailing_representation.items[5] = 40;
        state.trailing_representation.items[6] = 42;
        state.trailing_representation.items[7] = 46;

        return state;
    }
    
    pub fn readNextValue(state: *SerfXORDecompress) f64 {
        if (!state.have_stored_next_value) {
            state.readAndStoreNextValue();
        }
        state.have_stored_next_value = false;
        return state.stored_next_value;
    }

    // TODO: need to have this return an error so I can just keep doing it until it returns an error (that error being: can't read more, boss)
    pub fn readAndStoreNextValue(state: *SerfXORDecompress) void {
        if (state.have_stored_next_value) {
            return; // no-op
        }

        if (state.number_of_values >= state.window_size) {
            state.updatePositionsIfNeeded();
        }

        state.nextValueInternal();
        state.number_of_values += 1;
        state.stored_next_value = @as(f64, @bitCast(state.stored_val)) - @as(f64, @floatFromInt(state.adjust_digit));
        state.have_stored_next_value = true;
        return;
    }

    pub fn haveNextValue(state: *SerfXORDecompress) bool {
        if (state.buffer.buffer.items.len == 0) {
            return false;
        }

        if (state.buffer.fill >= (state.buffer.buffer.items.len * 8) - 1) {
            return false;
        }

        if (!state.have_stored_next_value) {
            state.readAndStoreNextValue();
        }
        // TODO: try to read ahead and see if there's a valid window + window end
        return !std.math.isNan(state.stored_next_value);

        // presumption: an entire window is gonna be no less than a byte
        //return state.buffer.fill == 0 or !(state.number_of_values == 0 and state.buffer.fill > ((state.input.len - 1) * 8));
    }

    pub fn nextValueInternal(state: *SerfXORDecompress) void {
        var value: u64 = 0;
        var center_bits: u64 = 0;
        if (state.buffer.readBit() == 1) {
            // case 1
            if (state.stored_leading_zeros + state.stored_trailing_zeros < 64) {
                center_bits = 64 - state.stored_leading_zeros - state.stored_trailing_zeros;
                value = state.buffer.read(@truncate(center_bits)) << @truncate(state.stored_trailing_zeros);
            } else {
                // no center bits to read
            }
            state.stored_val = state.stored_val ^ value;
        } else if (state.buffer.readBit() == 0) {
            // case 00
            const lead_and_trail = state.buffer.read(state.leading_bits_per_value + state.trailing_bits_per_value);
            const lead = lead_and_trail >> @truncate(state.trailing_bits_per_value);
            const trail = ~(@as(u64, 0xffffffffffffffff) << @truncate(state.trailing_bits_per_value)) & lead_and_trail;

            std.log.warn("lead_and_trail: {x}, lead: {x}, trail: {x},\n\tleading_bits_per_value: {}, trailing_bits_per_value: {}", .{lead_and_trail, lead, trail, state.leading_bits_per_value, state.trailing_bits_per_value});

            assert(state.leading_representation.len > lead);
            state.stored_leading_zeros = state.leading_representation.items[lead];

            assert(state.trailing_representation.len > trail);
            state.stored_trailing_zeros = state.trailing_representation.items[trail];

            std.log.warn("\tstored leading z: {}, trailing z: {}", .{state.stored_leading_zeros, state.stored_trailing_zeros});

            if (state.stored_leading_zeros + state.stored_trailing_zeros < 64) {
                center_bits = @as(u64, 64) - state.stored_leading_zeros - state.stored_trailing_zeros;
                value = state.buffer.read(@truncate(center_bits)) << @truncate(state.stored_trailing_zeros);
            } else {
                // not center bits to read
            }
            state.stored_val = state.stored_val ^ value;
        } else {
            // case 01
            // nothing to do, stored val doesn't change
        }
    }

    pub fn updatePositionsIfNeeded(state: *SerfXORDecompress) void {
        if (state.buffer.readBit() == 1) {
            state.updateLeadingRepresentation();
            state.updateTrailingRepresentation();
        }
        state.number_of_values = 0;
    }

    pub fn updateLeadingRepresentation(state: *SerfXORDecompress) void {
        var num = state.buffer.read(5);
        if (num == 0) {
            num = 32;
        }
        state.leading_bits_per_value = PostOfficeSolver.position_length_2bits[num];
        state.leading_representation.len = @truncate(num);
        for (0..num) |i| {
            state.leading_representation.items[i] = @truncate(state.buffer.read(6));
        }
    }

    pub fn updateTrailingRepresentation(state: *SerfXORDecompress) void {
        var num = state.buffer.read(5);
        if (num == 0) {
            num = 32;
        }
        state.trailing_bits_per_value = PostOfficeSolver.position_length_2bits[num];
        state.trailing_representation.len = @truncate(num);
        for (0..num) |i| {
            state.trailing_representation.items[i] = @truncate(state.buffer.read(6));
        }
    }
};

const SmallUnmanagedArrayList = struct {
    items: [64]u32,
    len: u32,

    pub fn init() SmallUnmanagedArrayList {
        return SmallUnmanagedArrayList {
            .items = [64]u32 {
                0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
            },
            .len = 0,
        };
    }

    pub fn initLen(len: u32) SmallUnmanagedArrayList {
        var self = init();
        self.len = len;
        return self;
    }

    pub fn zero(self: *SmallUnmanagedArrayList) void {
        for (0..64) |i| {
            self.items[i] = 0;
        }
        self.len = 0;
    }

    pub fn push(self: *SmallUnmanagedArrayList, v: u32) void {
        std.debug.assert(self.len < 64);
        self.items[self.len] = v;
        self.len += 1;
    }
};

const default_window_size: u64 = 1024;

fn maxDiffFromErrorBound(error_bound: f32) f64 {
    return @floatCast(error_bound);
}

fn adjustDigitFromInput(input: []const f64) u64 {
    var maximum: f64 = input[0];
    var minimum: f64 = input[0];
    for (input) |v| {
        maximum = @max(maximum, v);
        minimum = @min(minimum, v);
    }
    // NOTE (sio): taken from Serf/test/adjust_digit_calculator.cpp
    const u = @as(u6, @truncate(@as(u32, @intFromFloat(@ceil(@log2(@floor(maximum) - @floor(minimum) + 1))))));
    const lambda = (@as(i64, 1) << u) - @as(i64, @intFromFloat(@floor(minimum)));
    return @max(0, lambda);
}


fn writeU64ToArrayList(out: *ArrayList(u8), value: u64) !void {
    try out.append(@as(u8, @truncate(value >> 56)));
    try out.append(@as(u8, @truncate(value >> 48)));
    try out.append(@as(u8, @truncate(value >> 40)));
    try out.append(@as(u8, @truncate(value >> 32)));
    try out.append(@as(u8, @truncate(value >> 24)));
    try out.append(@as(u8, @truncate(value >> 16)));
    try out.append(@as(u8, @truncate(value >> 8)));
    try out.append(@as(u8, @truncate(value)));
}

fn readU64FromArray(input: []const u8) u64 {
    return (@as(u64, input[0]) << 56)
        | (@as(u64, input[1]) << 48)
        | (@as(u64, input[2]) << 40)
        | (@as(u64, input[3]) << 32)
        | (@as(u64, input[4]) << 24)
        | (@as(u64, input[5]) << 16)
        | (@as(u64, input[6]) << 8)
        | (@as(u64, input[7]));
}

test "serf-xor can compress and decompress random quantizable data" {
    var prng = std.Random.DefaultPrng.init(@bitCast(std.time.milliTimestamp()));
    const random = prng.random();
    for (1..100000) |inputs_length| {
        const error_bound = Tester.generateBoundedRandomValue(f64, 0.00001, 30.0, random);
        std.log.warn("inputs length {} error bound {}", .{ inputs_length, error_bound });

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

        std.log.warn("inputs len: {}, decompressed len: {}", .{ inputs.items.len, decompressed.items.len });
        for (inputs.items, decompressed.items) |expected, actual| {
            try testing.expectApproxEqAbs(expected, actual, error_bound);
        }
    }
}

// TODO: serf-xor tests
