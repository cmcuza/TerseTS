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

//! Implementation of the Sim-Piece algorithm from the paper
//! "Xenophon Kitsios, Panagiotis Liakos, Katia Papakonstantinopoulou, and Yannis Kotidis.
//! Sim-Piece: Highly Accurate Piecewise Linear Approximation through Similar Segment Merging.
//! Proc. VLDB Endow. 16, 8 2023.
//! https://doi.org/10.14778/3594512.3594521".

const std = @import("std");
const math = std.math;
const mem = std.mem;
const testing = std.testing;
const ArrayList = std.ArrayList;
const HashMap = std.HashMap;
const AutoHashMap = std.AutoHashMap;

const tersets = @import("../tersets.zig");
const Error = tersets.Error;
const DiscretePoint = tersets.DiscretePoint;

/// `SegmentMetadata` stores the information about a approximated segment during the execution
/// of Sim-Piece. It stores the starting time of the segment in `start_time` and the slopes of
/// the upper and lower bounds that constraint the linear approximation in that segment.
const SegmentMetadata = struct {
    start_time: usize,
    upper_bound_slope: f64,
    lower_bound_slope: f64,
};

const HashF64Context = struct {
    // const epsilon: f64 = 1e-12;

    pub fn hash(_: HashF64Context, value: f64) u64 {
        return @as(u64, @bitCast(value));
    }

    pub fn eql(_: HashF64Context, value_one: f64, value_two: f64) bool {
        return value_one == value_two;
    }
};

/// Compress `uncompressed_values` within `error_bound` using "Sim-Piece" and write the
/// result to `compressed_values`. If an error occurs it is returned.
pub fn compressSimPiece(
    uncompressed_values: []const f64,
    compressed_values: *ArrayList(u8),
    allocator: mem.Allocator,
    error_bound: f32,
) Error!void {
    var segments_metadata_map = HashMap(
        f64,
        ArrayList(SegmentMetadata),
        HashF64Context,
        std.hash_map.default_max_load_percentage,
    ).init(allocator);
    defer {
        var iterator = segments_metadata_map.iterator();
        while (iterator.next()) |entry| {
            entry.value_ptr.*.deinit();
        }
        segments_metadata_map.deinit();
    }

    // Sim-Piece Phase 1: Compute the Segment Metadata Map.
    try computeSegmentsMetadataMap(
        uncompressed_values,
        &segments_metadata_map,
        allocator,
        error_bound,
    );

    var merged_segments_metadata = ArrayList(SegmentMetadata).init(allocator);
    defer merged_segments_metadata.deinit();

    // Sim-Piece Phase 2: Merge the segments.
    try computeMergedSegmentsMetadata(segments_metadata_map, &merged_segments_metadata, allocator);

    // var reshaped_segments_metadata = AutoHashMap(usize, HashMap(
    //     f64,
    //     ArrayList(usize),
    //     HashF64Context,
    //     std.hash_map.default_max_load_percentage,
    // )).init(allocator);
    // reshaped_segments_metadata.deinit();

    // Sim-Piece Phase 3: Reshape segment metadata to output.
    // reshapeSegmentsMetadata(
    //     merged_segments_metadata,
    //     &reshaped_segments_metadata,
    // );
    try compressed_values.append(0);
}

/// Sim-Piece Phase 1: Create the HashMap between the intercepts and each segment information
/// stored on the `SegmentMetadata`.
fn computeSegmentsMetadataMap(
    uncompressed_values: []const f64,
    segments_metadata_map: *HashMap(
        f64,
        ArrayList(SegmentMetadata),
        HashF64Context,
        std.hash_map.default_max_load_percentage,
    ),
    allocator: mem.Allocator,
    error_bound: f32,
) !void {
    // Adjust the error bound to avoid exceeding it during decompression.
    const adjusted_error_bound = if (error_bound > 0)
        error_bound - tersets.ErrorBoundMargin
    else
        error_bound;

    var upper_bound_slope: f64 = math.floatMax(f64);
    var lower_bound_slope: f64 = -math.floatMax(f64);

    // Initialize the `start_point` with the first uncompressed value.
    var start_point: DiscretePoint = .{ .time = 0, .value = uncompressed_values[0] };

    var quantized_start_value = quantize(uncompressed_values[0], error_bound);

    // First point already part of `current_segment`, next point is at index one.
    var current_timestamp: usize = 1;
    while (current_timestamp < uncompressed_values.len) : (current_timestamp += 1) {
        const end_point: DiscretePoint = .{
            .time = current_timestamp,
            .value = uncompressed_values[current_timestamp],
        };

        const segment_size: f64 = @floatFromInt(current_timestamp - start_point.time);
        const upper_limit: f64 = upper_bound_slope * segment_size + quantized_start_value;
        const lower_limit: f64 = lower_bound_slope * segment_size + quantized_start_value;

        if ((upper_limit < (end_point.value - adjusted_error_bound)) or
            ((lower_limit > (end_point.value + adjusted_error_bound))))
        {
            // The new point is outside the upper and lower limit. Record a new segment metadata in
            // `segments_metadata_map` associated to `quantized_start_value`.
            std.debug.print("***Recording, start point {} ", .{start_point.time});
            std.debug.print("upper and lower bounds {:.4} {:.4} ", .{ upper_bound_slope, lower_bound_slope });
            std.debug.print("intercept {:.4} \n", .{quantized_start_value});
            try addSegmentMetadata(segments_metadata_map, .{
                .start_time = start_point.time,
                .upper_bound_slope = upper_bound_slope,
                .lower_bound_slope = lower_bound_slope,
            }, quantized_start_value, allocator);

            start_point = end_point;
            quantized_start_value = quantize(start_point.value, error_bound);
            upper_bound_slope = math.floatMax(f64);
            lower_bound_slope = -math.floatMax(f64);
        } else {
            // The new point is within the upper and lower bounds. Update the bounds' slopes.

            const new_upper_bound_slope: f64 =
                (end_point.value + adjusted_error_bound - quantized_start_value) / segment_size;
            const new_lower_bound_slope: f64 =
                (end_point.value - adjusted_error_bound - quantized_start_value) / segment_size;

            if (end_point.value + adjusted_error_bound < upper_limit)
                upper_bound_slope = @max(new_upper_bound_slope, lower_bound_slope);
            if (end_point.value - adjusted_error_bound > lower_limit)
                lower_bound_slope = @min(new_lower_bound_slope, upper_bound_slope);
        }
    }

    const segment_size = current_timestamp - start_point.time;
    if (segment_size >= 1) {
        if (segment_size == 1) {
            upper_bound_slope = 0;
            lower_bound_slope = 0;
        }
        std.debug.print("***Recording, start point {} ", .{start_point.time});
        std.debug.print("upper and lower bounds {:.4} {:.4} ", .{ upper_bound_slope, lower_bound_slope });
        std.debug.print("intercept {:.4} \n", .{quantized_start_value});
        try addSegmentMetadata(segments_metadata_map, .{
            .start_time = start_point.time,
            .upper_bound_slope = upper_bound_slope,
            .lower_bound_slope = lower_bound_slope,
        }, quantized_start_value, allocator);
    }
}

/// Sim-Piece Phase 2.
fn computeMergedSegmentsMetadata(
    segments_metadata_map: HashMap(
        f64,
        ArrayList(SegmentMetadata),
        HashF64Context,
        std.hash_map.default_max_load_percentage,
    ),
    merged_segments_metadata: *ArrayList(SegmentMetadata),
    allocator: mem.Allocator,
) !void {
    var timestamps_array = ArrayList(usize).init(allocator);
    defer timestamps_array.deinit();

    std.debug.print("\n", .{});
    std.debug.print("Count of segments {}\n", .{segments_metadata_map.count()});
    var iterator = segments_metadata_map.iterator();

    while (iterator.next()) |entry| {
        std.debug.print("B={} \n", .{entry.key_ptr.*});
        const metadata_array = entry.value_ptr;
        mem.sort(
            SegmentMetadata,
            metadata_array.items,
            {},
            comptime compareMetadata(SegmentMetadata),
        );
        var merge_metadata: SegmentMetadata = .{
            .start_time = 0.0,
            .lower_bound_slope = metadata_array.items[0].lower_bound_slope,
            .upper_bound_slope = metadata_array.items[0].upper_bound_slope,
        };
        for (1..metadata_array.items.len) |index| {
            const current_metadata = metadata_array.items[index];
            try timestamps_array.append(current_metadata.start_time);
            if ((current_metadata.lower_bound_slope <= merge_metadata.upper_bound_slope) and (current_metadata.upper_bound_slope >= merge_metadata.lower_bound_slope)) {
                // The merged segment metadata belongs to the current segment metadata.
                try timestamps_array.append(current_metadata.start_time);
                merge_metadata.lower_bound_slope = @max(
                    merge_metadata.lower_bound_slope,
                    current_metadata.lower_bound_slope,
                );
                merge_metadata.upper_bound_slope = @min(
                    merge_metadata.upper_bound_slope,
                    current_metadata.upper_bound_slope,
                );
            } else {
                // A new merged segment metadata needs to be created.
                for (timestamps_array.items) |timestamp| {
                    try merged_segments_metadata.append(SegmentMetadata{
                        .start_time = timestamp,
                        .lower_bound_slope = merge_metadata.lower_bound_slope,
                        .upper_bound_slope = merge_metadata.upper_bound_slope,
                    });
                }
            }
        }
    }
    std.debug.print("Count of segments {}\n", .{segments_metadata_map.count()});
}

/// Sim-Piece Phase 3. Reshape the input segment metadata array.
// fn reshapeSegmentsMetadata(merged_segments_metadata: ArrayList(SegmentMetadata), reshaped_segments_metadata: *ArrayHashMap(f64, ArrayHashMap(f64, ArrayList(usize)))) !void {}

/// Quantizes the given `value` by the specified `error_bound`. This process ensures that
/// the quantized value remains within the error bound of the original value.
fn quantize(value: f64, error_bound: f32) f64 {
    return @floor(value / error_bound) * error_bound;
}

fn addSegmentMetadata(
    metadata_map: *HashMap(
        f64,
        ArrayList(SegmentMetadata),
        HashF64Context,
        std.hash_map.default_max_load_percentage,
    ),
    metadata: SegmentMetadata,
    quantize_value: f64,
    allocator: mem.Allocator,
) !void {
    const get_result = try metadata_map.getOrPut(quantize_value);
    if (!get_result.found_existing) {
        get_result.value_ptr.* = ArrayList(SegmentMetadata).init(allocator);
    }
    try get_result.value_ptr.*.append(metadata);
}

fn compareMetadata(comptime T: type) fn (void, T, T) bool {
    return struct {
        pub fn inner(_: void, metadata_one: T, metadata_two: T) bool {
            return metadata_one.lower_bound_slope < metadata_two.lower_bound_slope;
        }
    }.inner;
}

/// Decompress `compressed_values` produced by "Swing Filter" and "Slide Filter" and write the
/// result to `decompressed_values`. If an error occurs it is returned.
pub fn decompress(
    compressed_values: []const u8,
    decompressed_values: *ArrayList(f64),
) Error!void {
    // The compressed representation is composed of three values: (start_value, end_value, end_time)
    // all of type 64-bit float.
    if (compressed_values.len % 24 != 0) return Error.IncorrectInput;

    const compressed_lines_and_index = mem.bytesAsSlice(f64, compressed_values);

    for (compressed_lines_and_index, 0..) |item, index| {
        decompressed_values.items[index] = item;
    }
}

test "f64 context can hash" {
    const allocator = testing.allocator;
    var f64_hash_map = HashMap(
        f64,
        f64,
        HashF64Context,
        std.hash_map.default_max_load_percentage,
    ).init(allocator);
    defer f64_hash_map.deinit();
    var rnd = std.rand.DefaultPrng.init(@as(u64, @bitCast(std.time.milliTimestamp())));

    // Even a small deviation like this should not impact the ability to hash the numbers.
    const deviation_margin = 1e-16;
    for (0..100) |_| {
        const rand_number = rnd.random().float(f64) - 0.5;
        try f64_hash_map.put(rand_number, rand_number);
        try f64_hash_map.put(rand_number - deviation_margin, rand_number - deviation_margin);
        try f64_hash_map.put(rand_number + deviation_margin, rand_number + deviation_margin);
    }

    // All elements are expected to be hashed independently, without collisions.
    try testing.expectEqual(300, f64_hash_map.count());

    var iterator = f64_hash_map.iterator();
    while (iterator.next()) |entry| {
        const key = entry.key_ptr.*;
        const value = entry.value_ptr.*;
        try testing.expectEqual(key, value);
    }
}

test "hashmap can map f64 to segment metadata array list" {
    const allocator = testing.allocator;
    var f64_metadata_hash_map = HashMap(
        f64,
        ArrayList(SegmentMetadata),
        HashF64Context,
        std.hash_map.default_max_load_percentage,
    ).init(allocator);
    defer {
        var iterator = f64_metadata_hash_map.iterator();
        while (iterator.next()) |entry| {
            entry.value_ptr.*.deinit();
        }
        f64_metadata_hash_map.deinit();
    }

    var f64_usize_hash_map = HashMap(
        f64,
        usize,
        HashF64Context,
        std.hash_map.default_max_load_percentage,
    ).init(allocator);
    defer f64_usize_hash_map.deinit();

    var rnd = std.rand.DefaultPrng.init(@as(u64, @bitCast(std.time.milliTimestamp())));

    for (0..200) |_| {
        const rand_number = @floor((rnd.random().float(f64) - 0.5) * 100) / 10;

        const count_map_result = try f64_usize_hash_map.getOrPut(rand_number);
        if (!count_map_result.found_existing) {
            count_map_result.value_ptr.* = 0;
        }
        count_map_result.value_ptr.* += 1;

        const metadata_map_result = try f64_metadata_hash_map.getOrPut(rand_number);
        if (!metadata_map_result.found_existing) {
            metadata_map_result.value_ptr.* = ArrayList(SegmentMetadata).init(allocator);
        }
        try metadata_map_result.value_ptr.*.append(SegmentMetadata{
            .start_time = count_map_result.value_ptr.*,
            .lower_bound_slope = rand_number,
            .upper_bound_slope = rand_number,
        });
    }

    var iterator_map = f64_metadata_hash_map.iterator();
    while (iterator_map.next()) |entry| {
        const expected_array_size: usize = f64_usize_hash_map.get(entry.key_ptr.*).?;
        try testing.expectEqual(expected_array_size, entry.value_ptr.*.items.len);

        for (entry.value_ptr.*.items, 1..) |item, i| {
            try testing.expectEqual(i, item.start_time);
            try testing.expectEqual(entry.key_ptr.*, item.lower_bound_slope);
        }
    }
}

test "sim-piece can compress and decompress" {
    const allocator = testing.allocator;

    var list_values = ArrayList(f64).init(allocator);
    defer list_values.deinit();
    var compressed_values = ArrayList(u8).init(allocator);
    defer compressed_values.deinit();
    const error_bound: f32 = 2;

    var rnd = std.rand.DefaultPrng.init(2);
    std.debug.print("\n", .{});
    for (0..10) |_| {
        try list_values.append(@floor(rnd.random().float(f64) * 100) / 10);
        std.debug.print("{} ", .{list_values.getLast()});
    }
    std.debug.print("\n", .{});

    const uncompressed_values = list_values.items;

    try compressSimPiece(
        uncompressed_values[0..],
        &compressed_values,
        allocator,
        error_bound,
    );
}
