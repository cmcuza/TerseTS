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
//! The implementation is partially based on the authors implementation at
//! https://github.com/xkitsios/Sim-Piece (accessed on 20-06-24). Few changes were made to support
//! an error bound equal to zero and to improve numerical stability. This is because Sim-Piece
//! does not support lossless compression. Setting `error_bound` to zero will cause an error due to
//! undefined quantization (`b=floor(value/error_bound)*error_bound`). This has a ripple effect
//! across the algorithm that demands the implementation of special instructions to handle a zero
//! error bound. Nevertheless, the numerical instabilities inherent to floating point operations
//! mean that decompressed values will not exactly match the original uncompressed values. To alert
//! the user, a warning is shown if the `error_bound` equals zero.

const std = @import("std");
const math = std.math;
const mem = std.mem;
const testing = std.testing;
const ArrayList = std.ArrayList;
const HashMap = std.HashMap;

const tersets = @import("../tersets.zig");
const Error = tersets.Error;
const DiscretePoint = tersets.DiscretePoint;

/// `SegmentMetadata` stores the information about an approximated segment during the execution
/// of Sim-Piece. It stores the starting time of the segment in `start_time`, the slopes of
/// the upper and lower bounds that constraint the linear approximation in that segment and the
/// `interception` point used to create the linear function approximation.
const SegmentMetadata = struct {
    start_time: usize,
    interception: f64,
    upper_bound_slope: f64,
    lower_bound_slope: f64,
};

/// `HashF64Context` is a struct providing context for hashing and comparing `f64` values.
const HashF64Context = struct {
    /// Hashes an `f64` `value` by bitcasting it to `u64`.
    pub fn hash(_: HashF64Context, value: f64) u64 {
        return @as(u64, @bitCast(value));
    }
    /// Compares two `f64` values for equality.
    pub fn eql(_: HashF64Context, value_one: f64, value_two: f64) bool {
        return value_one == value_two;
    }
};

/// Compresses `uncompressed_values` within `error_bound` using the "Sim-Piece" algorithm, writing
/// the result to `compressed_values`. The `allocator` is used for memory allocation of intermediate
/// data structures. If an error occurs, it is returned.
pub fn compressSimPiece(
    uncompressed_values: []const f64,
    compressed_values: *ArrayList(u8),
    allocator: mem.Allocator,
    error_bound: f32,
) Error!void {
    if (error_bound == 0.0) {
        std.debug.print("\nWarning: Sim-Piece does not support lossless compression. ", .{});
        std.debug.print("Expect deviations from the original data of at most 1e-6.\n", .{});
    }
    // Sim-Piece Phase 1: Compute the segments metadata.
    var segments_metadata = ArrayList(SegmentMetadata).init(allocator);
    defer segments_metadata.deinit();
    try computeSegmentsMetadata(
        uncompressed_values,
        &segments_metadata,
        error_bound,
    );

    // Sim-Piece Phase 2: Merge the segments metadata.
    var merged_segments_metadata = ArrayList(SegmentMetadata).init(allocator);
    defer merged_segments_metadata.deinit();
    try mergeSegmentsMetadata(segments_metadata, &merged_segments_metadata, allocator);

    // Sim-Piece Phase 3: Compute segment metadata hash map.
    var merged_segments_metadata_map = HashMap(
        f64,
        HashMap(
            f64,
            ArrayList(usize),
            HashF64Context,
            std.hash_map.default_max_load_percentage,
        ),
        HashF64Context,
        std.hash_map.default_max_load_percentage,
    ).init(allocator);
    defer {
        var hash_to_hash_iterator = merged_segments_metadata_map.iterator();
        while (hash_to_hash_iterator.next()) |hash_to_hash_entry| {
            var hash_to_array_iterator = hash_to_hash_entry.value_ptr.*.iterator();
            while (hash_to_array_iterator.next()) |hash_to_array_entry| {
                hash_to_array_entry.value_ptr.*.deinit();
            }
            hash_to_hash_entry.value_ptr.*.deinit();
        }
        merged_segments_metadata_map.deinit();
    }
    try computeSegmentsMetadataHashMap(
        merged_segments_metadata,
        &merged_segments_metadata_map,
        allocator,
    );

    // Sim-Piece Phase 4: Create the final compressed representation and store in compressed values.
    try createCompressedRepresentation(merged_segments_metadata_map, compressed_values);
    // The last timestamp needs to be stored in the compressed representaiton as well.
    try appendValue(usize, uncompressed_values.len, compressed_values);
}

/// Decompress `compressed_values` produced by "Sim-Piece" and write the result to
/// `decompressed_values`. If an error occurs it is returned.
pub fn decompress(
    compressed_values: []const u8,
    decompressed_values: *ArrayList(f64),
    allocator: mem.Allocator,
) Error!void {
    // The compressed representation of Sim-Piece is of variable length. We cannot assert the len
    // of the compressed representation to be equal to any specific number.
    var segments_metadata = ArrayList(SegmentMetadata).init(allocator);
    defer segments_metadata.deinit();
    const compressed_lines_and_index = mem.bytesAsSlice(f64, compressed_values);

    var general_index: usize = 0;
    while (general_index < compressed_lines_and_index.len - 1) {
        const interception: f64 = compressed_lines_and_index[general_index];
        const slopes_count = @as(usize, @bitCast(compressed_lines_and_index[general_index + 1]));
        general_index += 2;

        for (0..slopes_count) |_| {
            const slope = compressed_lines_and_index[general_index];
            const timestamps_count = @as(usize, @bitCast(compressed_lines_and_index[general_index + 1]));
            general_index += 2;
            var timestamp: usize = 0;
            for (0..timestamps_count) |_| {
                timestamp += @as(usize, @bitCast(compressed_lines_and_index[general_index]));
                try segments_metadata.append(.{
                    .start_time = timestamp,
                    .interception = interception,
                    .lower_bound_slope = slope,
                    .upper_bound_slope = slope,
                });
                general_index += 1;
            }
        }
    }

    const last_timestamp: usize = @as(usize, @bitCast(compressed_lines_and_index[general_index]));

    mem.sort(
        SegmentMetadata,
        segments_metadata.items,
        {},
        compareMetadataByStartTime(),
    );

    var current_timestamp: usize = 0;
    for (0..segments_metadata.items.len - 1) |index| {
        const current_metadata = segments_metadata.items[index];
        const next_metadata_start_time = segments_metadata.items[index + 1].start_time;
        while (current_timestamp < next_metadata_start_time) : (current_timestamp += 1) {
            const approximated_value = current_metadata.upper_bound_slope * @as(f64, @floatFromInt(
                current_timestamp - current_metadata.start_time,
            )) + current_metadata.interception;
            try decompressed_values.append(approximated_value);
        }
    }

    const current_metadata = segments_metadata.getLast();
    while (current_timestamp < last_timestamp) : (current_timestamp += 1) {
        const approximated_value = current_metadata.upper_bound_slope * @as(f64, @floatFromInt(
            current_timestamp - current_metadata.start_time,
        )) + current_metadata.interception;
        try decompressed_values.append(approximated_value);
    }
}

/// Sim-Piece Phase 1: Compute the `segments_metadata` for each segment that can be approximated
/// by a linear function within the `error_bound` from the `uncompressed_values`.
fn computeSegmentsMetadata(
    uncompressed_values: []const f64,
    segments_metadata: *ArrayList(SegmentMetadata),
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

    var quantized_interception = quantize(uncompressed_values[0], error_bound) +
        tersets.ErrorBoundMargin;

    // First point already part of `current_segment`, next point is at index one.
    var current_timestamp: usize = 1;
    while (current_timestamp < uncompressed_values.len) : (current_timestamp += 1) {
        const end_point: DiscretePoint = .{
            .time = current_timestamp,
            .value = uncompressed_values[current_timestamp],
        };

        const segment_size: f64 = @floatFromInt(current_timestamp - start_point.time);
        const upper_limit: f64 = upper_bound_slope * segment_size + quantized_interception;
        const lower_limit: f64 = lower_bound_slope * segment_size + quantized_interception;

        if ((upper_limit < (end_point.value - adjusted_error_bound)) or
            ((lower_limit > (end_point.value + adjusted_error_bound))))
        {
            // The new point is outside the upper and lower limit. Record a new segment metadata in
            // `segments_metadata_map` associated to `quantized_start_value`.
            try segments_metadata.append(.{
                .start_time = start_point.time,
                .interception = quantized_interception,
                .upper_bound_slope = upper_bound_slope,
                .lower_bound_slope = lower_bound_slope,
            });

            start_point = end_point;
            quantized_interception = quantize(start_point.value, error_bound) +
                tersets.ErrorBoundMargin;
            upper_bound_slope = math.floatMax(f64);
            lower_bound_slope = -math.floatMax(f64);
        } else {
            // The new point is within the upper and lower bounds. Update the bounds' slopes.
            const new_upper_bound_slope: f64 =
                (end_point.value + adjusted_error_bound - quantized_interception) / segment_size;
            const new_lower_bound_slope: f64 =
                (end_point.value - adjusted_error_bound - quantized_interception) / segment_size;

            if (end_point.value + adjusted_error_bound < upper_limit)
                upper_bound_slope = @max(new_upper_bound_slope, lower_bound_slope);
            if (end_point.value - adjusted_error_bound > lower_limit)
                lower_bound_slope = @min(new_lower_bound_slope, upper_bound_slope);
        }
    }

    const segment_size = current_timestamp - start_point.time;
    if (segment_size >= 1) {
        // Append the final segment.
        if (segment_size == 1) {
            upper_bound_slope = 0;
            lower_bound_slope = 0;
        }
        try segments_metadata.append(.{
            .start_time = start_point.time,
            .interception = quantized_interception,
            .upper_bound_slope = upper_bound_slope,
            .lower_bound_slope = lower_bound_slope,
        });
    }
}

/// Sim-Piece Phase 2. Merge the elements in `segments_metadata` based on Alg. 2 and store the
/// results in `merged_segments_metadata`. The `allocator` is used to allocate memory for the
/// intermediate representations needed.
fn mergeSegmentsMetadata(
    segments_metadata: ArrayList(SegmentMetadata),
    merged_segments_metadata: *ArrayList(SegmentMetadata),
    allocator: mem.Allocator,
) !void {
    var timestamps_array = ArrayList(usize).init(allocator);
    defer timestamps_array.deinit();

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

    // Iteratively populate `segments_metadata_map` based on the quantized interception values to
    // find the common interception between the segments. Segments falling in the same interception
    // can potentially be merged.
    for (segments_metadata.items) |segment_metadata| {
        try appendSegmentMetadata(
            &segments_metadata_map,
            segment_metadata,
            allocator,
        );
    }

    var iterator = segments_metadata_map.iterator();
    while (iterator.next()) |entry| {
        const metadata_array = entry.value_ptr.*;

        // Sort in asc order based on the lower bound's slope. Alg 2. Line 5.
        mem.sort(
            SegmentMetadata,
            metadata_array.items,
            {},
            compareMetadataBySlope(),
        );

        var merge_metadata: SegmentMetadata = .{
            .start_time = 0.0,
            .interception = metadata_array.items[0].interception,
            .lower_bound_slope = metadata_array.items[0].lower_bound_slope,
            .upper_bound_slope = metadata_array.items[0].upper_bound_slope,
        };
        try timestamps_array.append(metadata_array.items[0].start_time);

        for (1..metadata_array.items.len) |index| {
            const current_metadata = metadata_array.items[index];

            if ((current_metadata.lower_bound_slope <= merge_metadata.upper_bound_slope) and
                (current_metadata.upper_bound_slope >= merge_metadata.lower_bound_slope))
            {
                // The current segment metadata can be merged. Update the bounds's slopes.
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
                // The current segment metadata cannot be merged. Append the merged metadata and
                // create a new one.
                for (timestamps_array.items) |timestamp| {
                    try merged_segments_metadata.append(.{
                        .start_time = timestamp,
                        .interception = merge_metadata.interception,
                        .lower_bound_slope = merge_metadata.lower_bound_slope,
                        .upper_bound_slope = merge_metadata.upper_bound_slope,
                    });
                }
                timestamps_array.clearRetainingCapacity();
                merge_metadata = .{
                    .start_time = 0.0,
                    .interception = current_metadata.interception,
                    .lower_bound_slope = current_metadata.lower_bound_slope,
                    .upper_bound_slope = current_metadata.upper_bound_slope,
                };
                try timestamps_array.append(current_metadata.start_time);
            }
        }

        if (timestamps_array.items.len > 0) {
            // Append the final merged segment metadata.
            for (timestamps_array.items) |timestamp| {
                try merged_segments_metadata.append(.{
                    .start_time = timestamp,
                    .interception = merge_metadata.interception,
                    .lower_bound_slope = merge_metadata.lower_bound_slope,
                    .upper_bound_slope = merge_metadata.upper_bound_slope,
                });
            }
            timestamps_array.clearRetainingCapacity();
        }
    }

    // This step is needed since the timestamp order is lost due to the hashmap.
    mem.sort(
        SegmentMetadata,
        merged_segments_metadata.items,
        {},
        compareMetadataByStartTime(),
    );
}

/// Sim-Piece Phase 3. Compute a hash map from interception points in `merged_segments_metadata`
/// to a hash map from the approximation slope to an array list of timestamps and store in
/// `merged_segments_metadata_map`. The `allocator` is used to allocate memory of intermediates.
fn computeSegmentsMetadataHashMap(
    merged_segments_metadata: ArrayList(SegmentMetadata),
    merged_segments_metadata_map: *HashMap(f64, HashMap(
        f64,
        ArrayList(usize),
        HashF64Context,
        std.hash_map.default_max_load_percentage,
    ), HashF64Context, std.hash_map.default_max_load_percentage),
    allocator: mem.Allocator,
) !void {
    for (merged_segments_metadata.items) |segment_metadata| {
        const interception: f64 = segment_metadata.interception;
        const slope: f64 = (segment_metadata.lower_bound_slope +
            segment_metadata.upper_bound_slope) / 2;

        // Get or put the outer HashMap entry for the given interception
        const hash_to_hash_result = try merged_segments_metadata_map.getOrPut(interception);
        if (!hash_to_hash_result.found_existing) {
            hash_to_hash_result.value_ptr.* = HashMap(
                f64,
                ArrayList(usize),
                HashF64Context,
                std.hash_map.default_max_load_percentage,
            ).init(allocator);
        }
        // Get or put the inner HashMap entry for the given slope
        const hash_to_array_result = try hash_to_hash_result.value_ptr.*.getOrPut(slope);
        if (!hash_to_array_result.found_existing) {
            hash_to_array_result.value_ptr.* = ArrayList(usize).init(allocator);
        }
        try hash_to_array_result.value_ptr.*.append(segment_metadata.start_time);
    }
}

/// Sim-Piece Phase 4. Create from the `merged_segments_metadata_map` and byte array that can be
/// decoded during decompression and store it in `compressed_values`.
fn createCompressedRepresentation(
    merged_segments_metadata_map: HashMap(f64, HashMap(
        f64,
        ArrayList(usize),
        HashF64Context,
        std.hash_map.default_max_load_percentage,
    ), HashF64Context, std.hash_map.default_max_load_percentage),
    compressed_values: *ArrayList(u8),
) !void {
    // Iterate over the hash to hash map.
    var hash_to_hash_iterator = merged_segments_metadata_map.iterator();
    while (hash_to_hash_iterator.next()) |hash_to_hash_entry| {
        const current_interception: f64 = hash_to_hash_entry.key_ptr.*;
        // Append the `current_interception`.
        try appendValue(f64, current_interception, compressed_values);

        // Append the number of slopes hashed by the `current_interception`.
        try appendValue(usize, hash_to_hash_entry.value_ptr.*.count(), compressed_values);
        var hash_to_array_iterator = hash_to_hash_entry.value_ptr.*.iterator();

        while (hash_to_array_iterator.next()) |hash_to_array_entry| {
            const current_slope = hash_to_array_entry.key_ptr.*;
            // Append the `current_slope` that approximates the segment.
            try appendValue(f64, current_slope, compressed_values);

            // Append the number of timestamps that the `current_slope` is hashing.
            try appendValue(usize, hash_to_array_entry.value_ptr.*.items.len, compressed_values);
            var previous_timestamp: usize = 0;
            // Iterate over the timestamps and append them.
            for (hash_to_array_entry.value_ptr.*.items) |timestamp| {
                try appendValue(usize, timestamp - previous_timestamp, compressed_values);
                previous_timestamp = timestamp;
            }
        }
    }
}

/// Quantizes the given `value` by the specified `error_bound`. This process ensures that
/// the quantized value remains within the error bound of the original value. If the
/// `error_bound` is equal to zero, the value is directly returned.
fn quantize(value: f64, error_bound: f32) f64 {
    if (error_bound != 0)
        return @floor(value / error_bound) * error_bound;
    return value;
}

/// Appends the `metadata` to the hash map `metadata_map`. The `allocator` to use for allocating
/// the memory for a new ArrayList if the `metadata.interception` does not exist.
fn appendSegmentMetadata(
    metadata_map: *HashMap(
        f64,
        ArrayList(SegmentMetadata),
        HashF64Context,
        std.hash_map.default_max_load_percentage,
    ),
    metadata: SegmentMetadata,
    allocator: mem.Allocator,
) !void {
    const get_result = try metadata_map.getOrPut(metadata.interception);
    if (!get_result.found_existing) {
        get_result.value_ptr.* = ArrayList(SegmentMetadata).init(allocator);
    }
    try get_result.value_ptr.*.append(metadata);
}

/// Returns a comparator function that compares SegmentMetadata by `lower_bound_slope`.
fn compareMetadataBySlope() fn (void, SegmentMetadata, SegmentMetadata) bool {
    return struct {
        pub fn inner(_: void, metadata_one: SegmentMetadata, metadata_two: SegmentMetadata) bool {
            return metadata_one.lower_bound_slope < metadata_two.lower_bound_slope;
        }
    }.inner;
}

/// Returns a comparator function that compares SegmentMetadata by `start_time`.
fn compareMetadataByStartTime() fn (void, SegmentMetadata, SegmentMetadata) bool {
    return struct {
        pub fn inner(_: void, metadata_one: SegmentMetadata, metadata_two: SegmentMetadata) bool {
            return metadata_one.start_time < metadata_two.start_time;
        }
    }.inner;
}

/// Append `value` of `type` determined at compile time to `compressed_values`.
fn appendValue(comptime T: type, value: T, compressed_values: *std.ArrayList(u8)) !void {
    // Compile-time type check
    switch (@TypeOf(value)) {
        f64, usize => {
            const value_as_bytes: [8]u8 = @bitCast(value);
            try compressed_values.appendSlice(value_as_bytes[0..]);
        },
        f32 => {
            const value_as_bytes: [4]u8 = @bitCast(value);
            try compressed_values.appendSlice(value_as_bytes[0..]);
        },
        else => @compileError("Unsupported type for append value function"),
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
            .interception = rand_number,
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

test "sim-piece can compress, decompress and merge many segments with non-zero error bound" {
    const allocator = testing.allocator;

    var list_values = ArrayList(f64).init(allocator);
    defer list_values.deinit();
    var compressed_values = ArrayList(u8).init(allocator);
    defer compressed_values.deinit();
    var decompressed_values = ArrayList(f64).init(allocator);
    defer decompressed_values.deinit();
    var rnd = std.rand.DefaultPrng.init(@as(u64, @bitCast(std.time.milliTimestamp())));

    const error_bound: f32 = @floor(rnd.random().float(f32) * 100) / 10 + 0.1;

    for (0..200) |_| {
        // Generate floating points between numbers between 0 and 10. This will render many
        // merged segments by Sim-Piece.
        try list_values.append(@round(rnd.random().float(f64) * 100) / 10);
    }

    const uncompressed_values = list_values.items;

    try compressSimPiece(
        uncompressed_values[0..],
        &compressed_values,
        allocator,
        error_bound,
    );

    try decompress(compressed_values.items, &decompressed_values, allocator);

    try testing.expect(tersets.isWithinErrorBound(
        uncompressed_values,
        decompressed_values.items,
        error_bound,
    ));
}

test "sim-piece zero error bound and even size compress and decompress" {
    const allocator = testing.allocator;

    var list_values = ArrayList(f64).init(allocator);
    defer list_values.deinit();
    var compressed_values = ArrayList(u8).init(allocator);
    defer compressed_values.deinit();
    var decompressed_values = ArrayList(f64).init(allocator);
    defer decompressed_values.deinit();
    const error_bound: f32 = 0.0;

    var rnd = std.rand.DefaultPrng.init(0);

    var i: usize = 0;
    while (i < 100) : (i += 1) {
        const noise = rnd.random().float(f64) * 0.1 - 0.05;
        try list_values.append(@as(f64, @floatFromInt(i)) + noise);
    }

    const uncompressed_values = list_values.items;

    try compressSimPiece(uncompressed_values[0..], &compressed_values, allocator, error_bound);
    try decompress(compressed_values.items, &decompressed_values, allocator);

    try testing.expectEqual(uncompressed_values.len, decompressed_values.items.len);

    for (0..uncompressed_values.len) |index| {
        const uncompressed_value = uncompressed_values[index];
        const decompressed_value = decompressed_values.items[index];
        // Finding the maximum deviation level. A level lower than 1e-6 is not passing the test.
        try testing.expect(@abs(uncompressed_value - decompressed_value) < 1e-6);
    }
}

test "sim-piece zero error bound and odd size compress and decompress" {
    const allocator = testing.allocator;

    var list_values = ArrayList(f64).init(allocator);
    defer list_values.deinit();
    var compressed_values = ArrayList(u8).init(allocator);
    defer compressed_values.deinit();
    var decompressed_values = ArrayList(f64).init(allocator);
    defer decompressed_values.deinit();
    const error_bound: f32 = 0.0;

    var rnd = std.rand.DefaultPrng.init(0);

    var i: usize = 0;
    while (i < 101) : (i += 1) {
        const noise = rnd.random().float(f64) * 0.1 - 0.05;
        try list_values.append(@as(f64, @floatFromInt(i)) + noise);
    }

    const uncompressed_values = list_values.items;

    try compressSimPiece(uncompressed_values[0..], &compressed_values, allocator, error_bound);
    try decompress(compressed_values.items, &decompressed_values, allocator);

    try testing.expectEqual(uncompressed_values.len, decompressed_values.items.len);

    for (0..uncompressed_values.len) |index| {
        const uncompressed_value = uncompressed_values[index];
        const decompressed_value = decompressed_values.items[index];
        // Finding the maximum deviation level. A level lower than 1e-6 is not passing the test.
        try testing.expect(@abs(uncompressed_value - decompressed_value) < 1e-6);
    }
}

test "sim-piece random lines and error bound compress and decompress" {
    const allocator = testing.allocator;
    var rnd = std.rand.DefaultPrng.init(@as(u64, @bitCast(std.time.milliTimestamp())));

    var list_values = ArrayList(f64).init(allocator);
    defer list_values.deinit();
    var compressed_values = ArrayList(u8).init(allocator);
    defer compressed_values.deinit();
    var decompressed_values = ArrayList(f64).init(allocator);
    defer decompressed_values.deinit();
    const error_bound: f32 = rnd.random().float(f32) * 0.1;

    var i: usize = 0;
    var lineIndex: usize = 0;
    var slope: f64 = 2 * (rnd.random().float(f64) - 0.5);
    var intercept: f64 = 2 * (rnd.random().float(f64) - 0.5);
    while (i < 1000) : (i += 1) {
        if (i / 250 > lineIndex) {
            lineIndex = i / 250;
            slope = 2 * (rnd.random().float(f64) - 0.5);
            intercept = 2 * (rnd.random().float(f64) - 0.5);
        }
        const noise = rnd.random().float(f64) * 0.1 - 0.05;
        try list_values.append(slope * @as(f64, @floatFromInt(i)) + intercept + noise);
    }

    const uncompressed_values = list_values.items;

    try compressSimPiece(
        uncompressed_values[0..],
        &compressed_values,
        allocator,
        error_bound,
    );
    try decompress(compressed_values.items, &decompressed_values, allocator);

    try testing.expect(tersets.isWithinErrorBound(
        uncompressed_values,
        decompressed_values.items,
        error_bound,
    ));
}
