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
//! https://github.com/xkitsios/Sim-Piece (accessed on 20-06-24).

const std = @import("std");
const math = std.math;
const mem = std.mem;
const rand = std.Random;
const time = std.time;
const testing = std.testing;
const ArrayList = std.ArrayList;

const tersets = @import("../tersets.zig");
const Method = tersets.Method;
const Error = tersets.Error;
const shared = @import("../utilities/shared_structs.zig");
const DiscretePoint = shared.DiscretePoint;

const tester = @import("../tester.zig");

/// Compresses `uncompressed_values` within `error_bound` using the "Sim-Piece" algorithm.
/// The function writes the result to `compressed_values`. The `allocator` is used for memory
/// allocation of intermediate data structures. If an error occurs, it is returned.
pub fn compress(
    uncompressed_values: []const f64,
    compressed_values: *ArrayList(u8),
    allocator: mem.Allocator,
    error_bound: f32,
) Error!void {
    if (error_bound <= 0.0) {
        return Error.UnsupportedErrorBound;
    }
    // Sim-Piece Phase 1: Compute `SegmentMetadata` for all segments that can be approximated
    // by the given `error_bound`.
    var segments_metadata = ArrayList(shared.SegmentMetadata).init(allocator);
    defer segments_metadata.deinit();
    try computeSegmentsMetadata(
        uncompressed_values,
        &segments_metadata,
        error_bound,
    );

    // Sim-Piece Phase 2: Merge the `SegmentMetadata` that share the same intercept point.
    var merged_segments_metadata = ArrayList(shared.SegmentMetadata).init(allocator);
    defer merged_segments_metadata.deinit();
    try mergeSegmentsMetadata(segments_metadata, &merged_segments_metadata, allocator);

    // Sim-Piece Phase 3: Populate the `SegmentMetadata` HashMap based on the intercept point.
    var merged_segments_metadata_map = shared.HashMapf64(shared.HashMapf64(ArrayList(usize))).init(allocator);
    defer {
        // Deinit all ArrayList instances within the inner HashMaps and then deinit the inner HashMaps
        // themselves before finally deinit the outer HashMap.
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
    try populateSegmentsMetadataHashMap(
        merged_segments_metadata,
        &merged_segments_metadata_map,
        allocator,
    );

    // Sim-Piece Phase 4: Create the final compressed representation and store in compressed values.
    try createCompressedRepresentation(merged_segments_metadata_map, compressed_values);

    // The last timestamp must be stored, otherwise the end time during decompression is unknown.
    try appendValue(usize, uncompressed_values.len, compressed_values);
}

/// Decompress `compressed_values` produced by "Sim-Piece". The function writes the result to
/// `decompressed_values`. The `allocator` is used for memory allocation of intermediate
/// data structures. If an error occurs it is returned.
pub fn decompress(
    compressed_values: []const u8,
    decompressed_values: *ArrayList(f64),
    allocator: mem.Allocator,
) Error!void {
    // The compressed representation of Sim-Piece is of variable length. We cannot assert the len
    // of the compressed representation to be equal to any specific number.
    var segments_metadata = ArrayList(shared.SegmentMetadata).init(allocator);
    defer segments_metadata.deinit();
    const compressed_lines_and_index = mem.bytesAsSlice(f64, compressed_values);

    var compressed_index: usize = 0;
    while (compressed_index < compressed_lines_and_index.len - 1) {
        const intercept: f64 = compressed_lines_and_index[compressed_index];
        const slopes_count = @as(usize, @bitCast(compressed_lines_and_index[compressed_index + 1]));
        compressed_index += 2;

        for (0..slopes_count) |_| {
            const slope = compressed_lines_and_index[compressed_index];
            const timestamps_count = @as(usize, @bitCast(compressed_lines_and_index[compressed_index + 1]));
            compressed_index += 2;
            var timestamp: usize = 0;
            for (0..timestamps_count) |_| {
                timestamp += @as(usize, @bitCast(compressed_lines_and_index[compressed_index]));
                try segments_metadata.append(.{
                    .start_time = timestamp,
                    .intercept = intercept,
                    .lower_bound_slope = slope,
                    .upper_bound_slope = slope,
                });
                compressed_index += 1;
            }
        }
    }

    const last_timestamp: usize = @as(usize, @bitCast(compressed_lines_and_index[compressed_index]));

    mem.sort(
        shared.SegmentMetadata,
        segments_metadata.items,
        {},
        compareMetadataByStartTime,
    );

    var current_timestamp: usize = 0;
    for (0..segments_metadata.items.len - 1) |index| {
        const current_metadata = segments_metadata.items[index];
        const next_metadata_start_time = segments_metadata.items[index + 1].start_time;
        try decompressSegment(
            current_metadata,
            current_timestamp,
            next_metadata_start_time,
            decompressed_values,
        );
        current_timestamp = next_metadata_start_time;
    }

    const current_metadata = segments_metadata.getLast();
    try decompressSegment(
        current_metadata,
        current_timestamp,
        last_timestamp,
        decompressed_values,
    );
}

/// Sim-Piece Phase 1: Compute `SegmentMetadata` for each segment that can be approximated
/// by a linear function within the `error_bound` from `uncompressed_values`.
fn computeSegmentsMetadata(
    uncompressed_values: []const f64,
    segments_metadata: *ArrayList(shared.SegmentMetadata),
    error_bound: f32,
) Error!void {
    // Adjust the error bound to avoid exceeding it during decompression.
    const adjusted_error_bound = error_bound - shared.ErrorBoundMargin;

    var upper_bound_slope: f64 = math.floatMax(f64);
    var lower_bound_slope: f64 = -math.floatMax(f64);

    // Check if the first point is NaN, infinite or a reduced precision floating point.
    // If so, return an error.
    if (!math.isFinite(uncompressed_values[0]) or
        uncompressed_values[0] > 1e15) return Error.UnsupportedInput;

    // Initialize the `start_point` with the first uncompressed value.
    var start_point: DiscretePoint = .{ .time = 0, .value = uncompressed_values[0] };

    // The quantization can only be done using the original error bound. Afterwards, we add
    // `tersets.ErrorBoundMargin` to avoid exceeding the error bound during decompression.
    var quantized_intercept = quantize(uncompressed_values[0], error_bound) +
        shared.ErrorBoundMargin;

    // The first point is already part of `current_segment`, the next point is at index one.
    for (1..uncompressed_values.len) |current_timestamp| {

        // Check if the current point is NaN, infinite or a reduced precision floating point.
        // If so, return an error.
        if (!math.isFinite(uncompressed_values[current_timestamp]) or
            uncompressed_values[current_timestamp] > 1e15) return Error.UnsupportedInput;

        const end_point: DiscretePoint = .{
            .time = current_timestamp,
            .value = uncompressed_values[current_timestamp],
        };

        // `segment_size` of type f64 to avoid casting from usize when computing other variables.
        const segment_size: f64 = @floatFromInt(current_timestamp - start_point.time);
        const upper_limit: f64 = upper_bound_slope * segment_size + quantized_intercept;
        const lower_limit: f64 = lower_bound_slope * segment_size + quantized_intercept;

        if ((upper_limit < (end_point.value - adjusted_error_bound)) or
            ((lower_limit > (end_point.value + adjusted_error_bound))))
        {
            // The new point is outside the upper and lower limit. Record a new segment metadata in
            // `segments_metadata_map` associated to `intercept`.
            try segments_metadata.append(.{
                .start_time = start_point.time,
                .intercept = quantized_intercept,
                .upper_bound_slope = upper_bound_slope,
                .lower_bound_slope = lower_bound_slope,
            });

            start_point = end_point;
            quantized_intercept = quantize(start_point.value, error_bound) +
                shared.ErrorBoundMargin;
            upper_bound_slope = math.floatMax(f64);
            lower_bound_slope = -math.floatMax(f64);
        } else {
            // The new point is within the upper and lower bounds. Update the bounds' slopes.
            const new_upper_bound_slope: f64 =
                (end_point.value + adjusted_error_bound - quantized_intercept) / segment_size;
            const new_lower_bound_slope: f64 =
                (end_point.value - adjusted_error_bound - quantized_intercept) / segment_size;

            if (end_point.value + adjusted_error_bound < upper_limit)
                upper_bound_slope = @max(new_upper_bound_slope, lower_bound_slope);
            if (end_point.value - adjusted_error_bound > lower_limit)
                lower_bound_slope = @min(new_lower_bound_slope, upper_bound_slope);
        }
    }

    const segment_size = uncompressed_values.len - start_point.time;
    if (segment_size > 0) {
        // Append the final segment.
        if (segment_size == 1) {
            upper_bound_slope = 0;
            lower_bound_slope = 0;
        }
        try segments_metadata.append(.{
            .start_time = start_point.time,
            .intercept = quantized_intercept,
            .upper_bound_slope = upper_bound_slope,
            .lower_bound_slope = lower_bound_slope,
        });
    }
}

/// Sim-Piece Phase 2. Merge the elements in `segments_metadata` using Alg. 2 and store the
/// results in `merged_segments_metadata`. The segments are merged based on the intercept value.
/// The `allocator` is used to allocate memory for the intermediate representations needed.
fn mergeSegmentsMetadata(
    segments_metadata: ArrayList(shared.SegmentMetadata),
    merged_segments_metadata: *ArrayList(shared.SegmentMetadata),
    allocator: mem.Allocator,
) !void {
    var timestamps_array = ArrayList(usize).init(allocator);
    defer timestamps_array.deinit();

    var segments_metadata_map = shared.HashMapf64(ArrayList(shared.SegmentMetadata)).init(allocator);
    defer {
        // Deinit all ArrayList instances within the HashMap before deinit it.
        var iterator = segments_metadata_map.iterator();
        while (iterator.next()) |entry| {
            entry.value_ptr.*.deinit();
        }
        segments_metadata_map.deinit();
    }

    // Iteratively populate `segments_metadata_map` based on the quantized intercept values to
    // find the common intercept between the segments. Segments with the same intercept
    // can be merged if the upper and lower bounds of one segment are contained within the lower
    // and upper bounds of the other segment.
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

        // Sort in asc order based on the lower bound's slope. Alg 2. Line 5. This enables finding
        // the segments contained inside other segments and merge them.
        mem.sort(
            shared.SegmentMetadata,
            metadata_array.items,
            {},
            compareMetadataBySlope,
        );

        var merge_metadata: shared.SegmentMetadata = .{
            .start_time = 0.0,
            .intercept = metadata_array.items[0].intercept,
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
                        .intercept = merge_metadata.intercept,
                        .lower_bound_slope = merge_metadata.lower_bound_slope,
                        .upper_bound_slope = merge_metadata.upper_bound_slope,
                    });
                }
                timestamps_array.clearRetainingCapacity();
                merge_metadata = .{
                    .start_time = 0.0,
                    .intercept = current_metadata.intercept,
                    .lower_bound_slope = current_metadata.lower_bound_slope,
                    .upper_bound_slope = current_metadata.upper_bound_slope,
                };
                try timestamps_array.append(current_metadata.start_time);
            }
        }

        // Append the final merged segment metadata.
        for (timestamps_array.items) |timestamp| {
            try merged_segments_metadata.append(.{
                .start_time = timestamp,
                .intercept = merge_metadata.intercept,
                .lower_bound_slope = merge_metadata.lower_bound_slope,
                .upper_bound_slope = merge_metadata.upper_bound_slope,
            });
        }
        timestamps_array.clearRetainingCapacity();
    }

    // This step is needed since the timestamp order is lost due to the HashMap.
    mem.sort(
        shared.SegmentMetadata,
        merged_segments_metadata.items,
        {},
        compareMetadataByStartTime,
    );
}

/// Sim-Piece Phase 3. Populate the `SegmentMetadata` HashMap from intercept points in
/// `merged_segments_metadata` to a HashMap from the approximation slope to an array list of
/// timestamps and store it in `merged_segments_metadata_map`. The `allocator` is used to allocate
/// memory of intermediates.
fn populateSegmentsMetadataHashMap(
    merged_segments_metadata: ArrayList(shared.SegmentMetadata),
    merged_segments_metadata_map: *shared.HashMapf64(shared.HashMapf64(ArrayList(usize))),
    allocator: mem.Allocator,
) !void {
    for (merged_segments_metadata.items) |segment_metadata| {
        const intercept: f64 = segment_metadata.intercept;
        const slope: f64 = (segment_metadata.lower_bound_slope +
            segment_metadata.upper_bound_slope) / 2;

        // Get or put the inner HashMap entry for the given `intercept` wich will contain the
        // slopes and timestamps associated to it.
        const hash_to_hash_result = try merged_segments_metadata_map.getOrPut(intercept);
        if (!hash_to_hash_result.found_existing) {
            hash_to_hash_result.value_ptr.* = shared.HashMapf64(
                ArrayList(usize),
            ).init(allocator);
        }
        // Get or put the ArrayList of timestamps mapped to the given `slope` which is at the same
        // time associated to the given `intercept`.
        const hash_to_array_result = try hash_to_hash_result.value_ptr.*.getOrPut(slope);
        if (!hash_to_array_result.found_existing) {
            hash_to_array_result.value_ptr.* = ArrayList(usize).init(allocator);
        }
        try hash_to_array_result.value_ptr.*.append(segment_metadata.start_time);
    }
}

/// Sim-Piece Phase 4. Create compressed representation from the `merged_segments_metadata_map`
/// that can be decoded during decompression and store it in `compressed_values`. The compressed
/// representation is a byte array containing the intercepts, slopes and timestamps per segment.
/// Specifically, the compressed representation has the following structure:
/// [b_1, N_1, a_11, M_11, t_1, t_2, ..., a_12, M_12, t_1, ...., b_2, N_2, a_21, M_21, t1, t2, ...,
///  b_n, N_n, a_n1, M_n1, t_1, ...], where b_i are the intercepts, N_i are the number of slopes
/// associated to intercept b_i, a_ij are the slopes associated to intercept b_i, M_ij are
/// the number of timestamps associated to the slope a_ij, and t_k are the timestamps.
pub fn createCompressedRepresentation(
    merged_segments_metadata_map: shared.HashMapf64(shared.HashMapf64(ArrayList(usize))),
    compressed_values: *ArrayList(u8),
) !void {
    // Iterate over the outer HashMap to append the intercepts b_i.
    var hash_to_hash_iterator = merged_segments_metadata_map.iterator();
    while (hash_to_hash_iterator.next()) |hash_to_hash_entry| {
        const current_intercept: f64 = hash_to_hash_entry.key_ptr.*;

        try appendValue(f64, current_intercept, compressed_values);

        // Append the number of slopes N_i associated to `current_intercept` b_i.
        try appendValue(usize, hash_to_hash_entry.value_ptr.*.count(), compressed_values);

        // Iterate over the inner HashMap to append the slopes a_ij associated to b_i.
        var hash_to_array_iterator = hash_to_hash_entry.value_ptr.*.iterator();
        while (hash_to_array_iterator.next()) |hash_to_array_entry| {
            const current_slope = hash_to_array_entry.key_ptr.*;
            try appendValue(f64, current_slope, compressed_values);

            // Append the number of timestamps M_ij associated to `current_slope` a_ij.
            try appendValue(usize, hash_to_array_entry.value_ptr.*.items.len, compressed_values);
            var previous_timestamp: usize = 0;

            // Iterate over the ArrayList to append the timestamps t_k.
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
    if (error_bound != 0) {
        return @floor(value / error_bound) * error_bound;
    }
    return value;
}

/// Appends the `metadata` to the HashMap `metadata_map`. The `allocator` is used for allocating
/// the memory for a new ArrayList if the `metadata.intercept` does not exist.
pub fn appendSegmentMetadata(
    metadata_map: *shared.HashMapf64(
        ArrayList(shared.SegmentMetadata),
    ),
    metadata: shared.SegmentMetadata,
    allocator: mem.Allocator,
) !void {
    const get_result = try metadata_map.getOrPut(metadata.intercept);
    if (!get_result.found_existing) {
        get_result.value_ptr.* = ArrayList(shared.SegmentMetadata).init(allocator);
    }
    try get_result.value_ptr.*.append(metadata);
}

/// Compares `metadata_one` and `metadata_two` by their lower bound slope.
pub fn compareMetadataBySlope(
    _: void,
    metadata_one: shared.SegmentMetadata,
    metadata_two: shared.SegmentMetadata,
) bool {
    return metadata_one.lower_bound_slope < metadata_two.lower_bound_slope;
}

/// Compares `metadata_one` and `metadata_two` by their start time.
pub fn compareMetadataByStartTime(
    _: void,
    metadata_one: shared.SegmentMetadata,
    metadata_two: shared.SegmentMetadata,
) bool {
    return metadata_one.start_time < metadata_two.start_time;
}

/// Append `value` of `type` determined at compile time to `compressed_values`.
pub fn appendValue(comptime T: type, value: T, compressed_values: *std.ArrayList(u8)) !void {
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

/// Computes and stores in `decompressed_values` the decompress representation of the points from
/// the `start_time` to the `end_time` based on the information stored in `segment_metadata`.
pub fn decompressSegment(
    segment_metadata: shared.SegmentMetadata,
    start_time: usize,
    end_time: usize,
    decompressed_values: *ArrayList(f64),
) !void {
    for (start_time..end_time) |i| {
        const decompressed_value = segment_metadata.upper_bound_slope * @as(f64, @floatFromInt(
            i - segment_metadata.start_time,
        )) + segment_metadata.intercept;
        try decompressed_values.append(decompressed_value);
    }
}

test "f64 context can hash" {
    const allocator = testing.allocator;
    var f64_hash_map = shared.HashMapf64(
        f64,
    ).init(allocator);
    defer f64_hash_map.deinit();
    var rnd = std.Random.DefaultPrng.init(@as(u64, @bitCast(std.time.milliTimestamp())));

    // Add 100 elements into the HashMap. For each element, add two more with small deviation of
    // 1e-16 to test that the numbers are different and a new key is created.
    const deviation = 1e-16;
    for (0..100) |_| {
        const rand_number = rnd.random().float(f64) - 0.5;
        try f64_hash_map.put(rand_number, rand_number);
        try f64_hash_map.put(rand_number - deviation, rand_number - deviation);
        try f64_hash_map.put(rand_number + deviation, rand_number + deviation);
    }

    // All elements are expected to be added with an independent key, i.e., all be entries in the
    // for-loop must be stored as separate key-value pairs in the hash map.
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
    var f64_metadata_hash_map = shared.HashMapf64(ArrayList(shared.SegmentMetadata)).init(allocator);
    defer {
        var iterator = f64_metadata_hash_map.iterator();
        while (iterator.next()) |entry| {
            entry.value_ptr.*.deinit();
        }
        f64_metadata_hash_map.deinit();
    }

    var f64_usize_hash_map = shared.HashMapf64(usize).init(allocator);
    defer f64_usize_hash_map.deinit();

    var rnd = std.Random.DefaultPrng.init(@as(u64, @bitCast(std.time.milliTimestamp())));

    for (0..200) |_| {
        const rand_number = @floor((rnd.random().float(f64) - 0.5) * 100) / 10;

        const count_map_result = try f64_usize_hash_map.getOrPut(rand_number);
        if (!count_map_result.found_existing) {
            count_map_result.value_ptr.* = 0;
        }
        count_map_result.value_ptr.* += 1;

        const metadata_map_result = try f64_metadata_hash_map.getOrPut(rand_number);
        if (!metadata_map_result.found_existing) {
            metadata_map_result.value_ptr.* = ArrayList(shared.SegmentMetadata).init(allocator);
        }
        try metadata_map_result.value_ptr.*.append(shared.SegmentMetadata{
            .start_time = count_map_result.value_ptr.*,
            .intercept = rand_number,
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

test "sim-piece can compress and decompress bounded values with any error bound" {
    const allocator = testing.allocator;
    const data_distributions = &[_]tester.DataDistribution{
        .LinearFunctions,
        .BoundedRandomValues,
        .SinusoidalFunction,
    };
    try tester.testErrorBoundedCompressionMethod(
        allocator,
        Method.SimPiece,
        data_distributions,
    );
}

test "sim-piece can compress, decompress and merge many segments with non-zero error bound" {
    const allocator = testing.allocator;

    const error_bound = tester.generateBoundedRandomValue(f32, 0.5, 3, undefined);

    var uncompressed_values = ArrayList(f64).init(allocator);
    defer uncompressed_values.deinit();

    for (0..20) |_| {
        // Generate floating points numbers between 0 and 10. This will generate many merged
        // segments when applying Sim-Piece.
        try tester.generateBoundedRandomValues(&uncompressed_values, 0, 10, undefined);
    }

    try tester.testCompressAndDecompress(
        allocator,
        uncompressed_values.items,
        Method.SimPiece,
        error_bound,
        tersets.isWithinErrorBound,
    );
}

test "sim-piece cannot compress nan values" {
    const allocator = testing.allocator;

    const uncompressed_values = &[4]f64{ 19.0, 48.0, math.nan(f64), 3.0 };

    var compressed_values = ArrayList(u8).init(allocator);
    defer compressed_values.deinit();

    compress(
        uncompressed_values,
        &compressed_values,
        allocator,
        0.1,
    ) catch |err| {
        try testing.expectEqual(Error.UnsupportedInput, err);
        return;
    };

    try testing.expectFmt(
        "",
        "The Sim-Piece algorithm cannot compress nan values",
        .{},
    );
}

test "sim-piece cannot compress inf values" {
    const allocator = testing.allocator;

    const uncompressed_values = &[4]f64{ 19.0, 48.0, math.inf(f64), 3.0 };

    var compressed_values = ArrayList(u8).init(allocator);
    defer compressed_values.deinit();

    compress(
        uncompressed_values,
        &compressed_values,
        allocator,
        0.1,
    ) catch |err| {
        try testing.expectEqual(Error.UnsupportedInput, err);
        return;
    };

    try testing.expectFmt(
        "",
        "The Sim-Piece algorithm cannot compress inf values",
        .{},
    );
}

test "sim-piece cannot compress f64 with reduced precision" {
    const allocator = testing.allocator;

    const uncompressed_values = &[4]f64{ 19.0, 48.0, 1e17, 3.0 };

    var compressed_values = ArrayList(u8).init(allocator);
    defer compressed_values.deinit();

    compress(
        uncompressed_values,
        &compressed_values,
        allocator,
        0.1,
    ) catch |err| {
        try testing.expectEqual(Error.UnsupportedInput, err);
        return;
    };

    try testing.expectFmt(
        "",
        "The Sim-Piece algorithm cannot compress reduced precision floating point values",
        .{},
    );
}
