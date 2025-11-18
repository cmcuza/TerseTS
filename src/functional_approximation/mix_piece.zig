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

//! Implementation of the Mix-Piece algorithm from the paper
//! "Xenophon Kitsios, Panagiotis Liakos, Katia Papakonstantinopoulou, and Yannis Kotidis.
//! Flexible grouping of linear segments for highly accurate lossy compression of time series data.
//! Proc. VLDB Endow. 15, 7 2024.
//! https://doi.org/10.1007/s00778-024-00862-z".
//!
//! The implementation is partially based on the author's implementation at
//! https://github.com/xkitsios/Mix-Piece (accessed on 20-05-2025).

const std = @import("std");
const math = std.math;
const mem = std.mem;
const time = std.time;
const testing = std.testing;
const rand = std.Random;
const Method = tersets.Method;
const ArrayList = std.ArrayList;
const Allocator = mem.Allocator;

const tersets = @import("../tersets.zig");
const configuration = @import("../configuration.zig");
const shared_structs = @import("../utilities/shared_structs.zig");
const shared_functions = @import("../utilities/shared_functions.zig");
const tester = @import("../tester.zig");
const sp = @import("sim_piece.zig");

const Error = tersets.Error;
const DiscretePoint = shared_structs.DiscretePoint;

/// Struct for keeping ungrouped segments.
const UngroupedSegment = struct {
    slope: f64,
    intercept: f64,
    timestamp: usize,
};

/// Struct for the grouping of cross intercept groups.
const InterceptTimestampPair = struct {
    intercept: f64,
    timestamp: usize,
};

/// HashMap for cross_intercept_groups.
const CrossInterceptGroupsMap = shared_structs.HashMapf64(
    ArrayList(InterceptTimestampPair),
);

/// Compresses `uncompressed_values` within `error_bound` using the "Mix-Piece" algorithm.
/// The function writes the result to `compressed_values`. The `allocator` is used for memory
/// allocation of intermediate data structures and the `method_configuration` parser.
/// The `method_configuration` is expected to be of `AbsoluteErrorBound` type otherwise an
/// `InvalidConfiguration` error is return. If any other error occurs during the execution
/// of the method, it is returned.
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

    // It is save to access the error bound now.
    const error_bound: f32 = parsed_configuration.abs_error_bound;

    if (error_bound == 0.0) {
        return Error.UnsupportedErrorBound;
    }

    // Mix-Piece Phase 1: Compute segments metadata.
    var segments_metadata = ArrayList(shared_structs.SegmentMetadata).init(allocator);
    defer segments_metadata.deinit();
    try computeSegmentsMetadata(
        uncompressed_values,
        &segments_metadata,
        error_bound,
    );

    // Mix-Piece Phase 2: Merge segments metadata.
    var same_intercept_groups = ArrayList(shared_structs.SegmentMetadata).init(allocator);
    defer same_intercept_groups.deinit();

    var cross_intercept_groups = ArrayList(shared_structs.SegmentMetadata).init(allocator);
    defer cross_intercept_groups.deinit();

    var ungrouped_segments = ArrayList(shared_structs.SegmentMetadata).init(allocator);
    defer ungrouped_segments.deinit();

    try mergeSegmentsMetadata(
        segments_metadata,
        &same_intercept_groups,
        &cross_intercept_groups,
        &ungrouped_segments,
        allocator,
    );

    // Mix-Piece Phase 3: Populate the three data structures. Separately handle three parts.
    // Part 1: segment groups that share the same intercept.
    // Part 2: segment groups that don't share the same intercept but the same slope.
    // Part 3: segment unmerged segments.

    // Part 1: Handle same intercept groups.
    var same_intercept_groups_map = shared_structs.HashMapf64(shared_structs.HashMapf64(ArrayList(usize))).init(allocator);
    defer {
        // Deinit all ArrayList instances within the inner HashMaps and then deinit the inner HashMaps.
        // themselves before finally deinit the outer HashMap.
        var hash_to_hash_iterator = same_intercept_groups_map.iterator();
        while (hash_to_hash_iterator.next()) |hash_to_hash_entry| {
            var hash_to_array_iterator = hash_to_hash_entry.value_ptr.*.iterator();
            while (hash_to_array_iterator.next()) |hash_to_array_entry| {
                hash_to_array_entry.value_ptr.*.deinit();
            }
            hash_to_hash_entry.value_ptr.*.deinit();
        }
        same_intercept_groups_map.deinit();
    }
    try populateSameInterceptGroupsHashMap(
        same_intercept_groups,
        &same_intercept_groups_map,
        allocator,
    );

    // Part 2: Handle cross-intercept groups.
    var cross_intercept_groups_map = CrossInterceptGroupsMap.init(allocator);
    defer {
        var iterator = cross_intercept_groups_map.iterator();
        while (iterator.next()) |entry| {
            entry.value_ptr.*.deinit();
        }
        cross_intercept_groups_map.deinit();
    }

    try populateCrossInterceptGroupsAsHashMap(
        cross_intercept_groups,
        &cross_intercept_groups_map,
        allocator,
    );

    // Part 3: Handle ungrouped segments.
    var ungrouped_segments_array = ArrayList(UngroupedSegment).init(allocator);
    defer ungrouped_segments_array.deinit();

    try populateUngroupedSegmentsArray(
        ungrouped_segments,
        &ungrouped_segments_array,
    );

    // Mix-Piece Phase 4: Create compressed representation with boundaries.
    // Part 1: Store the number of intercept groups.
    try shared_functions.appendValue(
        usize,
        same_intercept_groups_map.count(),
        compressed_values,
    );

    // Part 2: Store the number of slope groups.
    try shared_functions.appendValue(
        usize,
        cross_intercept_groups_map.count(),
        compressed_values,
    );

    // part 3: Store the number of ungrouped segments.
    try shared_functions.appendValue(
        usize,
        ungrouped_segments_array.items.len,
        compressed_values,
    );

    // Write the actual data for each part.
    // Part 1: Same intercept groups.
    try sp.createCompressedRepresentation(
        same_intercept_groups_map,
        compressed_values,
    );

    // Part 2: Cross-intercept groups.
    try createCompressedRepresentationCrossInterceptGroups(
        cross_intercept_groups_map,
        compressed_values,
    );

    // Part 3: Ungrouped segments.
    try createCompressedRepresentationUngroupedSegments(
        ungrouped_segments_array,
        compressed_values,
    );

    // The last timestamp must be stored, otherwise the end time during decompression is unknown.
    try shared_functions.appendValue(usize, uncompressed_values.len, compressed_values);
}

/// Decompress `compressed_values` produced by "Mix-Piece". The function writes the result to
/// `decompressed_values`. The `allocator` is used for memory allocation of intermediate data
/// structures. If an error occurs, it is returned.
pub fn decompress(
    allocator: mem.Allocator,
    compressed_values: []const u8,
    decompressed_values: *ArrayList(f64),
) Error!void {

    // Initialize temp array to store all segments.
    var all_segments = ArrayList(shared_structs.SegmentMetadata).init(allocator);
    defer all_segments.deinit();

    // Read header to get structure counts.
    const header = mem.bytesAsSlice(usize, compressed_values[0 .. 3 * @sizeOf(usize)]);
    const part1_count = header[0]; // Number of intercept groups in Part 1.
    const part2_count = header[1]; // Number of slope groups in Part 2.
    const part3_count = header[2]; // Number of ungrouped segments in Part 3.

    var offset: usize = 3 * @sizeOf(usize);

    // Part 1. Parse Same Intercept Groups.
    if (part1_count > 0) {
        // We need to parse part1_count intercept groups.
        // Each group has: intercept, slopes_count,
        // then for each slope: slope, timestamps_count, timestamps.
        for (0..part1_count) |_| {
            const intercept = try shared_functions.readOffsetValue(f64, compressed_values, &offset);

            const slopes_count = try shared_functions.readOffsetValue(usize, compressed_values, &offset);

            for (0..slopes_count) |_| {
                const slope = try shared_functions.readOffsetValue(f64, compressed_values, &offset);

                const timestamps_count = try shared_functions.readOffsetValue(
                    usize,
                    compressed_values,
                    &offset,
                );

                var timestamp: usize = 0;
                for (0..timestamps_count) |_| {
                    const delta = try shared_functions.readOffsetValue(usize, compressed_values, &offset);

                    timestamp += delta;
                    try all_segments.append(.{
                        .start_time = timestamp,
                        .lower_bound_slope = slope,
                        .upper_bound_slope = slope,
                        .intercept = intercept,
                    });
                }
            }
        }
    }

    // Part 2. Parse Cross Intercept Groups.
    if (part2_count > 0) {
        // We need to parse part2_count slope groups.
        // Each group has: slope, pair_count, then for each pair: intercept, timestamp_delta.
        for (0..part2_count) |_| {
            const slope = try shared_functions.readOffsetValue(f64, compressed_values, &offset);

            const pair_count = try shared_functions.readOffsetValue(usize, compressed_values, &offset);

            var timestamp: usize = 0;
            for (0..pair_count) |_| {
                const intercept = try shared_functions.readOffsetValue(f64, compressed_values, &offset);

                const delta = try shared_functions.readOffsetValue(usize, compressed_values, &offset);

                timestamp += delta;
                try all_segments.append(.{
                    .start_time = timestamp,
                    .lower_bound_slope = slope,
                    .upper_bound_slope = slope,
                    .intercept = intercept,
                });
            }
        }
    }

    // Part 3. Parse Ungrouped Segments.
    if (part3_count > 0) {
        // Each segment has: slope, intercept, timestamp_delta.
        var timestamp: usize = 0;
        for (0..part3_count) |_| {
            const slope = try shared_functions.readOffsetValue(f64, compressed_values, &offset);

            const intercept = try shared_functions.readOffsetValue(f64, compressed_values, &offset);

            const delta = try shared_functions.readOffsetValue(usize, compressed_values, &offset);

            timestamp += delta;
            try all_segments.append(.{
                .start_time = timestamp,
                .lower_bound_slope = slope,
                .upper_bound_slope = slope,
                .intercept = intercept,
            });
        }
    }

    // Sort all segments by timestamp.
    mem.sort(shared_structs.SegmentMetadata, all_segments.items, {}, struct {
        fn compare(_: void, a: shared_structs.SegmentMetadata, b: shared_structs.SegmentMetadata) bool {
            return a.start_time < b.start_time;
        }
    }.compare);

    const final_timestamp = try shared_functions.readOffsetValue(usize, compressed_values, &offset);

    // Decompress each segment separately.
    for (0..all_segments.items.len) |i| {
        const segment = all_segments.items[i];
        const start_time = segment.start_time;
        const end_time = if (i + 1 < all_segments.items.len)
            all_segments.items[i + 1].start_time
        else
            final_timestamp;

        // Generate points for the current segment.
        try sp.decompressSegment(
            segment,
            start_time,
            end_time,
            decompressed_values,
        );
    }
}

/// Mix-Piece Phase 1: Compute `SegmentMetadata` for each segment that can be approximated
/// by a linear function within the `error_bound` from `uncompressed_values`.
/// Uses both floor and ceil quantization to find optimal segments.
fn computeSegmentsMetadata(
    uncompressed_values: []const f64,
    segments_metadata: *ArrayList(shared_structs.SegmentMetadata),
    error_bound: f32,
) Error!void {
    // Adjust the error bound to avoid exceeding it during decompression.
    const adjusted_error_bound = error_bound - shared_structs.ErrorBoundMargin;

    // Track bounds for floor quantization.
    var upper_bound_slope_floor: f64 = math.floatMax(f64);
    var lower_bound_slope_floor: f64 = -math.floatMax(f64);

    // Track bounds for ceil quantization.
    var upper_bound_slope_ceil: f64 = math.floatMax(f64);
    var lower_bound_slope_ceil: f64 = -math.floatMax(f64);

    // Check if the first point is NaN, infinite or a reduced precision f64.
    // If so, return an error.
    if (!math.isFinite(uncompressed_values[0]) or @abs(uncompressed_values[0]) > tester.max_test_value)
        return Error.UnsupportedInput;

    // Initialize the `start_point` with the first uncompressed value.
    var start_point: DiscretePoint = .{ .time = 0, .value = uncompressed_values[0] };

    // The quantization can only be done using the original error bound. Afterwards, we add
    // `shared_structs.ErrorBoundMargin` to avoid exceeding the error bound during decompression.
    var quantized_intercept_floor = quantizeFloor(uncompressed_values[0], error_bound) +
        shared_structs.ErrorBoundMargin;

    var quantized_intercept_ceil = quantizeCeil(uncompressed_values[0], error_bound) +
        shared_structs.ErrorBoundMargin;

    // Track which quantization mode is still valid.
    var floor_valid = true;
    var ceil_valid = true;

    // Track the last valid index for each quantization mode.
    var last_valid_floor: usize = 0;
    var last_valid_ceil: usize = 0;

    // The first point is already part of `current_segment`, the next point is at index one.
    for (1..uncompressed_values.len) |current_timestamp| {

        // Check if the current point is NaN, infinite or a reduced precision f64.
        // If so, return an error.
        if (!math.isFinite(uncompressed_values[current_timestamp]) or
            @abs(uncompressed_values[current_timestamp]) > tester.max_test_value)
            return Error.UnsupportedInput;

        const end_point: DiscretePoint = .{
            .time = current_timestamp,
            .value = uncompressed_values[current_timestamp],
        };

        // `segment_size` of type f64 to avoid casting from usize when computing other variables.
        const segment_size: f64 = @floatFromInt(current_timestamp - start_point.time);

        // Check floor quantization bounds.
        if (floor_valid) {
            const upper_limit_floor: f64 = upper_bound_slope_floor * segment_size +
                quantized_intercept_floor;
            const lower_limit_floor: f64 = lower_bound_slope_floor * segment_size +
                quantized_intercept_floor;

            if ((upper_limit_floor < (end_point.value - adjusted_error_bound)) or
                (lower_limit_floor > (end_point.value + adjusted_error_bound)))
            {
                floor_valid = false;
                last_valid_floor = current_timestamp - 1;
            } else {
                // Update floor bounds.
                const new_upper_bound_slope: f64 =
                    (end_point.value + adjusted_error_bound - quantized_intercept_floor) /
                    segment_size;
                const new_lower_bound_slope: f64 =
                    (end_point.value - adjusted_error_bound - quantized_intercept_floor) /
                    segment_size;

                if (end_point.value + adjusted_error_bound < upper_limit_floor)
                    upper_bound_slope_floor = @max(new_upper_bound_slope, lower_bound_slope_floor);
                if (end_point.value - adjusted_error_bound > lower_limit_floor)
                    lower_bound_slope_floor = @min(new_lower_bound_slope, upper_bound_slope_floor);
            }
        }

        // Check ceil quantization bounds.
        if (ceil_valid) {
            const upper_limit_ceil: f64 = upper_bound_slope_ceil * segment_size +
                quantized_intercept_ceil;
            const lower_limit_ceil: f64 = lower_bound_slope_ceil * segment_size +
                quantized_intercept_ceil;

            if ((upper_limit_ceil < (end_point.value - adjusted_error_bound)) or
                (lower_limit_ceil > (end_point.value + adjusted_error_bound)))
            {
                ceil_valid = false;
                last_valid_ceil = current_timestamp - 1;
            } else {
                // Update ceil bounds.
                const new_upper_bound_slope: f64 =
                    (end_point.value + adjusted_error_bound - quantized_intercept_ceil) /
                    segment_size;
                const new_lower_bound_slope: f64 =
                    (end_point.value - adjusted_error_bound - quantized_intercept_ceil) /
                    segment_size;

                if (end_point.value + adjusted_error_bound < upper_limit_ceil)
                    upper_bound_slope_ceil = @max(new_upper_bound_slope, lower_bound_slope_ceil);
                if (end_point.value - adjusted_error_bound > lower_limit_ceil)
                    lower_bound_slope_ceil = @min(new_lower_bound_slope, upper_bound_slope_ceil);
            }
        }

        // If both quantizations have failed, choose the one that went further.
        if (!floor_valid and !ceil_valid) {
            if (last_valid_floor >= last_valid_ceil) {
                // Use floor quantization.
                try segments_metadata.append(.{
                    .start_time = start_point.time,
                    .intercept = quantized_intercept_floor,
                    .upper_bound_slope = upper_bound_slope_floor,
                    .lower_bound_slope = lower_bound_slope_floor,
                });
            } else {
                // Use ceil quantization.
                try segments_metadata.append(.{
                    .start_time = start_point.time,
                    .intercept = quantized_intercept_ceil,
                    .upper_bound_slope = upper_bound_slope_ceil,
                    .lower_bound_slope = lower_bound_slope_ceil,
                });
            }

            // Reset for next segment.
            start_point = end_point;

            quantized_intercept_floor = quantizeFloor(start_point.value, error_bound) +
                shared_structs.ErrorBoundMargin;
            quantized_intercept_ceil = quantizeCeil(start_point.value, error_bound) +
                shared_structs.ErrorBoundMargin;

            upper_bound_slope_floor = math.floatMax(f64);
            lower_bound_slope_floor = -math.floatMax(f64);
            upper_bound_slope_ceil = math.floatMax(f64);
            lower_bound_slope_ceil = -math.floatMax(f64);
            floor_valid = true;
            ceil_valid = true;
            last_valid_floor = current_timestamp;
            last_valid_ceil = current_timestamp;
        }
    }

    // Handle the final segment.
    const segment_size = uncompressed_values.len - start_point.time;
    if (segment_size > 0) {
        // Choose the quantization that remained valid longer.
        if (floor_valid and !ceil_valid) {
            // Floor is still valid.
            if (segment_size == 1) {
                upper_bound_slope_floor = 0;
                lower_bound_slope_floor = 0;
            }
            try segments_metadata.append(.{
                .start_time = start_point.time,
                .intercept = quantized_intercept_floor,
                .upper_bound_slope = upper_bound_slope_floor,
                .lower_bound_slope = lower_bound_slope_floor,
            });
        } else if (ceil_valid and !floor_valid) {
            // Ceil is still valid.
            if (segment_size == 1) {
                upper_bound_slope_ceil = 0;
                lower_bound_slope_ceil = 0;
            }
            try segments_metadata.append(.{
                .start_time = start_point.time,
                .intercept = quantized_intercept_ceil,
                .upper_bound_slope = upper_bound_slope_ceil,
                .lower_bound_slope = lower_bound_slope_ceil,
            });
        } else {
            // Both are valid or both invalid - choose the one that is closer to the original value.
            const original_value = uncompressed_values[start_point.time];
            if (@round(original_value / error_bound) == @ceil(original_value / error_bound)) {
                // Ceil quantization.
                if (segment_size == 1) {
                    upper_bound_slope_ceil = 0;
                    lower_bound_slope_ceil = 0;
                }
                try segments_metadata.append(.{
                    .start_time = start_point.time,
                    .intercept = quantized_intercept_ceil,
                    .upper_bound_slope = upper_bound_slope_ceil,
                    .lower_bound_slope = lower_bound_slope_ceil,
                });
            } else {
                // Floor quantization.
                if (segment_size == 1) {
                    upper_bound_slope_floor = 0;
                    lower_bound_slope_floor = 0;
                }
                try segments_metadata.append(.{
                    .start_time = start_point.time,
                    .intercept = quantized_intercept_floor,
                    .upper_bound_slope = upper_bound_slope_floor,
                    .lower_bound_slope = lower_bound_slope_floor,
                });
            }
        }
    }
}

/// Mix-Piece Phase 2. Merge the elements in `segments_metadata` using Algorithm 4 from the paper.
/// The results are stored in three output arrays.
/// `same_intercept_groups`: Groups formed from segments sharing the same quantized intercept value.
/// `cross_intercept_groups`: Groups formed from segments with different intercept values.
/// `ungrouped_segments`: Segments that couldn't be grouped with any other segment.
/// The `allocator` is used to allocate memory for the intermediate representations needed.
fn mergeSegmentsMetadata(
    segments_metadata: ArrayList(shared_structs.SegmentMetadata),
    same_intercept_groups: *ArrayList(shared_structs.SegmentMetadata), // 'groups_b' in paper.
    cross_intercept_groups: *ArrayList(shared_structs.SegmentMetadata), // 'groups' in paper.
    ungrouped_segments: *ArrayList(shared_structs.SegmentMetadata), // 'rest' in paper.
    allocator: mem.Allocator,
) !void {
    // Temporary storage for timestamps being merged in Part 1.
    var timestamps_array = ArrayList(usize).init(allocator);
    defer timestamps_array.deinit();

    // Temporary storage for segments that couldn't be grouped in Part 1.
    var single_segment_groups = ArrayList(shared_structs.SegmentMetadata).init(allocator);
    defer single_segment_groups.deinit();

    // Group segments by their quantized intercept value.
    var segments_by_intercept = shared_structs.HashMapf64(ArrayList(shared_structs.SegmentMetadata)).init(allocator);
    defer {
        // Clean up all ArrayLists within the HashMap.
        var iterator = segments_by_intercept.iterator();
        while (iterator.next()) |entry| {
            entry.value_ptr.*.deinit();
        }
        segments_by_intercept.deinit();
    }

    // Populate the HashMap with segments grouped by their quantized intercept value.
    for (segments_metadata.items) |segment_metadata| {
        try sp.appendSegmentMetadata(
            &segments_by_intercept,
            segment_metadata,
            allocator,
        );
    }

    // Part 1: Group segments with the same starting point.
    var iterator = segments_by_intercept.iterator();
    while (iterator.next()) |entry| {
        const segments_same_intercept_val = entry.value_ptr.*;

        // Sort segments by ascending lower bound slope.
        // This ordering allows us to find the optimal grouping using a greedy approach.
        mem.sort(
            shared_structs.SegmentMetadata,
            segments_same_intercept_val.items,
            {},
            sp.compareMetadataBySlope,
        );

        // Initialize the first group with the first segment.
        var current_group: shared_structs.SegmentMetadata = .{
            .start_time = 0.0, // Not used for group metadata.
            .intercept = segments_same_intercept_val.items[0].intercept,
            .lower_bound_slope = segments_same_intercept_val.items[0].lower_bound_slope,
            .upper_bound_slope = segments_same_intercept_val.items[0].upper_bound_slope,
        };
        try timestamps_array.append(segments_same_intercept_val.items[0].start_time);

        // Process remaining segments for this intercept value.
        for (segments_same_intercept_val.items[1..]) |segment| {
            // Check if the current segment's interval overlaps with the group's interval.
            if (segment.lower_bound_slope <= current_group.upper_bound_slope and
                segment.upper_bound_slope >= current_group.lower_bound_slope)
            {
                // Segments can be grouped - update the group's interval.
                try timestamps_array.append(segment.start_time);
                current_group.lower_bound_slope = @max(
                    current_group.lower_bound_slope,
                    segment.lower_bound_slope,
                );
                current_group.upper_bound_slope = @min(
                    current_group.upper_bound_slope,
                    segment.upper_bound_slope,
                );
            } else {
                // Cannot merge - finalize current group.
                if (timestamps_array.items.len > 1) {
                    // Multiple segments in group - add to same_intercept_groups.
                    for (timestamps_array.items) |timestamp| {
                        try same_intercept_groups.append(.{
                            .start_time = timestamp,
                            .intercept = current_group.intercept,
                            .lower_bound_slope = current_group.lower_bound_slope,
                            .upper_bound_slope = current_group.upper_bound_slope,
                        });
                    }
                } else {
                    // Single segment - save for Part 2 processing.
                    try single_segment_groups.append(.{
                        .start_time = timestamps_array.items[0],
                        .intercept = current_group.intercept,
                        .lower_bound_slope = current_group.lower_bound_slope,
                        .upper_bound_slope = current_group.upper_bound_slope,
                    });
                }

                // Start new group with current segment.
                timestamps_array.clearRetainingCapacity();
                current_group = .{
                    .start_time = 0.0,
                    .intercept = segment.intercept,
                    .lower_bound_slope = segment.lower_bound_slope,
                    .upper_bound_slope = segment.upper_bound_slope,
                };
                try timestamps_array.append(segment.start_time);
            }
        }

        // Handle the final group for this intercept value.
        if (timestamps_array.items.len > 1) {
            for (timestamps_array.items) |timestamp| {
                try same_intercept_groups.append(.{
                    .start_time = timestamp,
                    .intercept = current_group.intercept,
                    .lower_bound_slope = current_group.lower_bound_slope,
                    .upper_bound_slope = current_group.upper_bound_slope,
                });
            }
        } else {
            try single_segment_groups.append(.{
                .start_time = timestamps_array.items[0],
                .intercept = current_group.intercept,
                .lower_bound_slope = current_group.lower_bound_slope,
                .upper_bound_slope = current_group.upper_bound_slope,
            });
        }
        timestamps_array.clearRetainingCapacity();
    }

    // Part 2: Group remaining ungrouped segments across different intercept values.
    // Sort ungrouped segments by lower bound slope.
    mem.sort(
        shared_structs.SegmentMetadata,
        single_segment_groups.items,
        {},
        sp.compareMetadataBySlope,
    );

    // For cross-intercept grouping, track both intercept and timestamp values for each segment.
    var cross_intercept_segment_info =
        ArrayList(struct { intercept: f64, start_time: usize }).init(allocator);
    defer cross_intercept_segment_info.deinit();

    if (single_segment_groups.items.len > 0) {
        // Initialize with the first ungrouped segment.
        var current_cross_group: shared_structs.SegmentMetadata = .{
            .start_time = 0.0, // Not used for group metadata.
            .intercept = 0.0, // Will vary for each segment in the group.
            .lower_bound_slope = single_segment_groups.items[0].lower_bound_slope,
            .upper_bound_slope = single_segment_groups.items[0].upper_bound_slope,
        };
        try cross_intercept_segment_info.append(.{
            .intercept = single_segment_groups.items[0].intercept,
            .start_time = single_segment_groups.items[0].start_time,
        });

        // Process remaining ungrouped segments.
        for (single_segment_groups.items[1..]) |segment| {
            // Check if intervals overlap.
            if (segment.lower_bound_slope <= current_cross_group.upper_bound_slope and
                segment.upper_bound_slope >= current_cross_group.lower_bound_slope)
            {
                // Can merge - update group interval.
                current_cross_group.lower_bound_slope = @max(
                    current_cross_group.lower_bound_slope,
                    segment.lower_bound_slope,
                );
                current_cross_group.upper_bound_slope = @min(
                    current_cross_group.upper_bound_slope,
                    segment.upper_bound_slope,
                );
                try cross_intercept_segment_info.append(.{
                    .intercept = segment.intercept,
                    .start_time = segment.start_time,
                });
            } else {
                // Cannot merge - finalize current cross-intercept group.
                if (cross_intercept_segment_info.items.len > 1) {
                    // Multiple segments - add to cross_intercept_groups.
                    for (cross_intercept_segment_info.items) |info| {
                        try cross_intercept_groups.append(.{
                            .start_time = info.start_time,
                            .intercept = info.intercept,
                            .lower_bound_slope = current_cross_group.lower_bound_slope,
                            .upper_bound_slope = current_cross_group.upper_bound_slope,
                        });
                    }
                } else {
                    // Single segment - add to ungrouped_segments.
                    for (cross_intercept_segment_info.items) |info| {
                        try ungrouped_segments.append(.{
                            .start_time = info.start_time,
                            .intercept = info.intercept,
                            .lower_bound_slope = current_cross_group.lower_bound_slope,
                            .upper_bound_slope = current_cross_group.upper_bound_slope,
                        });
                    }
                }

                // Start new cross-intercept group.
                cross_intercept_segment_info.clearRetainingCapacity();
                current_cross_group = .{
                    .start_time = 0.0,
                    .intercept = 0.0,
                    .lower_bound_slope = segment.lower_bound_slope,
                    .upper_bound_slope = segment.upper_bound_slope,
                };
                try cross_intercept_segment_info.append(.{
                    .intercept = segment.intercept,
                    .start_time = segment.start_time,
                });
            }
        }

        // Handle the final cross-intercept group.
        if (cross_intercept_segment_info.items.len > 1) {
            for (cross_intercept_segment_info.items) |info| {
                try cross_intercept_groups.append(.{
                    .start_time = info.start_time,
                    .intercept = info.intercept,
                    .lower_bound_slope = current_cross_group.lower_bound_slope,
                    .upper_bound_slope = current_cross_group.upper_bound_slope,
                });
            }
        } else if (cross_intercept_segment_info.items.len == 1) {
            for (cross_intercept_segment_info.items) |info| {
                try ungrouped_segments.append(.{
                    .start_time = info.start_time,
                    .intercept = info.intercept,
                    .lower_bound_slope = current_cross_group.lower_bound_slope,
                    .upper_bound_slope = current_cross_group.upper_bound_slope,
                });
            }
        }
    }

    // Sort all output arrays by start time.
    mem.sort(
        shared_structs.SegmentMetadata,
        same_intercept_groups.items,
        {},
        sp.compareMetadataByStartTime,
    );

    mem.sort(
        shared_structs.SegmentMetadata,
        cross_intercept_groups.items,
        {},
        sp.compareMetadataByStartTime,
    );

    mem.sort(
        shared_structs.SegmentMetadata,
        ungrouped_segments.items,
        {},
        sp.compareMetadataByStartTime,
    );
}

/// Mix-Piece Phase 3.1. Populate the `SegmentMetadata` HashMap from intercept points in
/// `same_intercept_groups` to a HashMap from the approximation slope to an array list of
/// timestamps and store it in `same_intercept_groups_map`. The `allocator` is used to
/// allocate memory of intermediates.
fn populateSameInterceptGroupsHashMap(
    same_intercept_groups: ArrayList(shared_structs.SegmentMetadata),
    same_intercept_groups_map: *shared_structs.HashMapf64(shared_structs.HashMapf64(ArrayList(usize))),
    allocator: mem.Allocator,
) !void {
    for (same_intercept_groups.items) |segment_metadata| {
        const intercept: f64 = segment_metadata.intercept;
        const slope: f64 = (segment_metadata.lower_bound_slope +
            segment_metadata.upper_bound_slope) / 2;

        // Get or put the inner HashMap entry for the given `intercept` wich will contain the
        // slopes and timestamps associated to it.
        const hash_to_hash_result = try same_intercept_groups_map.getOrPut(intercept);
        if (!hash_to_hash_result.found_existing) {
            hash_to_hash_result.value_ptr.* = shared_structs.HashMapf64(
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

/// Mix-Piece Phase 3.2. Populate HashMap of `cross_intercept_groups` by grouping the segments by
/// their slope and stores intercept-timestamp pairs and stores it in `cross_intercept_groups_map`.
/// The output format is a HashMap from slope to an ArrayList of corresponding to:
/// a_j; k_j; b_j,1; t_j,1; b_j,k_j; t_j,k_j, where a_j is the slope, k_j is the number of
/// intercept-timestamp pairs, b_j,i; t_j,i are the intercept and timestamp pairs.
/// The `allocator` is used to allocate memory for intermediates.
fn populateCrossInterceptGroupsAsHashMap(
    cross_intercept_groups: ArrayList(shared_structs.SegmentMetadata),
    cross_intercept_groups_map: *CrossInterceptGroupsMap,
    allocator: mem.Allocator,
) !void {
    for (cross_intercept_groups.items) |segment_metadata| {
        // Calculate the slope as the average of lower and upper bound slopes.
        const slope: f64 = (segment_metadata.lower_bound_slope + segment_metadata.upper_bound_slope) / 2;

        // Get or put the slope key into the hashmap.
        const hash_result = try cross_intercept_groups_map.getOrPut(slope);
        if (!hash_result.found_existing) {
            // Initialize a new ArrayList for this slope.
            hash_result.value_ptr.* = ArrayList(InterceptTimestampPair).init(allocator);
        }

        // Create an InterceptTimestampPair and add it to the list for this slope.
        const intercept_timestamp_pair = InterceptTimestampPair{
            .intercept = segment_metadata.intercept,
            .timestamp = segment_metadata.start_time,
        };

        try hash_result.value_ptr.*.append(intercept_timestamp_pair);
    }
}

/// Mix-Piece Phase 3.3. Populate `ungrouped_segments` of SegmentMetadata in `ungrouped_segments_array`
/// ArrayList of UngroupedSegment structs. The final representation is a byte array containing the
/// slopes a_i, intercepts b_j, and timestamps t_k per segment.
fn populateUngroupedSegmentsArray(
    ungrouped_segments: ArrayList(shared_structs.SegmentMetadata),
    ungrouped_segments_array: *ArrayList(UngroupedSegment),
) !void {
    for (0..ungrouped_segments.items.len) |index| {
        // Calculate slope as average of upper and lower bounds.
        const slope = (ungrouped_segments.items[index].upper_bound_slope +
            ungrouped_segments.items[index].lower_bound_slope) / 2.0;

        // Create UngroupedSegment struct - no allocation needed.
        const ungrouped_segment = UngroupedSegment{
            .slope = slope,
            .intercept = ungrouped_segments.items[index].intercept,
            .timestamp = ungrouped_segments.items[index].start_time,
        };

        // Append to the output array (ArrayList handles memory allocation internally).
        try ungrouped_segments_array.append(ungrouped_segment);
    }
}

/// Mix-Piece Phase 4.1. Create a compressed representation from the 'merged_segments_metadata_map'
/// that can be decoded during decompression and stored in `compressed_values`. The compressed
/// representation is a byte array containing the intercepts, slopes and timestamps per segment.
/// Specifically, the compressed representation has the following structure:
/// [b_1, N_1, a_11, M_11, t_1, t_2, ..., a_12, M_12, t_1, ...., b_2, N_2, a_21, M_21, t1, t2, ...,
///  b_n, N_n, a_n1, M_n1, t_1, ...], where b_i are the intercepts, N_i are the number of slopes
/// associated to intercept b_i, a_ij are the slopes associated to intercept b_i, M_ij are
/// the number of timestamps associated to the slope a_ij, and t_k are the timestamps.
fn createCompressedRepresentationMergedSegments(
    merged_segments_metadata_map: shared_structs.HashMapf64(shared_structs.HashMapf64(ArrayList(usize))),
    compressed_values: *ArrayList(u8),
) !void {
    // Iterate over the outer HashMap to append the intercepts b_i.
    var hash_to_hash_iterator = merged_segments_metadata_map.iterator();
    while (hash_to_hash_iterator.next()) |hash_to_hash_entry| {
        const current_intercept: f64 = hash_to_hash_entry.key_ptr.*;

        try shared_functions.appendValue(f64, current_intercept, compressed_values);

        // Append the number of slopes N_i associated to `current_intercept` b_i.
        try shared_functions.appendValue(usize, hash_to_hash_entry.value_ptr.*.count(), compressed_values);

        // Iterate over the inner HashMap to append the slopes a_ij associated to b_i.
        var hash_to_array_iterator = hash_to_hash_entry.value_ptr.*.iterator();
        while (hash_to_array_iterator.next()) |hash_to_array_entry| {
            const current_slope = hash_to_array_entry.key_ptr.*;
            try shared_functions.appendValue(f64, current_slope, compressed_values);

            // Append the number of timestamps M_ij associated to `current_slope` a_ij.
            try shared_functions.appendValue(usize, hash_to_array_entry.value_ptr.*.items.len, compressed_values);
            var previous_timestamp: usize = 0;

            // Iterate over the ArrayList to append the timestamps t_k.
            for (hash_to_array_entry.value_ptr.*.items) |timestamp| {
                try shared_functions.appendValue(usize, timestamp - previous_timestamp, compressed_values);
                previous_timestamp = timestamp;
            }
        }
    }
}

/// Mix-Piece Phase 4.2. Create compressed representation from the 'cross_intercept_groups_map'
/// that can decoded during decompression and store it in `compressed_values`. The compressed
/// representation is a byte array containing the slopes a_j, intercept-timestamps pairs and
/// necessary metadata for proper decoding. Specifically, the compressed representation has
/// the following structure: [a_j, k_j, b_j,1, t_j,1, b_j,2, t_j,2, ..., b_j,k_j, t_j,k_j]
/// where a_j is the  slope, k_j is the  number of intercept-timestamp pairs for this slope
/// b_j,i is the i-th intercept for slope a_j, t_j,i is the i-th timestamp for slope a_j.
fn createCompressedRepresentationCrossInterceptGroups(
    cross_intercept_groups_map: CrossInterceptGroupsMap,
    compressed_values: *ArrayList(u8),
) !void {
    // Iterate over the HashMap to append each slope and its associated intercept-timestamp pairs.
    var slope_iterator = cross_intercept_groups_map.iterator();
    while (slope_iterator.next()) |slope_entry| {
        const current_slope: f64 = slope_entry.key_ptr.*;

        // Append the slope a_j.
        try shared_functions.appendValue(f64, current_slope, compressed_values);

        // Append the number of intercept-timestamp pairs k_j for this slope.
        try shared_functions.appendValue(usize, slope_entry.value_ptr.*.items.len, compressed_values);

        // Store previous timestamp for delta encoding.
        var previous_timestamp: usize = 0;

        // Iterate over all intercept-timestamp pairs for this slope.
        for (slope_entry.value_ptr.*.items) |pair| {
            // Append the intercept b_j,i.
            try shared_functions.appendValue(f64, pair.intercept, compressed_values);

            // Append the timestamp t_j,i.
            try shared_functions.appendValue(usize, pair.timestamp - previous_timestamp, compressed_values);
            previous_timestamp = pair.timestamp;
        }
    }
}

/// Mix-Piece Phase 4.3. Create compressed representation from the 'ungrouped_segments_array'
/// that can be decoded during decompression and store it in `compressed_values`. The compressed
/// representation is a byte array containing the slopes a_j, intercept-timestamps pairs and
/// necessary metadata for proper decoding. Specifically, the compressed representation has
/// the following structure: [a_j, k_j, b_j,1, t_j,1, b_j,2, t_j,2, ..., b_j,k_j, t_j,k_j]
/// where a_j is the  slope, k_j is the  number of intercept-timestamp pairs for this slope
/// b_j,i is the i-th intercept for slope a_j, t_j,i is the i-th timestamp for slope a_j.
fn createCompressedRepresentationUngroupedSegments(
    ungrouped_segments_array: ArrayList(UngroupedSegment),
    compressed_values: *ArrayList(u8),
) !void {
    var previous_timestamp: usize = 0;

    // Iterate over all ungrouped segments.
    for (ungrouped_segments_array.items) |segment| {
        // Append the slope a_i.
        try shared_functions.appendValue(f64, segment.slope, compressed_values);

        // Append the intercept b_i.
        try shared_functions.appendValue(f64, segment.intercept, compressed_values);

        // Append the timestamp t_i (delta encoded for consistency with other phases).
        try shared_functions.appendValue(usize, segment.timestamp - previous_timestamp, compressed_values);
        previous_timestamp = segment.timestamp;
    }
}

/// Quantizes the given `value` by the specified `error_bound`. This process ensures that
/// the quantized value remains within the error bound of the original value. If the
/// `error_bound` is equal to zero, the value is directly returned.
fn quantizeFloor(value: f64, error_bound: f32) f64 {
    if (error_bound != 0) {
        return @floor(value / error_bound) * error_bound;
    }
    return value;
}

/// Quantizes the given `value` by the specified `error_bound`. This process ensures that
/// the quantized value remains within the error bound of the original value. If the
/// `error_bound` is equal to zero, the value is directly returned.
fn quantizeCeil(value: f64, error_bound: f32) f64 {
    if (error_bound != 0) {
        return @ceil(value / error_bound) * error_bound;
    }
    return value;
}

test "mix-piece can compress and decompress bounded values with positive error bound" {
    const allocator = testing.allocator;
    const data_distributions = &[_]tester.DataDistribution{
        .LinearFunctions,
        .BoundedRandomValues,
        .SinusoidalFunction,
    };
    // This function evaluates Mix-Piece using all data distribution stored in
    // `data_distribution` with a positive error bound ranging from [1e-4, 1)*range
    // of the generated uncompressed time series.
    try tester.testErrorBoundedCompressionMethod(
        allocator,
        Method.MixPiece,
        data_distributions,
    );
}

test "mix-piece can compress, decompress and merge many segments with non-zero error bound" {
    const allocator = testing.allocator;

    const error_bound = tester.generateBoundedRandomValue(f32, 0.5, 3, undefined);

    var uncompressed_values = ArrayList(f64).init(allocator);
    defer uncompressed_values.deinit();

    for (0..20) |_| {
        // Generate floating points numbers between 0 and 10. This will generate many merged
        // segments when applying Mix-Piece.
        try tester.generateBoundedRandomValues(&uncompressed_values, 0, 10, undefined);
    }

    try tester.testCompressAndDecompress(
        allocator,
        uncompressed_values.items,
        Method.MixPiece,
        error_bound,
        shared_functions.isWithinErrorBound,
    );
}

test "mix-piece cannot compress NaN values" {
    const allocator = testing.allocator;

    const uncompressed_values = &[4]f64{ 19.0, 48.0, math.nan(f64), 3.0 };

    var compressed_values = ArrayList(u8).init(allocator);
    defer compressed_values.deinit();

    const method_configuration =
        \\ {"abs_error_bound": 0.1}
    ;

    compress(
        allocator,
        uncompressed_values,
        &compressed_values,
        method_configuration,
    ) catch |err| {
        try testing.expectEqual(Error.UnsupportedInput, err);
        return;
    };

    try testing.expectFmt(
        "",
        "The Mix-Piece algorithm cannot compress NaN values",
        .{},
    );
}

test "mix-piece cannot compress inf values" {
    const allocator = testing.allocator;

    const uncompressed_values = &[4]f64{ 19.0, 48.0, math.inf(f64), 3.0 };

    var compressed_values = ArrayList(u8).init(allocator);
    defer compressed_values.deinit();

    const method_configuration =
        \\ {"abs_error_bound": 0.1}
    ;

    compress(
        allocator,
        uncompressed_values,
        &compressed_values,
        method_configuration,
    ) catch |err| {
        try testing.expectEqual(Error.UnsupportedInput, err);
        return;
    };

    try testing.expectFmt(
        "",
        "The Mix-Piece algorithm cannot compress inf values",
        .{},
    );
}

test "mix-piece cannot compress f64 with reduced precision" {
    const allocator = testing.allocator;

    const uncompressed_values = &[4]f64{ 19.0, 48.0, 1e17, 3.0 };

    var compressed_values = ArrayList(u8).init(allocator);
    defer compressed_values.deinit();

    const method_configuration =
        \\ {"abs_error_bound": 0.1}
    ;

    compress(
        allocator,
        uncompressed_values,
        &compressed_values,
        method_configuration,
    ) catch |err| {
        try testing.expectEqual(Error.UnsupportedInput, err);
        return;
    };

    try testing.expectFmt(
        "",
        "The Mix-Piece algorithm cannot compress reduced precision floating point values",
        .{},
    );
}

test "mix-piece handles time series with trend" {
    // Additional Mix-Piece specific tests to cover its unique features.
    const error_bound = tester.generateBoundedRandomValue(
        f32,
        0.01,
        1,
        undefined,
    );
    const allocator = testing.allocator;
    var uncompressed_values = ArrayList(f64).init(allocator);
    defer uncompressed_values.deinit();

    // Generate data with a clear upward trend.
    for (0..100) |i| {
        const trend_value = @as(f64, @floatFromInt(i)) * 0.1;
        const noise = (tester.generateBoundedRandomValue(
            f64,
            0,
            1,
            undefined,
        ) - 0.5) * 0.2;

        try uncompressed_values.append(trend_value + noise);
    }

    try tester.testCompressAndDecompress(
        allocator,
        uncompressed_values.items,
        Method.MixPiece,
        error_bound,
        shared_functions.isWithinErrorBound,
    );
}

test "mix-piece handles cross-intercept grouping" {
    const error_bound = 0.01;
    const allocator = testing.allocator;
    var uncompressed_values = ArrayList(f64).init(allocator);
    defer uncompressed_values.deinit();

    // Create segments with similar slopes but different intercepts.
    // This should trigger Mix-Piece's cross-intercept grouping.

    // First segment: y = 2x + 1
    for (0..10) |i| {
        try uncompressed_values.append(2.0 * @as(f64, @floatFromInt(i)) + 1.0);
    }

    // Second segment: y = 2x + 5 (same slope, different intercept)
    for (10..20) |i| {
        try uncompressed_values.append(2.0 * @as(f64, @floatFromInt(i)) + 5.0);
    }

    // Third segment: y = 2.1x + 10 (similar slope, different intercept)
    for (20..30) |i| {
        try uncompressed_values.append(2.1 * @as(f64, @floatFromInt(i)) + 10.0);
    }

    try tester.testCompressAndDecompress(
        allocator,
        uncompressed_values.items,
        Method.MixPiece,
        error_bound,
        shared_functions.isWithinErrorBound,
    );
}

test "mix-piece handles single point segments" {
    const error_bound = tester.generateBoundedRandomValue(
        f32,
        0.01,
        1,
        undefined,
    );
    const allocator = testing.allocator;

    var uncompressed_values = ArrayList(f64).init(allocator);
    defer uncompressed_values.deinit();

    // Generate highly variable data that will create many single-point segments.
    for (0..50) |_| {
        try uncompressed_values.append(tester.generateBoundedRandomValue(
            f32,
            0,
            1,
            undefined,
        ));
    }
    try tester.testCompressAndDecompress(
        allocator,
        uncompressed_values.items,
        Method.MixPiece,
        error_bound,
        shared_functions.isWithinErrorBound,
    );
}

test "mix-piece floor vs ceil quantization selection" {
    const error_bound = 0.01;
    const allocator = testing.allocator;
    var uncompressed_values = ArrayList(f64).init(allocator);
    defer uncompressed_values.deinit();

    // Create data to test floor vs. ceil quantization decisions.
    // Start with values that are close to quantization boundaries.
    try uncompressed_values.append(1.49);
    try uncompressed_values.append(1.51);
    try uncompressed_values.append(1.99);
    try uncompressed_values.append(2.01);
    try uncompressed_values.append(2.49);
    try uncompressed_values.append(2.51);

    try tester.testCompressAndDecompress(
        allocator,
        uncompressed_values.items,
        Method.MixPiece,
        error_bound,
        shared_functions.isWithinErrorBound,
    );
}

test "check mixpiece configuration parsing" {
    // Tests the configuration parsing and functionality of the `compress` function.
    // The test verifies that the provided configuration is correctly interpreted and
    // that the `configuration.AbsoluteErrorBound` is expected in the function.
    const allocator = testing.allocator;

    const uncompressed_values = &[4]f64{ 19.0, 48.0, 28.0, 3.0 };

    var compressed_values = ArrayList(u8).init(allocator);
    defer compressed_values.deinit();

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
}
