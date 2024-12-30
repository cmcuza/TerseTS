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

//! Implementation of the Min-Merge algorithm from the paper
//! "Buragohain, Chiranjeeb, Nisheeth Shrivastava, and Subhash Suri.
//! Space Efficient Streaming Algorithms for the Maximum Error Histogram.
//! IEEE ICDE 2006.
//! https://doi.org/10.1109/ICDE.2007.368961.
//! Min-Merge find a Piecewise Constant and Linear Histogram compressed representation of the
//! time series. Thus, the abbreviation PWCH and PWLH is used for the methods as in the paper:
//! https://doi.org/10.1109/TKDE.2012.237.

const std = @import("std");
const mem = std.mem;
const math = std.math;
const testing = std.testing;
const time = std.time;
const ArrayList = std.ArrayList;

const expectEqual = testing.expectEqual;

const HashedPriorityQueue = @import(
    "../utilities/hashed_priority_queue.zig",
).HashedPriorityQueue;

const ConvexHull = @import(
    "../utilities/convex_hull.zig",
).ConvexHull;

const shared = @import("../utilities/shared_structs.zig");

const LinearFunction = shared.LinearFunction;
const Segment = shared.Segment;

const tersets = @import("../tersets.zig");
const Error = tersets.Error;

const tester = @import("../tester.zig");

// Enum to determine the type of approximation of the buckets in the histogram.
const Approximation = enum(i8) { constant, linear };

/// Compress `uncompressed_values` with the maximum number of buckets defined by the `error_bound`
/// using "Piecewice Constant Histogram" compression method and write the result to
/// `compressed_values`. If an error occurs it is returned.
pub fn compressPWCH(
    uncompressed_values: []const f64,
    compressed_values: *ArrayList(u8),
    allocator: mem.Allocator,
    error_bound: f32,
) Error!void {
    if (error_bound <= 1.0) {
        return Error.UnsupportedErrorBound;
    }

    // The original implementation requires the maximum number of buckets which can be represented
    // by a usize instead of `error_bound: f32`. Changing this requires modifications in
    // `src/tersets.zig` and `src/capi.zig` files.
    // TODO: Find the right way of passing the maximum number of buckets.
    const max_buckets: usize = @as(usize, @intFromFloat(@floor(error_bound)));
    var histogram = try Histogram.init(allocator, max_buckets, .constant);
    defer histogram.deinit();

    for (uncompressed_values, 0..) |elem, index| {
        try histogram.insert(index, elem);
    }

    for (0..histogram.len()) |index| {
        var bucket: Bucket = histogram.at(index);
        try appendValueAndIndexToArrayList(
            bucket.computeConstantApproximation(),
            bucket.end + 1,
            compressed_values,
        );
    }
}

/// Compress `uncompressed_values` with the maximum number of buckets defined by the `error_bound`
/// using "Piecewice Linear Histogram" compression method and write the result to
/// `compressed_values`. If an error occurs it is returned.
pub fn compressPWLH(
    uncompressed_values: []const f64,
    compressed_values: *ArrayList(u8),
    allocator: mem.Allocator,
    error_bound: f32,
) Error!void {
    if (error_bound <= 1.0) {
        return Error.UnsupportedErrorBound;
    }

    // The original implementation requires the maximum number of buckets which can be represented
    // by a usize instead of `error_bound: f32`. Changing this requires modifications in
    // `src/tersets.zig` and `src/capi.zig` files.
    // TODO: Find the right way of passing the maximum number of buckets.
    const max_buckets: usize = @as(usize, @intFromFloat(@floor(error_bound)));
    var histogram = try Histogram.init(allocator, max_buckets, .linear);
    defer histogram.deinit();

    for (uncompressed_values, 0..) |elem, index| {
        try histogram.insert(index, elem);
    }

    for (0..histogram.len()) |index| {
        var bucket: Bucket = histogram.at(index);
        const linear_approximation = try bucket.computeLinearApproximation();

        if (bucket.end - bucket.begin > 1) {
            const begin_value: f64 = @floatCast(linear_approximation.slope *
                @as(f64, @floatFromInt(bucket.begin)) +
                linear_approximation.intercept);
            try appendValue(f64, begin_value, compressed_values);

            const end_value: f64 = @floatCast(linear_approximation.slope *
                @as(f64, @floatFromInt(bucket.end)) +
                linear_approximation.intercept);

            try appendValue(f64, end_value + 1, compressed_values);
        } else {
            try appendValue(f64, uncompressed_values[bucket.begin], compressed_values);
            try appendValue(f64, uncompressed_values[bucket.end], compressed_values);
        }

        try appendValue(usize, bucket.end + 1, compressed_values);
    }
}

/// Decompress `compressed_values` produced by "Piecewise Constant Histogram" and write the result
/// to `decompressed_values`. If an error occurs it is returned.
pub fn decompressPWCH(
    compressed_values: []const u8,
    decompressed_values: *ArrayList(f64),
) Error!void {
    // The compressed representation is pairs containing a 64-bit float value
    // and 64-bit integer end index.
    if (compressed_values.len % 16 != 0) return Error.IncorrectInput;

    const compressed_values_and_index = mem.bytesAsSlice(f64, compressed_values);

    var compressed_index: usize = 0;
    var uncompressed_index: usize = 0;
    while (compressed_index < compressed_values_and_index.len) : (compressed_index += 2) {
        const value = compressed_values_and_index[compressed_index];
        const index: usize = @bitCast(compressed_values_and_index[compressed_index + 1]);
        for (uncompressed_index..index) |_| {
            try decompressed_values.append(value);
        }
        uncompressed_index = index;
    }
}

/// Decompress `compressed_values` produced by "PWLH" and write the result to `decompressed_values`.
/// If an error occurs it is returned.
pub fn decompressPWLH(
    compressed_values: []const u8,
    decompressed_values: *ArrayList(f64),
) Error!void {
    // The compressed representation is composed of three values: (start_value, end_value, end_time)
    // all of type 64-bit float.
    if (compressed_values.len % 24 != 0) return Error.IncorrectInput;

    const compressed_lines_and_index = mem.bytesAsSlice(f64, compressed_values);

    var linear_approximation: LinearFunction = .{ .slope = undefined, .intercept = undefined };

    var first_timestamp: usize = 0;
    var index: usize = 0;
    while (index < compressed_lines_and_index.len) : (index += 3) {
        const current_segment: Segment = .{
            .start_point = .{ .time = first_timestamp, .value = compressed_lines_and_index[index] },
            .end_point = .{
                .time = @as(usize, @bitCast(compressed_lines_and_index[index + 2])) - 1,
                .value = compressed_lines_and_index[index + 1],
            },
        };

        if (current_segment.start_point.time < current_segment.end_point.time) {
            if (current_segment.end_point.time != current_segment.start_point.time) {
                const duration: f80 = @floatFromInt(current_segment.end_point.time -
                    current_segment.start_point.time);
                linear_approximation.slope = (current_segment.end_point.value -
                    current_segment.start_point.value) / duration;
                linear_approximation.intercept = current_segment.start_point.value - linear_approximation.slope *
                    @as(f64, @floatFromInt(current_segment.start_point.time));
            } else {
                linear_approximation.slope = 0.0;
                linear_approximation.intercept = current_segment.start_point.value;
            }
            try decompressed_values.append(current_segment.start_point.value);
            var current_timestamp: usize = current_segment.start_point.time + 1;
            while (current_timestamp < current_segment.end_point.time) : (current_timestamp += 1) {
                const y: f64 = @floatCast(linear_approximation.slope *
                    @as(f64, @floatFromInt(current_timestamp)) +
                    linear_approximation.intercept);
                try decompressed_values.append(y);
            }
            try decompressed_values.append(current_segment.end_point.value);
            first_timestamp = current_timestamp + 1;
        } else {
            try decompressed_values.append(current_segment.start_point.value);
            first_timestamp += 1;
        }
    }
}

/// `Bucket` stores information about a range of consecutives values in the time series. The
/// structure stores the indices `begin` and `end` indexing where the bucket starts and ends. It stores the
/// minimum and maximum values (`min_val` and `max_val`) in the bucket. Additionally, it stores
/// the `convex_hull` of type `ConvexHull` that represents the elements in the bucket. The
/// `convex_hull` is utilized when a linear approximation of the data points in the bucket is required. The
/// structure contains the function `computeConstantApproximation` which computes and returns the
/// constant approximation that minimizes the $L_\inf$ error associated to the bucket. Likewise,
/// it contains the function `computeLinearApproximation` which computes and returns the linear
/// approximation that minimizes the $L_\inf$ error associated to the bucket.
const Bucket = struct {
    // Begining of the bucket.
    begin: usize,
    // Ending of the bucket.
    end: usize,
    // Min value of the bucket.
    min_val: f64,
    // Max value of the bucket.
    max_val: f64,
    // Convex Hull of the elements in the bucket.
    convex_hull: ConvexHull,

    /// Initialize the bucket with the given indices and min/max values.
    pub fn init(begin: usize, end: usize, min_val: f64, max_val: f64, allocator: mem.Allocator) !Bucket {
        return Bucket{
            .begin = begin,
            .end = end,
            .min_val = min_val,
            .max_val = max_val,
            .convex_hull = try ConvexHull.init(allocator),
        };
    }

    /// Deinitialize the bucket.
    pub fn deinit(self: *Bucket) void {
        self.convex_hull.deinit();
    }

    /// Returns the absolute error of the values in the bucket.
    pub fn computeConstantApproximation(self: *Bucket) f64 {
        return (self.max_val - self.min_val) / 2.0;
    }

    pub fn computeLinearApproximation(self: *Bucket) Error!LinearFunction {
        return try self.convex_hull.computeMABRLinearFunction();
    }
};

/// Structure for holding merge information. This represents the error that results from merging
/// two adjacent buckets.
const MergeError = struct {
    // Index of the first bucket in the pair to be merged.
    index: usize,
    // Error resulting from merging this bucket with the next one.
    merge_error: f64,
};

/// `HashMergeErrorContext` provides context for hashing and comparing `MergeError` items for use
/// in `HashMap`. It defines how `MergeError` are hashed and compared for equality.
const HashMergeErrorContext = struct {
    /// Hashes the `index: usize` by bitcasting it to `u64`.
    pub fn hash(_: HashMergeErrorContext, merge_error: MergeError) u64 {
        return @as(u64, @intCast(merge_error.index));
    }
    /// Compares two `index` for equality.
    pub fn eql(_: HashMergeErrorContext, merge_error_one: MergeError, merge_error_two: MergeError) bool {
        return merge_error_one.index == merge_error_two.index;
    }
};

/// Comparison function for the `HashedPriorityQueue`. It compares merge errors, and also considers
///  bucket indices for equality.
fn compareMergeError(_: void, error_1: MergeError, error_2: MergeError) math.Order {
    if (error_1.index == error_2.index)
        return math.Order.eq;
    return math.order(error_1.merge_error, error_2.merge_error);
}

/// A Histogram structure containing an `ArrayList` of `buckets` and a `HashedPriorityQueue`
/// `merge_queue` that keeps track of the minimum merge error merging two adjacent buckets.
/// This structure contains the Min-Merge algorithm from https://doi.org/10.1109/ICDE.2007.368961
/// which maintain the histogram using only `max_buckets` buckets.
const Histogram = struct {
    const Self = @This();

    // Enum to determine the approximation type.
    approximation: Approximation,
    // Memory allocator for the convex hull in the buckets.
    allocator: mem.Allocator,
    // Target number of buckets.
    max_buckets: usize,
    // List of current buckets.
    buckets: ArrayList(Bucket),
    // Priority queue of merge errors.
    merge_queue: HashedPriorityQueue(
        MergeError,
        void,
        compareMergeError,
        HashMergeErrorContext,
    ),

    /// Initialize the histogram with a given allocator and maximum number of buckets. This
    /// parameter can be thought of as fixing a minimum compression ratio that users wants to achieve.
    /// The `minMerge` function will then find the optimal histogram under this constraint.
    /// For example, if `max_buckets=|N|/2`, then the compression ratio will be at least 2x.
    pub fn init(allocator: mem.Allocator, max_buckets: usize, approximation: Approximation) !Histogram {
        return Histogram{
            .approximation = approximation,
            .allocator = allocator,
            .max_buckets = max_buckets,
            .buckets = ArrayList(Bucket).init(allocator),
            .merge_queue = try HashedPriorityQueue(
                MergeError,
                void,
                compareMergeError,
                HashMergeErrorContext,
            ).init(allocator, {}),
        };
    }

    /// Deinitialize the histogram.
    pub fn deinit(self: *Self) void {
        // Deinitialize all buckets. Necessary to deinitialize the convex hulls within.
        for (self.buckets.items) |*bucket| {
            bucket.deinit();
        }
        self.buckets.deinit();
        self.merge_queue.deinit();
    }

    /// Insert a new value into the histogram.
    pub fn insert(self: *Self, index: usize, value: f64) !void {
        // Create a new bucket for the incoming value with start and end at 'index'.
        var bucket: Bucket = try Bucket.init(index, index, value, value, self.allocator);

        try bucket.convex_hull.add(.{ .time = index, .value = value });

        try self.buckets.append(bucket);

        if (self.buckets.items.len > 1) {
            const bucket_last_index: usize = self.buckets.items.len - 1;

            const merge_error: f64 = switch (self.approximation) {
                .constant => try self.calculateConstantApproximationMergeError(bucket_last_index - 1),
                .linear => try self.calculateLinearApproximationMergeError(bucket_last_index - 1),
            };
            try self.merge_queue.add(MergeError{ .index = bucket_last_index - 1, .merge_error = merge_error });
        }
        // If the number of buckets exceeds `max_buckets`, merge the least increasing pair.
        if (self.buckets.items.len > self.max_buckets) {
            try self.minMerge();
        }
    }

    /// Returns the bucket at `index`.
    pub fn at(self: *Self, index: usize) Bucket {
        return self.buckets.items[index];
    }

    /// Returns the len of the histogram.
    pub fn len(self: *Self) usize {
        return self.buckets.items.len;
    }

    /// Calculate the merge error for merging the bucket at `index` with the next bucket when
    /// a `constant` approximation is required. Returns `Error.ItemNotFound` if `index+1` does
    /// not exists.
    fn calculateConstantApproximationMergeError(self: *Self, index: usize) Error!f64 {
        if (index + 1 >= self.buckets.items.len) {
            return Error.ItemNotFound;
        }

        const bucket_one = self.buckets.items[index];
        const bucket_two = self.buckets.items[index + 1];
        const merged_min = @min(bucket_one.min_val, bucket_two.min_val);
        const merged_max = @max(bucket_one.max_val, bucket_two.max_val);
        return (merged_max - merged_min) / 2.0;
    }

    /// Calculate the merge error for merging the bucket at `index` with the next bucket when
    /// a `linear` approximation is required. Returns `Error.ItemNotFound` if `index+1` does
    /// not exists.
    fn calculateLinearApproximationMergeError(self: *Self, index: usize) Error!f64 {
        if (index + 1 >= self.buckets.items.len) {
            return Error.ItemNotFound;
        }

        var convex_hull_one = self.buckets.items[index].convex_hull;
        var convex_hull_two = self.buckets.items[index + 1].convex_hull;

        var convex_hull_merged: ConvexHull = try ConvexHull.init(self.allocator);
        defer convex_hull_merged.deinit();

        try convex_hull_one.merge(&convex_hull_two, &convex_hull_merged);

        const linear_approximation = try convex_hull_merged.computeMABRLinearFunction();
        const max_error = convex_hull_merged.computeMaxError(linear_approximation);

        return max_error;
    }

    /// Perform the minimum merge by finding the pair with the smallest merge error when a `constant`
    /// approximations is required. This function is only called when the number of buckets exceeds the
    /// `max_buckets`, which is always higher than 1. Thus, this function is only called with 2 or more
    /// elements in the `buckets` list.
    fn minMerge(self: *Self) !void {
        // Pop the smallest merge error (the least costly merge).
        const min_merge_error: MergeError = try self.merge_queue.remove();

        // Merge the buckets at min_merge_error.index and min_merge_error.index + 1.
        const index = min_merge_error.index;
        var bucket_one = &self.buckets.items[index];
        var bucket_two = &self.buckets.items[index + 1];

        // Merge the two buckets, updating their range and min/max values.
        const new_begin = bucket_one.begin;
        const new_end = bucket_two.end;
        const new_min = @min(bucket_one.min_val, bucket_two.min_val);
        const new_max = @max(bucket_one.max_val, bucket_two.max_val);

        // Replace the first bucket with the merged result and remove the second.
        bucket_one.begin = new_begin;
        bucket_one.end = new_end;
        bucket_one.min_val = new_min;
        bucket_one.max_val = new_max;

        if (self.approximation == .linear) {
            // Merge the convex hulls.
            try bucket_one.convex_hull.merge(&bucket_two.convex_hull, null);
        }

        var bucket = self.buckets.orderedRemove(index + 1); // Remove the merged bucket.
        bucket.deinit();

        if (index < self.buckets.items.len - 1) {
            const new_merge_error = switch (self.approximation) {
                .constant => try self.calculateConstantApproximationMergeError(index),
                .linear => try self.calculateLinearApproximationMergeError(index),
            };

            try self.merge_queue.add(MergeError{ .index = index, .merge_error = new_merge_error });

            if (index > 0) {
                // Update error for the previous bucket. Since the function is called at `index-1`,
                // the function will not return `Error.ItemNotFound` when accessing `index+1`.
                try self.merge(index - 1);
            }

            try self.updateAllIndex(index + 1);
        }
    }

    /// Update the merge error for the bucket at 'index'.
    /// Return `Error.ItemNotFound` if `index+1` does not exists.
    fn merge(self: *Self, index: usize) !void {
        const merge_error = switch (self.approximation) {
            .constant => try self.calculateConstantApproximationMergeError(index),
            .linear => try self.calculateLinearApproximationMergeError(index),
        };

        const new_merge_error = MergeError{ .index = index, .merge_error = merge_error };

        // Placeholder for the old merge, only the index is relevant.
        const old_merge_item: MergeError = MergeError{ .index = index, .merge_error = -1.0 };

        // Update the priority queue with the new merge error using the 'update' method.
        try self.merge_queue.update(old_merge_item, new_merge_error);
    }

    /// Update the indices of all merge error from `index` until `buckets.items.len`.
    fn updateAllIndex(self: *Self, index: usize) !void {
        for (index..self.buckets.items.len) |i| {
            const merge_error_index: usize = try self.merge_queue.getIndex(
                .{ .index = i, .merge_error = 0 },
            );
            const old_merge_error: MergeError = try self.merge_queue.get(merge_error_index);
            var new_merge_error: MergeError = .{
                .index = old_merge_error.index - 1,
                .merge_error = old_merge_error.merge_error,
            };

            if (i == index) {
                // To effectively remove an index from the queue without physically deleting it,
                // we perform a logical removal. This is done by setting its `index` to an
                // unreachable value (`max_buckets + 10`), ensuring it falls outside the normal range.
                // Additionally, we set `merge_error` to a very high value (1e16).
                // This combination guarantees that during the `update`, this entry will
                // be pushed to the end of the list, simulating its removal.
                new_merge_error.index = self.max_buckets + 10;
                new_merge_error.merge_error = 1e16;
            }
            try self.merge_queue.update(old_merge_error, new_merge_error);
        }
    }
};

/// Append `compressed_value` and `index` to `compressed_values`.
fn appendValueAndIndexToArrayList(
    compressed_value: f64,
    index: usize,
    compressed_values: *ArrayList(u8),
) !void {
    const value: f64 = compressed_value;
    const valueAsBytes: [8]u8 = @bitCast(value);
    try compressed_values.appendSlice(valueAsBytes[0..]);
    const indexAsBytes: [8]u8 = @bitCast(index); // No -1 due to 0 indexing.
    try compressed_values.appendSlice(indexAsBytes[0..]);
}

/// Append `value` of `type` determined at compile time to `compressed_values`.
fn appendValue(comptime T: type, value: T, compressed_values: *std.ArrayList(u8)) !void {
    // Compile-time type check.
    switch (@TypeOf(value)) {
        f64, usize => {
            const value_as_bytes: [8]u8 = @bitCast(value);
            try compressed_values.appendSlice(value_as_bytes[0..]);
        },
        else => @compileError("Unsupported type for append value function"),
    }
}

test "Hash PriorityQueue with hash_context for MergeError" {
    const allocator = testing.allocator;

    var pq = try HashedPriorityQueue(
        MergeError,
        void,
        compareMergeError,
        HashMergeErrorContext,
    ).init(allocator, {});
    defer pq.deinit();

    const error1 = MergeError{ .index = 1, .merge_error = 5.0 };
    const error2 = MergeError{ .index = 2, .merge_error = 3.0 };
    const error3 = MergeError{ .index = 3, .merge_error = 8.0 };

    try pq.add(error1);
    try pq.add(error2);
    try pq.add(error3);

    const top_value = try pq.remove();
    try std.testing.expect(top_value.index == 2);

    const next_top = try pq.remove();
    try std.testing.expect(next_top.index == 1);

    const final_top = try pq.remove();
    try std.testing.expect(final_top.index == 3);
}

test "Histogram insert, and merge test number buckets in PWCH" {
    // Initialize a random number generator.
    const seed: u64 = @bitCast(time.milliTimestamp());
    var prng = std.rand.DefaultPrng.init(seed);
    const random = prng.random();

    const allocator = testing.allocator;
    const max_buckets: usize = 100;
    var histogram = try Histogram.init(allocator, max_buckets, .constant);
    defer histogram.deinit();

    // Insert 1000 random numbers into the histogram.
    for (0..1000) |i| {
        const rand_number = tester.generateBoundedRandomValue(0, 1000, random);
        try histogram.insert(i, rand_number);
    }
    try expectEqual(max_buckets, histogram.buckets.items.len);
}

test "Simple fixed values test of PWCH" {
    const allocator = testing.allocator;

    // Input data points.
    const data_points = [_]f64{
        // Cluster 1.
        0.9,  1.1,  0.7, 1.0, 0.8,
        // Cluster 2.
        4.8,  5.2,  4.6, 5.0, 4.7,
        // Cluster 3.
        9.8,  10.2, 9.9, 9.7, 10.0,
        10.1,
    };

    // Initialize the histogram with a maximum of 3 buckets.
    var histogram = try Histogram.init(allocator, 3, .constant);
    defer histogram.deinit();

    // Insert data points into the histogram.
    for (data_points, 0..) |value, i| {
        try histogram.insert(i, value);
    }
    try expectEqual(@as(usize, @intCast(3)), histogram.buckets.items.len);

    var convex_hull: ConvexHull = try ConvexHull.init(allocator);
    defer convex_hull.deinit();

    const expected_histogram = [_]Bucket{
        Bucket{ .begin = 0, .end = 4, .min_val = 0.7, .max_val = 1.1, .convex_hull = convex_hull },
        Bucket{ .begin = 5, .end = 9, .min_val = 4.6, .max_val = 5.2, .convex_hull = convex_hull },
        Bucket{ .begin = 10, .end = 15, .min_val = 9.7, .max_val = 10.2, .convex_hull = convex_hull },
    };

    for (histogram.buckets.items, 0..) |bucket, i| {
        try expectEqual(expected_histogram[i].begin, bucket.begin);
        try expectEqual(expected_histogram[i].end, bucket.end);
        try expectEqual(expected_histogram[i].min_val, bucket.min_val);
        try expectEqual(expected_histogram[i].max_val, bucket.max_val);
    }
}

test "Fixed cluster number with random values for PWCH" {
    // Initialize a random number generator.
    const seed: u64 = @bitCast(time.milliTimestamp());
    var prng = std.rand.DefaultPrng.init(seed);
    const random = prng.random();

    const allocator = std.testing.allocator;

    const cluster_ranges = [_]struct {
        min: f64,
        max: f64,
        count: usize,
    }{
        // Cluster 1.
        .{ .min = 0.5, .max = 1.5, .count = 5 },
        // Cluster 2.
        .{ .min = 4.5, .max = 5.5, .count = 5 },
        // Cluster 3.
        .{ .min = 9.5, .max = 10.5, .count = 6 },
        // Cluster 4.
        .{ .min = 1.5, .max = 2.5, .count = 6 },
        // Cluster 5.
        .{ .min = 11.5, .max = 12.5, .count = 3 },
    };

    // Collect all data points.
    var data_points = std.ArrayList(f64).init(allocator);
    defer data_points.deinit();

    // Generate random values for each cluster.
    for (cluster_ranges) |cluster| {
        for (cluster.count) |_| {
            const value = tester.generateBoundedRandomValue(cluster.min, cluster.max, random);
            try data_points.append(value);
        }
    }

    // Initialize the histogram with a maximum of 3 buckets.
    var histogram = try Histogram.init(allocator, cluster_ranges.len, .constant);
    defer histogram.deinit();

    // Insert data points into the histogram.
    for (data_points.items, 0..) |value, i| {
        try histogram.insert(i, value);
    }

    // Build the expected histogram by iterating over cluster_ranges.
    var expected_histogram = std.ArrayList(Bucket).init(allocator);
    defer expected_histogram.deinit();

    var convex_hull: ConvexHull = try ConvexHull.init(allocator);
    defer convex_hull.deinit();

    var current_begin: usize = 0;
    for (cluster_ranges) |cluster| {
        const bucket = Bucket{
            .begin = current_begin,
            .end = current_begin + cluster.count - 1,
            .min_val = cluster.min,
            .max_val = cluster.max,
            .convex_hull = convex_hull,
        };
        try expected_histogram.append(bucket);
        current_begin += cluster.count;
    }

    // Verify that the output histogram matches the expected histogram.
    for (histogram.buckets.items, 0..) |bucket, i| {
        const expected_bucket = expected_histogram.items[i];

        try expectEqual(expected_bucket.begin, bucket.begin);
        try expectEqual(expected_bucket.end, bucket.end);
    }
}

test "Random clusters, elements per cluster and values for PWCH" {
    // Initialize a random number generator.
    var prng = std.rand.DefaultPrng.init(12345);
    const random = prng.random();

    const allocator = std.testing.allocator;
    const num_cluster: usize = prng.random().uintLessThan(usize, 100) + 10;

    var cluster_ranges = std.ArrayList(struct {
        min: f64,
        max: f64,
        count: usize,
    }).init(allocator);
    defer cluster_ranges.deinit();

    const min_value: f64 = -1e6;
    const cluster_width: f64 = tester.generateBoundedRandomValue(100, 1000, random);
    // Generate min gap.
    const gap: f64 = cluster_width + tester.generateBoundedRandomValue(100, 1000, random) + 100;
    const max_counts_per_cluster: usize = 100;

    var current_min_value = min_value;
    for (0..num_cluster) |i| {
        const momentum: f64 = if (i < num_cluster / 2) 1.0 else -1.0;
        const cluster_min = current_min_value + momentum * gap;
        const cluster_max = cluster_min + cluster_width;
        const count: usize = prng.random().uintLessThan(usize, max_counts_per_cluster) + 10;
        // Append the cluster to cluster_ranges.
        try cluster_ranges.append(.{
            .min = cluster_min,
            .max = cluster_max,
            .count = count,
        });
        current_min_value = cluster_min;
    }

    // Collect all data points.
    var data_points = std.ArrayList(f64).init(allocator);
    defer data_points.deinit();

    // Generate random values for each cluster.
    for (cluster_ranges.items) |cluster| {
        for (cluster.count) |_| {
            const value = tester.generateBoundedRandomValue(cluster.min, cluster.max, random);
            try data_points.append(value);
        }
    }

    // Initialize the histogram with a maximum of 'num_cluster' buckets.
    var histogram = try Histogram.init(allocator, num_cluster, .constant);
    defer histogram.deinit();

    // Insert data points into the histogram.
    for (data_points.items, 0..) |value, i| {
        try histogram.insert(i, value);
    }

    // Build the expected histogram by iterating over cluster_ranges.
    var expected_histogram = std.ArrayList(Bucket).init(allocator);
    defer expected_histogram.deinit();

    var convex_hull: ConvexHull = try ConvexHull.init(allocator);
    defer convex_hull.deinit();

    var current_begin: usize = 0;
    for (cluster_ranges.items) |cluster| {
        const bucket = Bucket{
            .begin = current_begin,
            .end = current_begin + cluster.count - 1,
            .min_val = cluster.min,
            .max_val = cluster.max,
            .convex_hull = convex_hull,
        };
        try expected_histogram.append(bucket);
        current_begin += cluster.count;
    }

    // Verify that the output histogram matches the expected histogram.
    for (histogram.buckets.items, 0..) |bucket, i| {
        const expected_bucket = expected_histogram.items[i];

        try expectEqual(expected_bucket.begin, bucket.begin);
        try expectEqual(expected_bucket.end, bucket.end);
    }
}

test "PWCH can compress and decompress with bounded values and expected maximum error" {
    var prng = std.rand.DefaultPrng.init(0);
    const random = prng.random();
    const error_bound: f32 = 10;

    const allocator = testing.allocator;
    var uncompressed_values = ArrayList(f64).init(allocator);
    defer uncompressed_values.deinit();
    try tester.generateBoundedRandomValues(&uncompressed_values, 0.0, error_bound, random);

    // This test internally calls the functions `compress()` and `decompress()`. However, the line
    // `testing.expect(withinErrorBound(uncompressed_values,decompressed_values.items,error_bound))`
    // inside the function, is nonsensical in the context of PWCH. This is because the `error_bound`
    // represents the number of bins in the histogram and not the maximum decompression error.
    // To solve this problem, we need to create another type of configuration for compression
    // algorithms like PWCH, which do not base their compression on a maximum decompression error
    // but a minimum (or maximum) compression ratio.
    // Nevertheless, since all `uncompressed_values` are between 0 and 10, the `error_bound=10`
    // should be fulfilled as the PWCH finds the mean value over the buckets.
    try tester.testCompressAndDecompress(
        uncompressed_values.items,
        allocator,
        tersets.Method.PiecewiseConstantHistogram,
        error_bound,
        tersets.isWithinErrorBound,
    );
}

test "Compute simple linear approximation merge error with known results" {
    const allocator = std.testing.allocator;

    // Create Convex Hulls for the buckets.
    // No need to deallocate memory because histogram will do it.
    var convex_hull_one = try ConvexHull.init(allocator);

    try convex_hull_one.add(.{ .time = 0.0, .value = 0.0 });
    try convex_hull_one.add(.{ .time = 1.0, .value = 1.0 });

    // No need to deallocate memory because histogram will do it.
    var convex_hull_two = try ConvexHull.init(allocator);

    try convex_hull_two.add(.{ .time = 2.0, .value = 2.0 });
    try convex_hull_two.add(.{ .time = 3.0, .value = 3.0 });

    // Insert into buckets.
    var histogram = try Histogram.init(allocator, 2, .linear);
    defer histogram.deinit();

    try histogram.buckets.append(.{
        .begin = 0,
        .end = 0,
        .min_val = 0,
        .max_val = 0,
        .convex_hull = convex_hull_one,
    });
    try histogram.buckets.append(.{
        .begin = 0,
        .end = 0,
        .min_val = 0,
        .max_val = 0,
        .convex_hull = convex_hull_two,
    });

    const merge_error = try histogram.calculateLinearApproximationMergeError(0);

    try std.testing.expectApproxEqAbs(0.0, merge_error, 1e-15);
}

test "Compute divergent linear approximation merge error with known results" {
    const allocator = std.testing.allocator;

    // No need to deallocate memory because `histogram` will do it.
    var convex_hull_one = try ConvexHull.init(allocator);

    try convex_hull_one.add(.{ .time = 0.0, .value = 0.0 });
    try convex_hull_one.add(.{ .time = 1.0, .value = 1.0 });

    // No need to deallocate memory because histogram will do it.
    var convex_hull_two = try ConvexHull.init(allocator);

    try convex_hull_two.add(.{ .time = 2.0, .value = 3.0 });
    try convex_hull_two.add(.{ .time = 3.0, .value = 4.0 });

    // Step 2: Insert into buckets.
    var histogram = try Histogram.init(allocator, 2, .linear);
    defer histogram.deinit();

    try histogram.buckets.append(.{
        .begin = 0,
        .end = 0,
        .min_val = 0,
        .max_val = 0,
        .convex_hull = convex_hull_one,
    });
    try histogram.buckets.append(.{
        .begin = 0,
        .end = 0,
        .min_val = 0,
        .max_val = 0,
        .convex_hull = convex_hull_two,
    });

    const merge_error = try histogram.calculateLinearApproximationMergeError(0);

    try std.testing.expectApproxEqAbs(0.25, merge_error, 1e-10);
}

test "Compute PWLH with a simple set of values with known results" {
    const allocator = std.testing.allocator;

    var histogram = try Histogram.init(allocator, 2, .linear);
    defer histogram.deinit();

    try histogram.insert(0, 0);
    try histogram.insert(1, 1);
    try histogram.insert(2, 3);
    try histogram.insert(3, 4);

    var convex_hull_one = histogram.at(0).convex_hull;
    const linear_approximation_one = try convex_hull_one.computeMABRLinearFunction();
    var convex_hull_two = histogram.at(1).convex_hull;
    const linear_approximation_two = try convex_hull_two.computeMABRLinearFunction();

    try testing.expectEqual(linear_approximation_one.slope, 1);
    try testing.expectEqual(linear_approximation_one.intercept, 0);

    try testing.expectEqual(linear_approximation_two.slope, 1);
    try testing.expectEqual(linear_approximation_two.intercept, 1);
}

test "Insert random values in an Histogram with expected number of buckets" {
    var prng = std.rand.DefaultPrng.init(0);
    const random = prng.random();

    const allocator = testing.allocator;
    const max_buckets: usize = 100;
    var histogram = try Histogram.init(allocator, max_buckets, .linear);
    defer histogram.deinit();

    // Insert 1000 random numbers into the histogram.
    for (0..1000) |i| {
        const rand_number = tester.generateBoundedRandomValue(0, 1000, random);
        try histogram.insert(i, rand_number);
    }
    try expectEqual(max_buckets, histogram.buckets.items.len);
}

test "PWLH can compress and decompress with bounded values and expected maximum error" {
    var prng = std.rand.DefaultPrng.init(0);
    const random = prng.random();
    const error_bound: f32 = 10;

    const allocator = testing.allocator;
    var uncompressed_values = ArrayList(f64).init(allocator);
    defer uncompressed_values.deinit();
    try tester.generateBoundedRandomValues(&uncompressed_values, 0.0, error_bound / 4.0, random);

    // This test internally calls the functions `compress()` and `decompress()`. However, the line
    // `testing.expect(withinErrorBound(uncompressed_values,decompressed_values.items,error_bound))`
    // inside the function, is nonsensical in the context of PWLH. This is because the `error_bound`
    // represents the number of bins in the histogram and not the maximum decompression error.
    // To solve this problem, we need to create another type of configuration for compression
    // algorithms like PWLH, which do not base their compression on a maximum decompression error
    // but a minimum (or maximum) compression ratio.
    // Nevertheless, since all `uncompressed_values` are between 0 and error_bound/4.0, the `error_bound=10`
    // should be fulfilled as the PWLH finds the mean value over the buckets.
    try tester.testCompressAndDecompress(
        uncompressed_values.items,
        allocator,
        tersets.Method.PiecewiseLinearHistogram,
        error_bound,
        tersets.isWithinErrorBound,
    );
}
