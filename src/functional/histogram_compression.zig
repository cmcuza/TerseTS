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
//! Space efficient streaming algorithms for the maximum error histogram.
//! IEEE ICDE 2006.
//! https://doi.org/10.1109/ICDE.2007.368961.
//! Min-Merge find a Piecewise Constant Histogram compressed representation
//! of the time series. Thus, we called the method PWCH as in the benchmark paper
//! https://doi.org/10.1109/TKDE.2012.237.

const std = @import("std");
const mem = std.mem;
const math = std.math;
const testing = std.testing;
const time = std.time;
const expectEqual = testing.expectEqual;
const ArrayList = std.ArrayList;

const HashedPriorityQueue = @import(
    "../data_structures/hashed_priority_queue.zig",
).HashedPriorityQueue;
const tersets = @import("../tersets.zig");
const Error = tersets.Error;

/// `Bucket` stores the necessary information about the buckets. It stores the
/// `begin` of the bucket (index where the bucket starts in the data stream).
/// `end` of the bucket (index where the bucket ends in the data stream).
///  The minimum value `min_val` and maximum value `max_value` within the bucket.
/// The structure also contains the function `computeError` which computes and
/// returns the $L_\inf$ error associated to the bucket.
const Bucket = struct {
    begin: usize,
    end: usize,
    min_val: f64,
    max_val: f64,

    /// Initialize the bucket with the given indices and min/max values.
    pub fn init(begin: usize, end: usize, min_val: f64, max_val: f64) !Bucket {
        return Bucket{ .begin = begin, .end = end, .min_val = min_val, .max_val = max_val };
    }

    /// Calculate the error for the bucket. Error is half the difference between max and min values.
    pub fn computeError(self: Bucket) f64 {
        return (self.max_val - self.min_val) / 2.0;
    }
};

/// Structure for holding merge information. This represents the error that results from merging
/// two adjacent buckets.
const MergeError = struct {
    index: usize, // Index of the first bucket in the pair to be merged.
    merge_error: f64, // Error resulting from merging this bucket with the next one.
};

/// `HashMergeErrorContext` provides context for hashing and comparing `merge_error` items for use
/// in `HashMap`. It defines how the `merge_error` are hashed and compared for equality.
const HashMergeErrorContext = struct {
    /// Hashes the `index:usize` by bitcasting it to `u64`.
    pub fn hash(_: HashMergeErrorContext, merge_error: MergeError) u64 {
        return @as(u64, @intCast(merge_error.index));
    }
    /// Compares two `index` for equality.
    pub fn eql(_: HashMergeErrorContext, merge_error_one: MergeError, merge_error_two: MergeError) bool {
        return merge_error_one.index == merge_error_two.index;
    }
};

/// Comparison function for the priority queue. It compares merge errors, and also considers
///  bucket indices for equality.
fn compareMergeError(_: void, error_1: MergeError, error_2: MergeError) math.Order {
    if (error_1.index == error_2.index)
        return math.Order.eq;
    return math.order(error_1.merge_error, error_2.merge_error);
}

/// A Histogram structure containing an ArrayList of `buckets` and a HashedPriorityQueue
/// `merge_queue` that keeps track of the minimum merge error merging two adjacent bucktes.
/// This structure contains the Min-Merge algorithm from the paper to maintaning the histogram
/// using only `max_buckets` buckets.
const Histogram = struct {
    const Self = @This();

    /// Target number of buckets.
    max_buckets: usize,
    /// List of current buckets.
    buckets: ArrayList(Bucket),
    /// Priority queue of merge errors.
    merge_queue: HashedPriorityQueue(
        MergeError,
        void,
        compareMergeError,
        HashMergeErrorContext,
    ),

    /// Initialize the histogram with a given allocator and target bucket count.
    pub fn init(allocator: mem.Allocator, target_buckets: usize) !Histogram {
        return Histogram{
            .max_buckets = target_buckets,
            .buckets = ArrayList(Bucket).init(allocator),
            .merge_queue = try HashedPriorityQueue(MergeError, void, compareMergeError, HashMergeErrorContext).init(allocator, {}),
        };
    }

    /// Deinitialize the histogram and release memory.
    pub fn deinit(self: *Self) void {
        self.buckets.deinit();
        self.merge_queue.deinit();
    }

    /// Insert a new value into the histogram.
    pub fn insert(self: *Self, value: f64, index: usize) !void {
        // Create a new bucket for the incoming value with start and end at 'index'.
        try self.buckets.append(try Bucket.init(index, index, value, value));

        if (self.buckets.items.len > 1) {
            const bucket_last_index: usize = self.buckets.items.len - 1;
            const merge_error: f64 = self.calculateMergeError(bucket_last_index - 1);
            try self.merge_queue.add(MergeError{ .index = bucket_last_index - 1, .merge_error = merge_error });
        }
        // If the number of buckets exceeds 2B, merge the least increasing pair (MIN-MERGE).
        if (self.buckets.items.len > self.max_buckets) {
            try self.minMerge();
        }
    }

    /// Returns the bucket at `index`.
    pub fn getBucket(self: *Self, index: usize) Bucket {
        return self.buckets[index];
    }

    /// Calculate the merge error for merging the bucket at `index` with the next bucket.
    fn calculateMergeError(self: *Self, index: usize) f64 {
        const bucket1 = self.buckets.items[index];
        const bucket2 = self.buckets.items[index + 1];
        const merged_min = @min(bucket1.min_val, bucket2.min_val);
        const merged_max = @max(bucket1.max_val, bucket2.max_val);
        return (merged_max - merged_min) / 2.0;
    }

    /// Perform the minimum merge by finding the pair with the smallest merge error.
    fn minMerge(self: *Self) !void {

        // Pop the smallest merge error (the least costly merge).
        const min_merge_error: MergeError = try self.merge_queue.remove();

        // Merge the buckets at min_merge_error.index and min_merge_error.index + 1.
        const index = min_merge_error.index;
        const bucket_one = self.buckets.items[index];
        const bucket_two = self.buckets.items[index + 1];

        // Merge the two buckets, updating their range and min/max values.
        const new_begin = bucket_one.begin;
        const new_end = bucket_two.end;
        const new_min = @min(bucket_one.min_val, bucket_two.min_val);
        const new_max = @max(bucket_one.max_val, bucket_two.max_val);

        // Replace the first bucket with the merged result and remove the second.
        self.buckets.items[index] = try Bucket.init(new_begin, new_end, new_min, new_max);
        _ = self.buckets.orderedRemove(index + 1); // Remove the merged bucket.

        if (index < self.buckets.items.len - 1) {
            const new_merge_error = self.calculateMergeError(index);
            try self.merge_queue.add(MergeError{ .index = index, .merge_error = new_merge_error });

            if (index > 0)
                try self.merge(index - 1); // Update error for the previous bucket.

            try self.updateAllIndex(index + 1);
        }
    }

    /// Update the merge error for the bucket at 'index'.
    fn merge(self: *Self, index: usize) !void {
        const merge_error = self.calculateMergeError(index);
        const new_merge_error = MergeError{ .index = index, .merge_error = merge_error };

        // Placeholder for the old merge.
        const old_merge_item: MergeError = MergeError{ .index = index, .merge_error = merge_error };

        // Update the priority queue with the new merge error using the 'update' method.
        try self.merge_queue.update(old_merge_item, new_merge_error);
    }

    /// Update the indices of all merge error from `index` until `buckets.items.len`.
    fn updateAllIndex(self: *Self, index: usize) !void {
        for (index..self.buckets.items.len) |i| {
            const merge_error_index: usize = try self.merge_queue.getItemPosition(
                .{ .index = i, .merge_error = 0 },
            );
            const old_merge_error: MergeError = try self.merge_queue.getItemAt(merge_error_index);
            var new_merge_error: MergeError = .{
                .index = old_merge_error.index - 1,
                .merge_error = old_merge_error.merge_error,
            };

            if (i == index) {
                new_merge_error.index = self.max_buckets + 10;
                new_merge_error.merge_error = 1e16;
            }
            try self.merge_queue.update(old_merge_error, new_merge_error);
        }
    }
};

/// Compress `uncompressed_values` with the maximum number of buckets defined by the `error_bound`
/// using "Piewice Constant Histogram" compression method and write the result to
/// `compressed_values`. If an error occurs it is returned.
pub fn compressPWCH(
    uncompressed_values: []const f64,
    compressed_values: *ArrayList(u8),
    allocator: mem.Allocator,
    error_bound: f32,
) Error!void {
    if (error_bound <= 0.0) {
        return Error.UnsupportedErrorBound;
    }

    // The original implementation requires the maximum number of buckets which can be represented
    // by a usize instead of `error_bound: f32`. Changing this requires modifications in
    // `src/tersets.zig` and `src/capi.zig` files.
    // TODO: Find the right of passing the maximum number of buckets.
    const target_buckets: usize = @as(usize, @intFromFloat(@floor(error_bound)));
    var histogram = try Histogram.init(allocator, target_buckets);
    defer histogram.deinit();

    for (uncompressed_values.items, 0..) |elem, index| {
        try histogram.insert(elem, index);
    }

    for (0..2 * target_buckets) |index| {
        const bucket: Bucket = histogram.getBucket(index);
        try appendValueAndIndexToArrayList(bucket.computeError(), bucket.end, compressed_values);
    }
}

/// Decompress `compressed_values` produced by "Piecewise Constant Histogram" and write the result
/// to `decompressed_values`. If an error occurs it is returned. The implementation is similar to
/// "Poor Man’s Compression" decompression function.
pub fn decompress(
    compressed_values: []const u8,
    decompressed_values: *ArrayList(f64),
) Error!void {
    // The compressed representation is pairs containing a 64-bit float value and 64-bit end index.
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

/// Append `compressed_value` and `index` to `compressed_values`.
fn appendValueAndIndexToArrayList(
    compressed_value: f64,
    index: usize,
    compressed_values: *ArrayList(u8),
) !void {
    const value: f64 = @floatCast(compressed_value);
    const valueAsBytes: [8]u8 = @bitCast(value);
    try compressed_values.appendSlice(valueAsBytes[0..]);
    const indexAsBytes: [8]u8 = @bitCast(index); // No -1 due to 0 indexing.
    try compressed_values.appendSlice(indexAsBytes[0..]);
}

test "PWCH: HashedPriorityQueue with hashcontext for MergeError" {
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

test "PWCH: histogram insert, and merge test number buckets" {
    const allocator = testing.allocator;
    const target_buckets: usize = 100;
    var histogram = try Histogram.init(allocator, target_buckets);
    defer histogram.deinit();

    // Initialize a random number generator.
    const seed: u64 = @bitCast(time.milliTimestamp());
    var rnd = std.rand.DefaultPrng.init(seed);

    // // Insert 200 random numbers into the histogram.
    for (0..1000) |i| {
        const rand_number = @floor((rnd.random().float(f64)) * 100) / 10;
        try histogram.insert(rand_number, i);
    }
    try expectEqual(target_buckets, histogram.buckets.items.len);
}

test "PWCH: simple fixed values test" {
    const allocator = testing.allocator;

    // Input data points
    const data_points = [_]f64{
        0.9, 1.1, 0.7, 1.0, 0.8, // Cluster 1
        4.8, 5.2,  4.6, 5.0, 4.7, // Cluster 2
        9.8, 10.2, 9.9, 9.7, 10.0,
        10.1, // Cluster 3
    };

    // Initialize the histogram with a maximum of 3 buckets
    var histogram = try Histogram.init(allocator, 3);
    defer histogram.deinit();

    // Insert data points into the histogram
    for (data_points, 0..) |value, i| {
        try histogram.insert(value, i);
    }
    try expectEqual(@as(usize, @intCast(3)), histogram.buckets.items.len);

    const expected_histogram = [_]Bucket{
        Bucket{ .begin = 0, .end = 4, .min_val = 0.7, .max_val = 1.1 },
        Bucket{ .begin = 5, .end = 9, .min_val = 4.6, .max_val = 5.2 },
        Bucket{ .begin = 10, .end = 15, .min_val = 9.7, .max_val = 10.2 },
    };

    for (histogram.buckets.items, 0..) |bucket, i| {
        try expectEqual(expected_histogram[i].begin, bucket.begin);
        try expectEqual(expected_histogram[i].end, bucket.end);
        try expectEqual(expected_histogram[i].min_val, bucket.min_val);
        try expectEqual(expected_histogram[i].max_val, bucket.max_val);
    }
}

test "PWCH: fixed cluster number with random values" {
    const allocator = std.testing.allocator;
    const seed: u64 = @bitCast(time.milliTimestamp());
    var rnd = std.rand.DefaultPrng.init(seed);

    const cluster_ranges = [_]struct {
        min: f64,
        max: f64,
        count: usize,
    }{
        // Cluster 1
        .{ .min = 0.5, .max = 1.5, .count = 5 },
        // Cluster 2
        .{ .min = 4.5, .max = 5.5, .count = 5 },
        // Cluster 3
        .{ .min = 9.5, .max = 10.5, .count = 6 },
        // Cluster 4
        .{ .min = 1.5, .max = 2.5, .count = 6 },
        // Cluster 5
        .{ .min = 11.5, .max = 12.5, .count = 3 },
    };

    // Collect all data points
    var data_points = std.ArrayList(f64).init(allocator);
    defer data_points.deinit();

    // Generate random values for each cluster
    for (cluster_ranges) |cluster| {
        for (cluster.count) |_| {
            const value = rnd.random().float(f64) * (cluster.max - cluster.min) + cluster.min;
            try data_points.append(value);
        }
    }

    // Initialize the histogram with a maximum of 3 buckets
    var histogram = try Histogram.init(allocator, cluster_ranges.len);
    defer histogram.deinit();

    // Insert data points into the histogram
    for (data_points.items, 0..) |value, i| {
        try histogram.insert(value, i);
    }

    // Verify the number of buckets
    try expectEqual(cluster_ranges.len, histogram.buckets.items.len);

    // Build the expected histogram by iterating over cluster_ranges
    var expected_histogram = std.ArrayList(Bucket).init(allocator);
    defer expected_histogram.deinit();

    var current_begin: usize = 0;
    for (cluster_ranges) |cluster| {
        const bucket = Bucket{
            .begin = current_begin,
            .end = current_begin + cluster.count - 1,
            .min_val = cluster.min,
            .max_val = cluster.max,
        };
        try expected_histogram.append(bucket);
        current_begin += cluster.count;
    }

    // Verify that the output histogram matches the expected histogram
    for (histogram.buckets.items, 0..) |bucket, i| {
        const expected_bucket = expected_histogram.items[i];

        try expectEqual(expected_bucket.begin, bucket.begin);
        try expectEqual(expected_bucket.end, bucket.end);
    }
}

test "PWCH: random clusters, elements per cluster and values" {
    const allocator = std.testing.allocator;
    const seed: u64 = @bitCast(time.milliTimestamp());
    var rnd = std.rand.DefaultPrng.init(seed);
    const num_cluster: usize = rnd.random().uintLessThan(usize, 100) + 10;

    var cluster_ranges = std.ArrayList(struct {
        min: f64,
        max: f64,
        count: usize,
    }).init(allocator);
    defer cluster_ranges.deinit();

    const min_value: f64 = -1e6;
    const cluster_width: f64 = rnd.random().float(f64) * 1000 / 10;
    const gap: f64 = cluster_width + rnd.random().float(f64) * 100 + 20; // min gag
    const max_counts_per_cluster: usize = 100;

    var current_min_value = min_value;
    for (0..num_cluster) |i| {
        const momentum: f64 = if (i < num_cluster / 2) 1.0 else -1.0;
        const cluster_min = current_min_value + momentum * gap;
        const cluster_max = cluster_min + cluster_width;
        const count: usize = rnd.random().uintLessThan(usize, max_counts_per_cluster) + 10;
        // Append the cluster to cluster_ranges
        try cluster_ranges.append(.{
            .min = cluster_min,
            .max = cluster_max,
            .count = count,
        });
        current_min_value = cluster_min;
    }

    // Collect all data points
    var data_points = std.ArrayList(f64).init(allocator);
    defer data_points.deinit();

    // Generate random values for each cluster
    for (cluster_ranges.items) |cluster| {
        for (cluster.count) |_| {
            const value = rnd.random().float(f64) * (cluster.max - cluster.min) + cluster.min;
            try data_points.append(value);
        }
    }

    // Initialize the histogram with a maximum of  'num_cluster' buckets
    var histogram = try Histogram.init(allocator, num_cluster);
    defer histogram.deinit();

    // Insert data points into the histogram
    for (data_points.items, 0..) |value, i| {
        try histogram.insert(value, i);
    }

    // Verify the number of buckets
    try expectEqual(cluster_ranges.items.len, histogram.buckets.items.len);

    // Build the expected histogram by iterating over cluster_ranges
    var expected_histogram = std.ArrayList(Bucket).init(allocator);
    defer expected_histogram.deinit();

    var current_begin: usize = 0;
    for (cluster_ranges.items) |cluster| {
        const bucket = Bucket{
            .begin = current_begin,
            .end = current_begin + cluster.count - 1,
            .min_val = cluster.min,
            .max_val = cluster.max,
        };
        try expected_histogram.append(bucket);
        current_begin += cluster.count;
    }

    // Verify that the output histogram matches the expected histogram
    for (histogram.buckets.items, 0..) |bucket, i| {
        const expected_bucket = expected_histogram.items[i];

        try expectEqual(expected_bucket.begin, bucket.begin);
        try expectEqual(expected_bucket.end, bucket.end);
    }
}
