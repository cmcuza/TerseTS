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
//! Min-Merge find a Piecewise Constant Histogram compressed representation of the time series.
//! Thus, the abbreviation PWCH is used for the method as in the paper:
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

const tersets = @import("../tersets.zig");
const Error = tersets.Error;

const tester = @import("../tester.zig");

/// `Bucket` stores information about a range of consecutives values in the time series.
/// The structure stores the indices `begin` and `end` where the bucket starts and ends.
/// It also stores the minimum and maximum values (`min_val` and `max_value`) in the bucket.
/// The structure also contains the function `computeError` which computes and returns the
/// $L_\inf$ error associated to the bucket.
const Bucket = struct {
    // Begining of the bucket.
    begin: usize,
    // Ending of the bucket.
    end: usize,
    // Min value of the bucket.
    min_val: f64,
    // Max value of the bucket.
    max_val: f64,

    /// Initialize the bucket with the given indices and min/max values.
    pub fn init(begin: usize, end: usize, min_val: f64, max_val: f64) !Bucket {
        return Bucket{ .begin = begin, .end = end, .min_val = min_val, .max_val = max_val };
    }

    /// Returns the absolute error of the values in the bucket.
    pub fn computeError(self: Bucket) f64 {
        return (self.max_val - self.min_val) / 2.0;
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
    /// parameter can be thought as fixing a minimum compression ratio that users wants to achieve.
    /// The `minMerge` function will then find the optimal histogram under this constraint.
    /// For example, if `max_buckets=|N|/2`, then the compression ratio will be at least 2x.
    pub fn init(allocator: mem.Allocator, max_buckets: usize) !Histogram {
        return Histogram{
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
        self.buckets.deinit();
        self.merge_queue.deinit();
    }

    /// Insert a new value into the histogram.
    pub fn insert(self: *Self, value: f64, index: usize) !void {
        // Create a new bucket for the incoming value with start and end at 'index'.
        try self.buckets.append(try Bucket.init(index, index, value, value));

        if (self.buckets.items.len > 1) {
            const bucket_last_index: usize = self.buckets.items.len - 1;
            const merge_error: f64 = try self.calculateMergeError(bucket_last_index - 1);
            try self.merge_queue.add(MergeError{ .index = bucket_last_index - 1, .merge_error = merge_error });
        }
        // If the number of buckets exceeds `max_buckets`, merge the least increasing pair.
        if (self.buckets.items.len > self.max_buckets) {
            try self.minMerge();
        }
    }

    /// Returns the bucket at `index`.
    pub fn get(self: *Self, index: usize) Bucket {
        return self.buckets[index];
    }

    /// Returns the len of the histogram.
    pub fn len(self: *Self) usize {
        return self.buckets.items.len;
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
            const new_merge_error = try self.calculateMergeError(index);
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
        const merge_error = try self.calculateMergeError(index);
        const new_merge_error = MergeError{ .index = index, .merge_error = merge_error };

        // Placeholder for the old merge.
        const old_merge_item: MergeError = MergeError{ .index = index, .merge_error = merge_error };

        // Update the priority queue with the new merge error using the 'update' method.
        try self.merge_queue.update(old_merge_item, new_merge_error);
    }

    /// Calculate the merge error for merging the bucket at `index` with the next bucket.
    /// Return `Error.ItemNotFound` if `index+1` does not exists.
    fn calculateMergeError(self: *Self, index: usize) Error!f64 {
        if (index + 1 > self.buckets.items.len) {
            return Error.ItemNotFound;
        }

        const bucket1 = self.buckets.items[index];
        const bucket2 = self.buckets.items[index + 1];
        const merged_min = @min(bucket1.min_val, bucket2.min_val);
        const merged_max = @max(bucket1.max_val, bucket2.max_val);
        return (merged_max - merged_min) / 2.0;
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

/// Compress `uncompressed_values` with the maximum number of buckets defined by the `error_bound`
/// using "Piecewice Constant Histogram" compression method and write the result to
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
    // TODO: Find the right way of passing the maximum number of buckets.
    const max_buckets: usize = @as(usize, @intFromFloat(@floor(error_bound)));
    var histogram = try Histogram.init(allocator, max_buckets);
    defer histogram.deinit();

    for (uncompressed_values.items, 0..) |elem, index| {
        try histogram.insert(elem, index);
    }

    for (0..histogram.len()) |index| {
        const bucket: Bucket = histogram.at(index);
        try appendValueAndIndexToArrayList(bucket.computeError(), bucket.end, compressed_values);
    }
}

/// Decompress `compressed_values` produced by "Piecewise Constant Histogram" and write the result
/// to `decompressed_values`. If an error occurs it is returned.
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

test "HashedPriorityQueue with hashcontext for MergeError in PWCH" {
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
    var histogram = try Histogram.init(allocator, max_buckets);
    defer histogram.deinit();

    // Insert 200 random numbers into the histogram.
    for (0..1000) |i| {
        const rand_number = tester.generateBoundedRandomValue(0, 1000, random);
        try histogram.insert(rand_number, i);
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
    var histogram = try Histogram.init(allocator, 3);
    defer histogram.deinit();

    // Insert data points into the histogram.
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
    var histogram = try Histogram.init(allocator, cluster_ranges.len);
    defer histogram.deinit();

    // Insert data points into the histogram.
    for (data_points.items, 0..) |value, i| {
        try histogram.insert(value, i);
    }

    // Build the expected histogram by iterating over cluster_ranges.
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

    // Verify that the output histogram matches the expected histogram.
    for (histogram.buckets.items, 0..) |bucket, i| {
        const expected_bucket = expected_histogram.items[i];

        try expectEqual(expected_bucket.begin, bucket.begin);
        try expectEqual(expected_bucket.end, bucket.end);
    }
}

test "Random clusters, elements per cluster and values for PWCH" {
    // Initialize a random number generator.
    const seed: u64 = @bitCast(time.milliTimestamp());
    var prng = std.rand.DefaultPrng.init(seed);
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
    var histogram = try Histogram.init(allocator, num_cluster);
    defer histogram.deinit();

    // Insert data points into the histogram.
    for (data_points.items, 0..) |value, i| {
        try histogram.insert(value, i);
    }

    // Build the expected histogram by iterating over cluster_ranges.
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

    // Verify that the output histogram matches the expected histogram.
    for (histogram.buckets.items, 0..) |bucket, i| {
        const expected_bucket = expected_histogram.items[i];

        try expectEqual(expected_bucket.begin, bucket.begin);
        try expectEqual(expected_bucket.end, bucket.end);
    }
}

test "PWCH can compress and decompress" {
    const seed: u64 = @bitCast(time.milliTimestamp());
    var prng = std.rand.DefaultPrng.init(seed);
    const random = prng.random();

    const allocator = testing.allocator;
    var uncompressed_values = ArrayList(f64).init(allocator);
    defer uncompressed_values.deinit();
    try tester.generateBoundedRandomValues(&uncompressed_values, 0.0, 1.0, random);

    const error_bound: f32 = @floatFromInt(uncompressed_values.items.len);

    try tester.testCompressAndDecompress(
        uncompressed_values.items,
        allocator,
        tersets.Method.PiecewiseConstantHistogram,
        error_bound,
        tersets.isWithinErrorBound,
    );
}
