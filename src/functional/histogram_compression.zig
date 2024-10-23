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

const std = @import("std");
const mem = std.mem;
const math = std.math;
const testing = std.testing;
const ArrayList = std.ArrayList;
const HashedPriorityQueue = @import(
    "../data_structures/hashed_priority_queue.zig",
).HashedPriorityQueue;

const Bucket = struct {
    begin: usize, // Begin of the bucket (index where the bucket starts in the data stream).
    end: usize, // End of the bucket (index where the bucket ends in the data stream).
    min_val: f64, // Minimum value within this bucket.
    max_val: f64, // Maximum value within this bucket.

    // Initialize the bucket with the given indices and min/max values.
    pub fn init(begin: usize, end: usize, min_val: f64, max_val: f64) !Bucket {
        return Bucket{ .begin = begin, .end = end, .min_val = min_val, .max_val = max_val };
    }

    // Calculate the error for the bucket. Error is half the difference between max and min values.
    pub fn getError(self: Bucket) f64 {
        return (self.max_val - self.min_val) / 2.0;
    }
};

// Structure for holding merge information.
// This represents the error that results from merging two adjacent buckets.
const MergeError = struct {
    index: usize, // Index of the first bucket in the pair to be merged.
    merge_error: f64, // Error resulting from merging this bucket with the next one.
};

const HashMergeErrorContext = struct {
    /// Hashes an the `index` by bitcasting it to `u64`.
    pub fn hash(_: HashMergeErrorContext, merge_error: MergeError) usize {
        return merge_error.index;
    }
    /// Compares two `index` for equality.
    pub fn eql(_: HashMergeErrorContext, merge_error_one: MergeError, merge_error_two: MergeError) bool {
        return merge_error_one.index == merge_error_two.index;
    }
};

// Comparison function for the priority queue.
// It compares merge errors, and also considers bucket indices for equality.
fn compareMergeError(_: void, error_1: MergeError, error_2: MergeError) math.Order {
    if (error_1.index == error_2.index)
        return math.Order.eq;
    return math.order(error_1.merge_error, error_2.merge_error);
}

const Histogram = struct {
    B: usize, // Target number of buckets.
    buckets: ArrayList(Bucket), // List of current buckets.
    // Priority queue of merge errors.
    merge_queue: HashedPriorityQueue(
        MergeError,
        void,
        compareMergeError,
        HashMergeErrorContext,
    ),

    // Initialize the histogram with a given allocator and target bucket count.
    pub fn init(allocator: mem.Allocator, target_buckets: usize) !Histogram {
        return Histogram{
            .B = target_buckets,
            .buckets = ArrayList(Bucket).init(allocator),
            .merge_queue = PriorityQueue(MergeError, void, compareMergeError).init(allocator, {}),
        };
    }

    // Deinitialize the histogram and release memory.
    pub fn deinit(self: *Histogram) void {
        self.buckets.deinit();
        self.merge_queue.deinit();
    }

    // Insert a new value into the histogram.
    pub fn insert(self: *Histogram, value: f64, index: usize) !void {
        // Create a new bucket for the incoming value with start and end at 'index'.
        try self.buckets.append(try Bucket.init(index, index, value, value));

        if (self.buckets.items.len > 1) {
            const bucket_last_index: usize = self.buckets.items.len - 1;
            const merge_error: f64 = self.calculateMergeError(bucket_last_index - 1);
            try self.merge_queue.add(MergeError{ .index = bucket_last_index - 1, .merge_error = merge_error });
        }
        // If the number of buckets exceeds 2B, merge the least increasing pair (MIN-MERGE).
        if (self.buckets.items.len > 2 * self.B) {
            for (self.merge_queue.items) |me| {
                std.debug.print("Merge Error Inserted {} and {}\n", .{ me.index, me.merge_error });
            }
            try self.minMerge();
        }
    }

    // Perform the minimum merge by finding the pair with the smallest merge error.
    fn minMerge(self: *Histogram) !void {

        // Pop the smallest merge error (the least costly merge).
        const min_merge_error: MergeError = self.merge_queue.remove();
        std.debug.print("MinMerge-Error {d} \n", .{min_merge_error.index});

        // Merge the buckets at min_merge_error.index and min_merge_error.index + 1.
        const index = min_merge_error.index;
        const bucket1 = self.buckets.items[index];
        const bucket2 = self.buckets.items[index + 1];

        // Merge the two buckets, updating their range and min/max values.
        const new_begin = bucket1.begin;
        const new_end = bucket2.end;
        const new_min = @min(bucket1.min_val, bucket2.min_val);
        const new_max = @max(bucket1.max_val, bucket2.max_val);

        // Replace the first bucket with the merged result and remove the second.
        self.buckets.items[index] = try Bucket.init(new_begin, new_end, new_min, new_max);
        _ = self.buckets.orderedRemove(index + 1); // Remove the merged bucket.

        const new_merge_error = self.calculateMergeError(index);
        try self.merge_queue.add(MergeError{ .index = index, .merge_error = new_merge_error });

        // Recompute the errors for the neighbors of the merged bucket.
        if (index > 0)
            try self.merge(index - 1); // Update error for the previous bucket.
    }

    // Update the merge error for the bucket at 'index'.
    fn merge(self: *Histogram, index: usize) !void {
        const merge_error = self.calculateMergeError(index);
        const new_merge_error = MergeError{ .index = index, .merge_error = merge_error };

        // Placeholder for the old merge.
        const old_merge_item: MergeError = MergeError{ .index = index, .merge_error = merge_error };

        // Update the priority queue with the new merge error using the 'update' method.
        try self.merge_queue.update(old_merge_item, new_merge_error);
    }

    // Calculate the merge error for merging the bucket at index 'i' with the next bucket.
    fn calculateMergeError(self: *Histogram, i: usize) f64 {
        std.debug.print("calculateMergeError-Index {} and {} \n", .{ i, i + 1 });
        const bucket1 = self.buckets.items[i];
        const bucket2 = self.buckets.items[i + 1];
        const merged_min = @min(bucket1.min_val, bucket2.min_val);
        const merged_max = @max(bucket1.max_val, bucket2.max_val);
        return (merged_max - merged_min) / 2.0;
    }

    // Print the current state of the histogram, showing each bucket's range, min, max, and error.
    pub fn printHistrogram(self: *Histogram) void {
        std.debug.print("Mything {}", .{self.merge_queue.mything});
        // for (self.buckets.items) |b| {
        //     std.debug.print("Bucket [{} - {}] -> min: {}, max: {}, error: {}\n", .{ b.begin, b.end, b.min_val, b.max_val, b.getError() });
        // }
    }
};

test "Generic PriorityQueue with Custom HashMap and HashContext for MergeError" {
    const allocator = std.heap.page_allocator;

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
    try std.testing.expect(top_value.index == 2); // Top element should be index 2 (smallest merge_error)

    const next_top = try pq.remove();
    try std.testing.expect(next_top.index == 1);

    const final_top = try pq.remove();
    try std.testing.expect(final_top.index == 3);
}

// Test case to verify that insert and merge operations work as expected.
test "histogram insert, and merge test" {
    const allocator = testing.allocator;
    const target_buckets: usize = 10;
    var histogram = try Histogram.init(allocator, target_buckets);
    defer histogram.deinit();

    histogram.printHistrogram();
    // Initialize a random number generator.
    // var rnd = std.rand.DefaultPrng.init(@as(u64, @bitCast(std.time.milliTimestamp())));

    // // Insert 200 random numbers into the histogram.
    // for (0..1000) |i| {
    //     const rand_number = @floor((rnd.random().float(f64) - 0.5) * 1000) / 10;
    //     try histogram.insert(rand_number, i);
    // }
    // try testing.expectEqual(2 * target_buckets, histogram.buckets.items.len);
}

// test "histogram compression algorithm test" {
//     const allocator = testing.allocator;

//     // Define the target number of buckets.
//     const target_buckets: usize = 10;
//     var histogram = try Histogram.init(allocator, target_buckets);
//     defer histogram.deinit();

//     // Insert a series of values in increasing order.
//     var values = ArrayList(f64).init(allocator);

//     // Insert values into the histogram.
//     for (0..20) |i| {
//         const value = @as(f64, @floatFromInt(i));
//         try histogram.insert(value, i);
//         try values.append(value);
//     }

//     // Step 1: Check that the total number of buckets does not exceed `2 * B`.
//     std.debug.assert(histogram.buckets.items.len <= 2 * target_buckets);

//     // Step 2: Check that bucket ranges and values are correct after merges.
//     // We should check that adjacent buckets maintain their min/max properties and have the correct error range.
//     var last_max_value: f64 = histogram.buckets.items[0].max_val;
//     for (histogram.buckets.items[1..]) |bucket| {
//         std.debug.assert(bucket.min_val >= last_max_value); // Ensure the next bucket starts at or above the previous bucket's max value.
//         last_max_value = bucket.max_val;
//     }

//     // Step 3: Verify that no bucket's error exceeds the expected maximum (L∞ norm).
//     const max_error: f64 = (@as(f64, @floatFromInt(values.items.len - 1))) / (2.0 * @as(
//         f64,
//         @floatFromInt(target_buckets),
//     )); // Expected maximum error.

//     for (histogram.buckets.items) |bucket| {
//         const merge_error = bucket.getError();
//         std.debug.assert(merge_error <= max_error); // Check that the error is within the expected bounds.
//     }

//     // Step 4: Test with random data (to simulate real use cases).
//     var rnd = std.rand.DefaultPrng.init(@as(u64, @bitCast(std.time.milliTimestamp())));
//     for (0..100) |i| {
//         const rand_value = rnd.random().float(f64) * 100.0; // Generate a random value between 0 and 100.
//         try histogram.insert(rand_value, i + values.items.len);
//     }

//     // After additional random inserts, the total number of buckets should still be <= `2 * B`.
//     std.debug.assert(histogram.buckets.items.len <= 2 * target_buckets);

//     // Final print of the histogram for manual inspection (optional).
//     histogram.printHistrogram();
// }
