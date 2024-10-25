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

//! Implementation of a Hashed Priority Queue based on Zig's standard `PriorityQueue` by
//! incorporating a hash map to track element positions, enabling efficient updates and removals
//! by key. The main heap operations (`add`, `siftUp`, `siftDown`, `remove`) are adapted from Zig's
//! implementation. Additionally, we have added new functions used by the compression algorithms.
//! This is particularly useful in applications where the priority of elements may change, or where
//! elements need to be efficiently accessed by key. This Hashed Priority Queue is used for the
//! compression algorithms implemented in: /src/functional/histogram_compression.zig and
//! /src/line_simplification/visvalingam–whyatt.zig (upcoming).

const std = @import("std");
const rand = std.rand;
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;
const Order = std.math.Order;
const testing = std.testing;
const expectEqual = testing.expectEqual;
const expectError = testing.expectError;

const tester = @import("../tester.zig");
const tersets = @import("../tersets.zig");
const Error = tersets.Error;

/// A generic priority queue with hashed indexing for fast updates.
/// `T`: The type of elements stored in the queue.
/// `Context`: The context type passed to the comparator function.
/// `compareFn`: A function to compare two elements of type `T`.
/// `HashContext`: A type providing `hash` and `eql` functions for elements of type `T`.
pub fn HashedPriorityQueue(
    comptime T: type,
    comptime Context: type,
    comptime compareFn: fn (context: Context, a: T, b: T) std.math.Order,
    comptime HashContext: type,
) type {
    return struct {
        const Self = @This();

        /// Dynamic array of elements forming the heap.
        items: []T,
        /// Number of elements in the queue.
        len: usize,
        /// Allocator used for memory management.
        allocator: std.mem.Allocator,
        /// Context passed to the comparator function.
        context: Context,
        /// Hash map for fast indexing and updates of elements.
        index_map: std.HashMap(T, usize, HashContext, std.hash_map.default_max_load_percentage),

        /// Initializes a new `HashedPriorityQueue`.
        pub fn init(allocator: std.mem.Allocator, context: Context) !Self {
            return Self{
                .items = &[_]T{}, // Initialize with an empty array.
                .len = 0,
                .allocator = allocator,
                .context = context,
                // Initialize the hash map with the given allocator.
                .index_map = std.HashMap(
                    T,
                    usize,
                    HashContext,
                    std.hash_map.default_max_load_percentage,
                ).init(allocator),
            };
        }

        /// Frees memory used by the queue.
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.items);
            self.index_map.deinit();
        }

        /// Inserts a new element, maintaining the heap property.
        pub fn add(self: *Self, elem: T) !void {
            // Ensure there is enough capacity for the new element.
            try self.ensureUnusedCapacity(1);
            // Add the element without checking capacity.
            try self.addUnchecked(elem);
        }

        /// Removes and returns the element with the highest priority (minimum value).
        pub fn remove(self: *Self) !T {
            return try self.removeIndex(0);
        }

        /// Removes and returns the element at the specified index.
        pub fn removeIndex(self: *Self, index: usize) !T {
            assert(self.len > index);
            const last = self.items[self.len - 1];
            const item = self.items[index];

            // Replace the removed element with the last element.
            self.items[index] = last;
            self.len -= 1;

            if (index == 0) {
                // Restore the heap property by sifting down.
                try self.siftDown(index);
            } else {
                const parent_index = ((index - 1) >> 1);
                const parent = self.items[parent_index];
                if (compareFn(self.context, last, parent) == .gt) {
                    // Sift down if the last element is greater than its parent.
                    try self.siftDown(index);
                } else {
                    // Otherwise, sift up.
                    try self.siftUp(index);
                }
            }

            // Remove the item from the hash map.
            _ = self.index_map.remove(item);
            return item;
        }

        /// Updates an element in the queue with a new value.
        pub fn update(self: *Self, elem: T, new_elem: T) !void {
            const update_index = self.index_map.get(elem) orelse return Error.ItemNotFound;
            const old_elem: T = self.items[update_index];

            // Remove old key from index_map
            _ = self.index_map.remove(elem);

            // Insert new key into index_map
            try self.index_map.put(new_elem, update_index);

            // Update the item in the heap
            self.items[update_index] = new_elem;

            // Determine whether to sift up or down based on the new value.
            switch (compareFn(self.context, new_elem, old_elem)) {
                .lt => try self.siftUp(update_index),
                .gt => try self.siftDown(update_index),
                .eq => {}, // No action needed if equal.
            }
        }

        /// Ensures there is capacity for at least `additional_count` more items.
        pub fn ensureUnusedCapacity(self: *Self, additional_count: usize) !void {
            return self.ensureTotalCapacity(self.len + additional_count);
        }

        /// Returns the `elem` position in the `items` arrays.
        pub fn getItemPosition(self: *Self, elem: T) !usize {
            return self.index_map.get(elem) orelse Error.ItemNotFound;
        }

        /// Returns the `elem` in the queue that has the same key as the elem added.
        pub fn getItemAt(self: *Self, index: usize) !T {
            if (index > self.items.len)
                return Error.ItemNotFound;

            return self.items[index];
        }

        /// Returns the current capacity of the internal array.
        pub fn capacity(self: *Self) usize {
            return self.items.len;
        }

        /// Adds an element assuming capacity is already ensured.
        fn addUnchecked(self: *Self, elem: T) !void {
            // Place the element at the end of the array.
            self.items[self.len] = elem;
            // Add the element's index to the hash map.
            try self.index_map.put(elem, self.len);
            // Restore the heap property by sifting up.
            try self.siftUp(self.len);
            self.len += 1;
        }

        /// Restores the heap property by sifting up from the given index.
        fn siftUp(self: *Self, start_index: usize) !void {
            const child = self.items[start_index];
            var child_index = start_index;
            while (child_index > 0) {
                const parent_index = ((child_index - 1) >> 1);
                const parent = self.items[parent_index];
                if (compareFn(self.context, child, parent) != .lt) break;
                self.items[child_index] = parent;
                // Update parent's position in the hash map.
                try self.index_map.put(parent, child_index);
                child_index = parent_index;
            }
            self.items[child_index] = child;
            // Update child's position in the hash map.
            try self.index_map.put(child, child_index);
        }

        /// Restores the heap property by sifting down from the given index.
        fn siftDown(self: *Self, target_index: usize) !void {
            const target_element = self.items[target_index];
            var index = target_index;
            while (true) {
                // Calculate the left child index.
                var lesser_child_i = (std.math.mul(usize, index, 2) catch break) | 1;
                if (!(lesser_child_i < self.len)) break;

                const next_child_i = lesser_child_i + 1;
                // Find the smaller of the two children.
                if (next_child_i < self.len and compareFn(self.context, self.items[next_child_i], self.items[lesser_child_i]) == .lt) {
                    lesser_child_i = next_child_i;
                }

                if (compareFn(self.context, target_element, self.items[lesser_child_i]) == .lt) break;

                // Move the lesser child up.
                self.items[index] = self.items[lesser_child_i];
                try self.index_map.put(self.items[index], index); // Update position in the hash map.
                index = lesser_child_i;
            }
            self.items[index] = target_element;
            try self.index_map.put(target_element, index); // Update position in the hash map.
        }

        /// Resizes the internal array to at least `new_capacity` elements.
        fn ensureTotalCapacity(self: *Self, new_capacity: usize) !void {
            var better_capacity = self.capacity();
            if (better_capacity >= new_capacity) return;
            while (true) {
                better_capacity += better_capacity / 2 + 8;
                if (better_capacity >= new_capacity) break;
            }
            // Reallocate the array with the new capacity.
            self.items = try self.allocator.realloc(self.items, better_capacity);
        }
    };
}

test "HashedPriorityQueue: add and remove min keys of simple values" {
    const allocator = testing.allocator;

    const CompareContext = struct {
        fn order(context: void, a: i64, b: i64) Order {
            _ = context;
            return std.math.order(a, b);
        }
    };

    var queue = try HashedPriorityQueue(
        i64,
        void,
        CompareContext.order,
        std.hash_map.AutoContext(i64),
    ).init(allocator, {});
    defer queue.deinit();

    // Generate an ArrayList of 100 i64 elements with random values.
    var list = std.ArrayList(i64).init(allocator);
    defer list.deinit();

    var rng = std.rand.DefaultPrng.init(12345); // Seed for reproducibility
    const random = rng.random();

    for (0..100) |_| {
        // Generate a random i64 value
        var value = @mod(random.int(i64), 1000);
        // Ensure value is positive (optional)
        value = if (value < 0) -value else value;
        try list.append(value);
        try queue.add(value);
    }

    // Create a copy of the list and sort it.
    var sorted_list = std.ArrayList(i64).init(allocator);
    defer sorted_list.deinit();
    try sorted_list.appendSlice(list.items);
    std.mem.sort(i64, sorted_list.items, {}, comptime std.sort.asc(i64));

    // Remove elements from the queue and verify the order.
    for (sorted_list.items) |expected_value| {
        const removed_value = try queue.remove();
        try expectEqual(expected_value, removed_value);
    }
}

test "HashedPriorityQueue: update min heap" {
    const allocator = testing.allocator;

    const CompareContext = struct {
        fn lessThan(context: void, a: u32, b: u32) Order {
            _ = context;
            return std.math.order(a, b);
        }
    };

    var queue = try HashedPriorityQueue(
        u32,
        void,
        CompareContext.lessThan,
        std.hash_map.AutoContext(u32),
    ).init(allocator, {});
    defer queue.deinit();

    try queue.add(55);
    try queue.add(44);
    try queue.add(11);
    try queue.update(55, 5);
    try queue.update(44, 4);
    try queue.update(11, 1);
    try expectEqual(@as(u32, 1), queue.remove());
    try expectEqual(@as(u32, 4), queue.remove());
    try expectEqual(@as(u32, 5), queue.remove());
    try queue.add(1);
    try queue.add(1);
    try queue.add(2);
    try queue.add(2);
    try queue.update(1, 5);
    try queue.update(2, 4);
    try expectEqual(@as(u32, 1), queue.remove());
    try expectEqual(@as(u32, 2), queue.remove());
    try expectEqual(@as(u32, 4), queue.remove());
    try expectEqual(@as(u32, 5), queue.remove());
    try expectError(Error.ItemNotFound, queue.update(1, 1));
}

test "HashedPriorityQueue: add and remove min key of structs" {
    const allocator = testing.allocator;

    const S = struct {
        key: i64,
        value: f64,
    };

    const HashContext = struct {
        pub fn hash(_: @This(), s: S) u64 {
            return @as(u64, @bitCast(s.key));
        }
        pub fn eql(_: @This(), s_one: S, s_two: S) bool {
            return s_one.key == s_two.key;
        }
    };

    const CompareContext = struct {
        fn order(context: void, a: S, b: S) Order {
            _ = context;
            return std.math.order(a.key, b.key);
        }
    };

    var queue = try HashedPriorityQueue(
        S,
        void,
        CompareContext.order,
        HashContext,
    ).init(allocator, {});
    defer queue.deinit();

    // Add elements with both `index` and `size`.
    try queue.add(.{ .key = 23, .value = 54.0 });
    try queue.add(.{ .key = 2, .value = 12.0 });
    try queue.add(.{ .key = 4, .value = 7.0 });
    try queue.add(.{ .key = 10, .value = 23.0 });
    try queue.add(.{ .key = 15, .value = 25.0 });
    try queue.add(.{ .key = 7, .value = 13.0 });

    // Remove elements and check if they are in the correct order.
    try expectEqual(2, (try queue.remove()).key);
    try expectEqual(4, (try queue.remove()).key);
    try expectEqual(7, (try queue.remove()).key);
    try expectEqual(10, (try queue.remove()).key);
    try expectEqual(15, (try queue.remove()).key);
    try expectEqual(23, (try queue.remove()).key);
}

test "HashedPriorityQueue: add, remove and element position for key of structs" {
    const allocator = testing.allocator;

    const S = struct {
        key: i64,
        value: f64,
    };

    const HashContext = struct {
        pub fn hash(_: @This(), s: S) u64 {
            return @as(u64, @bitCast(s.key));
        }
        pub fn eql(_: @This(), s_one: S, s_two: S) bool {
            return s_one.key == s_two.key;
        }
    };

    const CompareContext = struct {
        fn order(context: void, a: S, b: S) Order {
            _ = context;
            return std.math.order(a.value, b.value);
        }
    };

    var queue = try HashedPriorityQueue(
        S,
        void,
        CompareContext.order,
        HashContext,
    ).init(allocator, {});
    defer queue.deinit();

    // Add elements with both `index` and `size`.
    try queue.add(.{ .key = 23, .value = 54.0 });
    try queue.add(.{ .key = 2, .value = 12.0 });
    try queue.add(.{ .key = 4, .value = 7.0 });
    try queue.add(.{ .key = 10, .value = 23.0 });
    try queue.add(.{ .key = 15, .value = 25.0 });
    try queue.add(.{ .key = 7, .value = 13.0 });

    // Remove elements and check if they are in the correct order.
    try expectEqual(0, (try queue.getItemPosition(.{ .key = 4, .value = 7.0 })));
    try expectEqual(4, (try queue.remove()).key);
    try expectEqual(0, (try queue.getItemPosition(.{ .key = 2, .value = 12.0 })));
    try expectEqual(2, (try queue.remove()).key);

    try queue.update(.{ .key = 7, .value = 13.0 }, .{ .key = 8, .value = 13.0 });
    try expectEqual(0, (try queue.getItemPosition(.{ .key = 8, .value = 13.0 })));
    try expectEqual(8, (try queue.remove()).key);

    try expectError(Error.ItemNotFound, queue.update(
        .{ .key = 17, .value = 13.0 },
        .{ .key = 8, .value = 13.0 },
    ));

    try queue.update(.{ .key = 23, .value = 54.0 }, .{ .key = 23, .value = 5.0 });
    try expectEqual(0, (try queue.getItemPosition(.{ .key = 23, .value = 5.0 })));
    try expectEqual(23, (try queue.remove()).key);

    try queue.add(.{ .key = 7, .value = 3.0 });
    try expectEqual(0, (try queue.getItemPosition(.{ .key = 7, .value = 3.0 })));
    try expectEqual(7, (try queue.remove()).key);
    try expectError(Error.ItemNotFound, queue.update(
        .{ .key = 7, .value = 3.0 },
        .{ .key = 9, .value = 13.0 },
    ));
}
