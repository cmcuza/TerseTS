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
//! adding a hash map to track element positions, enabling efficient updates and removals by key.
//!
//! The main heap operations (`add`, `siftUp`, `siftDown`, `remove`) are adapted from Zig's
//! implementation at `$(ZIG_HOME)/lib/std/priority_queue.zig`. Additionally, the function `update`
//! was modified having now O(log N) complexity instead of O(N). We use the function `get()` to
//! access in O(1) any element in the queue. This is particularly useful in applications where the
//! priority of elements change, or where elements need to be accessed by key.

const std = @import("std");
const rand = std.Random;
const math = std.math;
const time = std.time;
const Order = std.math.Order;
const testing = std.testing;
const HashMap = std.HashMap;
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;

const expectEqual = testing.expectEqual;
const expectError = testing.expectError;

const tersets = @import("../tersets.zig");
const Error = tersets.Error;

/// A generic priority queue for storing generic data with hashed indexing for fast updates.
/// Similar as with Zig's std.PriorityQueue, initialize with `init`. Provide `compareFn` that
/// returns an `Order` enum to compare the second and third arguments, e.g., return `Order.lt`
/// if the second argument should be popped first. Provide `HashContext` to hash the elements in
/// the queue with their position. `HashContext` should contain a `hash` and `eql` function for
/// elements of type `T`.
pub fn HashedPriorityQueue(
    comptime T: type,
    comptime Context: type,
    comptime compareFn: fn (context: Context, a: T, b: T) Order,
    comptime HashContext: type,
) type {
    return struct {
        const Self = @This();

        /// Elements forming the heap.
        items: []T,
        /// Number of elements in the queue.
        len: usize,
        /// Allocator used for memory management.
        allocator: Allocator,
        /// Context passed to the comparator function.
        context: Context,
        /// Hash map for fast indexing and updates of elements.
        index_map: HashMap(T, usize, HashContext, std.hash_map.default_max_load_percentage),

        /// Initializes a new `HashedPriorityQueue`.
        pub fn init(allocator: Allocator, context: Context) !Self {
            return Self{
                .items = &[_]T{}, // Initialize with an empty slice.
                .len = 0,
                .allocator = allocator,
                .context = context,
                // Initialize the hash map with the given allocator.
                .index_map = HashMap(
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

        /// Returns the minimum element in the queue without removing it.
        pub fn peek(self: *Self) !T {
            if (self.len == 0) return Error.EmptyQueue;
            return self.items[0];
        }

        /// Removes and returns the element with the highest priority (at index 0).
        pub fn pop(self: *Self) !T {
            return try self.removeIndex(0);
        }

        /// Removes and returns the element at the specified index.
        pub fn removeIndex(self: *Self, index: usize) !T {
            if (index >= self.len) return Error.ItemNotFound;

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

        /// Returns the `elem` index in the queue.
        pub fn getIndex(self: *Self, elem: T) !usize {
            return self.index_map.get(elem) orelse Error.ItemNotFound;
        }

        /// Returns the `elem` in the queue at `index` in the queue.
        pub fn get(self: *Self, index: usize) !T {
            if (index >= self.len)
                return Error.ItemNotFound;

            return self.items[index];
        }

        /// Returns the current capacity of the queue (elements that can be added without
        /// reallocation).
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
                var lesser_child_i = (math.mul(usize, index, 2) catch break) | 1;
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

test "add and remove min keys of simple values in HashedPriorityQueue" {
    const allocator = testing.allocator;

    const CompareContext = struct {
        fn order(context: void, a: u64, b: u64) Order {
            _ = context;
            return math.order(a, b);
        }
    };

    var queue = try HashedPriorityQueue(
        u64,
        void,
        CompareContext.order,
        std.hash_map.AutoContext(u64),
    ).init(allocator, {});
    defer queue.deinit();

    // Generate an ArrayList of 100 i64 elements with random values.
    var list = ArrayList(u64).init(allocator);
    defer list.deinit();

    // Initialize a random number generator.
    const seed: u64 = @bitCast(time.milliTimestamp());
    var rnd = std.Random.DefaultPrng.init(seed);

    for (0..100) |_| {
        // Generate a random i64 value
        const value = @abs(@mod((rnd.random().int(i64)), 1000));
        try list.append(value);
        try queue.add(value);
    }

    // Create a copy of the list and sort it.
    var sorted_list = ArrayList(u64).init(allocator);
    defer sorted_list.deinit();
    try sorted_list.appendSlice(list.items);
    std.mem.sort(u64, sorted_list.items, {}, comptime std.sort.asc(u64));

    // Remove elements from the queue and verify the order.
    for (sorted_list.items) |expected_value| {
        const removed_value = try queue.pop();
        try expectEqual(expected_value, removed_value);
    }
}

test "test update function in HashedPriorityQueue" {
    const allocator = testing.allocator;

    const CompareContext = struct {
        fn lessThan(context: void, a: u32, b: u32) Order {
            _ = context;
            return math.order(a, b);
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
    try expectEqual(@as(u32, 1), queue.pop());
    try expectEqual(@as(u32, 4), queue.pop());
    try expectEqual(@as(u32, 5), queue.pop());
    try queue.add(1);
    try queue.add(1);
    try queue.add(2);
    try queue.add(2);
    try queue.update(1, 5);
    try queue.update(2, 4);
    try expectEqual(@as(u32, 1), queue.pop());
    try expectEqual(@as(u32, 2), queue.pop());
    try expectEqual(@as(u32, 4), queue.pop());
    try expectEqual(@as(u32, 5), queue.pop());
    try expectError(Error.ItemNotFound, queue.update(1, 1));
}

test "add and remove min key of custom struct in HashedPriorityQueue" {
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
            return math.order(a.key, b.key);
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
    try expectEqual(2, (try queue.pop()).key);
    try expectEqual(4, (try queue.pop()).key);
    try expectEqual(7, (try queue.pop()).key);
    try expectEqual(10, (try queue.pop()).key);
    try expectEqual(15, (try queue.pop()).key);
    try expectEqual(23, (try queue.pop()).key);
}

test "add, remove and element position for key of structs in HashedPriorityQueue" {
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
            return math.order(a.value, b.value);
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
    try expectEqual(0, (try queue.getIndex(.{ .key = 4, .value = 7.0 })));
    try expectEqual(4, (try queue.pop()).key);
    try expectEqual(0, (try queue.getIndex(.{ .key = 2, .value = 12.0 })));
    try expectEqual(2, (try queue.pop()).key);

    try queue.update(.{ .key = 7, .value = 13.0 }, .{ .key = 8, .value = 13.0 });
    try expectEqual(0, (try queue.getIndex(.{ .key = 8, .value = 13.0 })));
    try expectEqual(8, (try queue.pop()).key);

    try expectError(Error.ItemNotFound, queue.update(
        .{ .key = 17, .value = 13.0 },
        .{ .key = 8, .value = 13.0 },
    ));

    try queue.update(.{ .key = 23, .value = 54.0 }, .{ .key = 23, .value = 5.0 });
    try expectEqual(0, (try queue.getIndex(.{ .key = 23, .value = 5.0 })));
    try expectEqual(23, (try queue.pop()).key);

    try queue.add(.{ .key = 7, .value = 3.0 });
    try expectEqual(0, (try queue.getIndex(.{ .key = 7, .value = 3.0 })));
    try expectEqual(7, (try queue.pop()).key);
    try expectError(Error.ItemNotFound, queue.update(
        .{ .key = 7, .value = 3.0 },
        .{ .key = 9, .value = 13.0 },
    ));
}
