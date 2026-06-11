// Copyright 2026 TerseTS Contributors
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

//! Implementation of an indexed priority queue for dense `usize` indices.
//!
//! The queue is intended for algorithms whose items are identified by stable dense indices in the
//! range `0..capacity`. Unlike `HashedPriorityQueue`, this implementation does not use a hash map
//! to find or update existing elements. Instead, it stores the current state for each dense index
//! in arrays and keeps heap updates cheap by pushing a new version of the changed element. Older
//! heap entries are discarded lazily when they reach the top of the heap.
//!
//! This representation is useful for hot loops where the caller already has dense indices, such as
//! time-series algorithms operating over positions in an input slice. For sparse keys or keys that
//! are not naturally represented as dense `usize` values, use `HashedPriorityQueue` instead.

const std = @import("std");
const math = std.math;
const mem = std.mem;
const Order = std.math.Order;
const testing = std.testing;
const Allocator = mem.Allocator;
const ArrayList = std.ArrayList;

const expectEqual = testing.expectEqual;
const expectError = testing.expectError;

/// Errors returned by `IndexedPriorityQueue`.
pub const Error = error{
    EmptyQueue,
    ItemNotFound,
    OutOfMemory,
};

/// A priority queue for items identified by dense `usize` indices.
/// Similar as with Zig's std.PriorityQueue, initialize with `init`. Provide `compareFn` that
/// returns an `Order` enum to compare priorities, e.g., return `Order.lt` if the first priority
/// should be popped before the second priority. If two priorities compare equal, the item with
/// the smaller dense index is popped first to keep the behavior deterministic.
/// `capacity` passed to `init` defines the valid index range for the lifetime of the queue:
/// `0..capacity`. Indices outside that range return `Error.ItemNotFound`.
pub fn IndexedPriorityQueue(
    comptime T: type,
    comptime compareFn: fn (a: T, b: T) Order,
) type {
    return struct {
        const Self = @This();

        /// Public representation returned by `peek` and `pop`.
        pub const Entry = struct {
            /// Dense index associated with the priority.
            index: usize,
            /// Current priority for `index`.
            priority: T,
        };

        /// Internal heap entry. `version` identifies stale copies left in the heap after updates.
        const HeapEntry = struct {
            /// Dense index associated with the priority.
            index: usize,
            /// Priority captured when this heap entry was inserted.
            priority: T,
            /// Version captured when this heap entry was inserted.
            version: usize,
        };

        /// Elements forming the heap. Stale entries may exist until they reach the heap root.
        items: ArrayList(HeapEntry),
        /// Number of currently active dense indices in the queue.
        len: usize,
        /// Allocator used for memory management.
        allocator: Allocator,
        /// Current priority for each dense index.
        priorities: []T,
        /// Current version for each dense index.
        versions: []usize,
        /// Whether each dense index is currently present in the queue.
        is_active: []bool,

        /// Initializes a new `IndexedPriorityQueue` with valid indices in `0..initial_capacity`.
        pub fn init(allocator: Allocator, initial_capacity: usize) Error!Self {
            const priorities = try allocator.alloc(T, initial_capacity);
            errdefer allocator.free(priorities);

            const versions = try allocator.alloc(usize, initial_capacity);
            errdefer allocator.free(versions);

            const is_active = try allocator.alloc(bool, initial_capacity);
            errdefer allocator.free(is_active);

            @memset(versions, 0);
            @memset(is_active, false);

            var items = ArrayList(HeapEntry).empty;
            errdefer items.deinit(allocator);
            try items.ensureTotalCapacity(allocator, initial_capacity);

            return Self{
                .items = items,
                .len = 0,
                .allocator = allocator,
                .priorities = priorities,
                .versions = versions,
                .is_active = is_active,
            };
        }

        /// Frees memory used by the queue.
        pub fn deinit(self: *Self) void {
            self.items.deinit(self.allocator);
            self.allocator.free(self.priorities);
            self.allocator.free(self.versions);
            self.allocator.free(self.is_active);
        }

        /// Returns the number of active indices in the queue.
        pub fn count(self: *const Self) usize {
            return self.len;
        }

        /// Returns the current capacity of the queue, which is also the exclusive upper bound of
        /// valid dense indices.
        pub fn capacity(self: *const Self) usize {
            return self.is_active.len;
        }

        /// Returns whether `index` is currently present in the queue.
        pub fn contains(self: *const Self, index: usize) bool {
            return index < self.capacity() and self.is_active[index];
        }

        /// Inserts `index` with `priority`, maintaining the heap property.
        /// Returns `Error.ItemNotFound` if `index` is out of range or already present in the queue.
        pub fn add(self: *Self, index: usize, priority: T) Error!void {
            if (index >= self.capacity()) return Error.ItemNotFound;
            if (self.is_active[index]) return Error.ItemNotFound;

            self.is_active[index] = true;
            self.priorities[index] = priority;
            self.versions[index] += 1;
            self.len += 1;
            try self.pushHeap(.{
                .index = index,
                .priority = priority,
                .version = self.versions[index],
            });
        }

        /// Updates `index` with `priority`.
        /// Updates are implemented by pushing a new heap entry and incrementing the version for the
        /// index. Older heap entries become stale and are discarded lazily by `peek` or `pop`.
        pub fn update(self: *Self, index: usize, priority: T) Error!void {
            if (!self.contains(index)) return Error.ItemNotFound;

            self.priorities[index] = priority;
            self.versions[index] += 1;
            try self.pushHeap(.{
                .index = index,
                .priority = priority,
                .version = self.versions[index],
            });
        }

        /// Removes `index` from the queue.
        /// Any heap entries already inserted for `index` become stale and are discarded lazily.
        pub fn remove(self: *Self, index: usize) Error!void {
            if (!self.contains(index)) return Error.ItemNotFound;

            self.is_active[index] = false;
            self.versions[index] += 1;
            self.len -= 1;
        }

        /// Returns the minimum active entry without removing it.
        pub fn peek(self: *Self) Error!Entry {
            try self.discardStaleHeapEntries();
            if (self.items.items.len == 0) return Error.EmptyQueue;

            const entry = self.items.items[0];
            return Entry{ .index = entry.index, .priority = entry.priority };
        }

        /// Removes and returns the minimum active entry.
        pub fn pop(self: *Self) Error!Entry {
            try self.discardStaleHeapEntries();
            if (self.items.items.len == 0) return Error.EmptyQueue;

            const entry = try self.popHeap();
            self.is_active[entry.index] = false;
            self.versions[entry.index] += 1;
            self.len -= 1;
            return Entry{ .index = entry.index, .priority = entry.priority };
        }

        /// Returns whether `entry` still represents the current active state for its dense index.
        fn isActiveHeapEntry(self: *const Self, entry: HeapEntry) bool {
            return self.is_active[entry.index] and entry.version == self.versions[entry.index];
        }

        /// Removes stale heap roots until the root is active or the heap is empty.
        fn discardStaleHeapEntries(self: *Self) Error!void {
            while (self.items.items.len > 0 and !self.isActiveHeapEntry(self.items.items[0])) {
                _ = try self.popHeap();
            }
        }

        /// Inserts a heap entry, maintaining the heap property by sifting it up.
        fn pushHeap(self: *Self, entry: HeapEntry) Error!void {
            try self.items.append(self.allocator, entry);

            var child_index = self.items.items.len - 1;
            while (child_index > 0) {
                const parent_index = (child_index - 1) >> 1;
                if (!heapEntryLessThan(self.items.items[child_index], self.items.items[parent_index])) break;

                mem.swap(HeapEntry, &self.items.items[child_index], &self.items.items[parent_index]);
                child_index = parent_index;
            }
        }

        /// Removes and returns the heap root without checking whether it is active.
        fn popHeap(self: *Self) Error!HeapEntry {
            if (self.items.items.len == 0) return Error.EmptyQueue;

            const min_entry = self.items.items[0];
            const last_entry = self.items.pop().?;
            if (self.items.items.len == 0) return min_entry;

            self.items.items[0] = last_entry;
            var index: usize = 0;
            while (true) {
                const left_child_index = (math.mul(usize, index, 2) catch break) | 1;
                if (left_child_index >= self.items.items.len) break;

                const right_child_index = left_child_index + 1;
                var lesser_child_index = left_child_index;
                if (right_child_index < self.items.items.len and
                    heapEntryLessThan(self.items.items[right_child_index], self.items.items[left_child_index]))
                {
                    lesser_child_index = right_child_index;
                }

                if (!heapEntryLessThan(self.items.items[lesser_child_index], self.items.items[index])) break;

                mem.swap(HeapEntry, &self.items.items[index], &self.items.items[lesser_child_index]);
                index = lesser_child_index;
            }

            return min_entry;
        }

        /// Returns whether `entry_1` has higher priority than `entry_2`.
        fn heapEntryLessThan(entry_1: HeapEntry, entry_2: HeapEntry) bool {
            return switch (compareFn(entry_1.priority, entry_2.priority)) {
                .lt => true,
                .eq => entry_1.index < entry_2.index,
                .gt => false,
            };
        }
    };
}

fn compareF64(a: f64, b: f64) Order {
    return math.order(a, b);
}

test "indexed priority queue pops entries by priority" {
    const allocator = testing.allocator;

    var queue = try IndexedPriorityQueue(f64, compareF64).init(allocator, 5);
    defer queue.deinit();

    try queue.add(3, 3.0);
    try queue.add(1, 1.0);
    try queue.add(4, 4.0);
    try queue.add(2, 2.0);

    try expectEqual(@as(usize, 4), queue.count());
    try expectEqual(@as(usize, 5), queue.capacity());
    try expectEqual(@as(usize, 1), (try queue.pop()).index);
    try expectEqual(@as(usize, 2), (try queue.pop()).index);
    try expectEqual(@as(usize, 3), (try queue.pop()).index);
    try expectEqual(@as(usize, 4), (try queue.pop()).index);
    try expectError(Error.EmptyQueue, queue.pop());
}

test "indexed priority queue update discards stale priorities" {
    const allocator = testing.allocator;

    var queue = try IndexedPriorityQueue(f64, compareF64).init(allocator, 4);
    defer queue.deinit();

    try queue.add(0, 10.0);
    try queue.add(1, 20.0);
    try queue.add(2, 30.0);

    try queue.update(2, 1.0);
    try expectEqual(@as(usize, 2), (try queue.peek()).index);
    try expectEqual(@as(usize, 2), (try queue.pop()).index);
    try expectEqual(@as(usize, 0), (try queue.pop()).index);
    try expectEqual(@as(usize, 1), (try queue.pop()).index);
}

test "indexed priority queue removal invalidates pending entries" {
    const allocator = testing.allocator;

    var queue = try IndexedPriorityQueue(f64, compareF64).init(allocator, 4);
    defer queue.deinit();

    try queue.add(0, 1.0);
    try queue.add(1, 0.5);
    try queue.add(2, 2.0);
    try queue.remove(1);

    try testing.expect(!queue.contains(1));
    try expectEqual(@as(usize, 0), (try queue.pop()).index);
    try expectEqual(@as(usize, 2), (try queue.pop()).index);
    try expectError(Error.EmptyQueue, queue.pop());
}

test "indexed priority queue breaks equal-priority ties by index" {
    const allocator = testing.allocator;

    var queue = try IndexedPriorityQueue(f64, compareF64).init(allocator, 4);
    defer queue.deinit();

    try queue.add(3, 1.0);
    try queue.add(1, 1.0);
    try queue.add(2, 1.0);

    try expectEqual(@as(usize, 1), (try queue.pop()).index);
    try expectEqual(@as(usize, 2), (try queue.pop()).index);
    try expectEqual(@as(usize, 3), (try queue.pop()).index);
}
