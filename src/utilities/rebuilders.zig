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

//! This file provides utilities for rebuilding compressed representations produced by TerseTS
//! compression methods. Functions prefixed with `rebuild` recreate compressed buffers from
//! extracted components (i.e., indices and coefficients) that allow users to customize and reassemble
//! compression pipelines. These utilities support the construction of advanced compression pipelines
//! with TerseTS. However, misuse of these primitives can lead to unexpected loss of information,
//! such as when inputs are malformed, corrupted, or do not adhere to the expected representation.
//! To mitigate this problem, each `rebuild` function checks whether the input matches
//! the expected representation of the corresponding compression method, but users must ensure
//! that the data provided is semantically valid and consistent.

const std = @import("std");
const ArrayList = std.ArrayList;
const math = std.math;
const mem = std.mem;
const Allocator = mem.Allocator;
const testing = std.testing;

const tersets = @import("../tersets.zig");
const configuration = @import("../configuration.zig");
const tester = @import("../tester.zig");
const Error = tersets.Error;

const shared_functions = @import("shared_functions.zig");

/// Rebuilds the `compressed_values` array from the provided `indices` and
/// `coefficients` following a simple alternating-pair representation.
/// The function expects `indices.len == coefficients.len` and emits exactly
/// one `(coefficient, end_index)` pair per element. If the input lengths do
/// not match, `Error.CorruptedCompressedData` is returned. The `allocator`
/// handles the memory allocations of the output array. Allocation errors
/// are propagated.
pub fn rebuildCoefficientIndexPairs(
    allocator: Allocator,
    indices: []const u64,
    coefficients: []const f64,
    compressed_values: *ArrayList(u8),
) Error!void {
    // Validate input lengths: they must be equal.
    if (indices.len != coefficients.len) return Error.CorruptedCompressedData;

    // Each pair is 16 bytes. Ensure the total capacity once.
    try compressed_values.ensureTotalCapacity(allocator, coefficients.len * 16);

    const total_length = coefficients.len + indices.len;
    var timestamp_index: u64 = 0;
    var coefficient_index: u64 = 0;
    for (0..total_length) |i| {
        if (i % 2 == 0) {
            const coefficient = coefficients[coefficient_index];
            try shared_functions.appendValue(allocator, f64, coefficient, compressed_values);
            coefficient_index += 1;
        } else {
            const time = indices[timestamp_index];
            try shared_functions.appendValue(allocator, u64, time, compressed_values);
            timestamp_index += 1;
        }
    }
}

/// Rebuilds the `compressed_values` array from the provided `indices` and
/// `coefficients` following a coefficient–index representation starting
/// with a leading coefficient. The reconstructed layout consists of a
/// leading coefficient followed by alternating coefficient and index values.
/// The function therefore expects `coefficients.len == indices.len + 1`.
/// Any deviation from this representation results in
/// `Error.CorruptedCompressedData`. Any loss of information in the indices,
/// such as incorrect ordering or corrupted end indices, may result in
/// errors during decompression. Only basic structural validation is
/// performed. Semantic consistency must be ensured by the caller. The
/// `allocator` handles the memory allocations of the output array.
/// Allocation errors are propagated.
pub fn rebuildCoefficientIndexTuplesWithStartCoefficient(
    allocator: Allocator,
    indices: []const u64,
    coefficients: []const f64,
    compressed_values: *ArrayList(u8),
) Error!void {
    // Validate input lengths: coefficients must have at least one element,
    // and `indices` must have one less element than coefficients.
    if (coefficients.len == 0 or coefficients.len != indices.len + 1) {
        return Error.CorruptedCompressedData;
    }

    // Each pair is 16 bytes (two f64). Reserve once.
    try compressed_values.ensureTotalCapacity(allocator, coefficients.len * 16);

    const total_len = coefficients.len + indices.len;
    var timestamp_index: u64 = 0;
    var coefficient_index: u64 = 0;
    for (0..total_len) |i| {
        // Coefficients are at even indices (0, 1, 3, 5, ...),
        if ((i == 0) or (i % 2 == 1)) {
            const coefficient = coefficients[coefficient_index];
            try shared_functions.appendValue(allocator, f64, coefficient, compressed_values);
            coefficient_index += 1;
        } else {
            const time = indices[timestamp_index];
            try shared_functions.appendValue(allocator, u64, time, compressed_values);
            timestamp_index += 1;
        }
    }
}

/// Rebuilds the `compressed_values` array from the provided `indices` and
/// `coefficients`. The encoding follows a fixed representation of two
/// coefficients followed by one timestamp, repeating this pattern. The
/// function therefore expects `coefficients.len == indices.len * 2`.
/// If the input does not satisfy this requirement,
/// `Error.CorruptedCompressedData` is returned. Any loss of information in
/// the indices, such as incorrect alignment or modified end indices, may
/// result in errors during decompression. Only structural validation is
/// performed. Semantic consistency must be ensured by the caller. The
/// `allocator` handles the memory allocations of the output array.
///Allocation errors are propagated.
pub fn rebuildDoubleCoefficientIndexTriples(
    allocator: Allocator,
    indices: []const u64,
    coefficients: []const f64,
    compressed_values: *ArrayList(u8),
) Error!void {
    // Validate input lengths: each pair is 16 bytes (two f64). Reserve once.
    if (coefficients.len != indices.len * 2) {
        return Error.CorruptedCompressedData;
    }

    try compressed_values.ensureTotalCapacity(allocator, coefficients.len * 24);

    const total_len = coefficients.len + indices.len;
    var timestamp_index: u64 = 0;
    var coefficient_index: u64 = 0;
    for (0..total_len) |i| {
        if ((i + 1) % 3 != 0) {
            const coefficient = coefficients[coefficient_index];
            try shared_functions.appendValue(allocator, f64, coefficient, compressed_values);
            coefficient_index += 1;
        } else {
            const timestamp = indices[timestamp_index];
            try shared_functions.appendValue(allocator, u64, timestamp, compressed_values);
            timestamp_index += 1;
        }
    }
}
