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

//! This file provides utilities for extracting the indices and coefficients encoded in compressed
//! representations produced by TerseTS compression methods. Functions prefixed with `extract`
//! parse compressed buffers to retrieve indices and coefficients, enabling analysis or
//! transformation of the compressed data. These utilities support the construction of advanced
//! compression pipelines with TerseTS. However, misuse of these primitives can lead to unexpected
//! loss of information, such as when inputs are malformed, corrupted, or do not adhere to the
//! expected representation. To mitigate this problem, each `extract` function checks whether the
//! input matches the expected representation of the corresponding compression method, but users
//! must ensure that the data provided is semantically valid and consistent.

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

/// Extracts `indices` and `coefficients` from `compressed_values` that follow a
/// coefficient–end-index pair encoding `(coefficient: f64, end_index: u64)`.
/// The `indices` `ArrayList` is used to store the extracted end indices, and the
/// `coefficients` `ArrayList` is used to store the extracted coefficient values.
/// Therefore, the input must contain an even number of components (a multiple of
/// 16 bytes). On success, the function appends coefficients to `coefficients` and
/// end indices to `indices`. If the buffer does not match the expected layout,
/// `Error.CorruptedCompressedData` is returned. Allocation errors are propagated.
pub fn extractCoefficientIndexPairs(
    allocator: Allocator,
    compressed_values: []const u8,
    indices: *ArrayList(u64),
    coefficients: *ArrayList(f64),
) Error!void {
    // Validate input lengths: the representation consists of repeating pairs of
    // (f64 coefficient, f64 bit-cast of u64 end_index). Each pair is 16 bytes.
    if (compressed_values.len % 16 != 0) return Error.CorruptedCompressedData;
    const components = mem.bytesAsSlice(f64, compressed_values);

    // Pass through the components, extracting coefficients and indices.
    for (0..components.len) |i| {
        if (i % 2 == 0) {
            const coefficient = components[i];
            try coefficients.append(allocator, coefficient);
        } else {
            const timestamp = components[i];
            const end_index: u64 = @bitCast(timestamp);
            try indices.append(allocator, end_index);
        }
    }
}

/// Extracts `indices` and `coefficients` from `compressed_values` that follow a
/// coefficient–end-index encoding starting with a leading coefficient. This means
/// that the first compressed value is a coefficient that must be extracted, while
/// the remaining compressed values follow `(coefficient: f64, end_index: u64)`
/// pairs. The `indices` `ArrayList` is used to store the extracted end indices, and
/// the `coefficients` `ArrayList` is used to store the extracted coefficient values.
/// Any inconsistency or loss of information in the extracted indices may result in
/// errors when decompressing the reconstructed representation. If validation of
/// `compressed_values` fails, `Error.CorruptedCompressedData` is returned.
/// Allocation errors are propagated.
pub fn extractCoefficientIndexTuplesWithStartCoefficient(
    allocator: Allocator,
    compressed_values: []const u8,
    indices: *ArrayList(u64),
    coefficients: *ArrayList(f64),
) Error!void {
    // Validate input lengths: first value is coefficient,
    // then alternating coefficient and timestamp.
    if ((compressed_values.len - 8) % 16 != 0) return Error.CorruptedCompressedData;
    const components = mem.bytesAsSlice(f64, compressed_values);
    for (0..components.len) |i| {
        // Coefficients are at even indices (0, 1, 3, 5, ...),
        if ((i == 0) or (i % 2 == 1)) {
            const coefficient = components[i];
            try coefficients.append(allocator, coefficient);
        } else {
            const timestamp = components[i];
            const end_index: u64 = @bitCast(timestamp);
            try indices.append(allocator, end_index);
        }
    }
}

/// Extracts `indices` and `coefficients` from `compressed_values`.
/// The encoding follows repeating triples of `(coefficient: f64,
/// coefficient: f64, timestamp: f64)`. Every third `f64` value is interpreted
/// as a timestamp, while the remaining values are interpreted as coefficients.
/// Any loss of information in the extracted indices, including invalid bit
/// patterns or mismatched segment boundaries, may result in errors during
/// decompression. If the buffer length is not a multiple of 24 bytes, the
/// function returns `Error.CorruptedCompressedData`. The `allocator` handles
/// the memory allocations of the output arrays. Allocation errors are propagated.
pub fn extractDoubleCoefficientIndexTriples(
    allocator: Allocator,
    compressed_values: []const u8,
    indices: *ArrayList(u64),
    coefficients: *ArrayList(f64),
) Error!void {
    // Validate input lengths: the `compressed_values` contain every third
    // value as a timestamp, others are coefficients.
    if (compressed_values.len % 24 != 0) return Error.CorruptedCompressedData;
    const components = mem.bytesAsSlice(f64, compressed_values);
    for (0..components.len) |i| {
        if ((i + 1) % 3 != 0) {
            const coefficient = components[i];
            try coefficients.append(allocator, coefficient);
        } else {
            const time = components[i];
            const end_index: u64 = @bitCast(time);
            try indices.append(allocator, end_index);
        }
    }
}
