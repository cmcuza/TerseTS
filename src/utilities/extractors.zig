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

//! This file provides utilities for extracting and reconstructing compressed representations
//! produced by TerseTS compression methods. Functions prefixed with `extract` parse compressed
//! buffers to retrieve timestamps and coefficients, enabling analysis or transformation of the
//! compressed data. Functions prefixed with `rebuild` recreate compressed buffers from extracted
//! components (i.e., timestamps and coefficients) that allow users to customize and reassemble
//! compression pipelines. These utilities support the construction of advanced compression pipelines
//! with TerseTS. However, misuse of these primitives can lead to unexpected loss of information,
//! such as when inputs are malformed, corrupted, or do not adhere to the expected representation.
//! To mitigate this problem, each `extract` and `rebuild` function checks whether the input matches
//! the expected representation of the corresponding compression method, but users must ensure that the
//! data provided is semantically valid and consistent.

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

/// Extracts `timestamps` and `coefficients` from Poor Man's Compression (PMC)'s
/// `compressed_values`. The function works for both PMCMidrange and PMCMean.
/// A `timestamps` ArrayList is used to store the extracted end indices, and a
/// `coefficients` ArrayList is used to store the extracted coefficient values.
/// If validation of the `compressed_values` fails, `Error.CorruptedCompressedData` is
/// returned. The `allocator` handles the memory allocations of the output arrays.
/// Any memory allocation error is propagated to the caller.
pub fn extractPMC(
    allocator: Allocator,
    compressed_values: []const u8,
    timestamps: *ArrayList(u64),
    coefficients: *ArrayList(f64),
) Error!void {
    // Validate input lengths: The PMC's compressed representation consists of pairs of
    // (f64 value, f64 bit-cast of u64 end_index). Each pair is 16 bytes (two f64).
    if (compressed_values.len % 16 != 0) return Error.CorruptedCompressedData;
    const components = mem.bytesAsSlice(f64, compressed_values);

    // Pass through the components, extracting coefficients and timestamps.
    for (0..components.len) |i| {
        if (i % 2 == 0) {
            const coefficient = components[i];
            try coefficients.append(allocator, coefficient);
        } else {
            const timestamp = components[i];
            const end_index: u64 = @bitCast(timestamp);
            try timestamps.append(allocator, end_index);
        }
    }
}

/// Rebuilds Poor Man's Compression (PMC) `compressed_values` from the provided
/// `timestamps` and `coefficients`. The function expects both arrays to have
/// equal length. Each pair is encoded as an f64 coefficient and a u64 end_index
/// taken from the `coefficients` and `timestamps` arrays, respectively. Any
/// mismatch or loss of information in the timestamps can lead to failures when
/// decompressing the rebuilt representation. The `allocator` handles the memory
/// allocations of the output arrays. Returns `Error.CorruptedCompressedData`
/// if the array lengths differ, and propagates allocation errors otherwise.
pub fn rebuildPMC(
    allocator: Allocator,
    timestamps: []const u64,
    coefficients: []const f64,
    compressed_values: *ArrayList(u8),
) Error!void {
    // Validate input lengths: they must be equal.
    if (timestamps.len != coefficients.len) return Error.CorruptedCompressedData;

    // Each pair is 16 bytes. Ensure the total capacity once.
    try compressed_values.ensureTotalCapacity(allocator, coefficients.len * 16);

    const total_length = coefficients.len + timestamps.len;
    var timestamp_index: u64 = 0;
    var coefficient_index: u64 = 0;
    for (0..total_length) |i| {
        if (i % 2 == 0) {
            const coefficient = coefficients[coefficient_index];
            try shared_functions.appendValue(allocator, f64, coefficient, compressed_values);
            coefficient_index += 1;
        } else {
            const time = timestamps[timestamp_index];
            try shared_functions.appendValue(allocator, u64, time, compressed_values);
            timestamp_index += 1;
        }
    }
}

/// Extracts `timestamps` and coefficients from SwingFilter's `compressed_values`. A `timestamps`
/// ArrayList is used to store the extracted end indices, and a `coefficients` ArrayList is used
/// to store the coefficient values. Swing encodes a sequence of triples:
/// (f64 coefficient, f64 coefficient, u64 end_index). Any inconsistency or loss of information
/// in the `timestamps` may result in errors when decompressing the reconstructed representation.
/// If validation of the `compressed_values` fails, `Error.CorruptedCompressedData` is returned.
/// The `allocator` handles the memory allocations of the output arrays. Any memory allocation
/// error is propagated to the caller.
pub fn extractSwing(
    allocator: Allocator,
    compressed_values: []const u8,
    timestamps: *ArrayList(u64),
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
            try timestamps.append(allocator, end_index);
        }
    }
}

/// Rebuilds SwingFilter's `compressed_values` from the provided `timestamps` and
/// `coefficients`. SwingFilter encodes a leading coefficient, followed by
/// alternating coefficient and timestamp values. The function therefore expects
/// `coefficients.len == timestamps.len + 1`. Any deviation from this representation
/// results in `Error.CorruptedCompressedData`. Any loss of information on the timestamps,
/// such as incorrect ordering or corrupted end indices, can lead to unexpected
/// failures during decompression. Only basic structural validation is performed.
/// Semantic consistency must be ensured by the caller. The `allocator` handles the
/// memory allocations of the output arrays. Allocation errors are propagated to the caller.
pub fn rebuildSwing(
    allocator: Allocator,
    timestamps: []const u64,
    coefficients: []const f64,
    compressed_values: *ArrayList(u8),
) Error!void {
    // Validate input lengths: coefficients must have at least one element,
    // and `timestamps` must have one less element than coefficients.
    if (coefficients.len == 0 or coefficients.len != timestamps.len + 1) {
        return Error.CorruptedCompressedData;
    }

    // Each pair is 16 bytes (two f64). Reserve once.
    try compressed_values.ensureTotalCapacity(allocator, coefficients.len * 16);

    const total_len = coefficients.len + timestamps.len;
    var timestamp_index: u64 = 0;
    var coefficient_index: u64 = 0;
    for (0..total_len) |i| {
        // Coefficients are at even indices (0, 1, 3, 5, ...),
        if ((i == 0) or (i % 2 == 1)) {
            const coefficient = coefficients[coefficient_index];
            try shared_functions.appendValue(allocator, f64, coefficient, compressed_values);
            coefficient_index += 1;
        } else {
            const time = timestamps[timestamp_index];
            try shared_functions.appendValue(allocator, u64, time, compressed_values);
            timestamp_index += 1;
        }
    }
}

/// Extracts `timestamps` and `coefficients` from SlideFilter's `compressed_values`.
/// This function also applies to SwingFilterDisconnected, which follows the
/// same encoding representation. SlideFilter encodes repeating triples of
/// (coefficient, coefficient, timestamp). Every third f64 value is interpreted
/// as a timestamp, while the other two are coefficients. Any loss of information
/// on the timestamps, including invalid bit patterns or mismatched segment boundaries,
/// can lead to unexpected failures during decompression. If the buffer length is
/// not a multiple of 24 bytes, the function returns `Error.CorruptedCompressedData`.
/// The `allocator` handles the memory allocations of the output arrays.
/// Allocation errors are propagated.
pub fn extractSlide(
    allocator: Allocator,
    compressed_values: []const u8,
    timestamps: *ArrayList(u64),
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
            try timestamps.append(allocator, end_index);
        }
    }
}

/// Rebuilds SlideFilter's `compressed_values` from the provided `timestamps` and `coefficients`.
/// SlideFilter uses a fixed representation of two coefficients followed by one timestamp, repeating
/// this pattern. Thus, the function expects `coefficients.len == timestamps.len * 2`. If the input
/// does not satisfy this requirement, `Error.CorruptedCompressedData` is returned. Any loss of
/// information on the timestamps, such as incorrect alignment or modified end indices, may lead to
/// unexpected failures during decompression. Only structural checks are performed. The caller is
/// responsible for keeping the inputs semantically valid. The `allocator` handles the memory
/// of the output arrays. Allocation errors are propagated.
pub fn rebuildSlide(
    allocator: Allocator,
    timestamps: []const u64,
    coefficients: []const f64,
    compressed_values: *ArrayList(u8),
) Error!void {
    // Validate input lengths: each pair is 16 bytes (two f64). Reserve once.
    if (coefficients.len != timestamps.len * 2) {
        return Error.CorruptedCompressedData;
    }

    try compressed_values.ensureTotalCapacity(allocator, coefficients.len * 24);

    const total_len = coefficients.len + timestamps.len;
    var timestamp_index: u64 = 0;
    var coefficient_index: u64 = 0;
    for (0..total_len) |i| {
        if ((i + 1) % 3 != 0) {
            const coefficient = coefficients[coefficient_index];
            try shared_functions.appendValue(allocator, f64, coefficient, compressed_values);
            coefficient_index += 1;
        } else {
            const timestamp = timestamps[timestamp_index];
            try shared_functions.appendValue(allocator, u64, timestamp, compressed_values);
            timestamp_index += 1;
        }
    }
}

/// Extracts `timestamps` and `coefficients` from SimPiece's `compressed_values`.
/// The representation contains nested blocks of intercepts, slopes, timestamp counts,
/// and delta values of the form:
/// [intercept, slopes_count, (slope, index_count, deltas...)..., last_index].
/// A `timestamps` ArrayList is used to store extracted counts and deltas, while a `coefficients`
/// ArrayList stores intercepts and slopes. The structure is variable in length and must follow the
/// encoding defined in SimPiece code. Any loss of information on the timestamps, including
/// corrupted counts or altered delta values, can lead to unexpected failures when attempting to
/// decompress the resulting representation. The function performs only basic structural checks;
/// semantic correctness must be ensured by the caller. The final element of the payload must be a
/// trailing `last_timestamp`. If the buffer ends prematurely, or the structural expectations are
/// not satisfied, `Error.CorruptedCompressedData` is returned. The `allocator` handles the memory
/// of the output arrays. Allocation errors are propagated.
pub fn extractSimPiece(
    allocator: Allocator,
    compressed_values: []const u8,
    timestamps: *ArrayList(u64),
    coefficients: *ArrayList(f64),
) Error!void {

    // Validate input lengths: must contain at least one intercept and one last_timestamp.
    if (compressed_values.len < 16)
        return Error.CorruptedCompressedData;

    const compressed_values_slice = mem.bytesAsSlice(f64, compressed_values);

    var i: u64 = 0;
    while (i < compressed_values_slice.len - 1) {
        // intercept.
        try coefficients.append(allocator, compressed_values_slice[i]);
        i += 1;

        // slopes_count.
        try ensureIndexWithinLength(i, compressed_values_slice.len);
        const slopes_count: u64 = @bitCast(compressed_values_slice[i]);
        try timestamps.append(allocator, slopes_count);
        i += 1;

        // Process each slope block.
        var slope_i: u64 = 0;
        while (slope_i < slopes_count) : (slope_i += 1) {
            // slope.
            try ensureIndexWithinLength(i, compressed_values_slice.len);
            try coefficients.append(allocator, compressed_values_slice[i]);
            i += 1;

            // timestamps_count.
            try ensureIndexWithinLength(i, compressed_values_slice.len);
            const timestamps_count: u64 = @bitCast(compressed_values_slice[i]);
            try timestamps.append(allocator, timestamps_count);
            i += 1;

            // deltas.
            var time_i: u64 = 0;
            while (time_i < timestamps_count) : (time_i += 1) {
                try ensureIndexWithinLength(i, compressed_values_slice.len);
                const delta: u64 = @bitCast(compressed_values_slice[i]);
                try timestamps.append(allocator, delta);
                i += 1;
            }
        }
    }

    // Final timestamp must exist.
    if (i != compressed_values_slice.len - 1)
        return Error.CorruptedCompressedData;

    const last_timestamp: u64 = @bitCast(compressed_values_slice[i]);
    try timestamps.append(allocator, last_timestamp);
}

/// Rebuilds SimPiece's `compressed_values` from the provided `timestamps` and `coefficients`.
/// The structure is nested and variable in length, thus no simple extructural checking conditions.
/// The function reconstructs the encoded stream by iterating through both arrays in the expected
/// order. Any loss of information on the timestamps, such as incorrect counts or misaligned delta
/// values, can lead to unexpected failures during decompression. Only structural validation is
/// performed. It is the caller's responsibility to ensure the logical consistency of all fields.
/// A single trailing `last_timestamp` must remain after processing all nested blocks. Extra or
/// missing elements in either array result in `Error.CorruptedCompressedData`. The `allocator`
/// handles the memory allocations of the output arrays. Allocation errors are propagated.
pub fn rebuildSimPiece(
    allocator: Allocator,
    timestamps: []const u64,
    coefficients: []const f64,
    compressed_values: *ArrayList(u8),
) Error!void {
    // Validate input lengths: at least the final last_timestamp.
    if (timestamps.len == 0) return Error.CorruptedCompressedData;

    var coefficient_index: u64 = 0; // index into coefficients.
    var timestamps_index: u64 = 0; // index into timestamps.

    // For each intercept.
    while (coefficient_index < coefficients.len) {
        // Extract intercept.
        try shared_functions.appendValue(allocator, f64, coefficients[coefficient_index], compressed_values);
        coefficient_index += 1;

        if (timestamps_index >= timestamps.len) return Error.CorruptedCompressedData;
        const slopes_count = timestamps[timestamps_index];
        try shared_functions.appendValue(allocator, u64, slopes_count, compressed_values);
        timestamps_index += 1;

        // For each slope on this intercept.
        var slope_i: u64 = 0;
        while (slope_i < slopes_count) : (slope_i += 1) {
            if (coefficient_index >= coefficients.len) return Error.CorruptedCompressedData;
            // Extract slope.
            try shared_functions.appendValue(allocator, f64, coefficients[coefficient_index], compressed_values);
            coefficient_index += 1;

            if (timestamps_index >= timestamps.len) return Error.CorruptedCompressedData;
            const timestamps_count = timestamps[timestamps_index];
            try shared_functions.appendValue(allocator, u64, timestamps_count, compressed_values);
            timestamps_index += 1;

            // Extract deltas.
            var time_i: u64 = 0;
            while (time_i < timestamps_count) : (time_i += 1) {
                if (timestamps_index >= timestamps.len) return Error.CorruptedCompressedData;
                const delta = timestamps[timestamps_index];
                try shared_functions.appendValue(allocator, u64, delta, compressed_values);
                timestamps_index += 1;
            }
        }
    }

    // Must have exactly one trailing last_timestamp remaining.
    if (timestamps_index >= timestamps.len) return Error.CorruptedCompressedData;
    const last_timestamp = timestamps[timestamps_index];
    try shared_functions.appendValue(allocator, u64, last_timestamp, compressed_values);
    timestamps_index += 1;

    // No extra data should remain.
    if (timestamps_index != timestamps.len) return Error.CorruptedCompressedData;
}

/// Extracts `timestamps` and `coefficients` from MixPiece's `compressed_values`. MixPiece consists
/// of three parts, each beginning with a header count: (part1_count, part2_count, part3_count).
/// These counts determine how many variable-length groups follow in each part. A `timestamps`
/// ArrayList stores all counts and deltas, while a `coefficients` ArrayList stores intercepts
/// and slopes extracted from the representation. Because MixPiece contains several nested and
/// length-dependent structures, any loss of information on the timestamps, including incorrect
/// group counts or delta values, can lead to unexpected failures during decompression. Only
/// lightweight structural checks are performed. Deeper validation must be ensured by the caller.
/// The required details of the encoding are documented in `src/functional_approximation/mix_piece.zig`.
/// If the buffer does not match the expected header or runs out of data mid-structure, the function
/// returns `Error.CorruptedCompressedData`. The `allocator` handles the memory allocations of the
/// output arrays. Allocation errors are propagated.
pub fn extractMixPiece(
    allocator: Allocator,
    compressed_values: []const u8,
    timestamps: *ArrayList(u64),
    coefficients: *ArrayList(f64),
) Error!void {
    // Compressed representation must contain at least the 3 header fields.
    var offset: u64 = 3 * @sizeOf(u64);
    if (compressed_values.len < offset)
        return Error.CorruptedCompressedData;

    const header = mem.bytesAsSlice(u64, compressed_values[0..offset]);
    const part1_count = header[0];
    const part2_count = header[1];
    const part3_count = header[2];

    try timestamps.append(allocator, part1_count);
    try timestamps.append(allocator, part2_count);
    try timestamps.append(allocator, part3_count);

    // Mix-Piece Phase 3: Populate the three data structures. Separately handle three parts.
    // Part 1: segment groups that share the same intercept.
    // Part 2: segment groups that don't share the same intercept but the same slope.
    // Part 3: segment unmerged segments.

    // Part 1: Intercept groups.
    for (0..part1_count) |_| {
        try ensureEnoughBytesAreAvailable(compressed_values, offset, @sizeOf(f64));
        const intercept = try shared_functions.readOffsetValue(f64, compressed_values, &offset);
        try coefficients.append(allocator, intercept);

        try ensureEnoughBytesAreAvailable(compressed_values, offset, @sizeOf(u64));
        const slopes_count = try shared_functions.readOffsetValue(u64, compressed_values, &offset);
        try timestamps.append(allocator, slopes_count);

        for (0..slopes_count) |_| {
            try ensureEnoughBytesAreAvailable(compressed_values, offset, @sizeOf(f64));
            const slope = try shared_functions.readOffsetValue(f64, compressed_values, &offset);
            try coefficients.append(allocator, slope);

            try ensureEnoughBytesAreAvailable(compressed_values, offset, @sizeOf(u64));
            const timestamps_count = try shared_functions.readOffsetValue(u64, compressed_values, &offset);
            try timestamps.append(allocator, timestamps_count);

            for (0..timestamps_count) |_| {
                try ensureEnoughBytesAreAvailable(compressed_values, offset, @sizeOf(u64));
                const delta = try shared_functions.readOffsetValue(u64, compressed_values, &offset);
                try timestamps.append(allocator, delta);
            }
        }
    }

    // Part 2: Slope groups.
    for (0..part2_count) |_| {
        try ensureEnoughBytesAreAvailable(compressed_values, offset, @sizeOf(f64));
        const slope = try shared_functions.readOffsetValue(f64, compressed_values, &offset);
        try coefficients.append(allocator, slope);

        try ensureEnoughBytesAreAvailable(compressed_values, offset, @sizeOf(u64));
        const pair_count = try shared_functions.readOffsetValue(u64, compressed_values, &offset);
        try timestamps.append(allocator, pair_count);

        for (0..pair_count) |_| {
            try ensureEnoughBytesAreAvailable(compressed_values, offset, @sizeOf(f64));
            const intercept = try shared_functions.readOffsetValue(f64, compressed_values, &offset);
            try coefficients.append(allocator, intercept);

            try ensureEnoughBytesAreAvailable(compressed_values, offset, @sizeOf(u64));
            const delta = try shared_functions.readOffsetValue(u64, compressed_values, &offset);
            try timestamps.append(allocator, delta);
        }
    }

    // Part 3: Ungrouped segments.
    for (0..part3_count) |_| {
        try ensureEnoughBytesAreAvailable(compressed_values, offset, @sizeOf(f64) * 2 + @sizeOf(u64));

        const slope = try shared_functions.readOffsetValue(f64, compressed_values, &offset);
        try coefficients.append(allocator, slope);

        const intercept = try shared_functions.readOffsetValue(f64, compressed_values, &offset);
        try coefficients.append(allocator, intercept);

        const delta = try shared_functions.readOffsetValue(u64, compressed_values, &offset);
        try timestamps.append(allocator, delta);
    }

    // Final timestamp.
    try ensureEnoughBytesAreAvailable(compressed_values, offset, @sizeOf(u64));
    const final_timestamp = try shared_functions.readOffsetValue(u64, compressed_values, &offset);
    try timestamps.append(allocator, final_timestamp);

    // No extra data should remain.
    if (offset != compressed_values.len)
        return Error.CorruptedCompressedData;
}

/// Rebuilds MixPiece's `compressed_values` from the provided `timestamps` and `coefficients`.
/// The `timestamps` array must begin with the three part counts:
/// [part1_count, part2_count, part3_count].
/// Based on these counts, the function reconstructs the variable-length groups for each part
/// by consuming the arrays in the order required by MixPiece. Any loss of information on the
/// timestamps, including corrupted group counts or mismatched delta sequences, may cause unexpected
/// failures during decompression. The function assumes the input arrays are logically consistent
/// and performs only structural validation. If the reconstructed stream does not consume all
/// required coefficients or timestamps, or if extra data remains after processing a part, the
/// function returns `Error.UnsupportedInput`. The `allocator` handles the memory allocations of
/// the output arrays. Allocation errors are propagated.
pub fn rebuildMixPiece(
    allocator: Allocator,
    timestamps: []const u64,
    coefficients: []const f64,
    compressed_values: *ArrayList(u8),
) Error!void {
    // Must contain at least the three part counts.
    if (timestamps.len < 3)
        return Error.CorruptedCompressedData;

    const part1_count = timestamps[0];
    const part2_count = timestamps[1];
    const part3_count = timestamps[2];

    try shared_functions.appendValue(allocator, u64, part1_count, compressed_values);
    try shared_functions.appendValue(allocator, u64, part2_count, compressed_values);
    try shared_functions.appendValue(allocator, u64, part3_count, compressed_values);

    var timestamps_index: u64 = 3;
    var coefficients_index: u64 = 0;

    // Part 1: Intercept groups.
    for (0..part1_count) |_| {
        // intercept.
        try ensureIndexWithinLength(coefficients_index, coefficients.len);
        const intercept = coefficients[coefficients_index];
        try shared_functions.appendValue(allocator, f64, intercept, compressed_values);
        coefficients_index += 1;

        // slopes_count.
        try ensureIndexWithinLength(timestamps_index, timestamps.len);
        const slopes_count = timestamps[timestamps_index];
        try shared_functions.appendValue(allocator, u64, slopes_count, compressed_values);
        timestamps_index += 1;

        // slope groups.
        for (0..slopes_count) |_| {
            // slope.
            try ensureIndexWithinLength(coefficients_index, coefficients.len);
            const slope = coefficients[coefficients_index];
            try shared_functions.appendValue(allocator, f64, slope, compressed_values);
            coefficients_index += 1;

            // timestamps_count.
            try ensureIndexWithinLength(timestamps_index, timestamps.len);
            const timestamps_count = timestamps[timestamps_index];
            try shared_functions.appendValue(allocator, u64, timestamps_count, compressed_values);
            timestamps_index += 1;

            // deltas.
            for (0..timestamps_count) |_| {
                try ensureIndexWithinLength(timestamps_index, timestamps.len);
                const delta = timestamps[timestamps_index];
                try shared_functions.appendValue(allocator, u64, delta, compressed_values);
                timestamps_index += 1;
            }
        }
    }

    // Part 2: Slope groups.
    for (0..part2_count) |_| {
        // slope.
        try ensureIndexWithinLength(coefficients_index, coefficients.len);
        const slope = coefficients[coefficients_index];
        try shared_functions.appendValue(allocator, f64, slope, compressed_values);
        coefficients_index += 1;

        // pair_count.
        try ensureIndexWithinLength(timestamps_index, timestamps.len);
        const pair_count = timestamps[timestamps_index];
        try shared_functions.appendValue(allocator, u64, pair_count, compressed_values);
        timestamps_index += 1;

        // (intercept, delta) pairs.
        for (0..pair_count) |_| {
            // intercept.
            try ensureIndexWithinLength(coefficients_index, coefficients.len);
            const intercept = coefficients[coefficients_index];
            try shared_functions.appendValue(allocator, f64, intercept, compressed_values);
            coefficients_index += 1;

            // delta.
            try ensureIndexWithinLength(timestamps_index, timestamps.len);
            const delta = timestamps[timestamps_index];
            try shared_functions.appendValue(allocator, u64, delta, compressed_values);
            timestamps_index += 1;
        }
    }

    // Part 3: Ungrouped segments.
    for (0..part3_count) |_| {
        // slope.
        try ensureIndexWithinLength(coefficients_index, coefficients.len);
        const slope = coefficients[coefficients_index];
        try shared_functions.appendValue(allocator, f64, slope, compressed_values);
        coefficients_index += 1;

        // intercept.
        try ensureIndexWithinLength(coefficients_index, coefficients.len);
        const intercept = coefficients[coefficients_index];
        try shared_functions.appendValue(allocator, f64, intercept, compressed_values);
        coefficients_index += 1;

        // delta.
        try ensureIndexWithinLength(timestamps_index, timestamps.len);
        const delta = timestamps[timestamps_index];
        try shared_functions.appendValue(allocator, u64, delta, compressed_values);
        timestamps_index += 1;
    }

    // Final timestamp.
    try ensureIndexWithinLength(timestamps_index, timestamps.len);
    const final_timestamp = timestamps[timestamps_index];
    try shared_functions.appendValue(allocator, u64, final_timestamp, compressed_values);
    timestamps_index += 1;

    // No extra timestamps allowed.
    if (timestamps_index != timestamps.len)
        return Error.CorruptedCompressedData;

    // No extra coefficients allowed.
    if (coefficients_index != coefficients.len)
        return Error.CorruptedCompressedData;
}

/// Extracts `timestamps` and `coefficients` from NonLinearApproximation's `compressed_values`.
/// The `compressed_values` encodes: a shift amount (f64), the number of segments (u64), packed
/// function-type codes (two per byte), for each segment: (slope: f64, intercept: f64, end_index: u64).
/// A `timestamps` ArrayList stores the number of segments, all function type codes, and the end
/// indices. A `coefficients` ArrayList stores the shift amount and the per-segment (slope, intercept)
/// values. Any loss of information on the timestamps, for example, incorrect function codes,
/// wrong segment count, corrupted end indices, can lead to unexpected failures during decompression.
/// Only structural checks are performed. The caller must ensure semantic validity. If the compressed
/// stream does not follow the expected representation, `Error.CorruptedCompressedData` is returned.
/// The `allocator` handles the memory allocations of the output arrays. Allocation errors are propagated.
pub fn extractNonLinearApproximation(
    allocator: Allocator,
    compressed_values: []const u8,
    timestamps: *ArrayList(u64),
    coefficients: *ArrayList(f64),
) Error!void {
    // Must contain at least shift_amount (f64) and number_of_segments (u64).
    if (compressed_values.len < @sizeOf(f64) + @sizeOf(u64))
        return Error.CorruptedCompressedData;

    var offset: u64 = 0;

    // Read shift amount.
    const shift_amount =
        try shared_functions.readOffsetValue(f64, compressed_values, &offset);

    // Read number of segments.
    const number_of_segments: u64 =
        try shared_functions.readOffsetValue(u64, compressed_values, &offset);

    // Insert into output.
    try coefficients.append(allocator, shift_amount);
    try timestamps.append(allocator, number_of_segments);

    // Number of bytes containing packed 4-bit type codes.
    const type_bytes_len: u64 = (number_of_segments + 1) / 2;

    // Each segment has slope, intercept, end_index.
    const bytes_per_segment =
        @sizeOf(f64) * 2 + @sizeOf(u64);

    // Entire payload must match exactly the expected size.
    const expected_total_bytes =
        @sizeOf(f64) + @sizeOf(u64) + type_bytes_len + number_of_segments * bytes_per_segment;

    if (compressed_values.len != expected_total_bytes)
        return Error.CorruptedCompressedData;

    // Ensure availability of packed type data.
    if (offset + type_bytes_len > compressed_values.len)
        return Error.CorruptedCompressedData;

    const packed_function_types =
        compressed_values[offset .. offset + type_bytes_len];
    offset += type_bytes_len;

    // Unpack function types.
    for (0..number_of_segments) |segment_index| {
        const packed_information = packed_function_types[segment_index / 2];

        const code: u4 = if (segment_index % 2 != 0)
            @truncate(packed_information & 0x0F)
        else
            @truncate((packed_information >> 4) & 0x0F);

        try timestamps.append(allocator, @intCast(code));
    }

    // Read the per-segment (slope, intercept, end_index).
    for (0..number_of_segments) |_| {
        // Prevent buffer overrun.
        if (offset + bytes_per_segment > compressed_values.len)
            return Error.CorruptedCompressedData;

        const slope =
            try shared_functions.readOffsetValue(f64, compressed_values, &offset);
        const intercept =
            try shared_functions.readOffsetValue(f64, compressed_values, &offset);
        const end_index =
            try shared_functions.readOffsetValue(u64, compressed_values, &offset);

        try coefficients.append(allocator, slope);
        try coefficients.append(allocator, intercept);
        try timestamps.append(allocator, end_index);
    }

    // If offset does not end exactly here, data was malformed.
    if (offset != compressed_values.len)
        return Error.CorruptedCompressedData;
}

/// Rebuilds NonLinearApproximation's `compressed_values` from the given `timestamps` and
/// `coefficients`. The encoding consists of shift_amount (coefficients[0]), number_of_segments
/// (timestamps[0]), packed function types (timestamps[1 .. number_of_segments]), for each segment:
/// slope, intercept, end_index. Any loss or misalignment of timestamp information, for example,
/// incorrect function type, wrong segment count, missing end_index, can lead to failures during
/// decompression. The function checks for structural consistency and returns
/// `Error.CorruptedCompressedData` for malformed input. The `allocator` handles the memory
/// allocations of the output arrays. Allocation errors are propagated.
pub fn rebuildNonLinearApproximation(
    allocator: mem.Allocator,
    timestamps: []const u64,
    coefficients: []const f64,
    compressed_values: *ArrayList(u8),
) Error!void {
    // Must contain at least shift_amount and number_of_segments.
    if (timestamps.len < 1 or coefficients.len < 1)
        return Error.CorruptedCompressedData;

    // coefficients and timestamps must have same length:
    //   coefficients = [shift_amount, slope0, intercept0, slope1, intercept1, ...]
    //   timestamps   = [number_of_segments, type0, type1, ..., end0, end1, ...]
    if (coefficients.len != timestamps.len)
        return Error.CorruptedCompressedData;

    // shift_amount.
    try shared_functions.appendValue(allocator, f64, coefficients[0], compressed_values);

    // number_of_segments.
    const number_of_segments: u64 = timestamps[0];
    try shared_functions.appendValue(allocator, u64, number_of_segments, compressed_values);

    // There must be at least "number_of_segments" functios type.
    if (1 + number_of_segments > timestamps.len)
        return Error.CorruptedCompressedData;

    // Prepare the functions type packing.
    const packed_len = (number_of_segments + 1) / 2;
    var packed_function_types = try allocator.alloc(u8, packed_len);
    defer allocator.free(packed_function_types);
    // Allocate memory for packed functions type and initialize it to zero.
    // This enables the bitwise OR operations during packing.
    @memset(packed_function_types, 0);

    var coefficient_index: u64 = 1; // After shift_amount.
    var timestamps_index: u64 = 1; // After number_of_segments.

    // Pack the functions type.
    for (0..number_of_segments) |index| {
        const code_u64 = timestamps[timestamps_index];
        // Validate that the code fits in 4 bits.
        if (code_u64 > 0x0F)
            return Error.CorruptedCompressedData;

        const code: u8 = @intCast(code_u64);
        const byte_index = index / 2;
        const is_high = (index % 2) == 0;

        if (byte_index >= packed_len)
            return Error.CorruptedCompressedData;

        if (is_high) {
            packed_function_types[byte_index] |= (code << 4);
        } else {
            packed_function_types[byte_index] |= (code & 0x0F);
        }

        timestamps_index += 1;
    }

    try compressed_values.appendSlice(allocator, packed_function_types);

    // For each segment, we now need:
    //   coefficients: slope: f64, intercept: f64.
    //   timestamps:   end_index: u64.
    for (0..number_of_segments) |_| {
        // Must have slope and intercept available.
        if (coefficient_index + 1 >= coefficients.len)
            return Error.CorruptedCompressedData;

        const slope: f64 = coefficients[coefficient_index];
        const intercept: f64 = coefficients[coefficient_index + 1];

        try shared_functions.appendValue(allocator, f64, slope, compressed_values);
        try shared_functions.appendValue(allocator, f64, intercept, compressed_values);

        coefficient_index += 2;

        // Must have an end_index available.
        if (timestamps_index >= timestamps.len)
            return Error.CorruptedCompressedData;

        const end_index: u64 = timestamps[timestamps_index];
        try shared_functions.appendValue(allocator, u64, end_index, compressed_values);
        timestamps_index += 1;
    }

    // No extra timestamps allowed.
    if (timestamps_index != timestamps.len)
        return Error.CorruptedCompressedData;

    // No extra coefficients allowed.
    if (coefficient_index != coefficients.len)
        return Error.CorruptedCompressedData;
}

/// Extracts `timestamps` and `coefficients` from the Piecewise Constant Histogram's
/// `compressed_values`. The representation of PWCH is identical to that used by
/// Poor Man's Compression, so this function forwards its work to `extractPMC`. All structural
/// validation and corruption checks are handled internally by that routine. Any loss of
/// information on the timestamps can lead to unexpected failures during decompression.
/// The `allocator` handles the memory allocations of the output arrays. Allocation errors are propagated.
pub fn extractPWCH(
    allocator: Allocator,
    compressed_values: []const u8,
    timestamps: *ArrayList(u64),
    coefficients: *ArrayList(f64),
) Error!void {
    try extractPMC(allocator, compressed_values, timestamps, coefficients);
}

/// Rebuilds a Piecewise Constant Histogram representation from the provided `timestamps` and
/// `coefficients`. PWCH uses the same binary format as PMC, so this function forwards the work
/// to `rebuildPMC`. All structural and corruption checks are performed by the underlying function.
/// Any loss or misalignment of timestamp information can cause failures when decompressing
/// the rebuilt representation. The `allocator` handles the memory allocations of the output arrays.
/// Allocation errors are propagated.
pub fn rebuildPWCH(
    allocator: Allocator,
    timestamps: []const u64,
    coefficients: []const f64,
    compressed_values: *ArrayList(u8),
) Error!void {
    try rebuildPMC(allocator, timestamps, coefficients, compressed_values);
}

/// Extracts `timestamps` and `coefficients` from the Piecewise Linear Histogram's
/// `compressed_values`. PWLH uses the same triplet representation as SlideFilter, so this function
/// delegates to `extractSlide`. All validation and corruption detection handlesd by that routine.
/// Any loss of timestamp information may lead to unexpected failures during decompression.
/// The `allocator` handles the memory allocations of the output arrays. Allocation errors are propagated.
pub fn extractPWLH(
    allocator: Allocator,
    compressed_values: []const u8,
    timestamps: *ArrayList(u64),
    coefficients: *ArrayList(f64),
) Error!void {
    try extractSlide(allocator, compressed_values, timestamps, coefficients);
}

/// Rebuilds a Piecewise Linear Histogram representation from the provided `timestamps` and `coefficients`.
/// PWLH uses the SlideFilter representation, so this function forwards the work to `rebuildSlide`.
/// All correctness checks are performed internally by the delegated function. Any inconsistency in timestamp
/// counts or ordering may produce corrupted data that fails during decompression. The `allocator` handles
/// the memory of the output arrays. Allocation errors are propagated.
pub fn rebuildPWLH(
    allocator: Allocator,
    timestamps: []const u64,
    coefficients: []const f64,
    compressed_values: *ArrayList(u8),
) Error!void {
    try rebuildSlide(allocator, timestamps, coefficients, compressed_values);
}

/// Extracts `timestamps` and `coefficients` from Visvalingam-Whyatt's `compressed_values`. The binary
/// representation follows the same pattern as SwingFilter, so this function calls `extractSwing`.
/// All structural and corruption checks are performed by the delegated function. Any loss of
/// timestamp information can lead to failures during later decompression. The `allocator` handles
/// the memory of the output arrays. Allocation errors are propagated.
pub fn extractVisvalingamWhyatt(
    allocator: Allocator,
    compressed_values: []const u8,
    timestamps: *ArrayList(u64),
    coefficients: *ArrayList(f64),
) Error!void {
    try extractSwing(allocator, compressed_values, timestamps, coefficients);
}

/// Rebuilds Visvalingam-Whyatt's `compressed_values` from the provided `timestamps` and `coefficients`.
/// The representation matches SwingFilter, so the function delegates to `rebuildSwing`. All format
/// validation and corruption checks are performed by that routine. Any loss or misalignment of
/// timestamps may cause failures when decoding the rebuilt representation. The `allocator` handles
/// the memory of the output arrays. Allocation errors are propagated.
pub fn rebuildVisvalingamWhyatt(
    allocator: Allocator,
    timestamps: []const u64,
    coefficients: []const f64,
    compressed_values: *ArrayList(u8),
) Error!void {
    try rebuildSwing(allocator, timestamps, coefficients, compressed_values);
}

/// Extracts `timestamps` and `coefficients` from SlidingWindow's `compressed_values`. SlidingWindow
/// follows the same triplet representation as SlideFilter, so the function forwards to `extractSlide`.
/// All validation, including corruption detection, handlesd by that routine. Any loss of timestamp
/// information can lead to unexpected failures during decompression. The `allocator` handles the memory
/// of the output arrays. Allocation errors are propagated.
pub fn extractSlidingWindow(
    allocator: Allocator,
    compressed_values: []const u8,
    timestamps: *ArrayList(u64),
    coefficients: *ArrayList(f64),
) Error!void {
    try extractSlide(allocator, compressed_values, timestamps, coefficients);
}

/// Rebuilds SlidingWindow's `compressed_values` using the provided `timestamps` and `coefficients`.
/// The representation matches SlideFilter's representation, so this function delegates to
/// `rebuildSlide`. Structural and corruption checks are handled internally by the delegated function.
/// Incorrect or inconsistent timestamp information may cause decompression failures. The `allocator`
/// handles the memory allocations of the output arrays. Allocation errors are propagated.
pub fn rebuildSlindingWindow(
    allocator: Allocator,
    timestamps: []const u64,
    coefficients: []const f64,
    compressed_values: *ArrayList(u8),
) Error!void {
    try rebuildSlide(allocator, timestamps, coefficients, compressed_values);
}

/// Extracts `timestamps` and `coefficients` from BottomUp's `compressed_values`. BottomUp uses the
/// same three-value repeating representation as SlideFilter, so this function forwards to
/// `extractSlide`. All corruption checks and structural validation occur in that routine. Any loss
/// of information on timestamps can lead to failures when decoding. The `allocator` handles the memory
/// of the output arrays. Allocation errors are propagated.
pub fn extractBottomUp(
    allocator: Allocator,
    compressed_values: []const u8,
    timestamps: *ArrayList(u64),
    coefficients: *ArrayList(f64),
) Error!void {
    try extractSlide(allocator, compressed_values, timestamps, coefficients);
}

/// Rebuilds BottomUp's `compressed_values` from the provided `timestamps` and `coefficients`.
/// Because the format matches SlideFilter exactly, this wrapper forwards to `rebuildSlide`.
/// All format and corruption checks are performed internally. Incorrect or inconsistent
/// timestamps may produce corrupted output that cannot be decompressed. The `allocator`
/// handles the memory allocations of the output arrays. Allocation errors are propagated.
pub fn rebuildBottomUp(
    allocator: Allocator,
    timestamps: []const u64,
    coefficients: []const f64,
    compressed_values: *ArrayList(u8),
) Error!void {
    try rebuildSlide(allocator, timestamps, coefficients, compressed_values);
}

/// Extracts `timestamps` and `coefficients` from ConvexABC's `compressed_values`.
/// ConvexABC uses the same representation as SlideFilter, so this function delegates to
/// `extractSlide`. All corruption detection and validation checks are handled by
/// that routine. Any loss or modification of timestamp information can lead to
/// failures during decompression. Allocation errors are propagated.
pub fn extractABCLinearApproximation(
    allocator: Allocator,
    compressed_values: []const u8,
    timestamps: *ArrayList(u64),
    coefficients: *ArrayList(f64),
) Error!void {
    try extractSlide(allocator, compressed_values, timestamps, coefficients);
}

/// Rebuilds ConvexABC's `compressed_values` from the provided `timestamps` and
/// `coefficients`. The format matches SlideFilter, so this function forwards
/// to `rebuildSlide`. All structural validation and corruption checks occur
/// inside the delegated function. Any timestamp inconsistencies may cause failures
/// when decoding the rebuilt representation. Allocation errors are propagated.
pub fn rebuildABCLinearApproximation(
    allocator: Allocator,
    timestamps: []const u64,
    coefficients: []const f64,
    compressed_values: *ArrayList(u8),
) Error!void {
    try rebuildSlide(allocator, timestamps, coefficients, compressed_values);
}

/// Helper function to ensure that the `index` is within range of the `len`.
/// If not, returns `Error.CorruptedCompressedData`.
fn ensureIndexWithinLength(index: u64, len: u64) Error!void {
    if (index >= len) return Error.CorruptedCompressedData;
}

/// Helper function to ensure that the `offset` stays in-bounds of the `total_bytes_available`
/// when the `required_bytes` are read. If not, returns `Error.CorruptedCompressedData`.
fn ensureEnoughBytesAreAvailable(total_bytes_available: []const u8, offset: u64, required_bytes: u64) Error!void {
    if (offset + required_bytes > total_bytes_available.len)
        return Error.CorruptedCompressedData;
}

test "extract and rebuild works for any compression method supported" {
    const allocator = testing.allocator;
    const random = tester.getDefaultRandomGenerator();

    // Input data
    var uncompressed_values = ArrayList(f64).empty;
    defer uncompressed_values.deinit(allocator);
    try tester.generateBoundedRandomValues(
        allocator,
        &uncompressed_values,
        -100,
        100,
        random,
    );

    // Test each method.
    inline for (std.meta.fields(tersets.Method)) |method_field| {
        const method: tersets.Method = @enumFromInt(method_field.value);

        if (method == tersets.Method.BitPackedQuantization or
            method == tersets.Method.SerfQT or
            method == tersets.Method.RunLengthEncoding)
        {
            // These compression methods are not supported for extraction
            // of the coefficients and time indices. This is because even small
            // chages in the compressed representation can lead to large differences
            // or completely inconsistent decompressed values. For example, for
            // BitPackedQuantization, The decompression process relies on metadata
            // (e.g., min_val, bucket_size, and quantized indices) to reconstruct the
            // original values. If the coefficients are altered, the metadata no longer
            // aligns with the modified data, making it impossible to map the quantized
            // indices back to their original values. Finally, the bit-packing encodes
            // quantized values using a fixed-length scheme. If the coefficients are
            // modified, the bit-packed representation may no longer be valid, leading to
            // corrupted streams or misinterpretation of the data during decompression.
            // In case of RLE, modifying the coefficients can disrupt the run-length
            // encoding scheme, also leading to incorrect decompression results.
            continue;
        }

        const method_configuration = try configuration.defaultConfigurationBuilder(
            allocator,
            method,
        );
        defer allocator.free(method_configuration);

        var compressed_values = try tersets.compress(
            allocator,
            uncompressed_values.items,
            method,
            method_configuration,
        );
        defer compressed_values.deinit(allocator);

        var decompressed_values = try tersets.decompress(
            allocator,
            compressed_values.items,
        );
        defer decompressed_values.deinit(allocator);

        // Test extract and rebuild.
        var coefficient_values = ArrayList(f64).empty;
        defer coefficient_values.deinit(allocator);
        var timestamp_values = ArrayList(u64).empty;
        defer timestamp_values.deinit(allocator);

        try tersets.extract(
            allocator,
            compressed_values.items,
            &timestamp_values,
            &coefficient_values,
        );

        var rebuild_values = try tersets.rebuild(
            allocator,
            timestamp_values.items,
            coefficient_values.items,
            method,
        );
        defer rebuild_values.deinit(allocator);

        try testing.expectEqual(rebuild_values.items.len, compressed_values.items.len);
        try testing.expectEqualSlices(u8, rebuild_values.items, compressed_values.items);
    }
}

test "rebuildPMC rejects mismatched input lengths" {
    const allocator = testing.allocator;
    var compressed = ArrayList(u8).empty;
    defer compressed.deinit(allocator);

    const timestamps = [_]u64{1};
    const coefficients = [_]f64{ 1.0, 2.0 };

    try testing.expectError(Error.CorruptedCompressedData, rebuildPMC(
        allocator,
        timestamps[0..],
        coefficients[0..],
        &compressed,
    ));
}

test "rebuildSwing rejects mismatched input lengths" {
    const allocator = testing.allocator;
    var compressed = ArrayList(u8).empty;
    defer compressed.deinit(allocator);

    const timestamps = [_]u64{ 1, 2 };
    const coefficients = [_]f64{ 1.0, 2.0 };

    try testing.expectError(Error.CorruptedCompressedData, rebuildSwing(
        allocator,
        timestamps[0..],
        coefficients[0..],
        &compressed,
    ));
}

test "rebuildSlide rejects mismatched input lengths" {
    const allocator = testing.allocator;
    var compressed = ArrayList(u8).empty;
    defer compressed.deinit(allocator);

    const timestamps = [_]u64{ 1, 2, 3 };
    const coefficients = [_]f64{ 1.0, 2.0 };

    try testing.expectError(Error.CorruptedCompressedData, rebuildSlide(
        allocator,
        timestamps[0..],
        coefficients[0..],
        &compressed,
    ));
}

test "rebuildSimPiece rejects mismatched representation (too few timestamps)" {
    const allocator = testing.allocator;
    var compressed = ArrayList(u8).empty;
    defer compressed.deinit(allocator);

    // Minimal malformed input:
    // One intercept in coefficients but timestamps missing nested metadata.
    const coefficients = [_]f64{1.0}; // intercept only.
    const timestamps = [_]u64{0}; // slopes_count but missing final_timestamp.

    try testing.expectError(
        Error.CorruptedCompressedData,
        rebuildSimPiece(
            allocator,
            timestamps[0..],
            coefficients[0..],
            &compressed,
        ),
    );
}

test "rebuildSimPiece rejects mismatched representation (missing slope/intercept pairs)" {
    const allocator = testing.allocator;
    var compressed = ArrayList(u8).empty;
    defer compressed.deinit(allocator);

    // Only intercept is provided.
    const coefficients = [_]f64{1.0}; // missing slope.
    const timestamps = [_]u64{ 1, 0, 5 }; // slopes_count = 1, but no slope data.

    try testing.expectError(
        Error.CorruptedCompressedData,
        rebuildSimPiece(
            allocator,
            timestamps[0..],
            coefficients[0..],
            &compressed,
        ),
    );
}

test "rebuildMixPiece rejects malformed header-only input" {
    const allocator = testing.allocator;
    var compressed = ArrayList(u8).empty;
    defer compressed.deinit(allocator);

    // timestamps must contain at least 3 elements for header, but coefficients are missing.
    const timestamps = [_]u64{ 1, 0, 0 };
    const coefficients = [_]f64{}; // no intercept thus corrupted.

    try testing.expectError(
        Error.CorruptedCompressedData,
        rebuildMixPiece(allocator, timestamps[0..], coefficients[0..], &compressed),
    );
}

test "rebuildMixPiece rejects mismatched Part1 group structure" {
    // To fully understand this test, refer to the MixPiece implementation.
    const allocator = testing.allocator;
    var compressed = ArrayList(u8).empty;
    defer compressed.deinit(allocator);

    // part1_count = 1, but missing slope data.
    const timestamps = [_]u64{
        1, // part1_count.
        0, // part2_count.
        0, // part3_count.
        1, // slopes_count for group 0.
        0, // timestamps_count for slope (but missing deltas + final timestamp).
    };

    const coefficients = [_]f64{
        5.0, // intercept.
        // No slope thus the data is corrupted.
    };

    try testing.expectError(
        Error.CorruptedCompressedData,
        rebuildMixPiece(allocator, timestamps[0..], coefficients[0..], &compressed),
    );
}

test "rebuildMixPiece rejects missing final timestamp" {
    const allocator = testing.allocator;
    var compressed = ArrayList(u8).empty;
    defer compressed.deinit(allocator);

    const timestamps = [_]u64{
        0, 0, 0,
        // Missing final_timestamp.
    };

    const coefficients = [_]f64{}; // No coefficients needed for zero parts.
    try testing.expectError(
        Error.CorruptedCompressedData,
        rebuildMixPiece(allocator, timestamps[0..], coefficients[0..], &compressed),
    );
}

test "rebuildNonLinearApproximation rejects mismatched lengths" {
    const allocator = testing.allocator;
    var compressed = ArrayList(u8).empty;
    defer compressed.deinit(allocator);

    // number_of_segments = 1.
    const timestamps = [_]u64{
        1, // number_of_segments.
        2, // type code.
        // Missing end_index.
    };

    const coefficients = [_]f64{
        0.5, // shift.
        1.0, // slope.
        // Missing intercept.
    };

    try testing.expectError(
        Error.CorruptedCompressedData,
        rebuildNonLinearApproximation(allocator, timestamps[0..], coefficients[0..], &compressed),
    );
}

test "rebuildNonLinearApproximation rejects invalid function type code" {
    const allocator = testing.allocator;
    var compressed = ArrayList(u8).empty;
    defer compressed.deinit(allocator);

    const timestamps = [_]u64{
        1, // number_of_segments.
        999, // invalid > 0x0F thus must fail.
        5, // end_index.
    };

    const coefficients = [_]f64{
        0.0, // shift.
        1.0, // slope.
        2.0, // intercept.
    };

    try testing.expectError(
        Error.CorruptedCompressedData,
        rebuildNonLinearApproximation(allocator, timestamps[0..], coefficients[0..], &compressed),
    );
}

test "rebuildNonLinearApproximation rejects incomplete coefficient pair" {
    const allocator = testing.allocator;
    var compressed = ArrayList(u8).empty;
    defer compressed.deinit(allocator);

    const timestamps = [_]u64{
        1, // number_of_segments.
        2, // valid type.
        10, // end_index.
    };

    // Missing intercept.
    const coefficients = [_]f64{
        0.0, // shift.
        1.0, // slope.
    };

    try testing.expectError(
        Error.CorruptedCompressedData,
        rebuildNonLinearApproximation(allocator, timestamps[0..], coefficients[0..], &compressed),
    );
}
