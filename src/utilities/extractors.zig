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
//! components, allowing users to customize and reassemble compression pipelines. These utilities
//! are designed to support advanced workflows and interoperability with TerseTS's compression
//! techniques.

const std = @import("std");
const ArrayList = std.ArrayList;
const math = std.math;
const mem = std.mem;
const testing = std.testing;

const tersets = @import("../tersets.zig");
const Error = tersets.Error;

const shared_functions = @import("shared_functions.zig");

/// Extracts timestamps and coefficients from Poor Man's Compression (PMC)'s `compressed_values`.
/// The function works for both PMCMidrange and PMCMean compression methods.
/// The function accepts a `timestamps` ArrayList to store the extracted end indices,
/// and a `coefficients` ArrayList to store the extracted coefficient values.
/// If any validation of the `compressed_values` fails, and `Error.UnsupportedInput` is returned.
/// If any other memory allocation error occurs, it is propagated.
pub fn extractPMC(
    compressed_values: []const u8,
    timestamps: *ArrayList(usize),
    coefficients: *ArrayList(f64),
) Error!void {
    // The PMC layout consists of pairs of (f64 value, f64 bit-cast of usize end_index).
    // Each pair is 16 bytes (two f64).
    if (compressed_values.len % 16 != 0) return Error.UnsupportedInput;
    const components = mem.bytesAsSlice(f64, compressed_values);

    // Pass through the components, extracting coefficients and timestamps.
    for (0..components.len) |i| {
        if (i % 2 == 0) {
            const coeffs = components[i];
            try coefficients.append(coeffs);
        } else {
            const time = components[i];
            const end_idx: usize = @bitCast(time);
            try timestamps.append(end_idx);
        }
    }
}

/// Extracts timestamps and coefficients from SwingFilter's `compressed_values`.
/// The function accepts a `timestamps` ArrayList to store the extracted end indices,
/// and a `coefficients` ArrayList to store the extracted coefficient values.
/// If any validation of the `compressed_values` fails, and `Error.UnsupportedInput` is returned.
/// If any other memory allocation error occurs, it is propagated.
pub fn extractSwing(
    compressed_values: []const u8,
    timestamps: *ArrayList(usize),
    coefficients: *ArrayList(f64),
) Error!void {
    // First value is coefficient, then alternating coefficient and timestamp.
    if ((compressed_values.len - 8) % 16 != 0) return Error.UnsupportedInput;
    const components = mem.bytesAsSlice(f64, compressed_values);
    for (0..components.len) |i| {
        if ((i == 0) or (i % 2 == 1)) {
            const coeffs = components[i];
            try coefficients.append(coeffs);
        } else {
            const time = components[i];
            const end_idx: usize = @bitCast(time);
            try timestamps.append(end_idx);
        }
    }
}

/// Extracts timestamps and coefficients from SlideFilter's `compressed_values`.
/// This function can be used to extract the coefficients and timestamps from
/// SwingFilterDisconected function as well. The function accepts a `timestamps`
/// ArrayList to store the extracted end indices, and a `coefficients` ArrayList
/// to store the extracted coefficient values. If any validation of the
/// `compressed_values` fails, and `Error.UnsupportedInput` is returned.
/// If any other memory allocation error occurs, it is propagated.
pub fn extractSlide(
    compressed_values: []const u8,
    timestamps: *ArrayList(usize),
    coefficients: *ArrayList(f64),
) Error!void {
    // The `compressed_values` contain every third value as a timestamp, others are coefficients.
    if (compressed_values.len % 24 != 0) return Error.UnsupportedInput;
    const components = mem.bytesAsSlice(f64, compressed_values);
    for (0..components.len) |i| {
        if ((i + 1) % 3 != 0) {
            const coeffs = components[i];
            try coefficients.append(coeffs);
        } else {
            const time = components[i];
            const end_idx: usize = @bitCast(time);
            try timestamps.append(end_idx);
        }
    }
}

/// Extracts timestamps and coefficients from SimPiece's `compressed_values`.
/// The function accepts a `timestamps` ArrayList to store the extracted end indices,
/// and a `coefficients` ArrayList to store the extracted coefficient values.
/// If any validation of the `compressed_values` fails, and `Error.UnsupportedInput` is returned.
/// If any other memory allocation error occurs, it is propagated.
pub fn extractSimPiece(
    compressed_values: []const u8,
    timestamps: *ArrayList(usize),
    coefficients: *ArrayList(f64),
) Error!void {
    const items = mem.bytesAsSlice(f64, compressed_values);
    if (items.len == 0) return Error.UnsupportedInput;
    var i: usize = 0;

    // The layout of SimPiece is more intricate, with details outlined in
    // `src/functional_approximation/sim_piece.zig`. The following loop processes the data sequentially,
    // extracting components and populating the respective `timestamps` and `coefficients` ArrayLists.
    while (i < items.len - 1) {
        // intercept (f64).
        const intercept = items[i];
        try coefficients.append(intercept);
        i += 1;
        // slopes_count (usize as f64 bits).
        if (i >= items.len - 1) return Error.UnsupportedInput;
        const slopes_count: usize = @bitCast(items[i]);
        try timestamps.append(slopes_count);
        i += 1;
        // slopes blocks
        var s: usize = 0;
        while (s < slopes_count) : (s += 1) {
            if (i >= items.len - 1) return Error.UnsupportedInput;
            // slope (f64).
            const slope = items[i];
            try coefficients.append(slope);
            i += 1;
            // timestamps_count (usize).
            if (i >= items.len - 1) return Error.UnsupportedInput;
            const tcount: usize = @bitCast(items[i]);
            try timestamps.append(tcount);
            i += 1;
            // deltas (usize each).
            var t: usize = 0;
            while (t < tcount) : (t += 1) {
                if (i >= items.len - 1) return Error.UnsupportedInput;
                const delta: usize = @bitCast(items[i]);
                try timestamps.append(delta);
                i += 1;
            }
        }
    }
    // Final last_timestamp (usize) is the last f64 in the payload.
    if (i != items.len - 1) return Error.UnsupportedInput;
    const last_ts: usize = @bitCast(items[i]);
    try timestamps.append(last_ts);
}

/// Extracts timestamps and coefficients from MixPiece's `compressed_values`.
/// The function accepts a `timestamps` ArrayList to store the extracted counts and deltas,
/// and a `coefficients` ArrayList to store the extracted intercepts and slopes.
/// The layout consists of three header counts (part1, part2, part3), followed by
/// variable-length blocks for each part. See `src/functional_approximation/mix_piece.zig`
/// for details. Returns `Error.UnsupportedInput` on validation failure, or propagates
/// allocation errors.
pub fn extractMixPiece(
    compressed_values: []const u8,
    timestamps: *ArrayList(usize),
    coefficients: *ArrayList(f64),
) Error!void {
    // Read the header containing counts for the three parts of MixPiece.
    const header = mem.bytesAsSlice(usize, compressed_values[0 .. 3 * @sizeOf(usize)]);

    const part1_count = header[0]; // Number of intercept groups in Part 1.
    const part2_count = header[1]; // Number of slope groups in Part 2.
    const part3_count = header[2]; // Number of ungrouped segments in Part 3.

    // Append the header counts to the timestamps list.
    try timestamps.append(header[0]);
    try timestamps.append(header[1]);
    try timestamps.append(header[2]);

    // Initialize the offset to start reading after the header.
    var offset: usize = 3 * @sizeOf(usize);

    // Process Part 1: Intercept groups.
    if (part1_count > 0) {
        for (0..part1_count) |_| {
            // Read and append the intercept value.
            const intercept = try shared_functions.readOffsetValue(f64, compressed_values, &offset);
            try coefficients.append(intercept);

            // Read and append the number of slopes in this group.
            const slopes_count = try shared_functions.readOffsetValue(usize, compressed_values, &offset);
            try timestamps.append(slopes_count);

            // Process each slope in the group.
            for (0..slopes_count) |_| {
                // Read and append the slope value.
                const slope = try shared_functions.readOffsetValue(f64, compressed_values, &offset);
                try coefficients.append(slope);

                // Read and append the number of timestamps for this slope.
                const timestamps_count = try shared_functions.readOffsetValue(
                    usize,
                    compressed_values,
                    &offset,
                );
                try timestamps.append(timestamps_count);

                // Read and append each timestamp delta.
                for (0..timestamps_count) |_| {
                    const delta = try shared_functions.readOffsetValue(usize, compressed_values, &offset);
                    try timestamps.append(delta);
                }
            }
        }
    }

    // Process Part 2: Slope groups.
    if (part2_count > 0) {
        for (0..part2_count) |_| {
            // Read and append the slope value.
            const slope = try shared_functions.readOffsetValue(f64, compressed_values, &offset);
            try coefficients.append(slope);

            // Read and append the number of intercept-delta pairs in this group.
            const pair_count = try shared_functions.readOffsetValue(usize, compressed_values, &offset);
            try timestamps.append(pair_count);

            // Process each intercept-delta pair.
            for (0..pair_count) |_| {
                // Read and append the intercept value.
                const intercept = try shared_functions.readOffsetValue(f64, compressed_values, &offset);
                try coefficients.append(intercept);

                // Read and append the delta value.
                const delta = try shared_functions.readOffsetValue(usize, compressed_values, &offset);
                try timestamps.append(delta);
            }
        }
    }

    // Process Part 3: Ungrouped segments.
    if (part3_count > 0) {
        for (0..part3_count) |_| {
            // Read and append the slope value.
            const slope = try shared_functions.readOffsetValue(f64, compressed_values, &offset);
            try coefficients.append(slope);

            // Read and append the intercept value.
            const intercept = try shared_functions.readOffsetValue(f64, compressed_values, &offset);
            try coefficients.append(intercept);

            // Read and append the delta value.
            const delta = try shared_functions.readOffsetValue(usize, compressed_values, &offset);
            try timestamps.append(delta);
        }
    }

    // Read and append the final timestamp.
    const final_timestamp = try shared_functions.readOffsetValue(usize, compressed_values, &offset);
    try timestamps.append(final_timestamp);
}

/// Rebuilds Poor Man's Compression (PMC) `compressed_values` from `timestamps` and `coefficients`.
/// The function expects `timestamps` and `coefficients` to have equal length.
/// Each pair is encoded as (f64 coefficient, usize end_index as f64 bits).
/// Returns `Error.UnsupportedInput` if lengths mismatch, or propagates allocation errors.
pub fn rebuildPMC(
    timestamps: []const usize,
    coefficients: []const f64,
    compressed_values: *ArrayList(u8),
) Error!void {
    if (timestamps.len != coefficients.len) return Error.UnsupportedInput;

    // Each pair is 16 bytes (two f64). Reserve once.
    try compressed_values.ensureTotalCapacity(coefficients.len * 16);

    const total_len = coefficients.len + timestamps.len;
    var time_idx: usize = 0;
    var coeff_idx: usize = 0;
    for (0..total_len) |i| {
        if (i % 2 == 0) {
            const coeffs = coefficients[coeff_idx];
            try shared_functions.appendValue(f64, coeffs, compressed_values);
            coeff_idx += 1;
        } else {
            const time = timestamps[time_idx];
            try shared_functions.appendValue(usize, time, compressed_values);
            time_idx += 1;
        }
    }
}

/// Rebuilds SwingFilter's `compressed_values` from `timestamps` and `coefficients`.
/// The function expects the layout: first coefficient, then alternating coefficient and timestamp.
/// Returns `Error.UnsupportedInput` if input lengths are invalid, or propagates allocation errors.
pub fn rebuildSwing(
    timestamps: []const usize,
    coefficients: []const f64,
    compressed_values: *ArrayList(u8),
) Error!void {
    // Validate input lengths: coefficients must have at least one element,
    // and timestamps must have one less element than coefficients.
    if (coefficients.len == 0 or coefficients.len != timestamps.len + 1) {
        return Error.UnsupportedInput;
    }

    // Each pair is 16 bytes (two f64). Reserve once.
    try compressed_values.ensureTotalCapacity(coefficients.len * 16);

    const total_len = coefficients.len + timestamps.len;
    var time_idx: usize = 0;
    var coeff_idx: usize = 0;
    for (0..total_len) |i| {
        if ((i == 0) or (i % 2 == 1)) {
            const coeffs = coefficients[coeff_idx];
            try shared_functions.appendValue(f64, coeffs, compressed_values);
            coeff_idx += 1;
        } else {
            const time = timestamps[time_idx];
            try shared_functions.appendValue(usize, time, compressed_values);
            time_idx += 1;
        }
    }
}

/// Rebuilds SlideFilter's `compressed_values` from `timestamps` and `coefficients`.
/// Every third value is a `timestamp`, others are `coefficients`.
/// Returns `Error.UnsupportedInput` if input lengths are invalid, or propagates allocation errors.
pub fn rebuildSlide(
    timestamps: []const usize,
    coefficients: []const f64,
    compressed_values: *ArrayList(u8),
) Error!void {
    // Each pair is 16 bytes (two f64). Reserve once.
    try compressed_values.ensureTotalCapacity(coefficients.len * 24);

    if (coefficients.len != timestamps.len * 2) {
        return Error.UnsupportedInput;
    }

    const total_len = coefficients.len + timestamps.len;
    var time_idx: usize = 0;
    var coeff_idx: usize = 0;
    for (0..total_len) |i| {
        if ((i + 1) % 3 != 0) {
            const coeffs = coefficients[coeff_idx];
            try shared_functions.appendValue(f64, coeffs, compressed_values);
            coeff_idx += 1;
        } else {
            const time = timestamps[time_idx];
            try shared_functions.appendValue(usize, time, compressed_values);
            time_idx += 1;
        }
    }
}

/// Rebuilds SimPiece's `compressed_values` from `timestamps` and `coefficients`.
/// The function expects the input arrays to follow the SimPiece layout:
/// [intercept, slopes_count, (slope, timestamps_count, deltas...), ..., last_timestamp].
/// Returns `Error.UnsupportedInput` if the input is malformed, or propagates allocation errors.
pub fn rebuildSimPiece(
    timestamps: []const usize,
    coefficients: []const f64,
    compressed_values: *ArrayList(u8),
) Error!void {
    // We need at least the final last_timestamp.
    if (timestamps.len == 0) return Error.UnsupportedInput;

    var ci: usize = 0; // index into coefficients
    var ti: usize = 0; // index into timestamps

    // For each intercept:
    while (ci < coefficients.len) {
        // intercept
        try shared_functions.appendValue(f64, coefficients[ci], compressed_values);
        ci += 1;

        if (ti >= timestamps.len) return Error.UnsupportedInput;
        const slopes_count = timestamps[ti];
        try shared_functions.appendValue(usize, slopes_count, compressed_values);
        ti += 1;

        // slopes for this intercept
        var s: usize = 0;
        while (s < slopes_count) : (s += 1) {
            if (ci >= coefficients.len) return Error.UnsupportedInput;
            // slope
            try shared_functions.appendValue(f64, coefficients[ci], compressed_values);
            ci += 1;

            if (ti >= timestamps.len) return Error.UnsupportedInput;
            const tcount = timestamps[ti];
            try shared_functions.appendValue(usize, tcount, compressed_values);
            ti += 1;

            // deltas
            var t: usize = 0;
            while (t < tcount) : (t += 1) {
                if (ti >= timestamps.len) return Error.UnsupportedInput;
                const delta = timestamps[ti];
                try shared_functions.appendValue(usize, delta, compressed_values);
                ti += 1;
            }
        }
    }

    // Must have exactly one trailing last_timestamp remaining
    if (ti >= timestamps.len) return Error.UnsupportedInput;
    const last_ts = timestamps[ti];
    try shared_functions.appendValue(usize, last_ts, compressed_values);
    ti += 1;

    // No extra data should remain
    if (ti != timestamps.len) return Error.UnsupportedInput;
}

/// Rebuilds MixPiece's `compressed_values` from `timestamps` and `coefficients`.
/// The function expects the input arrays to follow the MixPiece layout:
/// [part1_count, part2_count, part3_count, ...] in `timestamps`, with `coefficients`
/// grouped accordingly. Returns `Error.UnsupportedInput` if the input is malformed,
/// or propagates allocation errors.
pub fn rebuildMixPiece(
    timestamps: []const usize,
    coefficients: []const f64,
    compressed_values: *ArrayList(u8),
) Error!void {
    // Extract the counts for the three parts from the timestamps array.
    const part1_count = timestamps[0];
    const part2_count = timestamps[1];
    const part3_count = timestamps[2];

    // Append the part counts to the compressed values buffer.
    try shared_functions.appendValue(usize, part1_count, compressed_values);
    try shared_functions.appendValue(usize, part2_count, compressed_values);
    try shared_functions.appendValue(usize, part3_count, compressed_values);

    // Initialize indices for iterating through timestamps and coefficients.
    var timestamps_idx: usize = 3;
    var coefficients_idx: usize = 0;

    // Process Part 1: Intercept groups.
    if (part1_count > 0) {
        for (0..part1_count) |_| {
            // Append the intercept value.
            const intercept = coefficients[coefficients_idx];
            try shared_functions.appendValue(f64, intercept, compressed_values);
            coefficients_idx += 1;

            // Append the number of slopes in this group.
            const slopes_count = timestamps[timestamps_idx];
            try shared_functions.appendValue(usize, slopes_count, compressed_values);
            timestamps_idx += 1;

            // Process each slope in the group.
            for (0..slopes_count) |_| {
                // Append the slope value.
                const slope = coefficients[coefficients_idx];
                try shared_functions.appendValue(f64, slope, compressed_values);
                coefficients_idx += 1;

                // Append the number of timestamps for this slope.
                const timestamps_count = timestamps[timestamps_idx];
                try shared_functions.appendValue(usize, timestamps_count, compressed_values);
                timestamps_idx += 1;

                // Append each timestamp delta for this slope.
                for (0..timestamps_count) |_| {
                    const delta = timestamps[timestamps_idx];
                    try shared_functions.appendValue(usize, delta, compressed_values);
                    timestamps_idx += 1;
                }
            }
        }
    }

    // Process Part 2: Slope groups.
    if (part2_count > 0) {
        for (0..part2_count) |_| {
            // Append the slope value.
            const slope = coefficients[coefficients_idx];
            try shared_functions.appendValue(f64, slope, compressed_values);
            coefficients_idx += 1;

            // Append the number of intercept-delta pairs in this group.
            const pair_count = timestamps[timestamps_idx];
            try shared_functions.appendValue(usize, pair_count, compressed_values);
            timestamps_idx += 1;

            // Process each intercept-delta pair.
            for (0..pair_count) |_| {
                // Append the intercept value.
                const intercept = coefficients[coefficients_idx];
                try shared_functions.appendValue(f64, intercept, compressed_values);
                coefficients_idx += 1;

                // Append the delta value.
                const delta = timestamps[timestamps_idx];
                try shared_functions.appendValue(usize, delta, compressed_values);
                timestamps_idx += 1;
            }
        }
    }

    // Process Part 3: Ungrouped segments.
    if (part3_count > 0) {
        for (0..part3_count) |_| {
            // Append the slope value.
            const slope = coefficients[coefficients_idx];
            try shared_functions.appendValue(f64, slope, compressed_values);
            coefficients_idx += 1;

            // Append the intercept value.
            const intercept = coefficients[coefficients_idx];
            try shared_functions.appendValue(f64, intercept, compressed_values);
            coefficients_idx += 1;

            // Append the delta value.
            const delta = timestamps[timestamps_idx];
            try shared_functions.appendValue(usize, delta, compressed_values);
            timestamps_idx += 1;
        }
    }

    // Append the final timestamp.
    const final_timestamp = timestamps[timestamps_idx];
    try shared_functions.appendValue(usize, final_timestamp, compressed_values);
}

test "extract and rebuild PMC round trip" {
    const allocator = testing.allocator;

    const timestamps = [_]usize{ 5, 10 };
    const coefficients = [_]f64{ 1.5, -0.25 };

    var compressed = ArrayList(u8).init(allocator);
    defer compressed.deinit();
    try rebuildPMC(timestamps[0..], coefficients[0..], &compressed);

    var extracted_ts = ArrayList(usize).init(allocator);
    defer extracted_ts.deinit();
    var extracted_coeffs = ArrayList(f64).init(allocator);
    defer extracted_coeffs.deinit();
    try extractPMC(compressed.items, &extracted_ts, &extracted_coeffs);

    try testing.expectEqualSlices(usize, timestamps[0..], extracted_ts.items);
    try testing.expectEqualSlices(f64, coefficients[0..], extracted_coeffs.items);
}

test "rebuildPMC rejects mismatched input lengths" {
    const allocator = testing.allocator;
    var compressed = ArrayList(u8).init(allocator);
    defer compressed.deinit();

    const timestamps = [_]usize{1};
    const coefficients = [_]f64{ 1.0, 2.0 };

    try testing.expectError(Error.UnsupportedInput, rebuildPMC(
        timestamps[0..],
        coefficients[0..],
        &compressed,
    ));
}

test "extract and rebuild Swing round trip" {
    const allocator = testing.allocator;

    const timestamps = [_]usize{ 7, 11 };
    const coefficients = [_]f64{ 2.0, 3.0, 4.0 };

    var compressed = ArrayList(u8).init(allocator);
    defer compressed.deinit();
    try rebuildSwing(timestamps[0..], coefficients[0..], &compressed);

    var extracted_ts = ArrayList(usize).init(allocator);
    defer extracted_ts.deinit();
    var extracted_coeffs = ArrayList(f64).init(allocator);
    defer extracted_coeffs.deinit();
    try extractSwing(compressed.items, &extracted_ts, &extracted_coeffs);

    try testing.expectEqualSlices(usize, timestamps[0..], extracted_ts.items);
    try testing.expectEqualSlices(f64, coefficients[0..], extracted_coeffs.items);
}

test "rebuildSwing rejects mismatched input lengths" {
    const allocator = testing.allocator;
    var compressed = ArrayList(u8).init(allocator);
    defer compressed.deinit();

    const timestamps = [_]usize{ 1, 2 };
    const coefficients = [_]f64{ 1.0, 2.0 };

    try testing.expectError(Error.UnsupportedInput, rebuildSwing(
        timestamps[0..],
        coefficients[0..],
        &compressed,
    ));
}

test "extract and rebuild Slide round trip" {
    const allocator = testing.allocator;

    const timestamps = [_]usize{ 3, 8 };
    const coefficients = [_]f64{ 0.5, -1.0, 2.5, 4.0 };

    var compressed = ArrayList(u8).init(allocator);
    defer compressed.deinit();
    try rebuildSlide(timestamps[0..], coefficients[0..], &compressed);

    var extracted_ts = ArrayList(usize).init(allocator);
    defer extracted_ts.deinit();
    var extracted_coeffs = ArrayList(f64).init(allocator);
    defer extracted_coeffs.deinit();
    try extractSlide(compressed.items, &extracted_ts, &extracted_coeffs);

    try testing.expectEqualSlices(usize, timestamps[0..], extracted_ts.items);
    try testing.expectEqualSlices(f64, coefficients[0..], extracted_coeffs.items);
}

test "rebuildSlide rejects mismatched input lengths" {
    const allocator = testing.allocator;
    var compressed = ArrayList(u8).init(allocator);
    defer compressed.deinit();

    const timestamps = [_]usize{ 1, 2, 3 };
    const coefficients = [_]f64{ 1.0, 2.0 };

    try testing.expectError(Error.UnsupportedInput, rebuildSlide(
        timestamps[0..],
        coefficients[0..],
        &compressed,
    ));
}

test "extract and rebuild SimPiece round trip" {
    const allocator = testing.allocator;

    const timestamps = [_]usize{ 1, 2, 4, 6, 12 };
    const coefficients = [_]f64{ 1.25, 0.75 };

    var compressed = ArrayList(u8).init(allocator);
    defer compressed.deinit();
    try rebuildSimPiece(timestamps[0..], coefficients[0..], &compressed);

    var extracted_ts = ArrayList(usize).init(allocator);
    defer extracted_ts.deinit();
    var extracted_coeffs = ArrayList(f64).init(allocator);
    defer extracted_coeffs.deinit();
    try extractSimPiece(compressed.items, &extracted_ts, &extracted_coeffs);

    try testing.expectEqualSlices(usize, timestamps[0..], extracted_ts.items);
    try testing.expectEqualSlices(f64, coefficients[0..], extracted_coeffs.items);
}

test "extract and rebuild MixPiece round trip" {
    const allocator = testing.allocator;

    const timestamps = [_]usize{ 1, 1, 1, 2, 2, 3, 4, 1, 8, 2, 5, 6, 10, 99 };
    const coefficients = [_]f64{ 1.1, 0.5, 1.5, 2.5, 0.25, 0.75, 3.5, -1.25 };

    var compressed = ArrayList(u8).init(allocator);
    defer compressed.deinit();
    try rebuildMixPiece(timestamps[0..], coefficients[0..], &compressed);

    var extracted_ts = ArrayList(usize).init(allocator);
    defer extracted_ts.deinit();
    var extracted_coeffs = ArrayList(f64).init(allocator);
    defer extracted_coeffs.deinit();
    try extractMixPiece(compressed.items, &extracted_ts, &extracted_coeffs);

    try testing.expectEqualSlices(usize, timestamps[0..], extracted_ts.items);
    try testing.expectEqualSlices(f64, coefficients[0..], extracted_coeffs.items);
}
