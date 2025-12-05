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
//! are designed to support advanced compression pipelines construction with TerseTS.

const std = @import("std");
const ArrayList = std.ArrayList;
const math = std.math;
const mem = std.mem;
const testing = std.testing;

const tersets = @import("../tersets.zig");
const configuration = @import("../configuration.zig");
const tester = @import("../tester.zig");
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
    timestamps: *ArrayList(u64),
    coefficients: *ArrayList(f64),
) Error!void {
    // The PMC layout consists of pairs of (f64 value, f64 bit-cast of u64 end_index).
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
            const end_idx: u64 = @bitCast(time);
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
    timestamps: *ArrayList(u64),
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
            const end_idx: u64 = @bitCast(time);
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
    timestamps: *ArrayList(u64),
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
            const end_idx: u64 = @bitCast(time);
            try timestamps.append(end_idx);
        }
    }
}

/// Extracts timestamps and coefficients from the ConvexABC's `compressed_values`.
/// The function accepts a `timestamps` ArrayList to store the extracted end indices,
/// and a `coefficients` ArrayList to store the extracted coefficient values.
/// If any validation of the `compressed_values` fails, and `Error.UnsupportedInput` is returned.
/// If any other memory allocation error occurs, it is propagated.
pub fn extractABCLinearApproximation(
    compressed_values: []const u8,
    timestamps: *ArrayList(u64),
    coefficients: *ArrayList(f64),
) Error!void {
    // The logic of this function is exactly the same as `extractSlide`.
    try extractSlide(compressed_values, timestamps, coefficients);
}

/// Extracts timestamps and coefficients from SimPiece's `compressed_values`.
/// The function accepts a `timestamps` ArrayList to store the extracted end indices,
/// and a `coefficients` ArrayList to store the extracted coefficient values.
/// If any validation of the `compressed_values` fails, and `Error.UnsupportedInput` is returned.
/// If any other memory allocation error occurs, it is propagated.
pub fn extractSimPiece(
    compressed_values: []const u8,
    timestamps: *ArrayList(u64),
    coefficients: *ArrayList(f64),
) Error!void {
    const items = mem.bytesAsSlice(f64, compressed_values);
    if (items.len == 0) return Error.UnsupportedInput;
    var i: u64 = 0;

    // The layout of SimPiece is more intricate, with details outlined in
    // `src/functional_approximation/sim_piece.zig`. The following loop processes the data sequentially,
    // extracting components and populating the respective `timestamps` and `coefficients` ArrayLists.
    while (i < items.len - 1) {
        // intercept (f64).
        const intercept = items[i];
        try coefficients.append(intercept);
        i += 1;
        // slopes_count (u64 as f64 bits).
        if (i >= items.len - 1) return Error.UnsupportedInput;
        const slopes_count: u64 = @bitCast(items[i]);
        try timestamps.append(slopes_count);
        i += 1;
        // slopes blocks
        var s: u64 = 0;
        while (s < slopes_count) : (s += 1) {
            if (i >= items.len - 1) return Error.UnsupportedInput;
            // slope (f64).
            const slope = items[i];
            try coefficients.append(slope);
            i += 1;
            // timestamps_count (u64).
            if (i >= items.len - 1) return Error.UnsupportedInput;
            const tcount: u64 = @bitCast(items[i]);
            try timestamps.append(tcount);
            i += 1;
            // deltas (u64 each).
            var t: u64 = 0;
            while (t < tcount) : (t += 1) {
                if (i >= items.len - 1) return Error.UnsupportedInput;
                const delta: u64 = @bitCast(items[i]);
                try timestamps.append(delta);
                i += 1;
            }
        }
    }
    // Final last_timestamp (u64) is the last f64 in the payload.
    if (i != items.len - 1) return Error.UnsupportedInput;
    const last_ts: u64 = @bitCast(items[i]);
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
    timestamps: *ArrayList(u64),
    coefficients: *ArrayList(f64),
) Error!void {
    // Read the header containing counts for the three parts of MixPiece.
    const header = mem.bytesAsSlice(u64, compressed_values[0 .. 3 * @sizeOf(u64)]);

    const part1_count = header[0]; // Number of intercept groups in Part 1.
    const part2_count = header[1]; // Number of slope groups in Part 2.
    const part3_count = header[2]; // Number of ungrouped segments in Part 3.

    // Append the header counts to the timestamps list.
    try timestamps.append(header[0]);
    try timestamps.append(header[1]);
    try timestamps.append(header[2]);

    // Initialize the offset to start reading after the header.
    var offset: u64 = 3 * @sizeOf(u64);

    // Process Part 1: Intercept groups.
    if (part1_count > 0) {
        for (0..part1_count) |_| {
            // Read and append the intercept value.
            const intercept = try shared_functions.readOffsetValue(f64, compressed_values, &offset);
            try coefficients.append(intercept);

            // Read and append the number of slopes in this group.
            const slopes_count = try shared_functions.readOffsetValue(u64, compressed_values, &offset);
            try timestamps.append(slopes_count);

            // Process each slope in the group.
            for (0..slopes_count) |_| {
                // Read and append the slope value.
                const slope = try shared_functions.readOffsetValue(f64, compressed_values, &offset);
                try coefficients.append(slope);

                // Read and append the number of timestamps for this slope.
                const timestamps_count = try shared_functions.readOffsetValue(
                    u64,
                    compressed_values,
                    &offset,
                );
                try timestamps.append(timestamps_count);

                // Read and append each timestamp delta.
                for (0..timestamps_count) |_| {
                    const delta = try shared_functions.readOffsetValue(u64, compressed_values, &offset);
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
            const pair_count = try shared_functions.readOffsetValue(u64, compressed_values, &offset);
            try timestamps.append(pair_count);

            // Process each intercept-delta pair.
            for (0..pair_count) |_| {
                // Read and append the intercept value.
                const intercept = try shared_functions.readOffsetValue(f64, compressed_values, &offset);
                try coefficients.append(intercept);

                // Read and append the delta value.
                const delta = try shared_functions.readOffsetValue(u64, compressed_values, &offset);
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
            const delta = try shared_functions.readOffsetValue(u64, compressed_values, &offset);
            try timestamps.append(delta);
        }
    }

    // Read and append the final timestamp.
    const final_timestamp = try shared_functions.readOffsetValue(u64, compressed_values, &offset);
    try timestamps.append(final_timestamp);
}

/// Extracts timestamps and coefficients from the Piecewice Constant Histogram's `compressed_values`.
/// The function accepts a `timestamps` ArrayList to store the extracted end indices,
/// and a `coefficients` ArrayList to store the extracted coefficient values.
/// If any validation of the `compressed_values` fails, and `Error.UnsupportedInput` is returned.
/// If any other memory allocation error occurs, it is propagated.
pub fn extractPWCH(
    compressed_values: []const u8,
    timestamps: *ArrayList(u64),
    coefficients: *ArrayList(f64),
) Error!void {
    // The logic of this function is exactly the same as `extractPMC`.
    try extractPMC(compressed_values, timestamps, coefficients);
}

/// Extracts timestamps and coefficients from the Piecewice Linear Histogram's `compressed_values`.
/// The function accepts a `timestamps` ArrayList to store the extracted end indices,
/// and a `coefficients` ArrayList to store the extracted coefficient values.
/// If any validation of the `compressed_values` fails, and `Error.UnsupportedInput` is returned.
/// If any other memory allocation error occurs, it is propagated.
pub fn extractPWLH(
    compressed_values: []const u8,
    timestamps: *ArrayList(u64),
    coefficients: *ArrayList(f64),
) Error!void {
    // The logic of this function is exactly the same as `extractSlide`.
    try extractSlide(compressed_values, timestamps, coefficients);
}

/// Extracts timestamps and coefficients from Visvalingam-Whyatt's `compressed_values`.
/// The function accepts a `timestamps` ArrayList to store the extracted end indices,
/// and a `coefficients` ArrayList to store the extracted coefficient values.
/// If any validation of the `compressed_values` fails, and `Error.UnsupportedInput` is returned.
/// If any other memory allocation error occurs, it is propagated.
pub fn extractVisvalingamWhyatt(
    compressed_values: []const u8,
    timestamps: *ArrayList(u64),
    coefficients: *ArrayList(f64),
) Error!void {
    // The logic of this function is exactly the same as `extractSwing`.
    try extractSwing(compressed_values, timestamps, coefficients);
}

/// Extracts timestamps and coefficients from SlidingWindows's `compressed_values`.
/// The function accepts a `timestamps` ArrayList to store the extracted end indices,
/// and a `coefficients` ArrayList to store the extracted coefficient values. If any
/// validation of the `compressed_values` fails, and `Error.UnsupportedInput` is returned.
/// If any other memory allocation error occurs, it is propagated.
pub fn extractSlidingWindow(
    compressed_values: []const u8,
    timestamps: *ArrayList(u64),
    coefficients: *ArrayList(f64),
) Error!void {
    // The logic of this function is exactly the same as `extractSlide`.
    try extractSlide(compressed_values, timestamps, coefficients);
}

/// Extracts timestamps and coefficients from BottomUp's `compressed_values`.
/// The function accepts a `timestamps` ArrayList to store the extracted end indices,
/// and a `coefficients` ArrayList to store the extracted coefficient values. If any
/// validation of the `compressed_values` fails, and `Error.UnsupportedInput` is returned.
/// If any other memory allocation error occurs, it is propagated.
pub fn extractBottomUp(
    compressed_values: []const u8,
    timestamps: *ArrayList(u64),
    coefficients: *ArrayList(f64),
) Error!void {
    // The logic of this function is exactly the same as `extractSlide`.
    try extractSlide(compressed_values, timestamps, coefficients);
}

/// Extracts timestamps and coefficients from NonLinearApproximation's `compressed_values`.
/// The function accepts a `timestamps` ArrayList to store the extracted end indices,
/// and a `coefficients` ArrayList to store the extracted coefficient values.
/// If any validation of the `compressed_values` fails, and `Error.UnsupportedInput` is returned.
/// If any other memory allocation error occurs, it is propagated.
pub fn extractNonLinearApproximation(
    compressed_values: []const u8,
    timestamps: *ArrayList(u64),
    coefficients: *ArrayList(f64),
) Error!void {
    // Validates that the compressed data contains some bytes to process.
    if (compressed_values.len < 12) return Error.CorruptedCompressedData;

    var offset: u64 = 0; // Tracks the current position in the compressed stream.
    const shift_amount = try shared_functions.readValue(f64, compressed_values[offset..]);
    offset += @sizeOf(f64);

    try coefficients.append(shift_amount);

    // Reads the number of segments that were used in the partitioning.
    const num_segments: u64 = try shared_functions.readValue(u32, compressed_values[offset..]);
    offset += @sizeOf(u32);

    try timestamps.append(num_segments);

    // Read packed function types (2 per byte, low nibble = even index, high nibble = odd).
    const type_bytes_len: u64 = (num_segments + 1) / 2;

    // Validate that the compressed stream contains exactly the expected number of bytes.
    // Each segment stores: 2 * f64 (slope, intercept) + u64 (end_idx).
    const bytes_per_segment = @sizeOf(f64) * 2 + @sizeOf(u64);
    const expected_total_bytes =
        @sizeOf(f64) + // shift_amount.
        @sizeOf(u32) + // num_segments.
        type_bytes_len + // packed function types.
        num_segments * bytes_per_segment;

    if (compressed_values.len != expected_total_bytes)
        return Error.CorruptedCompressedData;

    const packed_function_types = compressed_values[offset .. offset + type_bytes_len];
    offset += type_bytes_len;

    for (0..num_segments) |segment_idx| { // Iterates through each segment.
        const packed_code = packed_function_types[segment_idx / 2];
        const code: u4 = if (segment_idx % 2 != 0)
            @truncate(packed_code & 0x0F)
        else
            @truncate((packed_code >> 4) & 0x0F);

        const function_type: u64 = @intCast(code);

        try timestamps.append(function_type);
    }

    for (0..num_segments) |_| {
        // Reads the main function parameters (slope and intercept) and end index.
        const slope = try shared_functions.readValue(f64, compressed_values[offset..]);
        offset += @sizeOf(f64);
        const intercept = try shared_functions.readValue(f64, compressed_values[offset..]);
        offset += @sizeOf(f64);
        const end_idx: u64 = try shared_functions.readValue(u64, compressed_values[offset..]);
        offset += @sizeOf(u64);

        try coefficients.append(slope);
        try coefficients.append(intercept);
        try timestamps.append(end_idx);
    }
}

/// Rebuilds Poor Man's Compression (PMC) `compressed_values` from `timestamps` and `coefficients`.
/// The function expects `timestamps` and `coefficients` to have equal length.
/// Each pair is encoded as (f64 coefficient, u64 end_index as f64 bits).
/// Returns `Error.UnsupportedInput` if lengths mismatch, or propagates allocation errors.
pub fn rebuildPMC(
    timestamps: []const u64,
    coefficients: []const f64,
    compressed_values: *ArrayList(u8),
) Error!void {
    if (timestamps.len != coefficients.len) return Error.UnsupportedInput;

    // Each pair is 16 bytes (two f64). Reserve once.
    try compressed_values.ensureTotalCapacity(coefficients.len * 16);

    const total_len = coefficients.len + timestamps.len;
    var time_idx: u64 = 0;
    var coeff_idx: u64 = 0;
    for (0..total_len) |i| {
        if (i % 2 == 0) {
            const coeffs = coefficients[coeff_idx];
            try shared_functions.appendValue(f64, coeffs, compressed_values);
            coeff_idx += 1;
        } else {
            const time = timestamps[time_idx];
            try shared_functions.appendValue(u64, time, compressed_values);
            time_idx += 1;
        }
    }
}

/// Rebuilds SwingFilter's `compressed_values` from `timestamps` and `coefficients`.
/// The function expects the layout: first coefficient, then alternating coefficient and timestamp.
/// Returns `Error.UnsupportedInput` if input lengths are invalid, or propagates allocation errors.
pub fn rebuildSwing(
    timestamps: []const u64,
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
    var time_idx: u64 = 0;
    var coeff_idx: u64 = 0;
    for (0..total_len) |i| {
        if ((i == 0) or (i % 2 == 1)) {
            const coeffs = coefficients[coeff_idx];
            try shared_functions.appendValue(f64, coeffs, compressed_values);
            coeff_idx += 1;
        } else {
            const time = timestamps[time_idx];
            try shared_functions.appendValue(u64, time, compressed_values);
            time_idx += 1;
        }
    }
}

/// Rebuilds SlideFilter's `compressed_values` from `timestamps` and `coefficients`.
/// Every third value is a `timestamp`, while the other two are `coefficients`.
/// Returns `Error.UnsupportedInput` if input lengths are invalid, or propagates allocation errors.
pub fn rebuildSlide(
    timestamps: []const u64,
    coefficients: []const f64,
    compressed_values: *ArrayList(u8),
) Error!void {
    // Each pair is 16 bytes (two f64). Reserve once.
    try compressed_values.ensureTotalCapacity(coefficients.len * 24);

    if (coefficients.len != timestamps.len * 2) {
        return Error.UnsupportedInput;
    }

    const total_len = coefficients.len + timestamps.len;
    var time_idx: u64 = 0;
    var coeff_idx: u64 = 0;
    for (0..total_len) |i| {
        if ((i + 1) % 3 != 0) {
            const coeffs = coefficients[coeff_idx];
            try shared_functions.appendValue(f64, coeffs, compressed_values);
            coeff_idx += 1;
        } else {
            const time = timestamps[time_idx];
            try shared_functions.appendValue(u64, time, compressed_values);
            time_idx += 1;
        }
    }
}

/// Rebuilds ABCLinearApproximation's `compressed_values` from `timestamps` and `coefficients`.
/// Every third value is a `timestamp`, while the other two are `coefficients`.
/// Returns `Error.UnsupportedInput` if input lengths are invalid, or propagates allocation errors.
pub fn rebuildABCLinearApproximation(
    timestamps: []const u64,
    coefficients: []const f64,
    compressed_values: *ArrayList(u8),
) Error!void {
    // The logic of this function is exactly the same as `rebuildSlide`.
    try rebuildSlide(timestamps, coefficients, compressed_values);
}

/// Rebuilds SimPiece's `compressed_values` from `timestamps` and `coefficients`.
/// The function expects the input arrays to follow the SimPiece layout:
/// [intercept, slopes_count, (slope, timestamps_count, deltas...), ..., last_timestamp].
/// Returns `Error.UnsupportedInput` if the input is malformed, or propagates allocation errors.
pub fn rebuildSimPiece(
    timestamps: []const u64,
    coefficients: []const f64,
    compressed_values: *ArrayList(u8),
) Error!void {
    // We need at least the final last_timestamp.
    if (timestamps.len == 0) return Error.UnsupportedInput;

    var ci: u64 = 0; // index into coefficients
    var ti: u64 = 0; // index into timestamps

    // For each intercept.
    while (ci < coefficients.len) {
        // Extract intercept.
        try shared_functions.appendValue(f64, coefficients[ci], compressed_values);
        ci += 1;

        if (ti >= timestamps.len) return Error.UnsupportedInput;
        const slopes_count = timestamps[ti];
        try shared_functions.appendValue(u64, slopes_count, compressed_values);
        ti += 1;

        // For each slope on this intercept.
        var s: u64 = 0;
        while (s < slopes_count) : (s += 1) {
            if (ci >= coefficients.len) return Error.UnsupportedInput;
            // Extract slope.
            try shared_functions.appendValue(f64, coefficients[ci], compressed_values);
            ci += 1;

            if (ti >= timestamps.len) return Error.UnsupportedInput;
            const tcount = timestamps[ti];
            try shared_functions.appendValue(u64, tcount, compressed_values);
            ti += 1;

            // Extract deltas.
            var t: u64 = 0;
            while (t < tcount) : (t += 1) {
                if (ti >= timestamps.len) return Error.UnsupportedInput;
                const delta = timestamps[ti];
                try shared_functions.appendValue(u64, delta, compressed_values);
                ti += 1;
            }
        }
    }

    // Must have exactly one trailing last_timestamp remaining.
    if (ti >= timestamps.len) return Error.UnsupportedInput;
    const last_ts = timestamps[ti];
    try shared_functions.appendValue(u64, last_ts, compressed_values);
    ti += 1;

    // No extra data should remain.
    if (ti != timestamps.len) return Error.UnsupportedInput;
}

/// Rebuilds MixPiece's `compressed_values` from `timestamps` and `coefficients`.
/// The function expects the input arrays to follow the MixPiece layout:
/// [part1_count, part2_count, part3_count, ...] in `timestamps`, with `coefficients`
/// grouped accordingly. Returns `Error.UnsupportedInput` if the input is malformed,
/// or propagates allocation errors.
pub fn rebuildMixPiece(
    timestamps: []const u64,
    coefficients: []const f64,
    compressed_values: *ArrayList(u8),
) Error!void {
    // Extract the counts for the three parts from the timestamps array.
    const part1_count = timestamps[0];
    const part2_count = timestamps[1];
    const part3_count = timestamps[2];

    // Append the part counts to the compressed values buffer.
    try shared_functions.appendValue(u64, part1_count, compressed_values);
    try shared_functions.appendValue(u64, part2_count, compressed_values);
    try shared_functions.appendValue(u64, part3_count, compressed_values);

    // Initialize indices for iterating through timestamps and coefficients.
    var timestamps_idx: u64 = 3;
    var coefficients_idx: u64 = 0;

    // Process Part 1: Intercept groups.
    if (part1_count > 0) {
        for (0..part1_count) |_| {
            // Append the intercept value.
            const intercept = coefficients[coefficients_idx];
            try shared_functions.appendValue(f64, intercept, compressed_values);
            coefficients_idx += 1;

            // Append the number of slopes in this group.
            const slopes_count = timestamps[timestamps_idx];
            try shared_functions.appendValue(u64, slopes_count, compressed_values);
            timestamps_idx += 1;

            // Process each slope in the group.
            for (0..slopes_count) |_| {
                // Append the slope value.
                const slope = coefficients[coefficients_idx];
                try shared_functions.appendValue(f64, slope, compressed_values);
                coefficients_idx += 1;

                // Append the number of timestamps for this slope.
                const timestamps_count = timestamps[timestamps_idx];
                try shared_functions.appendValue(u64, timestamps_count, compressed_values);
                timestamps_idx += 1;

                // Append each timestamp delta for this slope.
                for (0..timestamps_count) |_| {
                    const delta = timestamps[timestamps_idx];
                    try shared_functions.appendValue(u64, delta, compressed_values);
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
            try shared_functions.appendValue(u64, pair_count, compressed_values);
            timestamps_idx += 1;

            // Process each intercept-delta pair.
            for (0..pair_count) |_| {
                // Append the intercept value.
                const intercept = coefficients[coefficients_idx];
                try shared_functions.appendValue(f64, intercept, compressed_values);
                coefficients_idx += 1;

                // Append the delta value.
                const delta = timestamps[timestamps_idx];
                try shared_functions.appendValue(u64, delta, compressed_values);
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
            try shared_functions.appendValue(u64, delta, compressed_values);
            timestamps_idx += 1;
        }
    }

    // Append the final timestamp.
    const final_timestamp = timestamps[timestamps_idx];
    try shared_functions.appendValue(u64, final_timestamp, compressed_values);
}

/// Rebuilds Piecewise Linear Histogram's `compressed_values` from `timestamps` and `coefficients`.
/// Every third value is a `timestamp`, while the other two are `coefficients`.
/// Returns `Error.UnsupportedInput` if lengths mismatch, or propagates allocation errors.
pub fn rebuildPWLH(
    timestamps: []const u64,
    coefficients: []const f64,
    compressed_values: *ArrayList(u8),
) Error!void {
    // The logic of this function is exactly the same as `rebuildSlide`.
    try rebuildSlide(timestamps, coefficients, compressed_values);
}

/// Rebuilds Piecewise Constant Histogram's `compressed_values` from `timestamps` and `coefficients`.
/// The function expects `timestamps` and `coefficients` to have equal length.
/// Each pair is encoded as (f64 coefficient, u64 end_index as f64 bits).
/// Returns `Error.UnsupportedInput` if lengths mismatch, or propagates allocation errors.
pub fn rebuildPWCH(
    timestamps: []const u64,
    coefficients: []const f64,
    compressed_values: *ArrayList(u8),
) Error!void {
    // The logic of this function is exactly the same as `rebuildPMC`.
    try rebuildPMC(timestamps, coefficients, compressed_values);
}

/// Rebuilds Visvalingam-Whyatt's `compressed_values` from `timestamps` and `coefficients`.
/// The function expects the layout: first coefficient, then alternating coefficient and timestamp.
/// Returns `Error.UnsupportedInput` if input lengths are invalid, or propagates allocation errors.
pub fn rebuildVisvalinganWhyatt(
    timestamps: []const u64,
    coefficients: []const f64,
    compressed_values: *ArrayList(u8),
) Error!void {
    // The logic of this function is exactly the same as `rebuildSwing`.
    try rebuildSwing(timestamps, coefficients, compressed_values);
}

/// Rebuilds SlidingWindow's `compressed_values` from `timestamps` and `coefficients`.
/// Every third value is a `timestamp`, while the other two are `coefficients`.
/// Returns `Error.UnsupportedInput` if input lengths are invalid, or propagates allocation errors.
pub fn rebuildSlindingWindow(
    timestamps: []const u64,
    coefficients: []const f64,
    compressed_values: *ArrayList(u8),
) Error!void {
    // The logic of this function is exactly the same as `rebuildSlide`.
    try rebuildSlide(timestamps, coefficients, compressed_values);
}

/// Rebuilds BottomUp's `compressed_values` from `timestamps` and `coefficients`.
/// Every third value is a `timestamp`, while the other two are `coefficients`.
/// Returns `Error.UnsupportedInput` if input lengths are invalid, or propagates allocation errors.
pub fn rebuildBottomUp(
    timestamps: []const u64,
    coefficients: []const f64,
    compressed_values: *ArrayList(u8),
) Error!void {
    // The logic of this function is exactly the same as `rebuildSlide`.
    try rebuildSlide(timestamps, coefficients, compressed_values);
}

/// Rebuild NonLinearApproximation's `compressed_values` from `timestamps` and `coefficients`.
/// The `compressed_values` encodes not just coefficients and timestamps but also function types.
/// The function rebuilds the `compressed_values` assuming that the `timestamps` contains the
/// function types encoding information. If non-feasible function appears an error is returned.
///
pub fn rebuildNonLinearApproximation(
    allocator: mem.Allocator,
    timestamps: []const u64,
    coefficients: []const f64,
    compressed_values: *ArrayList(u8),
) Error!void {
    var coefficient_index: u64 = 0;
    var timestamps_index: u64 = 0;

    try shared_functions.appendValue(f64, coefficients[coefficient_index], compressed_values);
    try shared_functions.appendValue(u64, timestamps[timestamps_index], compressed_values);

    coefficient_index += 1;
    timestamps_index += 1;

    if (coefficients.len != timestamps.len) {
        return Error.UnsupportedInput;
    }

    const total_len = coefficients.len + timestamps.len - 2;

    var packed_function_types = try allocator.alloc(u8, total_len);
    defer allocator.free(packed_function_types);
    @memset(packed_function_types, 0);

    // Each byte in packed_function_types stores two function types:
    // - The low nibble (bits 0-3) holds the function type for the approximation at odd index.
    // - The high nibble (bits 4-7) holds the function type for the approximation at even index.
    // Approximations are packed in order: [0,1], [2,3], ...
    for (0..total_len) |idx| {
        const code: u8 = @intCast(timestamps[timestamps_index]);
        const byte_idx: u64 = idx / 2;
        const is_high_nibble: bool = (idx % 2) == 0;
        if (is_high_nibble) {
            // Store the function type in the high nibble (bits 4-7).
            packed_function_types[byte_idx] |= @as(u8, code) << 4;
        } else {
            // Store the function type in the low nibble (bits 0-3).
            packed_function_types[byte_idx] |= @as(u8, code) & 0x0F;
        }
        timestamps_index += 1;
    }

    try compressed_values.appendSlice(packed_function_types);

    for (0..total_len) |_| {
        const slope = coefficients[coefficient_index];
        const intercept = coefficients[coefficient_index + 1];
        const end_idx = timestamps[timestamps_index];
        // Writes the two main function parameters (slope and intercept) and the end point.
        // The end point is exclusive, so it indicates where the next segment starts.
        try shared_functions.appendValue(f64, slope, compressed_values);
        try shared_functions.appendValue(f64, intercept, compressed_values);
        try shared_functions.appendValue(u64, end_idx, compressed_values);

        coefficient_index += 2;
        timestamps_index += 1;
    }
}

test "extract and rebuild PMC round trip" {
    const allocator = testing.allocator;

    const timestamps = [_]u64{ 5, 10 };
    const coefficients = [_]f64{ 1.5, -0.25 };

    var compressed = ArrayList(u8).init(allocator);
    defer compressed.deinit();
    try rebuildPMC(timestamps[0..], coefficients[0..], &compressed);

    var extracted_ts = ArrayList(u64).init(allocator);
    defer extracted_ts.deinit();
    var extracted_coeffs = ArrayList(f64).init(allocator);
    defer extracted_coeffs.deinit();
    try extractPMC(compressed.items, &extracted_ts, &extracted_coeffs);

    try testing.expectEqualSlices(u64, timestamps[0..], extracted_ts.items);
    try testing.expectEqualSlices(f64, coefficients[0..], extracted_coeffs.items);
}

test "rebuildPMC rejects mismatched input lengths" {
    const allocator = testing.allocator;
    var compressed = ArrayList(u8).init(allocator);
    defer compressed.deinit();

    const timestamps = [_]u64{1};
    const coefficients = [_]f64{ 1.0, 2.0 };

    try testing.expectError(Error.UnsupportedInput, rebuildPMC(
        timestamps[0..],
        coefficients[0..],
        &compressed,
    ));
}

test "extract and rebuild Swing round trip" {
    const allocator = testing.allocator;

    const timestamps = [_]u64{ 7, 11 };
    const coefficients = [_]f64{ 2.0, 3.0, 4.0 };

    var compressed = ArrayList(u8).init(allocator);
    defer compressed.deinit();
    try rebuildSwing(timestamps[0..], coefficients[0..], &compressed);

    var extracted_ts = ArrayList(u64).init(allocator);
    defer extracted_ts.deinit();
    var extracted_coeffs = ArrayList(f64).init(allocator);
    defer extracted_coeffs.deinit();
    try extractSwing(compressed.items, &extracted_ts, &extracted_coeffs);

    try testing.expectEqualSlices(u64, timestamps[0..], extracted_ts.items);
    try testing.expectEqualSlices(f64, coefficients[0..], extracted_coeffs.items);
}

test "rebuildSwing rejects mismatched input lengths" {
    const allocator = testing.allocator;
    var compressed = ArrayList(u8).init(allocator);
    defer compressed.deinit();

    const timestamps = [_]u64{ 1, 2 };
    const coefficients = [_]f64{ 1.0, 2.0 };

    try testing.expectError(Error.UnsupportedInput, rebuildSwing(
        timestamps[0..],
        coefficients[0..],
        &compressed,
    ));
}

test "extract and rebuild Slide round trip" {
    const allocator = testing.allocator;

    const timestamps = [_]u64{ 3, 8 };
    const coefficients = [_]f64{ 0.5, -1.0, 2.5, 4.0 };

    var compressed = ArrayList(u8).init(allocator);
    defer compressed.deinit();
    try rebuildSlide(timestamps[0..], coefficients[0..], &compressed);

    var extracted_ts = ArrayList(u64).init(allocator);
    defer extracted_ts.deinit();
    var extracted_coeffs = ArrayList(f64).init(allocator);
    defer extracted_coeffs.deinit();
    try extractSlide(compressed.items, &extracted_ts, &extracted_coeffs);

    try testing.expectEqualSlices(u64, timestamps[0..], extracted_ts.items);
    try testing.expectEqualSlices(f64, coefficients[0..], extracted_coeffs.items);
}

test "rebuildSlide rejects mismatched input lengths" {
    const allocator = testing.allocator;
    var compressed = ArrayList(u8).init(allocator);
    defer compressed.deinit();

    const timestamps = [_]u64{ 1, 2, 3 };
    const coefficients = [_]f64{ 1.0, 2.0 };

    try testing.expectError(Error.UnsupportedInput, rebuildSlide(
        timestamps[0..],
        coefficients[0..],
        &compressed,
    ));
}

test "extract and rebuild SimPiece round trip" {
    const allocator = testing.allocator;

    const timestamps = [_]u64{ 1, 2, 4, 6, 12 };
    const coefficients = [_]f64{ 1.25, 0.75 };

    var compressed = ArrayList(u8).init(allocator);
    defer compressed.deinit();
    try rebuildSimPiece(timestamps[0..], coefficients[0..], &compressed);

    var extracted_ts = ArrayList(u64).init(allocator);
    defer extracted_ts.deinit();
    var extracted_coeffs = ArrayList(f64).init(allocator);
    defer extracted_coeffs.deinit();
    try extractSimPiece(compressed.items, &extracted_ts, &extracted_coeffs);

    try testing.expectEqualSlices(u64, timestamps[0..], extracted_ts.items);
    try testing.expectEqualSlices(f64, coefficients[0..], extracted_coeffs.items);
}

test "extract and rebuild MixPiece round trip" {
    const allocator = testing.allocator;

    const timestamps = [_]u64{ 1, 1, 1, 2, 2, 3, 4, 1, 8, 2, 5, 6, 10, 99 };
    const coefficients = [_]f64{ 1.1, 0.5, 1.5, 2.5, 0.25, 0.75, 3.5, -1.25 };

    var compressed = ArrayList(u8).init(allocator);
    defer compressed.deinit();
    try rebuildMixPiece(timestamps[0..], coefficients[0..], &compressed);

    var extracted_ts = ArrayList(u64).init(allocator);
    defer extracted_ts.deinit();
    var extracted_coeffs = ArrayList(f64).init(allocator);
    defer extracted_coeffs.deinit();
    try extractMixPiece(compressed.items, &extracted_ts, &extracted_coeffs);

    try testing.expectEqualSlices(u64, timestamps[0..], extracted_ts.items);
    try testing.expectEqualSlices(f64, coefficients[0..], extracted_coeffs.items);
}

test "extract and rebuild PWCH round trip" {
    const allocator = testing.allocator;

    const timestamps = [_]u64{ 5, 10 };
    const coefficients = [_]f64{ 1.5, -0.25 };

    var compressed = ArrayList(u8).init(allocator);
    defer compressed.deinit();
    try rebuildPWCH(timestamps[0..], coefficients[0..], &compressed);

    var extracted_ts = ArrayList(u64).init(allocator);
    defer extracted_ts.deinit();
    var extracted_coeffs = ArrayList(f64).init(allocator);
    defer extracted_coeffs.deinit();
    try extractPWCH(compressed.items, &extracted_ts, &extracted_coeffs);

    try testing.expectEqualSlices(u64, timestamps[0..], extracted_ts.items);
    try testing.expectEqualSlices(f64, coefficients[0..], extracted_coeffs.items);
}

test "rebuildPWCH rejects mismatched input lengths" {
    const allocator = testing.allocator;
    var compressed = ArrayList(u8).init(allocator);
    defer compressed.deinit();

    const timestamps = [_]u64{1};
    const coefficients = [_]f64{ 1.0, 2.0 };

    try testing.expectError(Error.UnsupportedInput, rebuildPWCH(
        timestamps[0..],
        coefficients[0..],
        &compressed,
    ));
}

test "extract and rebuild works for any compression method" {
    const allocator = testing.allocator;
    const random = tester.getDefaultRandomGenerator();

    // Input data
    var uncompressed_values = ArrayList(f64).init(allocator);
    defer uncompressed_values.deinit();
    try tester.generateBoundedRandomValues(
        &uncompressed_values,
        -tester.max_test_value,
        tester.max_test_value,
        random,
    );

    // Test each method.
    inline for (std.meta.fields(tersets.Method)) |method_field| {
        const method: tersets.Method = @enumFromInt(method_field.value);

        const method_configuration = try configuration.defaultConfigurationBuilder(
            allocator,
            method,
        );
        defer allocator.free(method_configuration);

        const compressed_values = try tersets.compress(
            allocator,
            uncompressed_values.items,
            method,
            method_configuration,
        );
        defer compressed_values.deinit();

        const decompressed_values = try tersets.decompress(
            allocator,
            compressed_values.items,
        );
        defer decompressed_values.deinit();

        // Test extract and rebuild.
        var coefficient_values = ArrayList(f64).init(allocator);
        defer coefficient_values.deinit();
        var timestamp_values = ArrayList(u64).init(allocator);
        defer timestamp_values.deinit();

        try tersets.extract(
            compressed_values.items,
            &timestamp_values,
            &coefficient_values,
        );

        const rebuild_values = try tersets.rebuild(
            allocator,
            timestamp_values.items,
            coefficient_values.items,
            method,
        );
        defer rebuild_values.deinit();
    }
}
