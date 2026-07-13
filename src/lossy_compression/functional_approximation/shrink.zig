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

//! Implementation of the SHRINK algorithm from the paper
//! "Guoyou Sun, Panagiotis Karras, and Qi Zhang.
//! SHRINK: Data Compression by Semantic Extraction and Residuals Encoding.
//! arXiv:2410.06713, 2024.
//! https://arxiv.org/abs/2410.06713".
//!
//! This implementation reuses `shared_structs.SegmentMetadata` to represent cones
//! `(start_index, intercept = Theta, lower_bound_slope = Psi-, upper_bound_slope = Psi+)`
//! and reuses Sim-Piece's segment-merging routine, since both algorithms solve the same
//! sub-problem (greedily merging overlapping slope intervals that share an origin).

const std = @import("std");
const math = std.math;
const testing = std.testing;
const ArrayList = std.ArrayList;
const Allocator = std.mem.Allocator;

const tersets = @import("../../tersets.zig");
const configuration = @import("../../configuration.zig");

const Method = tersets.Method;
const Error = tersets.Error;
const shared_structs = @import("../../utilities/shared_structs.zig");
const shared_functions = @import("../../utilities/shared_functions.zig");
const DiscretePoint = shared_structs.DiscretePoint;
const SegmentMetadata = shared_structs.SegmentMetadata;

const sim_piece = @import("sim_piece.zig");

const tester = @import("../../tester.zig");

/// Compresses `uncompressed_values` using the "SHRINK" algorithm. The function writes the result
/// to `compressed_values`. The `allocator` is used for memory allocation of intermediate data
/// structures and the `method_configuration` parser. The `method_configuration` is expected to be
/// of `ShrinkConfiguration` type, otherwise `Error.InvalidConfiguration` is returned. If any other
/// error occurs during the execution of the method, it is returned.
pub fn compress(
    allocator: Allocator,
    uncompressed_values: []const f64,
    compressed_values: *ArrayList(u8),
    method_configuration: []const u8,
) Error!void {
    const parsed_configuration = try configuration.parse(
        allocator,
        configuration.ShrinkConfiguration,
        method_configuration,
    );

    const base_error_bound: f32 = parsed_configuration.base_error_bound;
    // The original SHRINK paper proposes a scheme with different error resolutions for the
    // residual part. In this implementation, the error bound is fixed for the whole sequence.
    const residual_error_bound: f32 = parsed_configuration.residual_error_bound;
    // `lambda` controls the default interval length `L = lambda * n * base_error_bound` used to
    // estimate the local fluctuation level of the data, see SHRINK paper Section III-B, Eq. (4).
    const lambda: f32 = parsed_configuration.lambda;

    if (base_error_bound == 0.0) {
        return Error.InvalidConfiguration;
    }
    if (residual_error_bound < 0.0 or residual_error_bound > base_error_bound) {
        return Error.InvalidConfiguration;
    }
    if (lambda <= 0.0 or lambda > 1.0) {
        return Error.InvalidConfiguration;
    }

    // SHRINK Phase 1 (Section III-B, Algorithms 2-3): compute cones using a base error threshold
    // that adapts to local data fluctuation.
    // This is based on `computeSegmentsMetadata` from sim_piece.zig.
    var segments_metadata = ArrayList(SegmentMetadata).empty;
    defer segments_metadata.deinit(allocator);
    try computeAdaptiveSegmentsMetadata(
        allocator,
        uncompressed_values,
        &segments_metadata,
        base_error_bound,
        lambda,
    );

    // SHRINK Phase 2 (Section III-C, Algorithm 4): merge cones that share a quantized origin and
    // whose slope intervals overlap into the knowledge base. This step is identical to Sim-Piece's
    // segment-merging phase, so it is reused directly.
    var base_segments_metadata = ArrayList(SegmentMetadata).empty;
    defer base_segments_metadata.deinit(allocator);
    try sim_piece.mergeSegmentsMetadata(allocator, segments_metadata, &base_segments_metadata);

    // SHRINK Phase 3 (Section III-D, Algorithms 5-6): pick a clean candidate slope per sub-base
    // segment and compute the residuals between the original values and the resulting line.
    var residuals = ArrayList(i64).empty;
    defer residuals.deinit(allocator);
    try computeResiduals(
        allocator,
        uncompressed_values,
        base_segments_metadata.items,
        residual_error_bound,
        &residuals,
    );

    // Serialize the knowledge base followed by the quantized, entropy-coded residual stream.
    try writeBase(allocator, base_segments_metadata.items, uncompressed_values.len, compressed_values);
    try writeResiduals(allocator, residuals.items, residual_error_bound, compressed_values);
}

/// Decompresses `compressed_values` produced by SHRINK. The function writes the result to
/// `decompressed_values`. The `allocator` is used for memory allocation of intermediate data
/// structures. If an error occurs it is returned.
pub fn decompress(
    allocator: Allocator,
    compressed_values: []const u8,
    decompressed_values: *ArrayList(f64),
) Error!void {
    var offset: usize = 0;

    var base_segments_metadata = ArrayList(SegmentMetadata).empty;
    defer base_segments_metadata.deinit(allocator);
    const series_length = try readBase(allocator, compressed_values, &offset, &base_segments_metadata);

    var stored = ArrayList(u64).empty;
    defer stored.deinit(allocator);
    const header = try readResiduals(allocator, compressed_values, &offset, &stored);

    var residuals = ArrayList(i64).empty;
    defer residuals.deinit(allocator);
    for (stored.items) |value| {
        try residuals.append(allocator, @as(i64, @intCast(value - 1)) + header.r_min);
    }

    // Combine residuals and base segments.
    var current_index: usize = 0;
    for (0..base_segments_metadata.items.len) |segment_index| {
        const current_metadata = base_segments_metadata.items[segment_index];
        const next_start_index = if (segment_index + 1 < base_segments_metadata.items.len)
            base_segments_metadata.items[segment_index + 1].start_index
        else
            series_length;

        try decompressSegment(
            allocator,
            current_metadata,
            current_index,
            next_start_index,
            residuals.items,
            header.error_bound,
            decompressed_values,
        );
        current_index = next_start_index;
    }
}

/// Extracts `indices` and `coefficients` from Shrink's `compressed_values`.
/// The compressed stream consists of a base section followed by a residuals section.
/// The base section encodes: a segment count (usize), for each segment: (start_index, intercept,
/// lower_bound_slope, upper_bound_slope), and the series length (usize).
/// The residuals section encodes: residual_error_bound (f32), r_min (i64), residuals_count (usize),
/// and optionally an Elias-Gamma-encoded block of shifted residual values.
/// A `indices` ArrayList stores the segment count, all start indices, the series length,
/// r_min (bitcast to u64), the residuals count, and the decoded Elias-Gamma stored values.
/// A `coefficients` ArrayList stores the per-segment (intercept, lower_bound_slope,
/// upper_bound_slope) values and the residual_error_bound.
/// Any loss or misalignment of the indices may produce a stream that cannot be decompressed or
/// yields incorrect results. Only structural checks are performed by the underlying reader
/// functions. The caller must ensure semantic validity. If the compressed stream does not
/// follow the expected representation, `Error.CorruptedCompressedData` is returned.
/// The `allocator` handles the memory allocations of the output arrays.
/// Allocation errors are propagated.
pub fn extract(
    allocator: Allocator,
    compressed_values: []const u8,
    indices: *ArrayList(u64),
    coefficients: *ArrayList(f64),
) Error!void {
    var offset: usize = 0;

    // Extract the base segments.
    var base_segments_metadata = ArrayList(SegmentMetadata).empty;
    defer base_segments_metadata.deinit(allocator);
    const series_length = try readBase(allocator, compressed_values, &offset, &base_segments_metadata);

    try indices.append(allocator, base_segments_metadata.items.len);
    for (base_segments_metadata.items) |segment| {
        try indices.append(allocator, segment.start_index);
        try coefficients.append(allocator, segment.intercept);
        try coefficients.append(allocator, segment.lower_bound_slope);
        try coefficients.append(allocator, segment.upper_bound_slope);
    }
    try indices.append(allocator, series_length);

    // Extract the residuals section.
    var stored = ArrayList(u64).empty;
    defer stored.deinit(allocator);
    const header = try readResiduals(allocator, compressed_values, &offset, &stored);

    try coefficients.append(allocator, @as(f64, @floatCast(header.error_bound)));
    try indices.append(allocator, @as(u64, @bitCast(header.r_min)));
    try indices.append(allocator, header.count);

    for (stored.items) |value| {
        try indices.append(allocator, value);
    }
}

/// Rebuilds Shrink's `compressed_values` from the provided `indices` and `coefficients`.
/// The encoding matches the layout produced by `extract`. The function reconstructs the
/// base section (segment count, per-segment metadata, series length) followed by the
/// residuals section (residual_error_bound, r_min, residuals count, Elias-Gamma-encoded
/// block). The `indices` array provides the integer metadata and stored residual values,
/// while the `coefficients` array provides the floating-point parameters.
/// Any loss or misalignment of the indices or coefficients, such as incorrect segment
/// count, mismatched residual count, or corrupted slope bounds, may produce a compressed
/// stream that cannot be decompressed or yields incorrect results. Only structural validation
/// is performed. Semantic consistency must be ensured by the caller. If the arrays do not
/// match the expected layout, `Error.CorruptedCompressedData` is returned. The `allocator`
/// handles the memory allocations of the output array. Allocation errors are propagated.
pub fn rebuild(
    allocator: Allocator,
    indices: []const u64,
    coefficients: []const f64,
    compressed_values: *ArrayList(u8),
) Error!void {
    // Write the base section.
    const segments_count = indices[0];
    try shared_functions.appendValue(allocator, usize, segments_count, compressed_values);

    var indices_offset: usize = 1;
    var coefficients_offset: usize = 0;
    for (0..segments_count) |_| {
        const start_index = indices[indices_offset];
        indices_offset += 1;
        const intercept = coefficients[coefficients_offset];
        const lower_bound_slope = coefficients[coefficients_offset + 1];
        const upper_bound_slope = coefficients[coefficients_offset + 2];
        coefficients_offset += 3;

        try shared_functions.appendValue(allocator, usize, start_index, compressed_values);
        try shared_functions.appendValue(allocator, f64, intercept, compressed_values);
        try shared_functions.appendValue(allocator, f64, lower_bound_slope, compressed_values);
        try shared_functions.appendValue(allocator, f64, upper_bound_slope, compressed_values);
    }

    const series_length = indices[indices_offset];
    indices_offset += 1;
    try shared_functions.appendValue(allocator, usize, series_length, compressed_values);

    // Write the residuals section.
    const residual_error_bound: f64 = coefficients[coefficients_offset];
    try shared_functions.appendValue(allocator, f32, @as(f32, @floatCast(residual_error_bound)), compressed_values);

    const r_min: i64 = @as(i64, @bitCast(indices[indices_offset]));
    indices_offset += 1;
    try shared_functions.appendValue(allocator, i64, r_min, compressed_values);

    const residuals_count = indices[indices_offset];
    indices_offset += 1;
    try shared_functions.appendValue(allocator, usize, residuals_count, compressed_values);

    if (residuals_count > 0) {
        const stored_values = indices[indices_offset .. indices_offset + residuals_count];

        var encoded = ArrayList(u8).empty;
        defer encoded.deinit(allocator);
        try shared_functions.encodeEliasGamma(allocator, stored_values, &encoded);

        try shared_functions.appendValue(allocator, usize, encoded.items.len, compressed_values);
        try compressed_values.appendSlice(allocator, encoded.items);
    }
}

/// SHRINK Phase 1. Computes the `segments_metadata` (cones) for `uncompressed_values` using an
/// adaptive base error threshold derived from `base_error_bound` and `lambda`. This mirrors
/// Sim-Piece's `computeSegmentsMetadata`, with the difference that the quantization step used at
/// the start of each new cone is recomputed per Eq. (4) of the SHRINK paper instead of staying
/// fixed for the whole series. `allocator` is used to allocate `segments_metadata`.
fn computeAdaptiveSegmentsMetadata(
    allocator: Allocator,
    uncompressed_values: []const f64,
    segments_metadata: *ArrayList(SegmentMetadata),
    base_error_bound: f32,
    lambda: f32,
) Error!void {
    if (uncompressed_values.len == 0) return;

    // Check if the first point is NaN, infinite, or a reduced precision f64. If so, return error.
    if (!math.isFinite(uncompressed_values[0]) or @abs(uncompressed_values[0]) > tester.max_test_value)
        return Error.UnsupportedInput;

    // Used for phase division.
    const global_range = computeRange(uncompressed_values);

    var upper_bound_slope: f64 = math.floatMax(f64);
    var lower_bound_slope: f64 = -math.floatMax(f64);

    // Initialize the `start_point` with the first uncompressed value.
    var start_point: DiscretePoint = .{ .index = 0, .value = uncompressed_values[0] };

    var adaptive_error_bound = adaptiveErrorBound(
        uncompressed_values,
        0,
        base_error_bound,
        lambda,
        global_range,
    );
    var quantized_intercept = quantize(uncompressed_values[0], adaptive_error_bound);

    for (1..uncompressed_values.len) |current_index| {
        if (!math.isFinite(uncompressed_values[current_index])) {
            return Error.UnsupportedInput;
        }

        const end_point: DiscretePoint = .{
            .index = current_index,
            .value = uncompressed_values[current_index],
        };

        const segment_size: f64 = @floatFromInt(current_index - start_point.index);
        const upper_limit: f64 = upper_bound_slope * segment_size + quantized_intercept;
        const lower_limit: f64 = lower_bound_slope * segment_size + quantized_intercept;

        if ((upper_limit < (end_point.value - adaptive_error_bound)) or
            (lower_limit > (end_point.value + adaptive_error_bound)))
        {
            // The cone cannot be extended to include `end_point`. Close the current cone and
            // start a new one rooted at `end_point`.
            try segments_metadata.append(allocator, .{
                .start_index = start_point.index,
                .intercept = quantized_intercept,
                .upper_bound_slope = upper_bound_slope,
                .lower_bound_slope = lower_bound_slope,
            });

            start_point = end_point;
            adaptive_error_bound = adaptiveErrorBound(
                uncompressed_values,
                current_index, // Same as end_point.index.
                base_error_bound,
                lambda,
                global_range,
            );
            quantized_intercept = quantize(start_point.value, adaptive_error_bound);
            upper_bound_slope = math.floatMax(f64);
            lower_bound_slope = -math.floatMax(f64);
        } else {
            // The new point is within the upper and lower bounds. Update the bounds' slopes.
            const new_upper_bound_slope: f64 =
                (end_point.value + adaptive_error_bound - quantized_intercept) / segment_size;
            const new_lower_bound_slope: f64 =
                (end_point.value - adaptive_error_bound - quantized_intercept) / segment_size;

            if (end_point.value + adaptive_error_bound < upper_limit)
                upper_bound_slope = @max(new_upper_bound_slope, lower_bound_slope);
            if (end_point.value - adaptive_error_bound > lower_limit)
                lower_bound_slope = @min(new_lower_bound_slope, upper_bound_slope);
        }
    }

    const segment_size = uncompressed_values.len - start_point.index;
    if (segment_size > 0) {
        // Append the final segment.
        if (segment_size == 1) {
            upper_bound_slope = 0;
            lower_bound_slope = 0;
        }
        try segments_metadata.append(allocator, .{
            .start_index = start_point.index,
            .intercept = quantized_intercept,
            .upper_bound_slope = upper_bound_slope,
            .lower_bound_slope = lower_bound_slope,
        });
    }
}

/// Computes the global value range (max - min) of `values`, used as Delta in Eq. (4). Returns
/// a minimum of a small positive epsilon to avoid division by zero for constant series.
fn computeRange(values: []const f64) f64 {
    var min_value = values[0];
    var max_value = values[0];
    for (values) |value| {
        if (math.isFinite(value)) {
            min_value = @min(min_value, value);
            max_value = @max(max_value, value);
        }
    }
    const range = max_value - min_value;
    return if (range > 0) range else 1e-7;
}

/// Computes the adaptive base error threshold `epsilon_hat_b` for the interval of default length
/// `L = lambda * n * base_error_bound` starting at `start_index`, following SHRINK Algorithm 2
/// (Phases Division).
fn adaptiveErrorBound(
    uncompressed_values: []const f64,
    start_index: usize,
    base_error_bound: f32,
    lambda: f32,
    global_range: f64,
) f32 {
    const n: f64 = @floatFromInt(uncompressed_values.len);
    const L: f64 = @as(f64, lambda) * n * @as(f64, base_error_bound);

    const interval_length: usize = @max(1, @as(usize, @intFromFloat(L)));
    const end_index = @min(uncompressed_values.len, start_index + interval_length);
    const interval = computeRange(uncompressed_values[start_index..end_index]);

    const beta_i = interval / global_range;
    const exponent: f64 = (2.0 / 3.0) - beta_i;
    const adaptive: f64 = @as(f64, base_error_bound) * @exp(exponent);
    return @floatCast(adaptive);
}

/// Quantizes `value` by the given `error_bound`, identical to Sim-Piece's `quantize`.
/// In the SHRINK paper this is line 12 of the of Algorithm 2.
fn quantize(value: f64, error_bound: f32) f64 {
    if (error_bound != 0) {
        return @floor(value / error_bound) * error_bound;
    }
    return value;
}

/// SHRINK Phase 3 (Section III-D). For each sub-base segment in `base_segments_metadata`, selects
/// a candidate slope (Algorithm 5) and computes the quantized residual (Algorithm 6, Eq. 6) for
/// every point covered by that segment. The residuals are appended, in series order, to
/// `residuals` as signed integers representing the quantization bucket index relative to the
/// segment's residual range, ready for r_min-shifted Elias-Gamma encoding.
/// If `residual_error_bound` is zero, no residuals are computed and `residuals` stays empty,
/// meaning decompression will only be able to reconstruct the (looser) base approximation.
fn computeResiduals(
    allocator: Allocator,
    uncompressed_values: []const f64,
    base_segments_metadata: []const SegmentMetadata,
    residual_error_bound: f32,
    residuals: *ArrayList(i64),
) Error!void {
    // No residual part.
    if (residual_error_bound == 0.0) return;

    for (0..base_segments_metadata.len) |segment_index| {
        const current_metadata = base_segments_metadata[segment_index];

        // Getting the interval.
        const next_start_index = if (segment_index + 1 < base_segments_metadata.len)
            base_segments_metadata[segment_index + 1].start_index
        else
            uncompressed_values.len;

        // Computing the slope.
        const slope = candidateSlope(
            current_metadata.lower_bound_slope,
            current_metadata.upper_bound_slope,
        );

        for (current_metadata.start_index..next_start_index) |i| {
            const predicted = slope * @as(f64, @floatFromInt(i - current_metadata.start_index)) +
                current_metadata.intercept;
            const residual = uncompressed_values[i] - predicted;

            // Section III-D, Eq. 6. The minimum quantized residual is subtracted in
            // `writeResiduals` to shift all values toward zero before
            // Elias-Gamma encoding, matching the paper's approach.
            const quantized: i64 = @intFromFloat(@floor(residual / residual_error_bound));
            try residuals.append(allocator, quantized);
        }
    }
}

/// SHRINK Algorithm 5 (Candidate Line Selection). Given the lower and upper slope bounds of a
/// sub-base segment, returns a "clean" candidate slope rather than the exact average, by
/// truncating to the longest common decimal prefix of `lower_bound_slope` and `upper_bound_slope`
/// when they share the same integer part, or to one decimal digit of their average otherwise.
/// This avoids carrying unnecessarily high floating-point precision in the slope while remaining
/// within the cone, matching the paper's worked example: `lower_bound_slope =
/// 0.12385382076923077`, `upper_bound_slope = 0.12389554722222222` yields `0.12387` rather than
/// the exact average `0.12387468399572649`.
fn candidateSlope(lower_bound_slope: f64, upper_bound_slope: f64) f64 {
    if (!math.isFinite(lower_bound_slope) or !math.isFinite(upper_bound_slope)) {
        if (math.isFinite(lower_bound_slope)) return lower_bound_slope;
        if (math.isFinite(upper_bound_slope)) return upper_bound_slope;
        return 0;
    }

    // Compare integer part (lead).
    const lead_lo = @floor(lower_bound_slope);
    const lead_hi = @floor(upper_bound_slope);
    if (lead_lo != lead_hi) {
        const average = (lower_bound_slope + upper_bound_slope) / 2.0;
        return @round(average * 10.0) / 10.0;
    }

    // Compare decimal part (tail).
    const tail_lo = lower_bound_slope - lead_lo;
    const tail_hi = upper_bound_slope - lead_hi;
    var scale: f64 = 1.0;
    for (0..5) |_| {
        const next_scale = scale * 10.0;
        const digit_lo = @floor(tail_lo * next_scale) - @floor(tail_lo * scale) * 10.0;
        const digit_hi = @floor(tail_hi * next_scale) - @floor(tail_hi * scale) * 10.0;
        if (digit_lo != digit_hi) {
            const avg_digit = @round((digit_lo + digit_hi) / 2.0);
            const prefix = @floor(tail_lo * scale);
            const tail = (prefix * 10.0 + avg_digit) / next_scale;
            return lead_lo + tail;
        }
        scale = next_scale;
    }
    // If everything matches within the defined precision compute the average directly.
    return (lower_bound_slope + upper_bound_slope) / 2.0;
}

/// Writes the knowledge `base_segments_metadata` to `compressed_values` prefixed by the
/// number of segments and followed by the total series length.
fn writeBase(
    allocator: Allocator,
    base_segments_metadata: []const SegmentMetadata,
    series_length: usize,
    compressed_values: *ArrayList(u8),
) Error!void {
    try shared_functions.appendValue(allocator, usize, base_segments_metadata.len, compressed_values);
    for (base_segments_metadata) |segment_metadata| {
        try shared_functions.appendValue(allocator, usize, segment_metadata.start_index, compressed_values);
        try shared_functions.appendValue(allocator, f64, segment_metadata.intercept, compressed_values);
        try shared_functions.appendValue(allocator, f64, segment_metadata.lower_bound_slope, compressed_values);
        try shared_functions.appendValue(allocator, f64, segment_metadata.upper_bound_slope, compressed_values);
    }
    try shared_functions.appendValue(allocator, usize, series_length, compressed_values);
}

/// Reads the knowledge base written by `writeBase` from `compressed_values`. Returns the
/// total series length encoded after the base and the offset where the residual encodings
/// start. The `allocator` is used to allocate `base_segments_metadata`.
fn readBase(
    allocator: Allocator,
    compressed_values: []const u8,
    offset: *usize,
    base_segments_metadata: *ArrayList(SegmentMetadata),
) Error!usize {
    const segments_count = try shared_functions.readOffsetValue(usize, compressed_values, offset);
    for (0..segments_count) |_| {
        const start_index = try shared_functions.readOffsetValue(usize, compressed_values, offset);
        const intercept = try shared_functions.readOffsetValue(f64, compressed_values, offset);
        const lower_bound_slope = try shared_functions.readOffsetValue(f64, compressed_values, offset);
        const upper_bound_slope = try shared_functions.readOffsetValue(f64, compressed_values, offset);
        try base_segments_metadata.append(allocator, .{
            .start_index = start_index,
            .intercept = intercept,
            .lower_bound_slope = lower_bound_slope,
            .upper_bound_slope = upper_bound_slope,
        });
    }
    return try shared_functions.readOffsetValue(usize, compressed_values, offset);
}

/// Writes `residuals` to `compressed_values`, prefixed by the `residual_error_bound` used to
/// quantize them, the minimum quantized residual `r_min` for shifting, and the number of residuals.
/// Residuals are shifted by `r_min` to produce unsigned values in `[0, K]`, incremented by one
/// since Elias-Gamma encoding is undefined for zero (see `shared_functions.encodeEliasGamma`),
/// and then Elias-Gamma encoded. This follows the SHRINK paper (Section III-D) which subtracts
/// the minimum residual before quantizing. If `residuals` is empty, only the (zero) residual error
/// bound, a zero r_min, and a zero count are written; decompression then returns the base
/// approximation unchanged.
fn writeResiduals(
    allocator: Allocator,
    residuals: []const i64,
    residual_error_bound: f32,
    compressed_values: *ArrayList(u8),
) Error!void {
    // Find the minimum quantized residual to shift residuals toward zero,
    // aligning with the SHRINK paper (Section III-D, Eq. 6).
    var r_min: i64 = 0;
    for (residuals) |r| {
        if (r < r_min) r_min = r;
    }

    try shared_functions.appendValue(allocator, f32, residual_error_bound, compressed_values);
    try shared_functions.appendValue(allocator, i64, r_min, compressed_values);
    try shared_functions.appendValue(allocator, usize, residuals.len, compressed_values);
    if (residuals.len == 0) return;

    var shifted_values = ArrayList(u64).empty;
    defer shifted_values.deinit(allocator);
    for (residuals) |residual| {
        // Shift by r_min to produce non-negative values, then add one since
        // Elias-Gamma encoding is undefined for zero.
        try shifted_values.append(allocator, @as(u64, @intCast(residual - r_min)) + 1);
    }

    var encoded_residuals = ArrayList(u8).empty;
    defer encoded_residuals.deinit(allocator);
    try shared_functions.encodeEliasGamma(allocator, shifted_values.items, &encoded_residuals);

    try shared_functions.appendValue(allocator, usize, encoded_residuals.items.len, compressed_values);
    try compressed_values.appendSlice(allocator, encoded_residuals.items);
}

/// Header fields from the residuals section of the compressed stream.
const ResidualsHeader = struct {
    error_bound: f32,
    /// The minimum quantized residual value, used to shift residuals toward zero.
    r_min: i64,
    count: usize,
};

/// Reads the residual section header and, if present, decodes the Elias-Gamma block into
/// `stored`. The allocator is used for decoding. Returns the header fields.
fn readResiduals(
    allocator: Allocator,
    compressed_values: []const u8,
    offset: *usize,
    stored: *ArrayList(u64),
) Error!ResidualsHeader {
    const error_bound = try shared_functions.readOffsetValue(f32, compressed_values, offset);
    const r_min = try shared_functions.readOffsetValue(i64, compressed_values, offset);
    const count = try shared_functions.readOffsetValue(usize, compressed_values, offset);
    if (count > 0) {
        const encoded_len = try shared_functions.readOffsetValue(usize, compressed_values, offset);
        const encoded_slice = compressed_values[offset.* .. offset.* + encoded_len];
        offset.* += encoded_len;
        try shared_functions.decodeEliasGamma(allocator, encoded_slice, stored);
        if (stored.items.len != count) return Error.CorruptedCompressedData;
    }
    return .{ .error_bound = error_bound, .r_min = r_min, .count = count };
}

/// Reconstructs the decompressed values for indices `[start_index, end_index)` covered by
/// `segment_metadata`, evaluating the base linear approximation and adding back the dequantized
/// residual at each point if `residuals` is non-empty.
fn decompressSegment(
    allocator: Allocator,
    segment_metadata: SegmentMetadata,
    start_index: usize,
    end_index: usize,
    residuals: []const i64,
    residual_error_bound: f32,
    decompressed_values: *ArrayList(f64),
) Error!void {
    const slope = candidateSlope(segment_metadata.lower_bound_slope, segment_metadata.upper_bound_slope);
    for (start_index..end_index) |i| {
        // Semantic part.
        const base_value = slope * @as(f64, @floatFromInt(
            i - segment_metadata.start_index,
        )) + segment_metadata.intercept;

        // Residual part.
        const decompressed_value = if (residuals.len > 0)
            base_value + dequantizeResidual(residuals[i], residual_error_bound)
        else
            base_value;

        try decompressed_values.append(allocator, decompressed_value);
    }
}

/// Reverses the quantization performed in `computeResiduals`.
fn dequantizeResidual(quantized_residual: i64, residual_error_bound: f32) f64 {
    const bucket: f64 = @as(f64, @floatFromInt(quantized_residual)) * residual_error_bound;
    return bucket + (@as(f64, residual_error_bound) / 2.0);
}

test "candidateSlope matches the paper's worked example" {
    const slope = candidateSlope(0.12385382076923077, 0.12389554722222222);
    try testing.expect(@abs(slope - 0.12387) < 1e-5);
}

test "candidateSlope handles unbounded cones from single-point segments" {
    try testing.expectEqual(@as(f64, 0), candidateSlope(0, 0));
}

test "shrink cannot compress NaN values" {
    const allocator = testing.allocator;

    const uncompressed_values = &[4]f64{ 19.0, 48.0, math.nan(f64), 3.0 };

    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);

    const method_configuration =
        \\ {"base_error_bound": 0.1, "residual_error_bound": 0.01, "lambda": 0.01}
    ;

    try testing.expectError(
        Error.UnsupportedInput,
        compress(allocator, uncompressed_values, &compressed_values, method_configuration),
    );
}

test "shrink rejects residual error bound larger than base error bound" {
    const allocator = testing.allocator;

    const uncompressed_values = &[4]f64{ 19.0, 48.0, 28.0, 3.0 };

    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);

    const method_configuration =
        \\ {"base_error_bound": 0.1, "residual_error_bound": 0.5, "lambda": 0.05}
    ;

    try testing.expectError(Error.InvalidConfiguration, compress(
        allocator,
        uncompressed_values,
        &compressed_values,
        method_configuration,
    ));
}

test "shrink handles an empty series" {
    const allocator = testing.allocator;

    const uncompressed_values = &[_]f64{};

    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);

    const method_configuration =
        \\ {"base_error_bound": 0.1, "residual_error_bound": 0.0, "lambda": 0.1}
    ;

    try compress(allocator, uncompressed_values, &compressed_values, method_configuration);

    var decompressed_values = ArrayList(f64).empty;
    defer decompressed_values.deinit(allocator);
    try decompress(allocator, compressed_values.items, &decompressed_values);

    try testing.expectEqual(@as(usize, 0), decompressed_values.items.len);
}

test "shrink handles a single-point series" {
    const allocator = testing.allocator;

    const uncompressed_values = &[_]f64{42.5};

    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);

    const method_configuration =
        \\ {"base_error_bound": 0.1, "residual_error_bound": 0.01, "lambda": 0.1}
    ;

    try compress(allocator, uncompressed_values, &compressed_values, method_configuration);

    var decompressed_values = ArrayList(f64).empty;
    defer decompressed_values.deinit(allocator);
    try decompress(allocator, compressed_values.items, &decompressed_values);

    try testing.expectEqual(@as(usize, 1), decompressed_values.items.len);
    try testing.expect(@abs(decompressed_values.items[0] - 42.5) <= 0.1);
}

test "shrink handles a constant series exactly" {
    const allocator = testing.allocator;

    var uncompressed_values: [50]f64 = undefined;
    for (0..50) |i| uncompressed_values[i] = 7.0;

    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);

    const method_configuration =
        \\ {"base_error_bound": 0.001, "residual_error_bound": 0.0001, "lambda": 0.1}
    ;

    try compress(allocator, &uncompressed_values, &compressed_values, method_configuration);

    var decompressed_values = ArrayList(f64).empty;
    defer decompressed_values.deinit(allocator);
    try decompress(allocator, compressed_values.items, &decompressed_values);

    try testing.expectEqual(uncompressed_values.len, decompressed_values.items.len);
    for (decompressed_values.items) |value| {
        try testing.expect(@abs(value - 7.0) <= 1e-4);
    }
}

test "shrink round trip preserves length for a noisy sinusoid with residual correction" {
    const allocator = testing.allocator;
    var uncompressed_values: [200]f64 = undefined;
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();
    for (0..200) |i| {
        const t: f64 = @as(f64, @floatFromInt(i));
        const signal: f64 = 5.0 * @sin(t * 0.1) + 2.0 * @cos(t * 0.05);
        const noise: f64 = (random.float(f64) - 0.5) * 2.0;
        uncompressed_values[i] = signal + noise;
    }
    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);
    const method_configuration =
        \\ {"base_error_bound": 1.0, "residual_error_bound": 0.15, "lambda": 0.1}
    ;
    try compress(allocator, &uncompressed_values, &compressed_values, method_configuration);
    var decompressed_values = ArrayList(f64).empty;
    defer decompressed_values.deinit(allocator);
    try decompress(allocator, compressed_values.items, &decompressed_values);
    try testing.expectEqual(uncompressed_values.len, decompressed_values.items.len);

    var max_error: f64 = 0.0;
    for (uncompressed_values, decompressed_values.items) |original, decompressed| {
        const err = @abs(original - decompressed);
        max_error = @max(max_error, err);
    }
    try testing.expect(max_error <= 0.15 + 1e-6);
}

test "shrink rejects lambda greater than 1.0" {
    const allocator = testing.allocator;

    const uncompressed_values = &[5]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };

    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);

    const method_configuration =
        \\ {"base_error_bound": 0.1, "residual_error_bound": 0.01, "lambda": 1.5}
    ;

    try testing.expectError(Error.InvalidConfiguration, compress(
        allocator,
        uncompressed_values,
        &compressed_values,
        method_configuration,
    ));
}

test "shrink rejects negative residual_error_bound" {
    const allocator = testing.allocator;

    const uncompressed_values = &[4]f64{ 19.0, 48.0, 28.0, 3.0 };

    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);

    const method_configuration =
        \\ {"base_error_bound": 0.1, "residual_error_bound": -0.01, "lambda": 0.05}
    ;

    try testing.expectError(Error.InvalidConfiguration, compress(
        allocator,
        uncompressed_values,
        &compressed_values,
        method_configuration,
    ));
}

test "shrink rejects base_error_bound of zero" {
    const allocator = testing.allocator;

    const uncompressed_values = &[4]f64{ 19.0, 48.0, 28.0, 3.0 };

    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);

    const method_configuration =
        \\ {"base_error_bound": 0.0, "residual_error_bound": 0.0, "lambda": 0.05}
    ;

    try testing.expectError(Error.InvalidConfiguration, compress(
        allocator,
        uncompressed_values,
        &compressed_values,
        method_configuration,
    ));
}

test "shrink candidateSlope stays within cone bounds across a range of inputs" {
    const pairs = [_][2]f64{
        .{ 0.1, 0.9 },
        .{ -0.5, 0.5 },
        .{ 1.999, 2.001 },
        .{ 0.94, 1.03 },
        .{ -2.2, -1.8 },
        .{ 0.0001, 0.0009 },
    };
    for (pairs) |pair| {
        const slope = candidateSlope(pair[0], pair[1]);
        try testing.expect(slope >= pair[0] - 1e-9);
        try testing.expect(slope <= pair[1] + 1e-9);
    }
}
