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

//! Implementation of the Mixed-Type PLA algorithm from the paper
//! "G. Luo, K. Yi, S.-W. Cheng, Z. Li, W. Fan, C. He, and Y. Mu.
//! Piecewise Linear Approximation of Streaming Time Series Data with Max-Error Guarantees.
//! Proc. IEEE 31st Int'l Conf. Data Engineering (ICDE), Seoul, South Korea 2015, pp. 173–184.
//! https://ieeexplore.ieee.org/document/7113282".
//! The implementation is partially based on the authors' C++ implementation generously
//! provided by Prof. Ke Yi of Hong Kong University of Science and Technology.

const std = @import("std");
const math = std.math;
const mem = std.mem;
const testing = std.testing;
const ArrayList = std.ArrayList;
const Allocator = std.mem.Allocator;

const tersets = @import("../../tersets.zig");
const configuration = @import("../../configuration.zig");

const Method = tersets.Method;
const Error = tersets.Error;

const shared_structs = @import("../../utilities/shared_structs.zig");
const shared_functions = @import("../../utilities/shared_functions.zig");
const ContinousPoint = shared_structs.ContinousPoint;
const LinearFunction = shared_structs.LinearFunction;
const ParameterSpacePoint = shared_structs.ParameterSpacePoint;

const tester = @import("../../tester.zig");

const ext_poly = @import("../../utilities/extended_polygon.zig");

/// Compresses `uncompressed_values` within `error_bound` using the "Mixed-Type PLA" algorithm.
/// The function writes the result to `compressed_values`.  The `allocator` is used for
/// memory allocation of  intermediate data structures and the `method_configuration` parser. The
/// `method_configuration` is expected to be of `AbsoluteErrorBound` type;
/// otherwise an `InvalidConfiguration` error is returned. If any other error
/// occurs during execution, it is returned.
pub fn compress(
    allocator: Allocator,
    uncompressed_values: []const f64,
    compressed_values: *ArrayList(u8),
    method_configuration: []const u8,
) Error!void {
    const parsed_configuration = try configuration.parse(
        allocator,
        configuration.AbsoluteErrorBound,
        method_configuration,
    );

    const error_bound: f64 = @floatCast(parsed_configuration.abs_error_bound);

    // Adjust the error bound to avoid exceeding it during decompression due to numerical instabilities.
    const adjusted_error_bound = if (error_bound > shared_structs.ErrorBoundMargin)
        error_bound - shared_structs.ErrorBoundMargin
    else
        error_bound;

    // Normalization that maps data to [-1, 1] centered at the midpoint, to improve numerical stability
    // and allow the algorithm to adapt to the data amplitude. The normalization parameters are stored
    // in the compressed blob header so that decompress() can undo it.
    var min_val: f64 = math.inf(f64);
    var max_val: f64 = -math.inf(f64);
    for (uncompressed_values) |v| {
        min_val = @min(min_val, v);
        max_val = @max(max_val, v);
    }
    const norm_offset: f64 = (min_val + max_val) / 2.0;
    const norm_scale: f64 = blk: {
        const s = (max_val - min_val) / 2.0;
        break :blk if (s < 1e-15) 1.0 else s;
    };
    var normalized_values = try allocator.alloc(f64, uncompressed_values.len);
    defer allocator.free(normalized_values);
    for (uncompressed_values, 0..) |v, i| {
        normalized_values[i] = (v - norm_offset) / norm_scale;
    }

    // Dynamic epsilon computed from the adjusted error bound and normalization scale, to ensure that
    // geometric tests in the algorithm are appropriately tight for the actual data amplitude and
    // error tolerance. A safety margin is subtracted to account for numerical instabilities.
    const normalized_error_bound = adjusted_error_bound / norm_scale;
    const dynamic_eps = @min(1e-7, @max(1e-14, normalized_error_bound * 1e-4));
    const safety_margin = dynamic_eps * 2.0;
    const safe_error_bound = normalized_error_bound - safety_margin;
    const tolerances = ext_poly.Tolerances{
        .val = dynamic_eps,
        .time = ext_poly.sign_time_diff,
    };

    // Run the Mixed-Type PLA algorithm on normalized data.
    var state =
        MixedTypePlaState.create(allocator, safe_error_bound, 0.0, tolerances);
    defer state.deinit();
    state.run(normalized_values);

    // Serialize with the normalization header.
    try serializeSegments(
        allocator,
        state.output_segments.items,
        state.connectivity_flags.items,
        uncompressed_values.len,
        norm_offset,
        norm_scale,
        compressed_values,
    );
}

/// Decompress `compressed_values` produced by "Mixed-Type PLA". The function
/// writes the result to `decompressed_values`. The `allocator` is used for
/// memory allocation of intermediate data structures. If an error occurs it
/// is returned.
pub fn decompress(
    allocator: Allocator,
    compressed_values: []const u8,
    decompressed_values: *ArrayList(f64),
) Error!void {

    // Read the normalization header (first 16 bytes: 2 × f64).
    if (compressed_values.len < 16) return Error.CorruptedCompressedData;
    var hdr_offset: usize = 0;
    const norm_offset =
        try shared_functions.readOffsetValue(f64, compressed_values, &hdr_offset);
    const norm_scale =
        try shared_functions.readOffsetValue(f64, compressed_values, &hdr_offset);

    // Deserialize parameter-space knot points and bit-packed knot flags from the remainder.
    var segment_data = ArrayList(ContinousPoint).empty;
    defer segment_data.deinit(allocator);
    var knot_flags = ArrayList(bool).empty;
    defer knot_flags.deinit(allocator);
    var original_length: usize = 0;

    try deserializeSegments(
        allocator,
        compressed_values[hdr_offset..],
        &segment_data,
        &knot_flags,
        &original_length,
    );

    // Convert knot points into explicit line segments.
    var line_segments = ArrayList(LineSegment).empty;
    defer line_segments.deinit(allocator);
    try readSegments(
        allocator,
        segment_data.items,
        knot_flags.items,
        &line_segments,
    );

    // Reconstruct the full series by evaluating segments at integer timestamps. When a timestamp
    // is covered by multiple segments, prefer the one whose start_time matches the timestamp.
    try decompressed_values.ensureTotalCapacity(allocator, original_length);
    for (0..original_length) |i| {
        const t: f64 = @floatFromInt(i);

        // Find preferred segment: one whose start_time == t, or the first segment that contains t.
        var preferred: ?LineSegment = null;
        for (line_segments.items) |seg| {
            if (seg.start_time <= t and t <= seg.end_time) {
                if (preferred == null) {
                    preferred = seg;
                }
                if (@abs(seg.start_time - t) < 1e-10) {
                    preferred = seg;
                    break;
                }
            }
        }

        if (preferred) |seg| {
            // Denormalize the values: normalized_val * scale + offset
            const normalized_val = seg.evaluate(t);
            const denormalized_val = normalized_val * norm_scale + norm_offset;
            try decompressed_values.append(allocator, denormalized_val);
            // } else {
            //     // No segment covers this timestamp — output norm_offset as fallback.
            //     try decompressed_values.append(allocator, norm_offset);
        }
    }
}

/// Extracts `indices` and `coefficients` from Mixed-Type PLA's `compressed_values`.
/// Mixed-Type PLA consists of a normalization header, knot points, packed connectivity flags,
/// and final original length metadata.
/// Following the extract/rebuild split used across methods:
/// `indices` stores structure and metadata fields required to rebuild layout exactly:
/// `num_segments`, each knot timestamp as the exact `u64` bit pattern of `f64`, `num_flags`,
/// `flag_bytes_len`, packed flag bytes, and `original_length`.
/// `coefficients` stores values that can be altered without changing stream structure:
/// normalization `offset`, normalization `scale`, and each knot value.
/// Because the format is length-dependent with packed boolean flags, any loss of information on
/// counts, timestamps, or flag bytes can lead to failures during decompression. Lightweight
/// structural validation is performed. If the buffer runs out of data mid-structure or the full
/// buffer is not consumed, `Error.CorruptedCompressedData` is returned. The `allocator` handles
/// memory allocations for the output arrays. Allocation errors are propagated.
pub fn extract(
    allocator: Allocator,
    compressed_values: []const u8,
    indices: *ArrayList(u64),
    coefficients: *ArrayList(f64),
) Error!void {
    // A minimum blob is 16-byte header + 8-byte num_segments + 8-byte num_flags.
    // + 8-byte flag_bytes_len + 8-byte original_length = 48 bytes (zero knots).
    if (compressed_values.len < 48) return Error.CorruptedCompressedData;

    var pos: usize = 0;

    // Add the normalization header to coefficients.
    const norm_offset = try shared_functions
        .readOffsetValue(f64, compressed_values, &pos);
    const norm_scale = try shared_functions
        .readOffsetValue(f64, compressed_values, &pos);
    try coefficients.append(allocator, norm_offset);
    try coefficients.append(allocator, norm_scale);

    // Add the num_segments to indices.
    const num_segments = try shared_functions
        .readOffsetValue(u64, compressed_values, &pos);
    const num_segments_usize =
        math.cast(usize, num_segments) orelse return Error.CorruptedCompressedData;
    try indices.append(allocator, num_segments);

    // Add knot timestamps to indices (as exact f64 bit patterns) and knot values to coefficients.
    for (0..num_segments_usize) |_| {
        const t = try shared_functions
            .readOffsetValue(f64, compressed_values, &pos);
        const m = try shared_functions
            .readOffsetValue(f64, compressed_values, &pos);
        try indices.append(allocator, @as(u64, @bitCast(t)));
        try coefficients.append(allocator, m);
    }

    // Add num_flags to indices.
    const num_flags = try shared_functions
        .readOffsetValue(u64, compressed_values, &pos);
    try indices.append(allocator, num_flags);

    // Add flag_bytes_len to indices.
    const flag_bytes_len = try shared_functions
        .readOffsetValue(u64, compressed_values, &pos);
    const flag_bytes_len_usize =
        math.cast(usize, flag_bytes_len) orelse return Error.CorruptedCompressedData;
    try indices.append(allocator, flag_bytes_len);

    // Add individual flag bytes to indices.
    for (0..flag_bytes_len_usize) |_| {
        const byte_val = try shared_functions
            .readOffsetValue(u8, compressed_values, &pos);
        try indices.append(allocator, @as(u64, byte_val));
    }

    // Add original_length to indices.
    const original_length =
        try shared_functions.readOffsetValue(u64, compressed_values, &pos);
    try indices.append(allocator, original_length);

    if (pos != compressed_values.len) return Error.CorruptedCompressedData;
}

/// Rebuilds Mixed-Type PLA's `compressed_values` from the provided `indices` and `coefficients`.
/// The `indices` array must contain metadata in this order: `num_segments`, one knot timestamp
/// bit pattern (`u64` cast of `f64`) per segment, `num_flags`, `flag_bytes_len`, packed flag
/// bytes, and `original_length`.
/// The `coefficients` array must contain normalization offset and scale, followed by one knot
/// value per segment.
/// This function is the exact inverse of `extract()`, reassembling the binary format expected
/// by `decompress()`. Any loss of information on counts or flag values may cause unexpected
/// failures during decompression. The function assumes the input arrays are logically consistent
/// and performs only structural validation. If array consumption is incomplete or inconsistent,
/// `Error.CorruptedCompressedData` is returned. The `allocator` handles memory allocations for
/// the output array. Allocation errors are propagated.
pub fn rebuild(
    allocator: Allocator,
    indices: []const u64,
    coefficients: []const f64,
    compressed_values: *ArrayList(u8),
) Error!void {
    if (coefficients.len < 2) return Error.CorruptedCompressedData;
    if (indices.len < 4) return Error.CorruptedCompressedData;

    var ci: usize = 0; // coefficient cursor
    var ii: usize = 0; // index cursor

    // Normalization header.
    try shared_functions
        .appendValue(allocator, f64, coefficients[ci], compressed_values);
    ci += 1;
    try shared_functions
        .appendValue(allocator, f64, coefficients[ci], compressed_values);
    ci += 1;

    // num_segments.
    const num_segments = indices[ii];
    try shared_functions
        .appendValue(allocator, u64, num_segments, compressed_values);
    ii += 1;

    // Knot points: timestamp from indices bit pattern + value from coefficients.
    const num_segments_usize = math.cast(usize, num_segments) orelse return Error.CorruptedCompressedData;
    if (coefficients.len < 2 + num_segments_usize)
        return Error.CorruptedCompressedData;
    if (indices.len < 4 + num_segments_usize)
        return Error.CorruptedCompressedData;
    for (0..num_segments_usize) |_| {
        const t_bits = indices[ii];
        const t: f64 = @bitCast(t_bits);
        try shared_functions
            .appendValue(allocator, f64, t, compressed_values);
        ii += 1;

        try shared_functions
            .appendValue(allocator, f64, coefficients[ci], compressed_values);
        ci += 1;
    }

    // num_flags.
    const num_flags = indices[ii];
    try shared_functions
        .appendValue(allocator, u64, num_flags, compressed_values);
    ii += 1;

    // flag_bytes_len.
    const flag_bytes_len = indices[ii];
    try shared_functions
        .appendValue(allocator, u64, flag_bytes_len, compressed_values);
    ii += 1;

    // flag bytes.
    const flag_bytes_len_usize =
        math.cast(usize, flag_bytes_len) orelse return Error.CorruptedCompressedData;
    if (indices.len < 4 + num_segments_usize + flag_bytes_len_usize)
        return Error.CorruptedCompressedData;
    for (0..flag_bytes_len_usize) |_| {
        const raw_byte = indices[ii];
        if (raw_byte > math.maxInt(u8)) return Error.CorruptedCompressedData;
        try compressed_values.append(allocator, @as(u8, @intCast(raw_byte)));
        ii += 1;
    }

    // Original_length.
    if (ii >= indices.len) return Error.CorruptedCompressedData;
    try shared_functions
        .appendValue(allocator, u64, indices[ii], compressed_values);
    ii += 1;

    // Validate that both arrays were fully consumed.
    if (ci != coefficients.len) return Error.CorruptedCompressedData;
    if (ii != indices.len) return Error.CorruptedCompressedData;
}

/// Serializes the `segments` (knot points) and `knot_flags` (connectivity information) into the
/// binary format used by Mixed-Type PLA. The output includes the normalization parameters,
/// followed by the knot point count and the time-value pairs as f64 values. Connectivity flags
/// are then packed into bytes (one bit per flag, LSB first) and written along with their byte
/// count. The original series length is appended at the end. The `allocator`
/// handles memory allocations for the output array. Allocation errors are propagated.
fn serializeSegments(
    allocator: Allocator,
    segments: []const ContinousPoint,
    knot_flags: []const bool,
    original_length: usize,
    norm_offset: f64,
    norm_scale: f64,
    compressed_values: *ArrayList(u8),
) Error!void {
    // Normalization header.
    try shared_functions
        .appendValue(allocator, f64, norm_offset, compressed_values);
    try shared_functions
        .appendValue(allocator, f64, norm_scale, compressed_values);

    // Number of segment knot points.
    try shared_functions.appendValue(
        allocator,
        u64,
        math.cast(u64, segments.len) orelse return Error.CorruptedCompressedData,
        compressed_values,
    );

    // Segment knot points as (f64 index, f64 value) pairs.
    for (segments) |seg| {
        try shared_functions
            .appendValue(allocator, f64, seg.index, compressed_values);
        try shared_functions
            .appendValue(allocator, f64, seg.value, compressed_values);
    }

    // Number of knot flags.
    try shared_functions.appendValue(
        allocator,
        u64,
        math.cast(u64, knot_flags.len) orelse return Error.CorruptedCompressedData,
        compressed_values,
    );

    // Packed boolean flags: 1 bit per flag, 8 per byte, LSB first.
    const num_flag_bytes_usize = (knot_flags.len + 7) / 8;
    const num_flag_bytes =
        math.cast(u64, num_flag_bytes_usize) orelse return Error.CorruptedCompressedData;
    try shared_functions
        .appendValue(allocator, u64, num_flag_bytes, compressed_values);

    for (0..num_flag_bytes_usize) |byte_idx| {
        var byte_val: u8 = 0;
        for (0..8) |bit_idx| {
            const flag_idx = byte_idx * 8 + bit_idx;
            if (flag_idx < knot_flags.len and knot_flags[flag_idx]) {
                byte_val |= @as(u8, 1) << @intCast(bit_idx);
            }
        }
        try compressed_values.append(allocator, byte_val);
    }

    // Original series length.
    try shared_functions.appendValue(
        allocator,
        u64,
        math.cast(u64, original_length) orelse return Error.CorruptedCompressedData,
        compressed_values,
    );
}

/// Deserializes the binary format produced by `serializeSegments` back into segment points,
/// knot flags, and the original series length. The `compressed_values` slice must start
/// immediately after the 16-byte normalization header (i.e., the caller has already extracted
/// `norm_offset` and `norm_scale`). The function reconstructs the knot points from time-value
/// pairs, unpacks the connectivity flags from their byte representation, and recovers the
/// original series length. Any inconsistency in byte boundaries, missing data, or flag counts
/// can lead to failures. Lightweight structural checks are performed to detect truncation.
/// The `allocator` handles memory allocations for the output arrays. Allocation errors are
/// propagated.
fn deserializeSegments(
    allocator: Allocator,
    compressed_values: []const u8,
    segment_data: *ArrayList(ContinousPoint),
    knot_flags: *ArrayList(bool),
    original_length: *usize,
) Error!void {
    var offset: usize = 0;

    // Number of segment knot points.
    const num_points = try shared_functions
        .readOffsetValue(u64, compressed_values, &offset);
    const num_points_usize =
        math.cast(usize, num_points) orelse return Error.CorruptedCompressedData;

    // Segment knot points as (f64 index, f64 value) pairs.
    for (0..num_points_usize) |_| {
        const idx = try shared_functions
            .readOffsetValue(f64, compressed_values, &offset);
        const val = try shared_functions
            .readOffsetValue(f64, compressed_values, &offset);
        segment_data
            .append(allocator, .{
            .index = idx,
            .value = val,
        }) catch return Error.CorruptedCompressedData;
    }

    // Number of knot flags.
    const num_flags = try shared_functions.readOffsetValue(u64, compressed_values, &offset);

    // Number of bytes used for packed flags.
    const num_flag_bytes = try shared_functions
        .readOffsetValue(u64, compressed_values, &offset);
    const num_flag_bytes_usize =
        math.cast(usize, num_flag_bytes) orelse return Error.CorruptedCompressedData;
    if (num_flag_bytes_usize > compressed_values.len - offset) return Error.CorruptedCompressedData;

    // Unpack boolean flags (LSB first, 8 per byte).
    var flags_read: u64 = 0;
    for (0..num_flag_bytes_usize) |_| {
        const byte_val = compressed_values[offset];
        offset += 1;
        for (0..8) |bit_idx| {
            if (flags_read >= num_flags) break;
            const flag = (byte_val & (@as(u8, 1) << @intCast(bit_idx))) != 0;
            knot_flags.append(allocator, flag) catch return Error.CorruptedCompressedData;
            flags_read += 1;
        }
    }

    // Original series length.
    const raw_len = try shared_functions.readOffsetValue(u64, compressed_values, &offset);
    original_length.* = math.cast(usize, raw_len) orelse return Error.CorruptedCompressedData;
}

/// Converts parameter-space knot points into primal-space line segments, respecting
/// the connectivity flags that indicate how segments are joined. Each knot point is a
/// time-value pair from the compressed representation. The `knot_flags` array determines
/// whether consecutive points form a connected segment or if there is a gap between them.
/// Connected segments start where the previous segment ends, while disconnected segments
/// consume an additional point to define their start position. The function builds the
/// reconstructed line segment representation expected by the decompression pipeline.
/// The `allocator` handles memory allocations for the output array. Allocation errors
/// are propagated.
fn readSegments(
    allocator: Allocator,
    segment_data: []const ContinousPoint,
    knot_flags: []const bool,
    line_segments: *ArrayList(LineSegment),
) Error!void {
    if (segment_data.len < 2) return;

    var it: usize = 0; // Iterator for segment_data.
    var knot_idx: usize = 1; // Skip first knot flag.

    var start_point = segment_data[it];
    it += 1;

    while (knot_idx < knot_flags.len and it < segment_data.len) {
        const end_point = segment_data[it];
        it += 1;

        const line = ext_poly.linearFromTwoPoints(start_point, end_point);
        const connected = knot_flags[knot_idx];

        try line_segments.append(allocator, .{
            .start_time = start_point.index,
            .end_time = end_point.index,
            .line = line,
        });

        if (connected) {
            // Next segment starts where this one ends.
            start_point = end_point;
        } else {
            // Disconnected: next segment starts at next point.
            if (it < segment_data.len) {
                start_point = segment_data[it];
                it += 1;
            }
        }

        knot_idx += 1;
    }

    // Handle last segment if there are remaining points.
    if (it < segment_data.len) {
        const end_point = segment_data[it];
        const line = ext_poly.linearFromTwoPoints(start_point, end_point);

        try line_segments.append(allocator, .{
            .start_time = start_point.index,
            .end_time = end_point.index,
            .line = line,
        });
    }
}

/// Linkage between consecutive pieces: `connected` continues from the previous end,
/// `disjoint` starts independently, and `mixed_link` lets the DP choose dynamically.
const PieceType = enum { connected, disjoint, mixed_link };

/// Compact fitting-window bounds used by DP; `tu`/`tg` define knot-placement
/// range and default to `-1` (unset).
const FittingWindow = struct {
    tu: f64 = -1.0,
    tg: f64 = -1.0,
};

/// Dynamic-programming knot node. `k` is the global knot index, `knot_type`
/// encodes connected/disconnected linkage, `prev` links to predecessor node,
/// and `references` tracks active references from bases/committed slots.
const Ck = struct {
    k: i64,
    knot_type: bool,
    references: i64,
    previous_knot: ?usize,
    last_knot: ContinousPoint,
    end_point: ContinousPoint,
    fw: FittingWindow,

    /// Create a `Ck` with knot index `k`, linkage `knot_type`, and optional
    /// predecessor index `previous_knot`; reference count starts at 1.
    fn create(
        k: i64,
        knotype: bool,
        prev: ?usize,
    ) Ck {
        return .{
            .k = k,
            .knot_type = knotype,
            .references = 1,
            .previous_knot = prev,
            .last_knot = .{ .index = 0.0, .value = 0.0 },
            .end_point = .{ .index = 0.0, .value = 0.0 },
            .fw = .{},
        };
    }

    fn decRef(self: *Ck) void {
        self.references -= 1;
    }

    fn incRef(self: *Ck) void {
        self.references += 1;
    }
};

/// Primal-space line segment over `[start_time, end_time]` with line model `line`;
/// used by decompression reconstruction.
const LineSegment = struct {
    start_time: f64,
    end_time: f64,
    line: ext_poly.LinearFunction,

    /// Evaluate segment line at time `t`.
    fn evaluate(self: LineSegment, t: f64) f64 {
        return ext_poly.evaluateLinear(self.line, t);
    }
};

/// A candidate fitting state explored in parallel by `MixedTypePlaState`.`Fittable` is a flat
/// struct that consolidates all per-candidate state needed to decide whether a single line segment
/// can be extended to cover a growing window of buffered samples within the error bound `delta`.
/// It maintains a `ExtendedPolygon` (the feasible region in slope/intercept space),
/// two `VisibleRegionChain` instances (`upper_visible_region` and `lower_visible_region`) that track the
/// extremal boundary of past constraints, and a sliding `active_error_tube` that bounds where the
/// next knot may fall. When a new sample makes the feasible polygon empty, `Fittable` computes the
/// best-fit line (`segment_start_point`, `segment_end_point`), stores a `restart_bias_segment` for
/// the connected restart, and records the `fitting_window` that
/// `MixedTypePlaState` uses to compare candidates.
const Fittable = struct {
    delta: f64,
    eps: f64,
    pieces_type: PieceType,
    /// Committed output knot points for this candidate.
    segments: ArrayList(ContinousPoint),
    /// The timestamp of the most-recently fed data point.
    current_time: f64,
    /// Time origin of the current fitting round (equals the timestamp of
    /// the first sample in `boundary_segments`).
    time_base: f64,
    /// Stash for the very first data point before the polygon is initialised
    /// (held between the first and second calls to `accumulateDataPoint`).
    data_point_buffer: ?ContinousPoint,
    delay_info: i64,
    apx_type: i32,

    // Feasible-region state.
    closed_direction: ext_poly.ChainType,
    boundary_segment: ArrayList(ext_poly.ErrorTubeSegment),
    feasible_polygon: ext_poly.ExtendedPolygon,
    active_error_tube: ext_poly.ErrorTubeSegment,
    upper_visible_region: ext_poly.VisibleRegionChain,
    lower_visible_region: ext_poly.VisibleRegionChain,

    // Knot and fitting-window state.
    knot_type: bool,
    fitting_window: FittingWindow,
    segment_start_point: ContinousPoint,
    segment_end_point: ContinousPoint,
    /// Boundary segment saved at termination for a connected restart.
    /// Set by `tryExtendOrTerminateNow`/`finalizeWithLastBufferedPoint`
    /// via `computeRestartBoundarySegment`; consumed and nulled by
    /// `restartConnectedRound`.
    restart_bias_segment: ?ext_poly.ErrorTubeSegment,
    /// `true` if the restart boundary came from `upper_visible_region`;
    /// `false` if it came from `lower_visible_region`. `null` when not set.
    restart_bias_uses_upper_region: ?bool,
    tolerances: ext_poly.Tolerances,

    allocator: Allocator,

    /// Create a new `Fittable` with error bound `delta`, size tolerance
    /// `eps`, and `connected` flag (true → connected piece, false → disjoint).
    fn create(
        allocator: Allocator,
        delta: f64,
        eps: f64,
        connected: bool,
        tolerances: ext_poly.Tolerances,
    ) Fittable {
        return .{
            .delta = delta,
            .eps = eps,
            .pieces_type = .connected,
            .segments = ArrayList(ContinousPoint).empty,
            .current_time = 0.0,
            .time_base = 0.0,
            .data_point_buffer = null,
            .delay_info = 0,
            .apx_type = 4,
            .closed_direction = .lower,
            .boundary_segment = ArrayList(ext_poly.ErrorTubeSegment).empty,
            .feasible_polygon = ext_poly.ExtendedPolygon.create(
                allocator,
                delta,
                eps,
                tolerances,
            ),
            .active_error_tube = ext_poly.ErrorTubeSegment.empty(),
            .upper_visible_region = ext_poly.VisibleRegionChain.create(
                allocator,
                .upper,
                delta,
                eps,
                tolerances,
            ),
            .lower_visible_region = ext_poly.VisibleRegionChain.create(
                allocator,
                .lower,
                delta,
                eps,
                tolerances,
            ),
            .knot_type = connected,
            .fitting_window = .{},
            .segment_start_point = .{ .index = 0.0, .value = 0.0 },
            .segment_end_point = .{ .index = 0.0, .value = 0.0 },
            .restart_bias_segment = null,
            .restart_bias_uses_upper_region = null,
            .tolerances = tolerances,
            .allocator = allocator,
        };
    }

    fn deinit(self: *Fittable) void {
        self.segments.deinit(self.allocator);
        self.boundary_segment.deinit(self.allocator);
        self.feasible_polygon.deinit();
        self.upper_visible_region.deinit();
        self.lower_visible_region.deinit();
    }

    /// Buffer an incoming data point into `boundary_segments`. On the first
    /// point, sets `time_base`, stashes `buffer_dp`, and initialises the
    /// `active_error_tube`. On the second point, initialises the
    /// `feasible_polygon` and seeds both visible region chains from the polygon's
    /// endmost vertices. Subsequent points (indices 2–3) are appended to
    /// `boundary_segments` without further processing; the sliding-window
    /// logic in `tryExtendOrTerminateNow` handles them once the buffer
    /// reaches four segments.
    fn buffer(self: *Fittable, dp: ContinousPoint) void {
        self.current_time = dp.index;
        const size = self.boundary_segment.items.len;

        if (size == 0) {
            self.time_base = dp.index;
            self.data_point_buffer = dp;
            self.initializeErrorTube(dp, self.delta);
            self.boundary_segment.append(self.allocator, ext_poly.ErrorTubeSegment
                .fromPointAndDelta(dp, self.delta)) catch unreachable;
        } else if (size == 1) {
            self.feasible_polygon.initializePolygon(
                self.data_point_buffer.?,
                dp,
                self.delta,
            ) catch unreachable;

            const sec = ext_poly.ErrorTubeSegment.fromPointAndDelta(dp, self.delta);
            self.upper_visible_region.resetFromSeedPoint(
                self.feasible_polygon.getEndmostVertex(.right_most),
                sec.upper,
                self.time_base,
            );
            self.lower_visible_region.resetFromSeedPoint(
                self.feasible_polygon.getEndmostVertex(.left_most),
                sec.lower,
                self.time_base,
            );

            self.data_point_buffer = null;
            self.boundary_segment.append(self.allocator, ext_poly.ErrorTubeSegment
                .fromPointAndDelta(dp, self.delta)) catch unreachable;
        } else if (size == 2 or size == 3) {
            self.boundary_segment.append(self.allocator, ext_poly.ErrorTubeSegment
                .fromPointAndDelta(dp, self.delta)) catch unreachable;
        }
    }

    /// Initialise `active_error_tube` to the ±`delta` tube around `dp`.
    /// Called once, on the very first buffered point, to establish the
    /// reference window from which `computeRestartBoundarySegment` will
    /// derive the next round's limiting segment.
    fn initializeErrorTube(self: *Fittable, dp: ContinousPoint, delta: f64) void {
        self.active_error_tube.upper = .{
            .index = dp.index,
            .value = dp.value + delta,
        };
        self.active_error_tube.lower = .{
            .index = dp.index,
            .value = dp.value - delta,
        };
    }

    /// Feed `dp` to this candidate. Buffers the point and, once
    /// `boundary_segments` contains more than three entries, delegates
    /// immediately to `tryExtendOrTerminateNow`. Returns `true` while the
    /// segment can still grow, `false` when a termination has been decided.
    fn updateFittable(self: *Fittable, dp: ContinousPoint) bool { //
        self.buffer(dp);

        if (self.boundary_segment.items.len <= 3) {
            return true;
        }

        return self.tryExtendOrTerminateNow();
    }

    /// Slide the boundary window one step and test whether the feasible
    /// polygon remains non-empty. Pops the front of `boundary_segments`,
    /// applies the upper and lower half-planes from the new leading segment
    /// `c` to `feasible_polygon`, and updates the visible region chains accordingly.
    ///
    /// If either half-plane empties the polygon (`contain_none`), the current
    /// segment is terminated: the surviving visible region chain's extremal line is used
    /// to set `segment_start_point` and `segment_end_point` via
    /// `computeSegmentEndpoints`, the depleted visible region chain is cleared,
    /// `restart_bias_segment` is saved via `computeRestartBoundarySegment`,
    /// and `fitting_window` is recorded. Returns `false`.
    ///
    /// If both half-planes leave the polygon non-empty, visible region chain vertices
    /// that survive `contain_all` are forwarded to `updateVisibleRegion`.
    /// Returns `true`.
    fn tryExtendOrTerminateNow(self: *Fittable) bool {
        if (self.boundary_segment.items.len > 0) {
            _ = self.boundary_segment.orderedRemove(0);
        }

        if (self.boundary_segment.items.len < 2) {
            return true;
        }

        const leading_segment = self.boundary_segment.items[1];

        const upper_halfplane = ext_poly.Halfplane{
            .sep = .{
                .slope = self.time_base - leading_segment.upper.index,
                .intercept = leading_segment.upper.value,
            },
            .direction = .point_to_below,
        };
        const upper_result = self.feasible_polygon.intersect(upper_halfplane) catch unreachable;
        if (upper_result == .contain_some) {
            self.upper_visible_region.resetFromSeedPoint(
                self.feasible_polygon.getEndmostVertex(.right_most),
                leading_segment.upper,
                self.time_base,
            );
        }

        const lower_halfplane = ext_poly.Halfplane{
            .sep = .{
                .slope = self.time_base - leading_segment.lower.index,
                .intercept = leading_segment.lower.value,
            },
            .direction = .point_to_above,
        };
        const lower_result = self.feasible_polygon.intersect(lower_halfplane) catch unreachable;
        if (lower_result == .contain_some) {
            self.lower_visible_region.resetFromSeedPoint(
                self.feasible_polygon.getEndmostVertex(.left_most),
                leading_segment.lower,
                self.time_base,
            );
        }

        if (upper_result == .contain_none or lower_result == .contain_none) {
            if (upper_result == .contain_none) {
                self.closed_direction = .upper;
                self.upper_visible_region.clear();
                self.restart_bias_uses_upper_region = false; // surviving chain is lower_visible_region
            } else {
                self.closed_direction = .lower;
                self.lower_visible_region.clear();
                self.restart_bias_uses_upper_region = true; // surviving chain is upper_visible_region
            }

            // Derive the best-fit line from the surviving visible region chain.
            const extremal_line = if (self.restart_bias_uses_upper_region.?)
                self.upper_visible_region.getExtremeLine()
            else
                self.lower_visible_region.getExtremeLine();
            self.computeSegmentEndpoints(extremal_line, leading_segment.upper.index);

            // Save restart boundary for the connected follow-on round.
            self.restart_bias_segment = self.computeRestartBoundarySegment(
                leading_segment,
                self.closed_direction,
            );

            // Record fitting window [tu, tg] for MixedTypePlaState comparison.
            if (self.closed_direction == .upper) {
                self.fitting_window.tu = leading_segment.upper.index;
                self.fitting_window.tg = self.active_error_tube.lower.index;
            } else {
                self.fitting_window.tu = self.active_error_tube.upper.index;
                self.fitting_window.tg = leading_segment.lower.index;
            }

            return false;
        } else {
            if (upper_result == .contain_all) {
                _ = self.upper_visible_region.addPoint(leading_segment.upper);
            }
            if (lower_result == .contain_all) {
                _ = self.lower_visible_region.addPoint(leading_segment.lower);
            }
            return true;
        }
    }

    /// Variant of `tryExtendOrTerminateNow` used when flushing the stream. Pops the front of
    /// `boundary_segments` and tests against the *last* (rather than second) remaining segment
    /// instead of index 1, so that  the tail of the input is processed correctly when the buffer is
    /// nearly empty.
    fn finalizeWithLastBufferedPoint(self: *Fittable) bool {
        if (self.boundary_segment.items.len > 0) {
            _ = self.boundary_segment.orderedRemove(0);
        }

        if (self.boundary_segment.items.len == 0) {
            return true;
        }

        const last_segment = self.boundary_segment.items[self.boundary_segment.items.len - 1];

        const upper_halfplane = ext_poly.Halfplane{
            .sep = .{
                .slope = self.time_base - last_segment.upper.index,
                .intercept = last_segment.upper.value,
            },
            .direction = .point_to_below,
        };
        const upper_result = self.feasible_polygon.intersect(upper_halfplane) catch unreachable;
        if (upper_result == .contain_some) {
            self.upper_visible_region.resetFromSeedPoint(
                self.feasible_polygon.getEndmostVertex(.right_most),
                last_segment.upper,
                self.time_base,
            );
        }

        const lower_halfplane = ext_poly.Halfplane{
            .sep = .{
                .slope = self.time_base - last_segment.lower.index,
                .intercept = last_segment.lower.value,
            },
            .direction = .point_to_above,
        };
        const lower_result = self.feasible_polygon.intersect(lower_halfplane) catch unreachable;
        if (lower_result == .contain_some) {
            self.lower_visible_region.resetFromSeedPoint(
                self.feasible_polygon.getEndmostVertex(.left_most),
                last_segment.lower,
                self.time_base,
            );
        }

        if (upper_result == .contain_none or lower_result == .contain_none) {
            if (upper_result == .contain_none) {
                self.closed_direction = .upper;
                self.upper_visible_region.clear();
                self.restart_bias_uses_upper_region = false; // TODO
            } else {
                self.closed_direction = .lower;
                self.lower_visible_region.clear();
                self.restart_bias_uses_upper_region = true;
            }

            const extremal_line = if (self.restart_bias_uses_upper_region.?)
                self.upper_visible_region.getExtremeLine()
            else
                self.lower_visible_region.getExtremeLine();
            self.computeSegmentEndpoints(extremal_line, last_segment.upper.index);

            self.restart_bias_segment = self.computeRestartBoundarySegment(
                last_segment,
                self.closed_direction,
            );

            if (self.closed_direction == .upper) {
                self.fitting_window.tu = last_segment.upper.index;
                self.fitting_window.tg = self.active_error_tube.lower.index;
            } else {
                self.fitting_window.tu = self.active_error_tube.upper.index;
                self.fitting_window.tg = last_segment.lower.index;
            }

            return false;
        } else {
            if (upper_result == .contain_all) {
                _ = self.upper_visible_region.addPoint(last_segment.upper);
            }
            if (lower_result == .contain_all) {
                _ = self.lower_visible_region.addPoint(last_segment.lower);
            }
            return true;
        }
    }

    /// Compute the restart boundary segment from the terminated segment's error tube and the
    /// surviving visible region chain. When the upper boundary caused termination (`up_or_low == .upper`),
    /// the extremal line from the lower polygon endpoint (`left_most`) is used to update
    /// `active_error_tube`, and the last lower-region vertex  becomes the lower endpoint of the
    /// restart segment. The symmetric  case applies when the lower boundary terminates.
    /// The returned `ErrorTubeSegment` is stored in `restart_bias_segment` and passed
    /// to `restartPolygonForNewRound` at the start of the next round.
    fn computeRestartBoundarySegment(
        self: *Fittable,
        terminating_segment: ext_poly.ErrorTubeSegment,
        upper_or_lower: ext_poly.ChainType,
    ) ext_poly.ErrorTubeSegment {
        var restart_segment = ext_poly.ErrorTubeSegment.empty();

        if (upper_or_lower == .upper) {
            const extreme_line = ext_poly.linearFromParamPoint(
                self.feasible_polygon.getEndmostVertex(.left_most),
                self.time_base,
            );

            self.active_error_tube.upper = .{
                .index = terminating_segment.upper.index,
                .value = ext_poly.evaluateLinear(extreme_line, terminating_segment.upper.index),
            };
            const ft = if (self.lower_visible_region.front()) |f| f.index else terminating_segment.upper.index;
            self.active_error_tube.lower = .{
                .index = ft,
                .value = ext_poly.evaluateLinear(extreme_line, ft),
            };

            restart_segment.upper = self.active_error_tube.upper;
            if (self.lower_visible_region.popBack()) |last| {
                restart_segment.lower = last;
            } else {
                restart_segment.lower = self.active_error_tube.lower;
            }
        } else {
            const extreme_line = ext_poly.linearFromParamPoint(
                self.feasible_polygon.getEndmostVertex(.right_most),
                self.time_base,
            );

            self.active_error_tube.lower = .{
                .index = terminating_segment.lower.index,
                .value = ext_poly.evaluateLinear(extreme_line, terminating_segment.lower.index),
            };
            const ft = if (self.upper_visible_region.front()) |f| f.index else terminating_segment.lower.index;
            self.active_error_tube.upper = .{
                .index = ft,
                .value = ext_poly.evaluateLinear(extreme_line, ft),
            };

            restart_segment.lower = self.active_error_tube.lower;
            if (self.upper_visible_region.popBack()) |last| {
                restart_segment.upper = last;
            } else {
                restart_segment.upper = self.active_error_tube.upper;
            }
        }

        return restart_segment;
    }

    /// Reinitialise the feasible polygon for the next segment, using `lseg`
    /// as the limiting (carry-over) boundary and the second entry of
    /// `boundary_segments` as the first new sample. Vertices still stored in
    /// the surviving visible region chain (`use_ceil` selects upper vs. lower) are
    /// replayed into the fresh polygon via `loadVisibleRegionConstraints` before both region
    /// chains are reseeded from the new polygon's endmost vertices.
    fn restartNewRound(self: *Fittable, lseg: ext_poly.ErrorTubeSegment, use_ceil: bool) void {
        if (self.boundary_segment.items.len < 2) {
            return;
        }

        const first_segment = self.boundary_segment.items[1];
        self.time_base = first_segment.upper.index;

        self.feasible_polygon.reInitializePolygon(
            lseg,
            first_segment,
            self.time_base,
            self.closed_direction,
        ) catch unreachable;

        // Replay surviving visible-region chain vertices as half-plane constraints.
        var surviving_chain =
            if (use_ceil) &self.upper_visible_region else &self.lower_visible_region;
        while (surviving_chain.popBack()) |dp| {
            const constraint_halfplane = ext_poly.Halfplane{
                .sep = .{
                    .slope = self.time_base - dp.index,
                    .intercept = dp.value,
                },
                .direction = surviving_chain.pointToDirection(),
            };
            _ = self.feasible_polygon.loadVisibleRegionConstraints(constraint_halfplane) catch unreachable;
        }

        self.upper_visible_region.resetFromSeedPoint(
            self.feasible_polygon.getEndmostVertex(.right_most),
            first_segment.upper,
            self.time_base,
        );
        self.lower_visible_region.resetFromSeedPoint(
            self.feasible_polygon.getEndmostVertex(.left_most),
            first_segment.lower,
            self.time_base,
        );
    }

    /// Intersect the extremal line `rsep` with `active_error_tube` to find
    /// the exact knot position and append it to `segments`. Also accumulates
    /// `accumulated_delay` to track the lag between the current time and
    /// the committed segment's base time.
    fn recordLastKnot(self: *Fittable, rsep: LinearFunction) void {
        var dp = ContinousPoint{ .index = 0.0, .value = 0.0 }; // TODO rename to knot_point
        self.active_error_tube.hittingLine(&dp, rsep, self.tolerances);
        self.segments.append(self.allocator, dp) catch unreachable;
        self.delay_info += @intFromFloat(self.current_time - self.time_base);
    }

    /// Finalise this candidate at end-of-stream. Selects the best-fit line
    /// from the current `feasible_polygon` and computes
    /// `segment_start_point`/`segment_end_point` via `computeSegmentEndpoints`.
    /// Degenerate case: if `boundary_segments` is empty (only one data point
    /// was ever buffered), a trivial horizontal segment is synthesised at the
    /// single buffered point so that the output is always well-formed.
    fn closeFitting(self: *Fittable) void {
        if (self.boundary_segment.items.len == 0) {
            if (self.data_point_buffer) |dp| {
                // Single-point degenerate case: emit a zero-length horizontal segment.
                self.segment_start_point = dp;
                self.segment_end_point = .{ .index = dp.index + 0.0001, .value = dp.value };
                self.fitting_window = .{ .tu = dp.index + 0.0001, .tg = dp.index + 0.0001 };
            }
            return;
        }

        const last = self.boundary_segment.items[self.boundary_segment.items.len - 1];
        const sol = self.feasible_polygon.selectSolution(
            self.time_base,
            last.midpointValue(),
        );

        const last_t = self.current_time + 0.0001;
        self.computeSegmentEndpoints(sol, last_t);
        self.fitting_window = .{ .tu = last_t, .tg = last_t };
    }
    /// Begin a new fitting round with the specified `link_type`. Marks the
    /// polygon as uninstantiated, then delegates to either
    /// `restartConnectedRound` or `restartDisconnectedRound` and updates
    /// `knot_type` accordingly.
    fn initializeNewRoundWithType(self: *Fittable, link_type: PieceType) void {
        self.feasible_polygon.setUninstantiated();
        if (link_type == .connected) {
            self.knot_type = true;
            self.restartConnectedNewRound();
        } else if (link_type == .disjoint) {
            self.knot_type = false;
            self.restartDisconnectedRound();
        }
    }

    /// Restart for a disjoint (disconnected) new round. Discards the oldest
    /// boundary segment, then — when exactly two segments remain —
    /// reinitialises the polygon from them and reseeds both visible region chains.
    fn restartDisconnectedRound(self: *Fittable) void {
        if (self.boundary_segment.items.len > 0) {
            _ = self.boundary_segment.orderedRemove(0);
        }

        if (self.boundary_segment.items.len == 0) {
            return;
        }

        const first_segment = self.boundary_segment.items[0];
        self.active_error_tube = first_segment;
        self.time_base = first_segment.upper.index;

        if (self.boundary_segment.items.len == 2) {
            const second_segment = self.boundary_segment.items[1];
            self.feasible_polygon
                .reInitializePolygon(
                first_segment,
                second_segment,
                self.time_base,
                .lower,
            ) catch unreachable;
            self.upper_visible_region.resetFromSeedPoint(
                self.feasible_polygon.getEndmostVertex(.right_most),
                second_segment.upper,
                self.time_base,
            );
            self.lower_visible_region.resetFromSeedPoint(
                self.feasible_polygon.getEndmostVertex(.left_most),
                second_segment.lower,
                self.time_base,
            );
        }
    }

    /// Restart for a connected new round. Delegates to `restartNewRound`
    /// with the saved `restart_bias_segment` and the surviving visible region chain
    /// indicator (`restart_bias_uses_upper_region`), then clears both fields
    /// so they are not replayed again.
    fn restartConnectedNewRound(self: *Fittable) void {
        if (self.restart_bias_segment != null and self.restart_bias_uses_upper_region != null) {
            self.restartNewRound(
                self.restart_bias_segment.?,
                self.restart_bias_uses_upper_region.?,
            );
        }
        self.restart_bias_uses_upper_region = null;
        self.restart_bias_segment = null;
    }

    /// Set `segment_start_point` to the intersection of `exl` with
    /// `active_error_tube`, and `segment_end_point` to the evaluation of
    /// `exl` at time `ctime`.
    fn computeSegmentEndpoints(self: *Fittable, exl: LinearFunction, ctime: f64) void {
        self.active_error_tube.hittingLine(
            &self.segment_start_point,
            exl,
            self.tolerances,
        );
        self.segment_end_point = .{
            .index = ctime,
            .value = ext_poly.evaluateLinear(exl, ctime),
        };
    }

    /// Deep-copy all state from `other` into `self`. Every `ArrayList` is
    /// cloned (not aliased), and both `feasible_polygon` and the two visible region
    /// chains are deep-copied via their own clone methods.
    /// `restart_bias_segment` and `restart_bias_uses_upper_region` are reset
    /// to `null`: restart state from the source candidate is not propagated.
    fn cloneFittable(self: *Fittable, other: *const Fittable) void {
        self.fitting_window = other.fitting_window;
        self.restart_bias_uses_upper_region = null;
        self.restart_bias_segment = null;

        // Clone boundary_segments.
        self.boundary_segment.clearRetainingCapacity();
        self.boundary_segment.appendSlice(
            self.allocator,
            other.boundary_segment.items,
        ) catch unreachable;

        // Clone scalar fields.
        self.active_error_tube = other.active_error_tube;
        self.current_time = other.current_time;
        self.time_base = other.time_base;
        self.apx_type = other.apx_type;
        self.pieces_type = other.pieces_type;

        // Clone segments.
        self.segments.clearRetainingCapacity();
        self.segments.appendSlice(self.allocator, other.segments.items) catch unreachable;

        self.data_point_buffer = other.data_point_buffer;

        // Clone feasible_polygon (deep copy both chains).
        self.feasible_polygon.deinit();
        self.feasible_polygon = other.feasible_polygon.cloneExtendedPolygon(
            self.allocator,
        ) catch unreachable;

        // Clone visible region chains.
        self.upper_visible_region.deinit();
        self.upper_visible_region = other.upper_visible_region.cloneVisibleRegionChain(
            self.allocator,
        ) catch unreachable;
        self.lower_visible_region.deinit();
        self.lower_visible_region = other.lower_visible_region.cloneVisibleRegionChain(
            self.allocator,
        ) catch unreachable;
    }
};

/// Orchestrator for Mixed-Type PLA. Holds three horizon knots (`horizon_knots[0..2]`,
/// corresponding to committed positions C[k], C[k+1], C[k+2]) and five parallel
/// `Fittable` candidates (`candidates[0..4]`). The five candidates represent a short
/// dynamic-programming horizon that lets the algorithm defer the connected/disjoint
/// choice until enough evidence has accumulated, avoiding greedy suboptimal commitments.
const MixedTypePlaState = struct {
    // Shared approximation parameters.
    delta: f64,
    eps: f64,
    pieces_type: PieceType,
    output_segments: ArrayList(ContinousPoint),
    current_time: f64,
    time_base: f64,
    accumulated_delay: i64,
    approximation_type: i32,
    // Mixed-type DP horizon state.
    horizon_index: i64,
    committed_knot_queue: ArrayList(Ck),
    horizon_knots: [3]?usize, // Indices into ck_list (or null) for C[k], C[k+1], C[k+2].
    knot_pool: ArrayList(Ck), // The master array of all Ck nodes. Indices into this array are used.
    candidates: [5]Fittable,
    candidate_active: [5]bool,
    chosen_candidate: usize,
    connectivity_flags: ArrayList(bool),
    tolerances: ext_poly.Tolerances,
    allocator: Allocator,

    /// Create a new `MixedTypePlaState` with the given error bound and tolerance.
    fn create(allocator: Allocator, delta: f64, eps: f64, tols: ext_poly.Tolerances) MixedTypePlaState {
        return .{
            .delta = delta,
            .eps = eps,
            .pieces_type = .mixed_link,
            .output_segments = ArrayList(ContinousPoint).empty,
            .current_time = 0.0,
            .time_base = 0.0,
            .accumulated_delay = 0,
            .approximation_type = 4,
            .horizon_index = -1,
            .committed_knot_queue = ArrayList(Ck).empty,
            .knot_pool = ArrayList(Ck).empty,
            .horizon_knots = .{ null, null, null },
            // Candidate index semantics:
            //   - `candidates[0]` = extend C[k]   with a disjoint piece
            //   - `candidates[1]` = extend C[k+1] with a connected piece
            //   - `candidates[2]` = extend C[k+1] with a disjoint piece
            //   - `candidates[3]` = extend C[k+2] with a connected piece
            //   - `candidates[4]` = extend C[k+2] with a disjoint piece
            .candidates = .{
                Fittable.create(
                    allocator,
                    delta,
                    eps,
                    false,
                    tols,
                ),
                Fittable.create(
                    allocator,
                    delta,
                    eps,
                    true,
                    tols,
                ),
                Fittable.create(
                    allocator,
                    delta,
                    eps,
                    false,
                    tols,
                ),
                Fittable.create(
                    allocator,
                    delta,
                    eps,
                    true,
                    tols,
                ),
                Fittable.create(
                    allocator,
                    delta,
                    eps,
                    false,
                    tols,
                ),
            },
            .candidate_active = .{ true, true, true, true, true },
            .chosen_candidate = 0,
            .connectivity_flags = ArrayList(bool).empty,
            .tolerances = tols,
            .allocator = allocator,
        };
    }

    fn deinit(self: *MixedTypePlaState) void {
        self.output_segments.deinit(self.allocator);
        self.committed_knot_queue.deinit(self.allocator);
        self.knot_pool.deinit(self.allocator);
        self.connectivity_flags.deinit(self.allocator);
        for (&self.candidates) |*b| {
            b.deinit();
        }
    }

    /// Feed a data point to all five candidates and trigger DP-horizon advancement as
    /// needed. Calls `updateFittable` on each active candidate; when both `candidates[0]`
    /// (disjoint extension of C[k]) and `candidates[1]` (connected extension of C[k+1])
    /// are simultaneously exhausted, advances the DP horizon via `advanceDpHorizon`.
    fn updateState(self: *MixedTypePlaState, p: ContinousPoint) void {
        for (0..5) |i| {
            if (self.candidate_active[i]) {
                self.candidate_active[i] = self.candidates[i].updateFittable(p);
            }
        }

        self.current_time = p.index;

        while (!self.candidate_active[0] and !self.candidate_active[1]) {
            self.advanceDpHorizon();
        }
    }

    /// Flush remaining buffered points through all active candidates via
    /// `finalizeWithLastBufferedPoint`, advance the DP horizon until stable,
    /// then close the winning candidate and emit the final committed pieces.
    fn closeFitting(self: *MixedTypePlaState) void {
        for (0..5) |i| {
            if (self.candidate_active[i]) {
                self.candidate_active[i] = self.candidates[i].finalizeWithLastBufferedPoint();
            }
        }

        while (!self.candidate_active[0] and !self.candidate_active[1]) {
            self.advanceDpHorizon();
        }

        self.chosen_candidate = self.IndexOfBetterBase(0, 1);
        self.candidates[self.chosen_candidate].closeFitting();

        const final_knot_index = self.tryCreateHorizonKnot(self.chosen_candidate);

        const beg_k = self.commitKnotAndPrune(self.horizon_knots[0]);
        _ = beg_k;
        self.emitFixedPieces();

        self.horizon_index += 1;
        self.horizon_knots[0] = self.horizon_knots[1];
        self.horizon_knots[1] = self.horizon_knots[2];
        self.horizon_knots[2] = final_knot_index;

        if (self.horizon_knots[2] == null) {
            self.horizon_knots[2] = self.horizon_knots[1];
        }

        self.emitKnotChain(self.horizon_knots[2]);
        self.clearData();

        if (self.connectivity_flags.items.len > 0) {
            self.accumulated_delay = @divTrunc(
                self.accumulated_delay,
                @as(i64, @intCast(self.connectivity_flags.items.len)),
            );
        }
    }

    /// Decision and transition step of the DP horizon. Called when both `candidates[0]`
    /// (disjoint extension of C[k]) and `candidates[1]` (connected extension of C[k+1])
    /// are exhausted. Selects the better option via `IndexOfBetterBase`, optionally creates
    /// a new horizon knot for position C[k+3] via `tryCreateHorizonKnot`, commits the
    /// oldest knot via `commitKnotAndPrune`, emits any now-fixed output pieces, then slides
    /// all five candidate slots and the three horizon-knot references one step forward.
    fn advanceDpHorizon(self: *MixedTypePlaState) void {
        // 1. Prepare C[k+3] by comparing base[0] vs base[1].
        self.chosen_candidate = self.IndexOfBetterBase(0, 1);
        const new_knot_index = self.tryCreateHorizonKnot(self.chosen_candidate);

        // Push oldest committed knot and possibly free unreferenced knots.
        const oldest_knot_result = self.commitKnotAndPrune(self.horizon_knots[0]);
        _ = oldest_knot_result;

        // 2. Prepare two new fittable bases rooted at the chosen option.
        if (new_knot_index != null) {
            // Duplicate the chosen base into the opposite slot.
            self.candidates[1 - self.chosen_candidate].cloneFittable(&self.candidates[self.chosen_candidate]);

            // chosen → connected extension, unchosen → disjoint extension for C[k+3].
            self.candidates[self.chosen_candidate].initializeNewRoundWithType(.connected);
            self.candidates[1 - self.chosen_candidate].initializeNewRoundWithType(.disjoint);
        } else {
            // No ck3: mark windows invalid.
            self.candidates[self.chosen_candidate].fitting_window = .{
                .tu = -1.0,
                .tg = -1.0,
            };
            self.candidates[1 - self.chosen_candidate].fitting_window = .{
                .tu = -1.0,
                .tg = -1.0,
            };
        }

        // 3. Slide the DP horizon forward.
        self.horizon_index += 1;

        // Move flags.
        self.candidate_active[0] = self.candidate_active[2];
        self.candidate_active[1] = self.candidate_active[3];
        self.candidate_active[2] = self.candidate_active[4];
        if (new_knot_index != null) {
            self.candidate_active[3] = true;
            self.candidate_active[4] = true;
        } else {
            self.candidate_active[3] = false;
            self.candidate_active[4] = false;
        }

        // Move base instances: swap references via temporary storage.
        // Slide the candidates window: [2,3,4] become the new [0,1,2], and the
        // chosen/other candidates from C[k+3] fill slots [3] and [4] respectively.
        // Use saved temporaries to avoid overwriting before the copy is complete.
        const chosen_base = self.candidates[self.chosen_candidate];
        const other_base = self.candidates[1 - self.chosen_candidate];
        self.candidates[0] = self.candidates[2];
        self.candidates[1] = self.candidates[3];
        self.candidates[2] = self.candidates[4];
        self.candidates[3] = chosen_base;
        self.candidates[4] = other_base;

        // Move C[k].
        self.horizon_knots[0] = self.horizon_knots[1];
        self.horizon_knots[1] = self.horizon_knots[2];
        self.horizon_knots[2] = new_knot_index;
        if (new_knot_index) |idx| {
            self.knot_pool.items[idx].incRef();
        }

        // Try to output fixed pieces.
        self.emitFixedPieces();
    }

    /// Attempt to create a new committed knot at horizon position C[k+3] for the winning
    /// candidate `win`. Compares the candidate's fitting window against the window held by
    /// the current C[k+2] knot: if C[k+2] already dominates on both `tu` and `tg`, no new
    /// knot is needed and `null` is returned. Otherwise allocates a fresh `Ck` node in
    /// `knot_pool`, wires its predecessor reference, copies the fitting window and boundary
    /// points from the winning candidate, and returns the new node's index into `knot_pool`.
    fn tryCreateHorizonKnot(self: *MixedTypePlaState, win: usize) ?usize {
        const new_window = self.candidates[win].fitting_window;

        var existing_window = FittingWindow{};
        if (self.horizon_knots[2]) |c2_idx| {
            existing_window = self.knot_pool.items[c2_idx].fw;
        }

        // If the existing window f2 already dominates f3, no need to create ck3.
        if (existing_window.tg >= new_window.tg and existing_window.tu >= new_window.tu) {
            return null;
        }

        // Create a new Ck node.
        const predecessor_index: ?usize = if (win == 0) self.horizon_knots[0] else self.horizon_knots[1];

        // Increment ref on prev.
        if (predecessor_index) |pi| {
            self.knot_pool.items[pi].incRef();
        }

        var new_knot = Ck.create(self.horizon_index + 3, win == 1, predecessor_index);
        new_knot.last_knot = self.candidates[win].segment_start_point;
        new_knot.end_point = self.candidates[win].segment_end_point;
        new_knot.fw = self.candidates[win].fitting_window;

        self.knot_pool.append(self.allocator, new_knot) catch unreachable;
        return self.knot_pool.items.len - 1;
    }

    /// Append the knot at `knot_index` to `committed_knot_queue` and walk its `prev`
    /// chain, decrementing reference counts and removing nodes whose `refn` drops to zero.
    /// Returns the `refn` of the first still-live ancestor (positive value), `-1` if the
    /// chain was fully pruned or the input index was `null`, or a negative error code if
    /// an inconsistency is detected (knot not found in the committed queue).
    fn commitKnotAndPrune(self: *MixedTypePlaState, knot_index: ?usize) i64 {
        if (knot_index == null) {
            return -1;
        }

        self.committed_knot_queue.append(
            self.allocator,
            self.knot_pool.items[knot_index.?],
        ) catch unreachable;

        var current_index = knot_index;
        while (current_index) |node_index| {
            // Find the index of this node in ck_list from the end.
            var list_position: ?usize = null;
            var i: usize = self.committed_knot_queue.items.len;
            while (i > 0) {
                i -= 1;
                // Compare by knot index k (unique identifier).
                if (self.committed_knot_queue.items[i].k == self.knot_pool.items[node_index].k) {
                    list_position = i;
                    break;
                }
            }

            if (list_position == null) {
                // Locating error — should not happen.
                return -2;
            }

            self.knot_pool.items[node_index].decRef();
            // Also update the copy in ck_list.
            self.committed_knot_queue.items[list_position.?].references = self.knot_pool.items[node_index].references;

            if (self.knot_pool.items[node_index].references < 0) {
                // Technical error.
                return self.knot_pool.items[node_index].k;
            } else if (self.knot_pool.items[node_index].references > 0) {
                return self.knot_pool.items[node_index].references;
            }

            // refn == 0: remove from ck_list and continue to prev.
            const prev = self.knot_pool.items[node_index].previous_knot;
            _ = self.committed_knot_queue.orderedRemove(list_position.?);
            current_index = prev;
        }

        return -1;
    }

    /// Flush any knots at the head of `committed_knot_queue` that are now uniquely
    /// referenced and therefore fully committed. A knot is fixed when its `refn == 1`
    /// and the next queue entry's `prev` index confirms the chain relationship. For each
    /// fixed knot, its type flag is appended to `connectivity_flags`, its `lastknot` is
    /// appended to `output_segments` (followed by `end_point` when the successor piece is
    /// disconnected), and the delay statistic is accumulated. The knot is then removed
    /// from the queue and the successor's `prev` reference is cleared.
    fn emitFixedPieces(self: *MixedTypePlaState) void {
        while (self.committed_knot_queue.items.len > 0) {
            const leading_knot = self.committed_knot_queue.items[0];
            if (leading_knot.references != 1) {
                return;
            }

            // Find the successor.
            var successor_knot: Ck = undefined;
            if (self.committed_knot_queue.items.len > 1) {
                successor_knot = self.committed_knot_queue.items[1];
            } else {
                // Fallback: try to find successor among committed c nodes.
                if (self.horizon_knots[0]) |c0| {
                    successor_knot = self.knot_pool.items[c0];
                } else if (self.horizon_knots[1]) |c1| {
                    successor_knot = self.knot_pool.items[c1];
                } else if (self.horizon_knots[2]) |c2| {
                    successor_knot = self.knot_pool.items[c2];
                } else {
                    // Error in fix — should not happen.
                    return;
                }
            }

            // Check if sec.prev points to fir (by k index).
            const second_previous_knot: ?i64 = if (successor_knot.previous_knot) |sp|
                self.knot_pool.items[sp].k
            else
                null;

            if (second_previous_knot != null and second_previous_knot.? == leading_knot.k) {
                // Append the knot type and lastknot point.
                self.connectivity_flags.append(self.allocator, leading_knot.knot_type) catch unreachable;
                self.output_segments.append(self.allocator, leading_knot.last_knot) catch unreachable;

                // If the successor is disconnected, also append leading_knot.end_point.
                if (!successor_knot.knot_type) {
                    self.output_segments.append(self.allocator, leading_knot.end_point) catch unreachable;
                }

                // Accumulate delay statistics.
                self.accumulated_delay += @intFromFloat(self.current_time - leading_knot.last_knot.index);

                // Remove the fixed knot from the queue.
                _ = self.committed_knot_queue.orderedRemove(0);

                // Clear successor_knot.prev (set to null in the ck_nodes array).
                if (successor_knot.previous_knot) |sp| {
                    _ = sp; // The prev reference is now consumed.
                }
                // Update the actual ck_node's prev to null.
                if (self.committed_knot_queue.items.len > 0) {
                    // Find successor_knot in ck_nodes and clear its prev.
                    for (self.knot_pool.items) |*node| {
                        if (node.k == successor_knot.k) {
                            node.previous_knot = null;
                            break;
                        }
                    }
                }
                // Also update ck_list[0] if it exists.
                if (self.committed_knot_queue.items.len > 0) {
                    self.committed_knot_queue.items[0].previous_knot = null;
                }
            } else {
                return;
            }
        }
    }

    /// Walk the `prev` chain starting at `ck_idx` (newest-to-oldest traversal), collect
    /// each node's connectivity flag and boundary points into temporary lists, then reverse
    /// and append them to `connectivity_flags` and `output_segments` in chronological order.
    /// Also decrements the reference count of each visited `knot_pool` node and accumulates
    /// the round-trip delay statistic for each knot.
    fn emitKnotChain(self: *MixedTypePlaState, ck_idx: ?usize) void {
        if (ck_idx == null) return;

        var type_flags = ArrayList(bool).empty;
        defer type_flags.deinit(self.allocator);
        var knot_points = ArrayList(ContinousPoint).empty;
        defer knot_points.deinit(self.allocator);

        var current_index = ck_idx;
        while (current_index) |ci| {
            var knot_node = &self.knot_pool.items[ci];
            knot_node.decRef();

            // Append end_point for the segment if appropriate.
            if (type_flags.items.len == 0) {
                knot_points.append(self.allocator, knot_node.end_point) catch unreachable;
            } else if (!type_flags.items[type_flags.items.len - 1]) {
                // Last was disconnected.
                knot_points.append(self.allocator, knot_node.end_point) catch unreachable;
            }

            // Append the knot point.
            knot_points.append(self.allocator, knot_node.last_knot) catch unreachable;

            // Record delay statistics.
            self.accumulated_delay += @intFromFloat(self.current_time - knot_node.last_knot.index);

            type_flags.append(self.allocator, knot_node.knot_type) catch unreachable;
            current_index = knot_node.previous_knot;
        }

        // Reverse and add accumulated flags & points to main lists.
        while (type_flags.items.len > 0) {
            self.connectivity_flags.append(self.allocator, type_flags.pop().?) catch unreachable;
        }
        while (knot_points.items.len > 0) {
            self.output_segments.append(self.allocator, knot_points.pop().?) catch unreachable;
        }
    }

    /// Compare candidates `first` and `second` and return the index of the preferred one.
    /// If exactly one is still active, that one is returned. If both are inactive, the one
    /// with the wider fitting window dominates; when the windows cross on `tu` and `tg`,
    /// the candidate with the smaller endpoint excursion (distance from segment start to end)
    /// is preferred as a proxy for tighter fit quality. If both are still active, `second`
    /// is returned as the default.
    fn IndexOfBetterBase(self: *MixedTypePlaState, first: usize, second: usize) usize {
        if (!self.candidate_active[first] and self.candidate_active[second]) {
            return second;
        } else if (self.candidate_active[first] and !self.candidate_active[second]) {
            return first;
        } else if (!self.candidate_active[first] and !self.candidate_active[second]) {
            if (self.candidates[first].fitting_window.tu > self.candidates[second].fitting_window.tu) {
                if (self.candidates[first].fitting_window.tg >= self.candidates[second].fitting_window.tg) {
                    return first;
                } else {
                    // FW crossing — pick base with smaller endpoint excursion
                    const first_excursion =
                        @abs(self.candidates[first].segment_end_point.value -
                        self.candidates[first].segment_start_point.value);
                    const second_excursion =
                        @abs(self.candidates[second].segment_end_point.value -
                        self.candidates[second].segment_start_point.value);
                    return if (first_excursion <= second_excursion) first else second;
                }
            } else if (self.candidates[first].fitting_window.tu < self.candidates[second].fitting_window.tu) {
                if (self.candidates[first].fitting_window.tg <= self.candidates[second].fitting_window.tg) {
                    return second;
                } else {
                    // FW crossing — pick base with smaller endpoint excursion
                    const first_excursion =
                        @abs(self.candidates[first].segment_end_point.value -
                        self.candidates[first].segment_start_point.value);
                    const second_excursion =
                        @abs(self.candidates[second].segment_end_point.value -
                        self.candidates[second].segment_start_point.value);
                    return if (first_excursion <= second_excursion) first else second;
                }
            } else {
                // Equal tu.
                if (self.candidates[first].fitting_window.tg > self.candidates[second].fitting_window.tg) {
                    return first;
                } else {
                    return second;
                }
            }
        } else {
            return 1;
        }
    }

    /// Reset the `horizon_knots` triple to `null` and drain `committed_knot_queue`,
    /// preparing `MixedTypePlaState` for the next fitting call after `closeFitting`
    /// has flushed all remaining output.
    fn clearData(self: *MixedTypePlaState) void {
        self.horizon_knots = .{ null, null, null };
        self.committed_knot_queue.clearRetainingCapacity();
    }

    /// Feed a full slice of raw `f64` values to the Mixed-Type PLA algorithm. Each value
    /// is paired with its zero-based integer index as the time coordinate and forwarded
    /// to `updateState`; once all values are processed, `closeFitting` is called to flush
    /// any remaining buffered candidates and finalise the `output_segments` list.
    fn run(self: *MixedTypePlaState, values: []const f64) void {
        for (values, 0..) |val, i| {
            self.updateState(.{
                .index = @as(f64, @floatFromInt(i)),
                .value = val,
            });
        }
        self.closeFitting();
    }
};

test "check mixed-type PLA configuration parsing" {
    // Tests the configuration parsing and functionality of the `compress` function.
    // The test verifies that the provided configuration is correctly interpreted and
    // that the `configuration.AbsoluteErrorBound` is expected in the function.
    const allocator = testing.allocator;

    const uncompressed_values = &[4]f64{ 19.0, 48.0, 28.0, 3.0 };

    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);

    const method_configuration =
        \\ {"abs_error_bound": 0.1}
    ;

    // The configuration is properly defined. No error expected.
    try compress(
        allocator,
        uncompressed_values,
        &compressed_values,
        method_configuration,
    );
}

test "mixed-type PLA compress and decompress roundtrip with known values" {
    const allocator = testing.allocator;

    const uncompressed_values = &[8]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0 };

    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);

    const method_configuration =
        \\ {"abs_error_bound": 0.5}
    ;

    try compress(
        allocator,
        uncompressed_values,
        &compressed_values,
        method_configuration,
    );

    var decompressed_values = ArrayList(f64).empty;
    defer decompressed_values.deinit(allocator);

    try decompress(
        allocator,
        compressed_values.items,
        &decompressed_values,
    );
    try testing.expectEqual(uncompressed_values.len, decompressed_values.items.len);
    for (uncompressed_values, decompressed_values.items) |orig, dec| {
        try testing.expect(@abs(orig - dec) <= 0.5);
    }
}

test "mixed-type PLA compress and decompress roundtrip with linear ramp" {
    const allocator = testing.allocator;

    // A perfectly linear ramp should compress well with a small error bound.
    var uncompressed_values: [50]f64 = undefined;
    for (0..50) |i| {
        uncompressed_values[i] = @as(f64, @floatFromInt(i)) * 0.3;
    }

    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);

    const method_configuration =
        \\ {"abs_error_bound": 0.2}
    ;

    try compress(
        allocator,
        &uncompressed_values,
        &compressed_values,
        method_configuration,
    );

    var decompressed_values = ArrayList(f64).empty;
    defer decompressed_values.deinit(allocator);

    try decompress(
        allocator,
        compressed_values.items,
        &decompressed_values,
    );

    try testing.expectEqual(uncompressed_values.len, decompressed_values.items.len);

    for (&uncompressed_values, decompressed_values.items) |orig, dec| {
        try testing.expect(@abs(orig - dec) <= 0.2);
    }
}

test "mixed-type PLA compress and decompress roundtrip with sinusoidal data" {
    const allocator = testing.allocator;

    var uncompressed_values: [100]f64 = undefined;
    for (0..100) |i| {
        const t: f64 = @as(f64, @floatFromInt(i));
        uncompressed_values[i] = 10.0 * @sin(t * 0.1) + 5.0;
    }

    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);

    const error_bound: f64 = 0.3;
    const method_configuration =
        \\ {"abs_error_bound": 0.3}
    ;

    try compress(
        allocator,
        &uncompressed_values,
        &compressed_values,
        method_configuration,
    );

    var decompressed_values = ArrayList(f64).empty;
    defer decompressed_values.deinit(allocator);

    try decompress(
        allocator,
        compressed_values.items,
        &decompressed_values,
    );

    try testing.expectEqual(uncompressed_values.len, decompressed_values.items.len);

    for (&uncompressed_values, decompressed_values.items) |orig, dec| {
        try testing.expect(@abs(orig - dec) <= error_bound);
    }
}

test "mixed-type PLA compress and decompress with two values" {
    const allocator = testing.allocator;

    const uncompressed_values = &[2]f64{ 10.0, 20.0 };

    var compressed_values = ArrayList(u8).empty;
    defer compressed_values.deinit(allocator);

    const method_configuration =
        \\ {"abs_error_bound": 0.5}
    ;

    try compress(
        allocator,
        uncompressed_values,
        &compressed_values,
        method_configuration,
    );

    var decompressed_values = ArrayList(f64).empty;
    defer decompressed_values.deinit(allocator);

    try decompress(
        allocator,
        compressed_values.items,
        &decompressed_values,
    );

    try testing.expectEqual(uncompressed_values.len, decompressed_values.items.len);

    for (uncompressed_values, decompressed_values.items) |orig, dec| {
        try testing.expect(@abs(orig - dec) <= 0.5);
    }
}

test "mixed-type PLA can always compress and decompress with positive error bound" {
    const allocator = testing.allocator;
    const data_distributions = &[_]tester.DataDistribution{
        .LinearFunctions,
        .BoundedRandomValues,
        .SinusoidalFunction,
    };

    try tester.testErrorBoundedCompressionMethod(
        allocator,
        Method.MixedTypePLA,
        data_distributions,
    );
}

test "mixed-type PLA can compress and decompress bounded values with many segments" {
    const allocator = testing.allocator;
    const error_bound = tester.generateBoundedRandomValue(f32, 0.5, 3, null);
    var uncompressed_values = ArrayList(f64).empty;
    defer uncompressed_values.deinit(allocator);
    for (0..20) |_| {
        try tester.generateBoundedRandomValues(allocator, &uncompressed_values, 0, 10, null);
    }

    try tester.testCompressAndDecompress(
        allocator,
        uncompressed_values.items,
        Method.MixedTypePLA,
        error_bound,
        shared_functions.isWithinErrorBound,
    );
}

test "extract rejects compressed data shorter than the minimum 48-byte header" {
    // The binary format requires at least 16 bytes of normalization header, 8 bytes for
    // num_segments, 8 for num_flags, 8 for flag_bytes_len, and 8 for original_length = 48 bytes.
    // Anything shorter must be rejected immediately.
    const allocator = testing.allocator;
    const too_short = [_]u8{0} ** 12; // 12 bytes — well below the 48-byte minimum.

    var indices = ArrayList(u64).empty;
    defer indices.deinit(allocator);
    var coefficients = ArrayList(f64).empty;
    defer coefficients.deinit(allocator);

    try testing.expectError(
        Error.CorruptedCompressedData,
        extract(allocator, &too_short, &indices, &coefficients),
    );
}

test "extract rejects compressed data with trailing bytes after the expected end" {
    // Compress a small input to obtain a valid binary blob, then append a spurious byte.
    // extract() must reject this because it validates that the entire slice is consumed.
    const allocator = testing.allocator;

    const uncompressed_values = &[4]f64{ 1.0, 2.0, 3.0, 4.0 };
    var compressed = ArrayList(u8).empty;
    defer compressed.deinit(allocator);

    try compress(allocator, uncompressed_values, &compressed,
        \\ {"abs_error_bound": 0.5}
    );

    // Append one spurious trailing byte to make the buffer larger than expected.
    try compressed.append(allocator, 0xFF);

    var indices = ArrayList(u64).empty;
    defer indices.deinit(allocator);
    var coefficients = ArrayList(f64).empty;
    defer coefficients.deinit(allocator);

    try testing.expectError(
        Error.CorruptedCompressedData,
        extract(allocator, compressed.items, &indices, &coefficients),
    );
}

test "extract stores knot timestamps in indices and knot values in coefficients" {
    const allocator = testing.allocator;

    const uncompressed_values = &[6]f64{ 1.0, 2.5, 3.0, 2.0, 1.5, 1.0 };
    var compressed = ArrayList(u8).empty;
    defer compressed.deinit(allocator);

    try compress(allocator, uncompressed_values, &compressed,
        \\ {"abs_error_bound": 0.5}
    );

    var indices = ArrayList(u64).empty;
    defer indices.deinit(allocator);
    var coefficients = ArrayList(f64).empty;
    defer coefficients.deinit(allocator);

    try extract(allocator, compressed.items, &indices, &coefficients);

    const num_segments = indices.items[0];
    const num_segments_usize = math.cast(usize, num_segments) orelse return Error.CorruptedCompressedData;

    // coefficients = [norm_offset, norm_scale] + one value per segment.
    try testing.expectEqual(@as(usize, 2 + num_segments_usize), coefficients.items.len);

    // indices = [num_segments] + one timestamp per segment + [num_flags, flag_bytes_len, ...flag_bytes, original_length].
    try testing.expect(indices.items.len >= 1 + num_segments_usize + 3);
}

// ── rebuild ───────────────────────────────────────────────────────────────────

test "rebuild rejects coefficients array that is too short (fewer than 2 entries)" {
    // The coefficients array must contain at least the normalization offset and scale (2 entries).
    const allocator = testing.allocator;
    var rebuilt = ArrayList(u8).empty;
    defer rebuilt.deinit(allocator);

    const coefficients = [_]f64{0.0}; // only 1 entry, need at least 2.
    const indices = [_]u64{ 0, 0, 0, 2 }; // structurally otherwise valid.

    try testing.expectError(
        Error.CorruptedCompressedData,
        rebuild(allocator, &indices, &coefficients, &rebuilt),
    );
}

test "rebuild rejects indices array that is too short (fewer than 4 entries)" {
    // The indices array must contain at least num_segments, num_flags, flag_bytes_len, and
    // original_length (4 entries, assuming zero flag bytes and zero knot points).
    const allocator = testing.allocator;
    var rebuilt = ArrayList(u8).empty;
    defer rebuilt.deinit(allocator);

    const coefficients = [_]f64{ 0.0, 1.0 }; // valid normalization header.
    const indices = [_]u64{ 0, 0 }; // only 2 entries, need at least 4.

    try testing.expectError(
        Error.CorruptedCompressedData,
        rebuild(allocator, &indices, &coefficients, &rebuilt),
    );
}

test "rebuild rejects num_segments that would require more coefficients than provided" {
    // num_segments = 1 requires one knot value coefficient in addition to offset/scale,
    // but only offset/scale are provided here.
    const allocator = testing.allocator;
    var rebuilt = ArrayList(u8).empty;
    defer rebuilt.deinit(allocator);

    const coefficients = [_]f64{ 0.0, 1.0 }; // only offset and scale, no knot values.
    const indices = [_]u64{ 1, 0, 0, 0, 4 }; // num_segments=1, t_bits=0.0, num_flags=0, flag_bytes_len=0, original_length=4.

    try testing.expectError(
        Error.CorruptedCompressedData,
        rebuild(allocator, &indices, &coefficients, &rebuilt),
    );
}

test "rebuild rejects flag_bytes_len that would require more indices than provided" {
    // flag_bytes_len = 3 requires indices[3], [4], [5] for the three flag bytes plus indices[6]
    // for original_length — but only 4 entries are provided, so the array is too short.
    const allocator = testing.allocator;
    var rebuilt = ArrayList(u8).empty;
    defer rebuilt.deinit(allocator);

    const coefficients = [_]f64{ 0.0, 1.0 }; // valid: offset and scale only, zero knots.
    const indices = [_]u64{ 0, 8, 3, 10 }; // num_segments=0, num_flags=8, flag_bytes_len=3:
    // needs indices[3..5] for flag bytes + indices[6] for original_length, but len=4.

    try testing.expectError(
        Error.CorruptedCompressedData,
        rebuild(allocator, &indices, &coefficients, &rebuilt),
    );
}

test "rebuild rejects extra trailing coefficients not accounted for by the structure" {
    // A perfectly valid structure (0 segments, 0 flags) but with one extra coefficient
    // beyond what the structure describes. rebuild() must reject this.
    const allocator = testing.allocator;
    var rebuilt = ArrayList(u8).empty;
    defer rebuilt.deinit(allocator);

    const coefficients = [_]f64{ 0.0, 1.0, 99.0 }; // extra 99.0 at the end.
    const indices = [_]u64{ 0, 0, 0, 4 }; // num_segments=0, num_flags=0, flag_bytes_len=0, original_length=4.

    try testing.expectError(
        Error.CorruptedCompressedData,
        rebuild(allocator, &indices, &coefficients, &rebuilt),
    );
}

test "rebuild rejects extra trailing indices not accounted for by the structure" {
    // A perfectly valid structure (0 segments, 0 flag bytes) but with one extra index
    // appended after original_length. rebuild() must reject this.
    const allocator = testing.allocator;
    var rebuilt = ArrayList(u8).empty;
    defer rebuilt.deinit(allocator);

    const coefficients = [_]f64{ 0.0, 1.0 }; // offset and scale only.
    const indices = [_]u64{ 0, 0, 0, 4, 999 }; // extra 999 after original_length=4.

    try testing.expectError(
        Error.CorruptedCompressedData,
        rebuild(allocator, &indices, &coefficients, &rebuilt),
    );
}
