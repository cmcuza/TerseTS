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

//! Provides a Zig API for TerseTS.

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;

const poor_mans_compression = @import("functional_approximation/poor_mans_compression.zig");
const swing_slide_filter = @import("functional_approximation/swing_slide_filter.zig");
const sim_piece = @import("functional_approximation/sim_piece.zig");
const mix_piece = @import("functional_approximation/mix_piece.zig");
const piecewise_histogram = @import("histogram_representation/constant_linear_representation.zig");
const abc_linear_approximation = @import("functional_approximation/abc_linear_approximation.zig");
const vw = @import("line_simplification/visvalingam_whyatt.zig");
const sliding_window = @import("line_simplification/sliding_window.zig");
const bottom_up = @import("line_simplification/bottom_up.zig");
const rle_enconding = @import("lossless_encoding/run_length_encoding.zig");
const bitpacked_quantization = @import("quantization/bitpacked_quantization.zig");
const non_linear_approximation = @import("functional_approximation/non_linear_approximation.zig");
const extractors = @import("utilities/extractors.zig");

/// The errors that can occur in TerseTS.
pub const Error = error{
    UnknownMethod,
    UnsupportedInput,
    UnsupportedErrorBound,
    CorruptedCompressedData,
    ItemNotFound,
    OutOfMemory,
    EmptyConvexHull,
    EmptyQueue,
    ByteStreamError,
};

/// The compression methods in TerseTS.
pub const Method = enum {
    PoorMansCompressionMidrange,
    PoorMansCompressionMean,
    SwingFilter,
    SwingFilterDisconnected,
    SlideFilter,
    SimPiece,
    PiecewiseConstantHistogram,
    PiecewiseLinearHistogram,
    ABCLinearApproximation,
    VisvalingamWhyatt,
    SlidingWindow,
    BottomUp,
    MixPiece,
    BitPackedQuantization,
    RunLengthEncoding,
    NonLinearApproximation,
};

/// Compress `uncompressed_values` within `error_bound` using `method` and returns the results
/// as a ArrayList of bytes returned by the compression methods. `allocator` is passed to the
/// compression functions for memory management. If the compression is sucessful, the `method`
/// is encoded in the compressed values last byte. If an error occurs it is returned.
pub fn compress(
    allocator: Allocator,
    uncompressed_values: []const f64,
    method: Method,
    error_bound: f32,
) Error!ArrayList(u8) {
    if (error_bound < 0) return Error.UnsupportedErrorBound;

    var compressed_values = ArrayList(u8).init(allocator);

    // If the input is one or zero elements, just store them uncompressed.
    if (uncompressed_values.len < 2) {
        if (uncompressed_values.len == 1) {
            const value_as_bytes: [8]u8 = @bitCast(uncompressed_values[0]);
            try compressed_values.appendSlice(value_as_bytes[0..]);
        }
        return compressed_values;
    }

    switch (method) {
        .PoorMansCompressionMidrange => {
            try poor_mans_compression.compressMidrange(
                uncompressed_values,
                &compressed_values,
                error_bound,
            );
        },
        .PoorMansCompressionMean => {
            try poor_mans_compression.compressMean(
                uncompressed_values,
                &compressed_values,
                error_bound,
            );
        },
        .SwingFilter => {
            try swing_slide_filter.compressSwingFilter(
                uncompressed_values,
                &compressed_values,
                error_bound,
            );
        },
        .SwingFilterDisconnected => {
            try swing_slide_filter.compressSwingFilterDisconnected(
                uncompressed_values,
                &compressed_values,
                error_bound,
            );
        },
        .SlideFilter => {
            try swing_slide_filter.compressSlideFilter(
                allocator,
                uncompressed_values,
                &compressed_values,
                error_bound,
            );
        },
        .SimPiece => {
            try sim_piece.compress(
                allocator,
                uncompressed_values,
                &compressed_values,
                error_bound,
            );
        },
        .MixPiece => {
            try mix_piece.compress(
                allocator,
                uncompressed_values,
                &compressed_values,
                error_bound,
            );
        },
        .PiecewiseConstantHistogram => {
            try piecewise_histogram.compressPWCH(
                allocator,
                uncompressed_values,
                &compressed_values,
                error_bound,
            );
        },
        .PiecewiseLinearHistogram => {
            try piecewise_histogram.compressPWLH(
                allocator,
                uncompressed_values,
                &compressed_values,
                error_bound,
            );
        },
        .ABCLinearApproximation => {
            try abc_linear_approximation.compress(
                allocator,
                uncompressed_values,
                &compressed_values,
                error_bound,
            );
        },
        .VisvalingamWhyatt => {
            try vw.compress(
                allocator,
                uncompressed_values,
                &compressed_values,
                error_bound,
            );
        },
        .SlidingWindow => {
            try sliding_window.compress(
                uncompressed_values,
                &compressed_values,
                error_bound,
            );
        },
        .BottomUp => {
            try bottom_up.compress(
                allocator,
                uncompressed_values,
                &compressed_values,
                error_bound,
            );
        },
        .RunLengthEncoding => {
            try rle_enconding.compress(uncompressed_values, &compressed_values);
        },
        .BitPackedQuantization => {
            try bitpacked_quantization.compress(
                allocator,
                uncompressed_values,
                &compressed_values,
                error_bound,
            );
        },
        .NonLinearApproximation => {
            try non_linear_approximation.compress(
                allocator,
                uncompressed_values,
                &compressed_values,
                error_bound,
            );
        },
    }
    try compressed_values.append(@intFromEnum(method));
    return compressed_values;
}

/// Decompress `compressed_values` using `method` and write the result to `decompressed_values`.
/// The compression `method` to use is encoded in the last byte of the `compressed_values`. If
/// an error occurs it is returned.
pub fn decompress(
    allocator: Allocator,
    compressed_values: []const u8,
) Error!ArrayList(f64) {
    if (compressed_values.len == 0) return Error.CorruptedCompressedData;

    const method_index: u8 = compressed_values[compressed_values.len - 1];
    if (method_index > getMaxMethodIndex()) return Error.UnknownMethod;

    const method: Method = @enumFromInt(method_index);
    const compressed_values_slice = compressed_values[0 .. compressed_values.len - 1];

    var decompressed_values = ArrayList(f64).init(allocator);

    // Handle the trivial case of one or zero elements.
    if (compressed_values.len < 9) {
        if (compressed_values.len == 8) {
            const value: f64 = @bitCast(compressed_values_slice[0..8].*);
            try decompressed_values.append(value);
        }
        return decompressed_values;
    }

    switch (method) {
        .PoorMansCompressionMidrange, .PoorMansCompressionMean => {
            try poor_mans_compression.decompress(compressed_values_slice, &decompressed_values);
        },
        .SwingFilter => {
            try swing_slide_filter.decompressSwingFilter(compressed_values_slice, &decompressed_values);
        },
        .SwingFilterDisconnected, .SlideFilter => {
            try swing_slide_filter.decompressSlideFilter(compressed_values_slice, &decompressed_values);
        },
        .SimPiece => {
            try sim_piece.decompress(allocator, compressed_values_slice, &decompressed_values);
        },
        .MixPiece => {
            try mix_piece.decompress(allocator, compressed_values_slice, &decompressed_values);
        },
        .PiecewiseConstantHistogram => {
            try piecewise_histogram.decompressPWCH(compressed_values_slice, &decompressed_values);
        },
        .PiecewiseLinearHistogram => {
            try piecewise_histogram.decompressPWLH(compressed_values_slice, &decompressed_values);
        },
        .ABCLinearApproximation => {
            try abc_linear_approximation.decompress(compressed_values_slice, &decompressed_values);
        },
        .VisvalingamWhyatt => {
            try vw.decompress(compressed_values_slice, &decompressed_values);
        },
        .SlidingWindow => {
            try sliding_window.decompress(compressed_values_slice, &decompressed_values);
        },
        .BottomUp => {
            try bottom_up.decompress(compressed_values_slice, &decompressed_values);
        },
        .RunLengthEncoding => {
            try rle_enconding.decompress(compressed_values_slice, &decompressed_values);
        },
        .BitPackedQuantization => {
            try bitpacked_quantization.decompress(compressed_values_slice, &decompressed_values);
        },
        .NonLinearApproximation => {
            try non_linear_approximation.decompress(allocator, compressed_values_slice, &decompressed_values);
        },
    }

    return decompressed_values;
}

/// Extracts `timestamps` and `coefficients` from a `compressed_values` using the method encoded in
/// the last byte. The function dispatches to the appropriate extractor in
/// `src/utilities/extractors.zig` based on the method. The function returns `Error.UnsupportedInput`
/// for unknown or unsupported methods. Moreover, the propagated errors from the extractors
/// are also returned.
pub fn extract(
    compressed_values: []const u8,
    timestamps: *ArrayList(usize),
    coefficients: *ArrayList(f64),
) Error!void {
    if (compressed_values.len == 0) return Error.UnsupportedInput;

    const method_index: u8 = compressed_values[compressed_values.len - 1];
    if (method_index > getMaxMethodIndex()) return Error.UnknownMethod;

    const method: Method = @enumFromInt(method_index);
    const compressed_values_slice = compressed_values[0 .. compressed_values.len - 1];

    switch (method) {
        // Both PMC methods use the same extractor.
        .PoorMansCompressionMean, .PoorMansCompressionMidrange => {
            try extractors.extractPMC(
                compressed_values_slice,
                timestamps,
                coefficients,
            );
        },
        .SwingFilter => {
            try extractors.extractSwing(
                compressed_values_slice,
                timestamps,
                coefficients,
            );
        },
        // Both SlideFilter and SwingFilterDisconnected use the same extractor.
        .SlideFilter, .SwingFilterDisconnected => {
            try extractors.extractSlide(
                compressed_values_slice,
                timestamps,
                coefficients,
            );
        },
        .SimPiece => {
            try extractors.extractSimPiece(
                compressed_values_slice,
                timestamps,
                coefficients,
            );
        },
        .MixPiece => {
            try extractors.extractMixPiece(
                compressed_values_slice,
                timestamps,
                coefficients,
            );
        },
        // Unsupported methods for extraction.
        // TODO: Implement extractors for the remaining methods.
        else => return Error.UnsupportedInput,
    }
}

/// Rebuilds `timestamps` and `coefficients` from a `compressed_values` using the `method` parameter.
/// The function dispatches to the appropriate extractor in `src/utilities/extractors.zig` based on
/// the method. The function appends the method byte to the rebuilt buffer for compatibility with
/// TerseTS APIs. The function returns `Error.UnsupportedInput` for unknown or unsupported methods.
/// Moreover, the propagated errors from the extractors are also returned.
pub fn rebuild(
    allocator: Allocator,
    timestamps: []const usize,
    coefficients: []const f64,
    method: Method,
) Error!ArrayList(u8) {
    if (timestamps.len == 0) return Error.UnsupportedInput;
    if (coefficients.len == 0) return Error.UnsupportedInput;

    var compressed_values = ArrayList(u8).init(allocator);

    switch (method) {
        // Both PMC methods use the same rebuilder.
        .PoorMansCompressionMean, .PoorMansCompressionMidrange => {
            try extractors.rebuildPMC(
                timestamps,
                coefficients,
                &compressed_values,
            );
        },
        .SwingFilter => {
            try extractors.rebuildSwing(
                timestamps,
                coefficients,
                &compressed_values,
            );
        },
        // Both SlideFilter and SwingFilterDisconnected use the same rebuilder.
        .SlideFilter, .SwingFilterDisconnected => {
            try extractors.rebuildSlide(
                timestamps,
                coefficients,
                &compressed_values,
            );
        },
        .SimPiece => {
            try extractors.rebuildSimPiece(
                timestamps,
                coefficients,
                &compressed_values,
            );
        },
        .MixPiece => {
            try extractors.rebuildMixPiece(
                timestamps,
                coefficients,
                &compressed_values,
            );
        },
        // Unsupported methods for rebuilding.
        // TODO: Implement rebuilders for the remaining methods.
        else => return Error.UnsupportedInput,
    }
    try compressed_values.append(@intFromEnum(method));
    return compressed_values;
}

/// Get the maximum index of the available methods in TerseTS.
pub fn getMaxMethodIndex() usize {
    const method_info = @typeInfo(Method).@"enum";

    var max_index: usize = 0;
    for (method_info.fields, 0..) |_, i| {
        max_index = if (i > max_index) i else max_index;
    }

    return max_index;
}
