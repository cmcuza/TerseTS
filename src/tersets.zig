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
    InvalidConfiguration,
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

/// Compress `uncompressed_values` using `method` and its `configuration` and returns the results
/// as a ArrayList of bytes returned by the compression methods. `allocator` is passed to the
/// compression functions for memory management. If the compression is sucessful, the `method`
/// is encoded in the compressed values last byte. If an error occurs it is returned.
pub fn compress(
    allocator: Allocator,
    uncompressed_values: []const f64,
    method: Method,
    configuration: []const u8,
) Error!ArrayList(u8) {
    var compressed_values = ArrayList(u8).init(allocator);

    // If the input is one or zero elements, just store them uncompressed disregarding
    // the compression method.
    if (uncompressed_values.len < 2) {
        if (uncompressed_values.len == 1) {
            const value_as_bytes: [8]u8 = @bitCast(uncompressed_values[0]);
            try compressed_values.appendSlice(value_as_bytes[0..]);
            return compressed_values;
        }
        // The uncompressed_values is empty.
        return Error.UnsupportedInput;
    }

    switch (method) {
        .PoorMansCompressionMidrange => {
            try poor_mans_compression.compressMidrange(
                allocator,
                uncompressed_values,
                &compressed_values,
                configuration,
            );
        },
        .PoorMansCompressionMean => {
            try poor_mans_compression.compressMean(
                allocator,
                uncompressed_values,
                &compressed_values,
                configuration,
            );
        },
        .SwingFilter => {
            try swing_slide_filter.compressSwingFilter(
                allocator,
                uncompressed_values,
                &compressed_values,
                configuration,
            );
        },
        .SwingFilterDisconnected => {
            try swing_slide_filter.compressSwingFilterDisconnected(
                allocator,
                uncompressed_values,
                &compressed_values,
                configuration,
            );
        },
        .SlideFilter => {
            try swing_slide_filter.compressSlideFilter(
                allocator,
                uncompressed_values,
                &compressed_values,
                configuration,
            );
        },
        .SimPiece => {
            try sim_piece.compress(
                allocator,
                uncompressed_values,
                &compressed_values,
                configuration,
            );
        },
        .MixPiece => {
            try mix_piece.compress(
                allocator,
                uncompressed_values,
                &compressed_values,
                configuration,
            );
        },
        .PiecewiseConstantHistogram => {
            // Again, we need to extract the actual value from the optional `number_histogram_bins`.
            try piecewise_histogram.compressPWCH(
                allocator,
                uncompressed_values,
                &compressed_values,
                configuration,
            );
        },
        .PiecewiseLinearHistogram => {
            // Again, we need to extract the actual value from the optional `number_histogram_bins`.
            try piecewise_histogram.compressPWLH(
                allocator,
                uncompressed_values,
                &compressed_values,
                configuration,
            );
        },
        .ABCLinearApproximation => {
            try abc_linear_approximation.compress(
                allocator,
                uncompressed_values,
                &compressed_values,
                configuration,
            );
        },
        .VisvalingamWhyatt => {
            try vw.compress(
                allocator,
                uncompressed_values,
                &compressed_values,
                configuration,
            );
        },
        .SlidingWindow => {
            try sliding_window.compress(
                allocator,
                uncompressed_values,
                &compressed_values,
                configuration,
            );
        },
        .BottomUp => {
            try bottom_up.compress(
                allocator,
                uncompressed_values,
                &compressed_values,
                configuration,
            );
        },
        .RunLengthEncoding => {
            try rle_enconding.compress(
                allocator,
                uncompressed_values,
                &compressed_values,
                configuration,
            );
        },
        .BitPackedQuantization => {
            try bitpacked_quantization.compress(
                allocator,
                uncompressed_values,
                &compressed_values,
                configuration,
            );
        },
        .NonLinearApproximation => {
            try non_linear_approximation.compress(
                allocator,
                uncompressed_values,
                &compressed_values,
                configuration,
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

    var decompressed_values = ArrayList(f64).init(allocator);

    // Handle the trivial case of one element.
    if (compressed_values.len == 8) {
        const value: f64 = @bitCast(compressed_values[0..8].*);
        try decompressed_values.append(value);
        return decompressed_values;
    }

    const method_index: u8 = compressed_values[compressed_values.len - 1];

    if (method_index > getMaxMethodIndex()) return Error.UnknownMethod;

    const method: Method = @enumFromInt(method_index);
    const compressed_values_slice = compressed_values[0 .. compressed_values.len - 1];

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

/// Get the maximum index of the available methods in TerseTS.
pub fn getMaxMethodIndex() usize {
    const method_info = @typeInfo(Method).@"enum";

    var max_index: usize = 0;
    for (method_info.fields, 0..) |_, i| {
        max_index = if (i > max_index) i else max_index;
    }

    return max_index;
}
