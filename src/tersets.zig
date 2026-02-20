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
const testing = std.testing;

// Import functional approximation methods.
const poor_mans_compression = @import(
    "lossy_compression/functional_approximation/poor_mans_compression.zig",
);
const swing_slide_filter = @import(
    "lossy_compression/functional_approximation/swing_slide_filter.zig",
);
const abc_linear_approximation = @import(
    "lossy_compression/functional_approximation/abc_linear_approximation.zig",
);
const sim_piece = @import("lossy_compression/functional_approximation/sim_piece.zig");
const mix_piece = @import("lossy_compression/functional_approximation/mix_piece.zig");
const non_linear_approximation = @import(
    "lossy_compression/functional_approximation/non_linear_approximation.zig",
);

// Import value approximation methods.
const piecewise_histogram = @import(
    "lossy_compression/value_representation/histogram_representation.zig",
);
const bitpacked_quantization = @import(
    "lossy_compression/value_representation/bitpacked_quantization.zig",
);
const serqt = @import("lossy_compression/value_representation/serf_qt.zig");
const buff = @import("lossy_compression/value_representation/bounded_fast_floats.zig");

// Import line simplification methods.
const vw = @import("lossy_compression/line_simplification/visvalingam_whyatt.zig");
const sliding_window = @import("lossy_compression/line_simplification/sliding_window.zig");
const bottom_up = @import("lossy_compression/line_simplification/bottom_up.zig");
const rle_enconding = @import("lossless_compression/run_length_encoding.zig");

const extractors = @import("utilities/extractors.zig");
const tester = @import("tester.zig");
const configuration_file = @import("configuration.zig");

/// The errors that can occur in TerseTS.
pub const Error = error{
    UnknownMethod,
    UnsupportedInput,
    InvalidConfiguration,
    CorruptedCompressedData,
    ItemNotFound,
    EmptyConvexHull,
    EmptyQueue,
    ByteStreamError,
    UnsupportedMethod,
    OutOfMemory,
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
    SerfQT,
    BitPackedBUFF,
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
    var compressed_values = ArrayList(u8).empty;

    // If the input is one or zero elements, just store them uncompressed disregarding
    // the compression method.
    if (uncompressed_values.len < 2) {
        if (uncompressed_values.len == 1) {
            const value_as_bytes: [8]u8 = @bitCast(uncompressed_values[0]);
            try compressed_values.appendSlice(allocator, value_as_bytes[0..]);
            return compressed_values;
        }
        // The `uncompressed_values` is empty.
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
            try piecewise_histogram.compressPWCH(
                allocator,
                uncompressed_values,
                &compressed_values,
                configuration,
            );
        },
        .PiecewiseLinearHistogram => {
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
        .SerfQT => {
            try serqt.compress(
                allocator,
                uncompressed_values,
                &compressed_values,
                configuration,
            );
        },
        .BitPackedBUFF => {
            try buff.compressBitPackedBUFF(
                allocator,
                uncompressed_values,
                &compressed_values,
                configuration,
            );
        },
    }
    try compressed_values.append(allocator, @intFromEnum(method));
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

    var decompressed_values = ArrayList(f64).empty;

    // Handle the trivial case of one element.
    if (compressed_values.len == 8) {
        const value: f64 = @bitCast(compressed_values[0..8].*);
        try decompressed_values.append(allocator, value);
        return decompressed_values;
    }

    const method_index: u8 = compressed_values[compressed_values.len - 1];

    if (method_index > getMaxMethodIndex()) return Error.UnknownMethod;

    const method: Method = @enumFromInt(method_index);
    const compressed_values_slice = compressed_values[0 .. compressed_values.len - 1];

    switch (method) {
        .PoorMansCompressionMidrange, .PoorMansCompressionMean => {
            try poor_mans_compression.decompress(allocator, compressed_values_slice, &decompressed_values);
        },
        .SwingFilter => {
            try swing_slide_filter.decompressSwingFilter(allocator, compressed_values_slice, &decompressed_values);
        },
        .SwingFilterDisconnected, .SlideFilter => {
            try swing_slide_filter.decompressSlideFilter(allocator, compressed_values_slice, &decompressed_values);
        },
        .SimPiece => {
            try sim_piece.decompress(allocator, compressed_values_slice, &decompressed_values);
        },
        .MixPiece => {
            try mix_piece.decompress(allocator, compressed_values_slice, &decompressed_values);
        },
        .PiecewiseConstantHistogram => {
            try piecewise_histogram.decompressPWCH(allocator, compressed_values_slice, &decompressed_values);
        },
        .PiecewiseLinearHistogram => {
            try piecewise_histogram.decompressPWLH(allocator, compressed_values_slice, &decompressed_values);
        },
        .ABCLinearApproximation => {
            try abc_linear_approximation.decompress(allocator, compressed_values_slice, &decompressed_values);
        },
        .VisvalingamWhyatt => {
            try vw.decompress(allocator, compressed_values_slice, &decompressed_values);
        },
        .SlidingWindow => {
            try sliding_window.decompress(allocator, compressed_values_slice, &decompressed_values);
        },
        .BottomUp => {
            try bottom_up.decompress(allocator, compressed_values_slice, &decompressed_values);
        },
        .RunLengthEncoding => {
            try rle_enconding.decompress(allocator, compressed_values_slice, &decompressed_values);
        },
        .BitPackedQuantization => {
            try bitpacked_quantization.decompress(allocator, compressed_values_slice, &decompressed_values);
        },
        .NonLinearApproximation => {
            try non_linear_approximation.decompress(allocator, compressed_values_slice, &decompressed_values);
        },
        .SerfQT => {
            try serqt.decompress(allocator, compressed_values_slice, &decompressed_values);
        },
        .BitPackedBUFF => {
            try buff.decompressBitPackedBUFF(
                allocator,
                compressed_values_slice,
                &decompressed_values,
            );
        },
    }

    return decompressed_values;
}

/// Extract `indices` and `coefficients` from `compressed_values` using the method encoded in the last byte.
/// The function dispatches to the appropriate extractor based on the method. On success, the extracted
/// `indices` and `coefficients` are populated. The `allocator` handles the memory for the output arrays.
/// If an error occurs it is returned.
pub fn extract(
    allocator: Allocator,
    compressed_values: []const u8,
    indices: *ArrayList(u64),
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
            try poor_mans_compression.extract(
                allocator,
                compressed_values_slice,
                indices,
                coefficients,
            );
        },
        .SwingFilter => {
            try swing_slide_filter.extractSwing(
                allocator,
                compressed_values_slice,
                indices,
                coefficients,
            );
        },
        // Both SlideFilter and SwingFilterDisconnected use the same extractor.
        .SlideFilter, .SwingFilterDisconnected => {
            try swing_slide_filter.extractSlide(
                allocator,
                compressed_values_slice,
                indices,
                coefficients,
            );
        },
        .ABCLinearApproximation => {
            try abc_linear_approximation.extract(
                allocator,
                compressed_values_slice,
                indices,
                coefficients,
            );
        },
        .SimPiece => {
            try sim_piece.extract(
                allocator,
                compressed_values_slice,
                indices,
                coefficients,
            );
        },
        .MixPiece => {
            try mix_piece.extract(
                allocator,
                compressed_values_slice,
                indices,
                coefficients,
            );
        },
        .PiecewiseConstantHistogram => {
            try piecewise_histogram.extractPWCH(
                allocator,
                compressed_values_slice,
                indices,
                coefficients,
            );
        },
        .PiecewiseLinearHistogram => {
            try piecewise_histogram.extractPWLH(
                allocator,
                compressed_values_slice,
                indices,
                coefficients,
            );
        },
        .VisvalingamWhyatt => {
            try vw.extract(
                allocator,
                compressed_values_slice,
                indices,
                coefficients,
            );
        },
        .SlidingWindow => {
            try sliding_window.extract(
                allocator,
                compressed_values_slice,
                indices,
                coefficients,
            );
        },
        .BottomUp => {
            try bottom_up.extract(
                allocator,
                compressed_values_slice,
                indices,
                coefficients,
            );
        },
        .NonLinearApproximation => {
            try non_linear_approximation.extract(
                allocator,
                compressed_values_slice,
                indices,
                coefficients,
            );
        },
        // For the following three methods, it is not possible to guarantee
        // that the pipeline will work as intended. This is because even small
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
        .BitPackedQuantization,
        .RunLengthEncoding,
        .BitPackedBUFF,
        .SerfQT,
        => {
            return Error.UnsupportedMethod;
        },
    }
}

/// Rebuild `compressed_values` from `indices` and `coefficients` using the specified `method`.
/// The function dispatches to the appropriate rebuilder based on the method. On success, the rebuilt
/// `compressed_values` are returned with the method byte appended. If an error occurs it is returned.
pub fn rebuild(
    allocator: Allocator,
    indices: []const u64,
    coefficients: []const f64,
    method: Method,
) Error!ArrayList(u8) {
    if (coefficients.len == 0) return Error.UnsupportedInput;

    var compressed_values = ArrayList(u8).empty;

    switch (method) {
        // Both PMC methods use the same rebuilder.
        .PoorMansCompressionMean, .PoorMansCompressionMidrange => {
            try poor_mans_compression.rebuild(
                allocator,
                indices,
                coefficients,
                &compressed_values,
            );
        },
        .SwingFilter => {
            try swing_slide_filter.rebuildSwing(
                allocator,
                indices,
                coefficients,
                &compressed_values,
            );
        },
        // Both SlideFilter and SwingFilterDisconnected use the same rebuilder.
        .SlideFilter, .SwingFilterDisconnected => {
            try swing_slide_filter.rebuildSlide(
                allocator,
                indices,
                coefficients,
                &compressed_values,
            );
        },
        .ABCLinearApproximation => {
            try abc_linear_approximation.rebuild(
                allocator,
                indices,
                coefficients,
                &compressed_values,
            );
        },
        .SimPiece => {
            try sim_piece.rebuild(
                allocator,
                indices,
                coefficients,
                &compressed_values,
            );
        },
        .MixPiece => {
            try mix_piece.rebuild(
                allocator,
                indices,
                coefficients,
                &compressed_values,
            );
        },
        .PiecewiseConstantHistogram => {
            try piecewise_histogram.rebuildPWCH(
                allocator,
                indices,
                coefficients,
                &compressed_values,
            );
        },
        .PiecewiseLinearHistogram => {
            try piecewise_histogram.rebuildPWLH(
                allocator,
                indices,
                coefficients,
                &compressed_values,
            );
        },
        .SlidingWindow => {
            try sliding_window.rebuild(
                allocator,
                indices,
                coefficients,
                &compressed_values,
            );
        },
        .BottomUp => {
            try bottom_up.rebuild(
                allocator,
                indices,
                coefficients,
                &compressed_values,
            );
        },
        .VisvalingamWhyatt => {
            try vw.rebuild(
                allocator,
                indices,
                coefficients,
                &compressed_values,
            );
        },
        .NonLinearApproximation => {
            try non_linear_approximation.rebuild(
                allocator,
                indices,
                coefficients,
                &compressed_values,
            );
        },
        // For the following three methods, it is not possible to guarantee
        // that the pipeline will work as intended. This is because even small
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
        .BitPackedQuantization,
        .BitPackedBUFF,
        .SerfQT,
        .RunLengthEncoding,
        => {
            return Error.UnsupportedMethod;
        },
    }
    try compressed_values.append(allocator, @intFromEnum(method));
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
    inline for (std.meta.fields(Method)) |method_field| {
        const method: Method = @enumFromInt(method_field.value);

        if (method == Method.BitPackedQuantization or
            method == Method.SerfQT or
            method == Method.RunLengthEncoding or
            method == Method.BitPackedBUFF)
        {
            // These compression methods are not supported for extraction
            // of the coefficients and indices. This is because even small
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

        const method_configuration = try configuration_file.defaultConfigurationBuilder(
            allocator,
            method,
        );
        defer allocator.free(method_configuration);

        var compressed_values = try compress(
            allocator,
            uncompressed_values.items,
            method,
            method_configuration,
        );
        defer compressed_values.deinit(allocator);

        var decompressed_values = try decompress(
            allocator,
            compressed_values.items,
        );
        defer decompressed_values.deinit(allocator);

        // Test extract and rebuild.
        var coefficient_values = ArrayList(f64).empty;
        defer coefficient_values.deinit(allocator);
        var index_values = ArrayList(u64).empty;
        defer index_values.deinit(allocator);

        try extract(
            allocator,
            compressed_values.items,
            &index_values,
            &coefficient_values,
        );

        var rebuild_values = try rebuild(
            allocator,
            index_values.items,
            coefficient_values.items,
            method,
        );
        defer rebuild_values.deinit(allocator);

        try testing.expectEqual(rebuild_values.items.len, compressed_values.items.len);
        try testing.expectEqualSlices(u8, rebuild_values.items, compressed_values.items);
    }
}
