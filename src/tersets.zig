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

const params = @import("params.zig");
const poor_mans_compression = @import("functional/poor_mans_compression.zig");
const swing_slide_filter = @import("functional/swing_slide_filter.zig");
const sim_piece = @import("functional/sim_piece.zig");
const piecewise_histogram = @import("functional/histogram_compression.zig");
const vw = @import("line_simplification/visvalingam_whyatt.zig");
const indentity = @import("lossless/identity.zig");

/// The errors that can occur in TerseTS.
pub const Error = error{
    UnknownMethod,
    UnsupportedInput,
    UnsupportedErrorBound,
    UnsupportedParameters,
    ItemNotFound,
    OutOfMemory,
    EmptyConvexHull,
    EmptyQueue,
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
    VisvalingamWhyatt,
    IdentityCompression,
};

/// Compress `uncompressed_values` within `error_bound` using `method` and returns the results
/// as a ArrayList of bytes returned by the compression methods. `allocator` is passed to the
/// compression functions for memory management. If the compression is sucessful, the `method`
/// is encoded in the compressed values last byte. If an error occurs it is returned.
pub fn compress(
    uncompressed_values: []const f64,
    allocator: Allocator,
    method: Method,
    parameters: ?*const anyopaque,
) Error!ArrayList(u8) {
    if (uncompressed_values.len == 0) return Error.UnsupportedInput;

    var compressed_values = ArrayList(u8).init(allocator);

    switch (method) {
        .PoorMansCompressionMidrange => {
            const param = try castParams(params.FunctionalParams, parameters);
            try poor_mans_compression.compressMidrange(
                uncompressed_values,
                &compressed_values,
                param.error_bound,
            );
        },
        .PoorMansCompressionMean => {
            const param = try castParams(params.FunctionalParams, parameters);
            try poor_mans_compression.compressMean(
                uncompressed_values,
                &compressed_values,
                param.error_bound,
            );
        },
        .SwingFilter => {
            const param = try castParams(params.FunctionalParams, parameters);
            try swing_slide_filter.compressSwingFilter(
                uncompressed_values,
                &compressed_values,
                param.error_bound,
            );
        },
        .SwingFilterDisconnected => {
            const param = try castParams(params.FunctionalParams, parameters);
            try swing_slide_filter.compressSwingFilterDisconnected(
                uncompressed_values,
                &compressed_values,
                param.error_bound,
            );
        },
        .SlideFilter => {
            const param = try castParams(params.FunctionalParams, parameters);
            try swing_slide_filter.compressSlideFilter(
                uncompressed_values,
                &compressed_values,
                allocator,
                param.error_bound,
            );
        },
        .SimPiece => {
            const param = try castParams(params.FunctionalParams, parameters);
            try sim_piece.compressSimPiece(
                uncompressed_values,
                &compressed_values,
                allocator,
                param.error_bound,
            );
        },
        .PiecewiseConstantHistogram => {
            const param = try castParams(params.HistogramParams, parameters);
            try piecewise_histogram.compressPWCH(
                uncompressed_values,
                &compressed_values,
                allocator,
                param.maximum_buckets,
            );
        },
        .PiecewiseLinearHistogram => {
            const param = try castParams(params.HistogramParams, parameters);
            try piecewise_histogram.compressPWLH(
                uncompressed_values,
                &compressed_values,
                allocator,
                param.maximum_buckets,
            );
        },
        .VisvalingamWhyatt => {
            const param = try castParams(params.LineSimplificationParams, parameters);
            try vw.compress(
                uncompressed_values,
                &compressed_values,
                allocator,
                param.error_bound,
            );
        },
        .IdentityCompression => {
            try indentity.compress(uncompressed_values, &compressed_values);
        },
    }
    try compressed_values.append(@intFromEnum(method));
    return compressed_values;
}

/// Decompress `compressed_values` using `method` and write the result to `decompressed_values`.
/// The compression `method` to use is encoded in the last byte of the `compressed_values`. If
/// an error occurs it is returned.
pub fn decompress(
    compressed_values: []const u8,
    allocator: Allocator,
) Error!ArrayList(f64) {
    if (compressed_values.len == 0) return Error.UnsupportedInput;

    const method_index: u8 = compressed_values[compressed_values.len - 1];
    if (method_index > getMaxMethodIndex()) return Error.UnknownMethod;

    const method: Method = @enumFromInt(method_index);
    const compressed_values_slice = compressed_values[0 .. compressed_values.len - 1];

    var decompressed_values = ArrayList(f64).init(allocator);

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
            try sim_piece.decompress(compressed_values_slice, &decompressed_values, allocator);
        },
        .PiecewiseConstantHistogram => {
            try piecewise_histogram.decompressPWCH(compressed_values_slice, &decompressed_values);
        },
        .PiecewiseLinearHistogram => {
            try piecewise_histogram.decompressPWLH(compressed_values_slice, &decompressed_values);
        },
        .VisvalingamWhyatt => {
            try vw.decompress(compressed_values_slice, &decompressed_values);
        },
        .IdentityCompression => {
            try indentity.decompress(compressed_values_slice, &decompressed_values);
        },
    }

    return decompressed_values;
}

/// Auxiliary function to validate of the decompressed time series is within the error bound of the
/// uncompressed time series. The function returns true if all elements are within the error bound,
/// false otherwise.
pub fn isWithinErrorBound(
    uncompressed_values: []const f64,
    decompressed_values: []const f64,
    error_bound: f32,
) bool {
    if (uncompressed_values.len != decompressed_values.len) {
        return false;
    }

    for (0..uncompressed_values.len) |index| {
        const uncompressed_value = uncompressed_values[index];
        const decompressed_value = decompressed_values[index];
        if (@abs(uncompressed_value - decompressed_value) > error_bound) return false;
    }
    return true;
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

fn castParams(comptime ParamsType: type, parameters: ?*const anyopaque) !*const ParamsType {
    if (parameters == null) return Error.UnsupportedParameters;
    return @alignCast(@ptrCast(parameters.?));
}
