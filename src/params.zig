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

//! This module defines all compression parameter structures used in TerseTS.
//! It provides strongly-typed, FFI-compatible structs for different compression methods,
//! and a helper to convert from a generic `BasicParams` to method-specific parameters
//! when default configuration is used.

const std = @import("std");
const Method = @import("tersets.zig").Method;
const Allocator = std.mem.Allocator;

/// The type of error metric used to control compression accuracy.
pub const ErrorBoundType = enum(u8) {
    /// Absolute error threshold.
    abs_error_bound,
    /// Relative error threshold, typically a ratio.
    relative_error_bound,
};

/// The cost function used to control compression process.
pub const CostFunction = enum(u8) {
    /// Root mean square error (RMSE).
    root_mean_square_error,
    /// L-infinity norm (maximum absolute error).
    l_infinity_norm,
};

/// A basic parameter structure used as a default for compression methods that only require a
/// scalar error bound. This struct is often used when the user wants to compress a time series
/// without configuring method-specific details.
pub const BasicParams = extern struct {
    /// The error bound used for compression, interpreted as absolute by default.
    error_bound: f32,
};

/// Parameters for methods using a functional approximation model, such as linear or piecewise
/// regression techniques.
pub const FunctionalParams = extern struct {
    /// The type of error bound to apply.
    error_bound_type: ErrorBoundType,
    /// The numeric threshold for the chosen error bound type.
    error_bound: f32,
};

/// Parameters for line simplification methods (e.g., Visvalingam-Whyatt).
pub const LineSimplificationParams = extern struct {
    /// The type of cost function to apply.
    cost_function: CostFunction,
    /// The numeric error threshold.
    error_bound: f32,
};

/// Parameters for histogram-based compression methods.
pub const HistogramParams = extern struct {
    /// The maximum number of buckets to use in the histogram.
    maximum_buckets: usize,
};

/// Converts a `BasicParams` structure to a method-specific parameter struct for a given
/// compression `method`. This function is used internally to support default compression
/// configurations where users supply only a basic error bound. If the method supports
/// basic parameters, this function returns a pointer to the corresponding method-specific
/// parameter struct. Otherwise, it returns `null`.
pub fn mapBasicParamsToMethodParams(
    method: Method,
    basic: *const BasicParams,
    allocator: Allocator,
) !?*const anyopaque {
    switch (method) {
        .PoorMansCompressionMidrange,
        .PoorMansCompressionMean,
        .SlideFilter,
        .SimPiece,
        .SwingFilter,
        .SwingFilterDisconnected,
        .ABCLinearApproximation,
        => {
            const concrete = try allocator.create(FunctionalParams);
            concrete.* = FunctionalParams{
                .error_bound_type = .abs_error_bound,
                .error_bound = basic.error_bound,
            };
            return @ptrCast(@as(*const anyopaque, concrete));
        },
        .PiecewiseConstantHistogram, .PiecewiseLinearHistogram => {
            const concrete = try allocator.create(HistogramParams);
            concrete.* = HistogramParams{
                .maximum_buckets = @intFromFloat(basic.error_bound),
            };
            return @ptrCast(@as(*const anyopaque, concrete));
        },
        .VisvalingamWhyatt => {
            const concrete = try allocator.create(LineSimplificationParams);
            concrete.* = LineSimplificationParams{
                .error_bound = basic.error_bound,
                .cost_function = .root_mean_square_error,
            };
            return @ptrCast(@as(*const anyopaque, concrete));
        },
        .IdentityCompression => {
            return null;
        },
    }
}

/// Cast a *anyopaque pointer to a mutable pointer of the given type.
/// Must only be used when the actual underlying type is known.
pub fn castOpaquePtr(comptime T: type, ptr: *anyopaque) *T {
    return @alignCast(@ptrCast(ptr));
}

/// Destroys method-specific parameters previously returned by mapBasicParamsToMethodParams.
/// The caller must ensure `ptr` was allocated via the `allocator` and matches the `method`.
pub fn destroyMappedParams(
    allocator: Allocator,
    method: Method,
    ptr: *const anyopaque,
) void {
    // Unsafe, but caller promises ownership so this is OK:
    const mutable_ptr: *anyopaque = @constCast(ptr);

    switch (method) {
        .SimPiece,
        .SlideFilter,
        .SwingFilter,
        .SwingFilterDisconnected,
        .PoorMansCompressionMidrange,
        .PoorMansCompressionMean,
        .ABCLinearApproximation,
        => {
            const typed: *FunctionalParams = castOpaquePtr(FunctionalParams, mutable_ptr);
            allocator.destroy(typed);
        },
        .PiecewiseConstantHistogram, .PiecewiseLinearHistogram => {
            const typed: *HistogramParams = castOpaquePtr(HistogramParams, mutable_ptr);
            allocator.destroy(typed);
        },
        .VisvalingamWhyatt => {
            const typed: *LineSimplificationParams = castOpaquePtr(LineSimplificationParams, mutable_ptr);
            allocator.destroy(typed);
        },
        .IdentityCompression => {},
    }
}
