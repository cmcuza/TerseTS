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

//! Provides the parameters structures for all compression methods in TerseTS.

const std = @import("std");
const Method = @import("tersets.zig").Method;
const Allocator = std.mem.Allocator;

pub const ErrorBoundType = enum(u8) {
    abs_error_bound,
    relative_error_bound,
    roo_mean_square_error,
    l_infinity_norm,
};

pub const BasicParams = extern struct {
    error_bound: f32,
};

pub const FunctionalParams = extern struct {
    error_bound_type: ErrorBoundType,
    error_bound: f32,
};

pub const LineSimplificationParams = extern struct {
    error_bound_type: ErrorBoundType,
    error_bound: f32,
};

pub const HistogramParams = extern struct {
    maximum_buckets: usize,
};

/// Converts BasicParams into method-specific param struct, heap-allocating it.
pub fn mapBasicParamsToMethodParams(
    method: Method,
    basic: *const BasicParams,
) !?*const anyopaque {
    switch (method) {
        .PoorMansCompressionMidrange, .PoorMansCompressionMean, .SlideFilter, .SimPiece, .SwingFilter, .SwingFilterDisconnected => {
            const concrete = FunctionalParams{
                .error_bound_type = .abs_error_bound,
                .error_bound = basic.error_bound,
            };
            return &concrete;
        },
        .PiecewiseConstantHistogram, .PiecewiseLinearHistogram => {
            const concrete = HistogramParams{
                .maximum_buckets = @intFromFloat(basic.error_bound),
            };
            return &concrete;
        },
        .VisvalingamWhyatt => {
            const concrete = LineSimplificationParams{
                .error_bound = basic.error_bound,
                .error_bound_type = .roo_mean_square_error,
            };
            return &concrete;
        },
        .IdentityCompression => {
            return null;
        },
    }
}
