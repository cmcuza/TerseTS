// Copyright 2025 TerseTS Contributors
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

//! Provides the available configurations for compression methods in TerseTS.
//!
//! This file defines the configuration types and parsing logic for the various compression
//! methods supported by TerseTS. Each compression method requires a configuration
//! structure, such as absolute or relative error bounds, histogram bin counts, or aggregate
//! error parameters. The configuration is parsed from JSON and validated according to the
//! requirements of each method. Invalid or missing configurations are reported with a descriptive
//! error type. All configuration types are documented below.

const std = @import("std");
const json = std.json;
const Allocator = std.mem.Allocator;
const tersets = @import("tersets.zig");

/// Configuration for methods that require an absolute error bound.
/// Example: { "abs_error_bound": 0.1 }
pub const AbsoluteErrorBound = struct {
    abs_error_bound: f32,
};

/// Configuration for methods that require a relative error bound.
/// Example: { "rel_error_bound": 0.01 }
pub const RelativeErrorBound = struct {
    rel_error_bound: f32,
};

/// Configuration for histogram-based methods specifying the number of bins.
/// Example: { "histogram_bins_number": 32 }
pub const HistogramBinsNumber = struct {
    histogram_bins_number: u32,
};

/// Configuration for methods that require an aggregate error type and bound.
/// Example: { "aggregate_error_type": "rmse", "aggregate_error_bound": 5.0 }
pub const AggregateError = struct {
    aggregate_error_type: []const u8,
    aggregate_error_bound: f32,
};

/// Configuration for methods that require an area under the curve error bound.
/// Example: { "area_under_curve_error": 0.01 }
pub const AreaUnderCurveError = struct {
    area_under_curve_error: f32,
};

/// Empty configuration for methods that do not require any parameters.
pub const EmptyConfiguration = struct {};

/// Used to indicate that the provided configuration is invalid or missing required fields.
/// The expected configuration is described in the error.
pub const InvalidConfiguration = struct {
    expected_configuration: []const u8,
};

const Configuration = enum {
    AbsoluteErrorBound,
    RelativeErrorBound,
    HistogramBinsNumber,
    AggregateError,
    AreaUnderCurveError,
    EmptyConfiguration,
    InvalidConfiguration,
};

/// This is a small convenience wrapper around json.parseFromSlice that accepts a
/// `ConfigurationType` and parses the the JSON text `configuration`. The function
/// returns the parsed value on success, or `null` on failure. The `allocator` is
/// used by the JSON parser.
pub fn parse(
    allocator: Allocator,
    comptime ConfigurationType: type,
    configuration: []const u8,
) ?ConfigurationType {
    const parsed = json.parseFromSlice(
        ConfigurationType,
        allocator,
        configuration,
        .{},
    ) catch return null;
    defer parsed.deinit();
    return parsed.value;
}

/// Get valid configuration given a `method`. If the method does not exist in TerseTS,
/// the function retuns the `InvalidConfiguration`.
fn getConfiguration(method: tersets.Method) type {
    return switch (method) {
        .PoorMansCompressionMean,
        .PoorMansCompressionMidrange,
        .BitPackedQuantization,
        .MixPiece,
        .SimPiece,
        .SlideFilter,
        .SwingFilterDisconnected,
        .SwingFilter,
        .ABCLinearApproximation,
        .NonLinearApproximation,
        => AbsoluteErrorBound,
        .BottomUp,
        .SlidingWindow,
        => AggregateError,
        .VisvalingamWhyatt => AreaUnderCurveError,
        .PiecewiseConstantHistogram,
        .PiecewiseLinearHistogram,
        => HistogramBinsNumber,
        .RunLengthEncoding => EmptyConfiguration,
    };
}

test "check configuration parsing for all methods" {
    const allocator = std.testing.allocator;

    const Case = struct {
        method: tersets.Method,
        configuration: []const u8,
        expected_tag: Configuration,
    };

    // List all methods and their valid/invalid configurations.
    const cases = [_]Case{
        // AbsoluteErrorBound methods (valid/invalid)
        .{
            .method = .PoorMansCompressionMean,
            .configuration =
            \\ { "abs_error_bound": 0.1 }
            ,
            .expected_tag = .AbsoluteErrorBound,
        },
        .{
            .method = .PoorMansCompressionMean,
            .configuration =
            \\ { "rel_error_bound": 0.1 }
            ,
            .expected_tag = .InvalidConfiguration,
        },
        .{
            .method = .SwingFilter,
            .configuration =
            \\ { "abs_error_bound": 10.0 }
            ,
            .expected_tag = .AbsoluteErrorBound,
        },
        .{
            .method = .SwingFilter,
            .configuration =
            \\ { "abs_error_bound": "bad" }
            ,
            .expected_tag = .InvalidConfiguration,
        },
        .{
            .method = .NonLinearApproximation,
            .configuration =
            \\ { "abs_error_bound": 1.0 }
            ,
            .expected_tag = .AbsoluteErrorBound,
        },
        .{
            .method = .NonLinearApproximation,
            .configuration =
            \\ {}
            ,
            .expected_tag = .InvalidConfiguration,
        },
        .{
            .method = .ABCLinearApproximation,
            .configuration =
            \\ {}
            ,
            .expected_tag = .InvalidConfiguration,
        },

        // AggregateError methods (valid/invalid)
        .{
            .method = .BottomUp,
            .configuration =
            \\ { "aggregate_error_type": "rmse", "aggregate_error_bound": 5.0 }
            ,
            .expected_tag = .AggregateError,
        },
        .{
            .method = .BottomUp,
            .configuration =
            \\ { "aggregate_error_type": "rmse" }
            ,
            .expected_tag = .InvalidConfiguration,
        },
        .{
            .method = .SlidingWindow,
            .configuration =
            \\ { "aggregate_error_type": "mae", "aggregate_error_bound": 3.0 }
            ,
            .expected_tag = .AggregateError,
        },
        .{
            .method = .SlidingWindow,
            .configuration =
            \\ { "aggregate_error_bound": 3.0 }
            ,
            .expected_tag = .InvalidConfiguration,
        },
        .{
            .method = .BottomUp,
            .configuration =
            \\ {}
            ,
            .expected_tag = .InvalidConfiguration,
        },

        // AreaUnderCurveError methods (valid/invalid)
        .{
            .method = .VisvalingamWhyatt,
            .configuration =
            \\ { "area_under_curve_error": 0.01 }
            ,
            .expected_tag = .AreaUnderCurveError,
        },
        .{
            .method = .VisvalingamWhyatt,
            .configuration =
            \\ { "the_area_in_curve": 123 }
            ,
            .expected_tag = .InvalidConfiguration,
        },
        .{
            .method = .VisvalingamWhyatt,
            .configuration =
            \\ {}
            ,
            .expected_tag = .InvalidConfiguration,
        },

        // HistogramBinsNumber methods (valid/invalid)
        .{
            .method = .PiecewiseConstantHistogram,
            .configuration =
            \\ { "histogram_bins_number": 32 }
            ,
            .expected_tag = .HistogramBinsNumber,
        },
        .{
            .method = .PiecewiseConstantHistogram,
            .configuration =
            \\ { "wrong_parameter": 12 }
            ,
            .expected_tag = .InvalidConfiguration,
        },
        .{
            .method = .PiecewiseLinearHistogram,
            .configuration =
            \\ { "histogram_bins_number": 16 }
            ,
            .expected_tag = .HistogramBinsNumber,
        },
        .{
            .method = .PiecewiseLinearHistogram,
            .configuration =
            \\ {}
            ,
            .expected_tag = .InvalidConfiguration,
        },

        // EmptyConfiguration methods (valid/invalid)
        .{
            .method = .RunLengthEncoding,
            .configuration =
            \\ {}
            ,
            .expected_tag = .EmptyConfiguration,
        },
        .{
            .method = .RunLengthEncoding,
            .configuration =
            \\ { "abs_error_bound": 1.0 }
            ,
            .expected_tag = .InvalidConfiguration,
        },
    };

    inline for (cases) |case| { // <— inline makes each case comptime-known
        const ConfigurationType = getConfiguration(case.method);
        const parsed = parse(
            allocator,
            ConfigurationType,
            case.configuration,
        );
        const tag: Configuration = if (parsed) |*val| switch (@TypeOf(val.*)) {
            AbsoluteErrorBound => .AbsoluteErrorBound,
            RelativeErrorBound => .RelativeErrorBound,
            AggregateError => .AggregateError,
            AreaUnderCurveError => .AreaUnderCurveError,
            HistogramBinsNumber => .HistogramBinsNumber,
            EmptyConfiguration => .EmptyConfiguration,
            else => .InvalidConfiguration,
        } else .InvalidConfiguration;

        try std.testing.expectEqual(case.expected_tag, tag);
    }
}
