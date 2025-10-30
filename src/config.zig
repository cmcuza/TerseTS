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

//! Provides the available configurations for compression methods in TerseTS.
//!
//! This file defines the configuration types and parsing logic for the various compression
//! methods supported by TerseTS. Each compression method may require a different configuration
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

/// Union of all possible configuration types for compression methods.
/// The tag indicates which configuration is active.
pub const Configuration = union(enum) {
    AbsoluteErrorBound: AbsoluteErrorBound,
    RelativeErrorBound: RelativeErrorBound,
    HistogramBinsNumber: HistogramBinsNumber,
    AggregateError: AggregateError,
    AreaUnderCurveError: AreaUnderCurveError,
    InvalidConfiguration: InvalidConfiguration,
    EmptyConfiguration: EmptyConfiguration,
};

/// Parses the `configuration` json string for the given compression `method`.
/// Returns the appropriate configuration type or an `InvalidConfiguration` if parsing fails.
/// The `allocator` is used for temporary allocations during parsing.
pub fn parse(
    allocator: Allocator,
    method: tersets.Method,
    configuration: []const u8,
) Configuration {
    switch (method) {
        // Methods that require an absolute error bound configuration.
        .PoorMansCompressionMean,
        .PoorMansCompressionMidrange,
        .ABCLinearApproximation,
        .SimPiece,
        .MixPiece,
        .BitPackedQuantization,
        .SwingFilter,
        .SlideFilter,
        .SwingFilterDisconnected,
        .NonLinearApproximation,
        => {
            // Attempt to parse the configuration as AbsoluteErrorBound.
            // If parsing fails, return InvalidConfiguration with the expected field.
            const parsed = json.parseFromSlice(
                AbsoluteErrorBound,
                allocator,
                configuration,
                .{},
            ) catch {
                return Configuration{
                    .InvalidConfiguration = .{
                        .expected_configuration = "abs_error_bound",
                    },
                };
            };
            defer parsed.deinit();
            return Configuration{
                .AbsoluteErrorBound = parsed.value,
            };
        },
        // Methods that require an aggregate error type and bound.
        .BottomUp, .SlidingWindow => {
            // Attempt to parse the configuration as AggregateError.
            // If parsing fails, return InvalidConfiguration with the expected fields.
            const parsed = json.parseFromSlice(
                AggregateError,
                allocator,
                configuration,
                .{},
            ) catch {
                return Configuration{
                    .InvalidConfiguration = .{
                        .expected_configuration = "aggregate_error_type and aggregate_error_bound",
                    },
                };
            };
            defer parsed.deinit();
            return Configuration{
                .AggregateError = parsed.value,
            };
        },
        // Methods that require an area under the curve error bound.
        .VisvalingamWhyatt => {
            // Attempt to parse the configuration as AreaUnderCurveError.
            // If parsing fails, return InvalidConfiguration with the expected field.
            const parsed = json.parseFromSlice(
                AreaUnderCurveError,
                allocator,
                configuration,
                .{},
            ) catch {
                return Configuration{
                    .InvalidConfiguration = .{
                        .expected_configuration = "area_under_curve_error",
                    },
                };
            };
            defer parsed.deinit();
            return Configuration{
                .AreaUnderCurveError = parsed.value,
            };
        },
        // Histogram-based methods require the number of bins.
        .PiecewiseLinearHistogram, .PiecewiseConstantHistogram => {
            // Attempt to parse the configuration as HistogramBinsNumber.
            // If parsing fails, return InvalidConfiguration with the expected field.
            const parsed = json.parseFromSlice(
                HistogramBinsNumber,
                allocator,
                configuration,
                .{},
            ) catch {
                return Configuration{
                    .InvalidConfiguration = .{
                        .expected_configuration = "histogram_bins_number",
                    },
                };
            };
            defer parsed.deinit();
            return Configuration{
                .HistogramBinsNumber = parsed.value,
            };
        },
        // Methods that do not require any configuration.
        .RunLengthEncoding => {
            // No configuration needed; return EmptyConfiguration.
            return Configuration{
                .EmptyConfiguration = .{},
            };
        },
    }
}

test "check configuration parsing for a few method" {
    const allocator = std.testing.allocator;

    const Case = struct {
        method: tersets.Method,
        json: []const u8,
        expected_tag: std.meta.Tag(Configuration),
    };

    // Valid cases.
    const valid_cases = [_]Case{
        .{
            .method = .PoorMansCompressionMean,
            .json = "{ \"abs_error_bound\": 0.1 }",
            .expected_tag = .AbsoluteErrorBound,
        },
        .{
            .method = .SwingFilter,
            .json = "{ \"abs_error_bound\": 10.0 }",
            .expected_tag = .AbsoluteErrorBound,
        },
        .{
            .method = .BottomUp,
            .json = "{ \"aggregate_error_type\": \"rmse\", \"aggregate_error_bound\": 5.0 }",
            .expected_tag = .AggregateError,
        },
        .{
            .method = .VisvalingamWhyatt,
            .json = "{ \"area_under_curve_error\": 0.01 }",
            .expected_tag = .AreaUnderCurveError,
        },
        .{
            .method = .PiecewiseConstantHistogram,
            .json = "{ \"histogram_bins_number\": 32 }",
            .expected_tag = .HistogramBinsNumber,
        },
        .{
            .method = .RunLengthEncoding,
            .json = "{}", // ignored
            .expected_tag = .EmptyConfiguration,
        },
    };

    // Invalid cases.
    const invalid_cases = [_]Case{
        .{
            .method = .PoorMansCompressionMean,
            .json = "{ \"rel_error_bound\": 0.1 }",
            .expected_tag = .InvalidConfiguration,
        },
        .{
            .method = .BottomUp,
            .json = "{ \"aggregate_error_type\": \"rmse\" }",
            .expected_tag = .InvalidConfiguration,
        },
        .{
            .method = .VisvalingamWhyatt,
            .json = "{ \"the_area_in_curve\": 123 }",
            .expected_tag = .InvalidConfiguration,
        },
        .{
            .method = .PiecewiseConstantHistogram,
            .json = "{ \"wrong_parameter\": 12 }",
            .expected_tag = .InvalidConfiguration,
        },
    };

    // Check the valid cases.
    for (valid_cases) |tc| {
        const cfg = parse(allocator, tc.method, tc.json);
        try std.testing.expectEqual(tc.expected_tag, @as(std.meta.Tag(Configuration), cfg));
    }

    // Check the invalid cases.
    for (invalid_cases) |tc| {
        const cfg = parse(allocator, tc.method, tc.json);
        try std.testing.expectEqual(tc.expected_tag, @as(std.meta.Tag(Configuration), cfg));
    }
}
