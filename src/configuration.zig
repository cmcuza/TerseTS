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
const testing = std.testing;
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

/// Configuration for domain transformation methods like DFT and DWT.
/// The number of coefficients refers to the number of coefficients to retain.
/// Example: { "number_of_coefficients": 10 }
pub const DomainTransformation = struct {
    number_of_coefficients: u32,
};

/// Empty configuration for methods that do not require any parameters.
pub const EmptyConfiguration = struct {};

/// This is a small convenience wrapper around json.parseFromSlice that accepts a
/// `ConfigurationType` and parses the the JSON text `configuration`. The function
/// returns the parsed value on success, or `null` on failure. The `allocator` is
/// used by the JSON parser. The function also checks that all numerical error bounds
/// or number of histogram bins are positive numbers.
pub fn parse(
    allocator: Allocator,
    comptime ConfigurationType: type,
    configuration: []const u8,
) !ConfigurationType {
    const parsed = json.parseFromSlice(
        ConfigurationType,
        allocator,
        configuration,
        .{},
    ) catch return error.InvalidConfiguration;
    defer parsed.deinit();
    const parsed_value = parsed.value;
    switch (@TypeOf(parsed_value)) {
        AbsoluteErrorBound => {
            if (parsed_value.abs_error_bound < 0)
                return error.InvalidConfiguration;
        },
        RelativeErrorBound => {
            if (parsed_value.rel_error_bound < 0)
                return error.InvalidConfiguration;
        },
        AreaUnderCurveError => {
            if (parsed_value.area_under_curve_error < 0)
                return error.InvalidConfiguration;
        },
        HistogramBinsNumber => {
            if (parsed_value.histogram_bins_number < 0)
                return error.InvalidConfiguration;
        },
        AggregateError => {
            if (parsed_value.aggregate_error_bound < 0)
                return error.InvalidConfiguration;
        },
        EmptyConfiguration => {},
        DomainTransformation => {
            if (parsed_value.number_of_coefficients < 0)
                return error.InvalidConfiguration;
        },
        else => return error.InvalidConfiguration,
    }
    return parsed_value;
}

test "parse valid AbsoluteErrorBound" {
    const allocator = std.testing.allocator;
    const config = try parse(
        allocator,
        AbsoluteErrorBound,
        \\{ "abs_error_bound": 0.1 }
        ,
    );
    try testing.expectEqual(@TypeOf(config), AbsoluteErrorBound);
    try testing.expectEqual(config.abs_error_bound, 0.1);
}

test "parse AbsoluteErrorBound with wrong field name" {
    const allocator = std.testing.allocator;
    _ = parse(
        allocator,
        AbsoluteErrorBound,
        \\{ "ab_eror_bund": 0.1 }
        ,
    ) catch |err| {
        try testing.expectEqual(error.InvalidConfiguration, err);
        return;
    };
    unreachable;
}

test "parse AbsoluteErrorBound with invalid float" {
    const allocator = std.testing.allocator;
    _ = parse(
        allocator,
        AbsoluteErrorBound,
        \\{ "abs_error_bound": 0.1.2 }
        ,
    ) catch |err| {
        try testing.expectEqual(error.InvalidConfiguration, err);
        return;
    };
    unreachable;
}

test "parse AbsoluteErrorBound with negative value" {
    const allocator = std.testing.allocator;
    _ = parse(
        allocator,
        AbsoluteErrorBound,
        \\{ "abs_error_bound": -0.1 }
        ,
    ) catch |err| {
        try testing.expectEqual(error.InvalidConfiguration, err);
        return;
    };
    unreachable;
}

test "parse valid RelativeErrorBound" {
    const allocator = std.testing.allocator;
    const config = try parse(
        allocator,
        RelativeErrorBound,
        \\{ "rel_error_bound": 0.1 }
        ,
    );
    try testing.expectEqual(@TypeOf(config), RelativeErrorBound);
    try testing.expectEqual(config.rel_error_bound, 0.1);
}

test "parse RelativeErrorBound with wrong field name" {
    const allocator = std.testing.allocator;
    _ = parse(
        allocator,
        RelativeErrorBound,
        \\{ "are_eror_bund": 0.1 }
        ,
    ) catch |err| {
        try testing.expectEqual(error.InvalidConfiguration, err);
        return;
    };
    unreachable;
}

test "parse RelativeErrorBound with invalid float" {
    const allocator = std.testing.allocator;
    _ = parse(
        allocator,
        RelativeErrorBound,
        \\{ "rel_error_bound": 0.1.2 }
        ,
    ) catch |err| {
        try testing.expectEqual(error.InvalidConfiguration, err);
        return;
    };
    unreachable;
}

test "parse RelativeErrorBound with negative value" {
    const allocator = std.testing.allocator;
    _ = parse(
        allocator,
        RelativeErrorBound,
        \\{ "rel_error_bund": -0.1 }
        ,
    ) catch |err| {
        try testing.expectEqual(error.InvalidConfiguration, err);
        return;
    };
    unreachable;
}

test "parse valid AreaUnderCurveError" {
    const allocator = std.testing.allocator;
    const config = try parse(
        allocator,
        AreaUnderCurveError,
        \\{ "area_under_curve_error": 0.1 }
        ,
    );
    try testing.expectEqual(@TypeOf(config), AreaUnderCurveError);
    try testing.expectEqual(config.area_under_curve_error, 0.1);
}

test "parse AreaUnderCurveError with wrong field name" {
    const allocator = std.testing.allocator;
    _ = parse(
        allocator,
        AreaUnderCurveError,
        \\{ "auc_error": 0.1 }
        ,
    ) catch |err| {
        try testing.expectEqual(error.InvalidConfiguration, err);
        return;
    };
    unreachable;
}

test "parse AreaUnderCurveError with invalid float" {
    const allocator = std.testing.allocator;
    _ = parse(
        allocator,
        AreaUnderCurveError,
        \\{ "area_under_curve_error": 0.1.2 }
        ,
    ) catch |err| {
        try testing.expectEqual(error.InvalidConfiguration, err);
        return;
    };
    unreachable;
}

test "parse AreaUnderCurveError with negative value" {
    const allocator = std.testing.allocator;
    _ = parse(
        allocator,
        AreaUnderCurveError,
        \\{ "area_under_curve_error": -0.1 }
        ,
    ) catch |err| {
        try testing.expectEqual(error.InvalidConfiguration, err);
        return;
    };
    unreachable;
}

test "parse valid HistogramBinsNumber" {
    const allocator = std.testing.allocator;
    const config = try parse(
        allocator,
        HistogramBinsNumber,
        \\{ "histogram_bins_number": 6 }
        ,
    );
    try testing.expectEqual(@TypeOf(config), HistogramBinsNumber);
    try testing.expectEqual(config.histogram_bins_number, 6);
}

test "parse HistogramBinsNumber with wrong field name" {
    const allocator = std.testing.allocator;
    _ = parse(
        allocator,
        HistogramBinsNumber,
        \\{ "histogram_bins_numr": 6 }
        ,
    ) catch |err| {
        try testing.expectEqual(error.InvalidConfiguration, err);
        return;
    };
    unreachable;
}

test "parse HistogramBinsNumber with float value" {
    const allocator = std.testing.allocator;
    _ = parse(
        allocator,
        HistogramBinsNumber,
        \\{ "histogram_bins_numr": 6.1 }
        ,
    ) catch |err| {
        try testing.expectEqual(error.InvalidConfiguration, err);
        return;
    };
    unreachable;
}

test "parse HistogramBinsNumber with negative value" {
    const allocator = std.testing.allocator;
    _ = parse(
        allocator,
        HistogramBinsNumber,
        \\{ "histogram_bins_numr": -6 }
        ,
    ) catch |err| {
        try testing.expectEqual(error.InvalidConfiguration, err);
        return;
    };
    unreachable;
}

test "parse valid AggregateError" {
    const allocator = std.testing.allocator;
    const config = try parse(
        allocator,
        AggregateError,
        \\{ "aggregate_error_type": "rmse", "aggregate_error_bound": 5.2 }
        ,
    );
    try testing.expectEqual(@TypeOf(config), AggregateError);
    try testing.expectEqual(config.aggregate_error_bound, 5.2);
    try testing.expect(std.mem.eql(u8, config.aggregate_error_type, "rmse"));
}

test "parse AggregateError with wrong field names" {
    const allocator = std.testing.allocator;
    _ = parse(
        allocator,
        AggregateError,
        \\{ "aggregate_errr_type": "rmse", "aggregate_eror_bound": 5.2 }
        ,
    ) catch |err| {
        try testing.expectEqual(error.InvalidConfiguration, err);
        return;
    };
    unreachable;
}

test "parse AggregateError with invalid float" {
    const allocator = std.testing.allocator;
    _ = parse(
        allocator,
        AggregateError,
        \\{ "aggregate_error_type": "rmse", "aggregate_error_bound": 5.2.2 }
        ,
    ) catch |err| {
        try testing.expectEqual(error.InvalidConfiguration, err);
        return;
    };
    unreachable;
}

test "parse AggregateError with negative value" {
    const allocator = std.testing.allocator;
    _ = parse(
        allocator,
        AggregateError,
        \\{ "aggregate_error_type": "rmse", "aggregate_error_bound": -2.2 }
        ,
    ) catch |err| {
        try testing.expectEqual(error.InvalidConfiguration, err);
        return;
    };
    unreachable;
}

test "parse valid EmptyConfiguration" {
    const allocator = std.testing.allocator;
    const config = try parse(
        allocator,
        EmptyConfiguration,
        \\{}
        ,
    );
    try testing.expectEqual(@TypeOf(config), EmptyConfiguration);
}

test "parse EmptyConfiguration with extra field" {
    const allocator = std.testing.allocator;
    _ = parse(
        allocator,
        EmptyConfiguration,
        \\{ "abs_error_bound": 5.2 }
        ,
    ) catch |err| {
        try testing.expectEqual(error.InvalidConfiguration, err);
        return;
    };
    unreachable;
}
