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
        else => return error.InvalidConfiguration,
    }
    return parsed_value;
}

pub fn checkAbsErrorBoundConfiguration() !void {
    const allocator = std.testing.allocator;
    const valid_configuration = try parse(
        allocator,
        AbsoluteErrorBound,
        \\ { "abs_error_bound": 0.1 }
        ,
    );
    try testing.expectEqual(@TypeOf(valid_configuration), AbsoluteErrorBound);
    try testing.expectEqual(valid_configuration.abs_error_bound, 0.1);

    _ = parse(
        allocator,
        AbsoluteErrorBound,
        \\ { "ab_eror_bund": 0.1 }
        ,
    ) catch |err| {
        try testing.expectEqual(error.InvalidConfiguration, err);
        return;
    };

    try testing.expectFmt(
        "",
        "The configuration is supposed to be invalid",
        .{},
    );

    _ = parse(
        allocator,
        AbsoluteErrorBound,
        \\ { "abs_error_bound": 0.1.2 }
        ,
    ) catch |err| {
        try testing.expectEqual(error.InvalidConfiguration, err);
        return;
    };

    try testing.expectFmt(
        "",
        "The configuration is supposed to be invalid",
        .{},
    );
}

pub fn checkRelErrorBoundConfiguration() !void {
    const allocator = std.testing.allocator;
    const valid_configuration = try parse(
        allocator,
        RelativeErrorBound,
        \\ { "rel_error_bound": 0.1 }
        ,
    );
    try testing.expectEqual(@TypeOf(valid_configuration), RelativeErrorBound);
    try testing.expectEqual(valid_configuration.rel_error_bound, 0.1);

    _ = parse(
        allocator,
        RelativeErrorBound,
        \\ { "are_eror_bund": 0.1 }
        ,
    ) catch |err| {
        try testing.expectEqual(error.InvalidConfiguration, err);
        return;
    };

    try testing.expectFmt(
        "",
        "The configuration is supposed to be invalid",
        .{},
    );

    _ = parse(
        allocator,
        RelativeErrorBound,
        \\ { "rel_error_bound": 0.1.2 }
        ,
    ) catch |err| {
        try testing.expectEqual(error.InvalidConfiguration, err);
        return;
    };

    try testing.expectFmt(
        "",
        "The configuration is supposed to be invalid",
        .{},
    );
}

pub fn checkAUCErrorConfiguration() !void {
    const allocator = std.testing.allocator;
    const valid_configuration = try parse(
        allocator,
        AreaUnderCurveError,
        \\ { "area_under_curve_error": 0.1 }
        ,
    );
    try testing.expectEqual(@TypeOf(valid_configuration), AreaUnderCurveError);
    try testing.expectEqual(valid_configuration.area_under_curve_error, 0.1);

    _ = parse(
        allocator,
        AreaUnderCurveError,
        \\ { "auc_error": 0.1 }
        ,
    ) catch |err| {
        try testing.expectEqual(error.InvalidConfiguration, err);
        return;
    };

    try testing.expectFmt(
        "",
        "The configuration is supposed to be invalid",
        .{},
    );

    _ = parse(
        allocator,
        AreaUnderCurveError,
        \\ { "area_under_curve_error": 0.1.2 }
        ,
    ) catch |err| {
        try testing.expectEqual(error.InvalidConfiguration, err);
        return;
    };

    try testing.expectFmt(
        "",
        "The configuration is supposed to be invalid",
        .{},
    );
}

pub fn checkHistogramErrorConfiguration() !void {
    const allocator = std.testing.allocator;
    const valid_configuration = try parse(
        allocator,
        HistogramBinsNumber,
        \\ { "histogram_bins_number": 6 }
        ,
    );
    try testing.expectEqual(@TypeOf(valid_configuration), HistogramBinsNumber);
    try testing.expectEqual(valid_configuration.histogram_bins_number, 6);

    _ = parse(
        allocator,
        HistogramBinsNumber,
        \\ { "histogram_bins_numr": 6 }
        ,
    ) catch |err| {
        try testing.expectEqual(error.InvalidConfiguration, err);
        return;
    };

    try testing.expectFmt(
        "",
        "The configuration is supposed to be invalid",
        .{},
    );

    _ = parse(
        allocator,
        HistogramBinsNumber,
        \\ { "histogram_bins_numr": 6.1 }
        ,
    ) catch |err| {
        try testing.expectEqual(error.InvalidConfiguration, err);
        return;
    };

    try testing.expectFmt(
        "",
        "The configuration is supposed to be invalid",
        .{},
    );
}

pub fn checkAggregatedErrorConfiguration() !void {
    const allocator = std.testing.allocator;
    const valid_configuration = try parse(
        allocator,
        AggregateError,
        \\ { "aggregate_error_type": "rmse", "aggregate_error_bound": 5.2 }
        ,
    );
    try testing.expectEqual(@TypeOf(valid_configuration), AggregateError);
    try testing.expectEqual(valid_configuration.aggregate_error_bound, 5.2);
    try testing.expect(std.mem.eql(u8, valid_configuration.aggregate_error_type, "rmse"));

    _ = parse(
        allocator,
        AggregateError,
        \\ { "aggregate_errr_type": "rmse", "aggregate_eror_bound": 5.2 }
        ,
    ) catch |err| {
        try testing.expectEqual(error.InvalidConfiguration, err);
        return;
    };

    try testing.expectFmt(
        "",
        "The configuration is supposed to be invalid",
        .{},
    );

    _ = parse(
        allocator,
        AggregateError,
        \\ { "aggregate_error_type": "rmse", "aggregate_error_bound": 5.2.2 }
        ,
    ) catch |err| {
        try testing.expectEqual(error.InvalidConfiguration, err);
        return;
    };

    try testing.expectFmt(
        "",
        "The configuration is supposed to be invalid",
        .{},
    );

    _ = parse(
        allocator,
        AggregateError,
        \\ { "aggregate_error_type": "343", "aggregate_error_bound": 2.2 }
        ,
    ) catch |err| {
        try testing.expectEqual(error.InvalidConfiguration, err);
        return;
    };

    try testing.expectFmt(
        "",
        "The configuration is supposed to be invalid",
        .{},
    );
}

pub fn checkEmptyConfiguration() !void {
    const allocator = std.testing.allocator;
    const valid_configuration = try parse(
        allocator,
        EmptyConfiguration,
        \\ {}
        ,
    );
    try testing.expectEqual(@TypeOf(valid_configuration), EmptyConfiguration);

    _ = parse(
        allocator,
        EmptyConfiguration,
        \\ { "abs_error_bound": 5.2 }
        ,
    ) catch |err| {
        try testing.expectEqual(error.InvalidConfiguration, err);
        return;
    };

    try testing.expectFmt(
        "",
        "The configuration is supposed to be invalid",
        .{},
    );
}
