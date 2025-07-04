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

//! Contains all shared variables and structures used across TerseTS.

const std = @import("std");
const HashMap = std.HashMap;

/// Margin to adjust the error bound for numerical stability. Reducing the error bound by this
/// margin ensures that all the elements of the decompressed time series are within the error bound
/// with respect to the uncompressed time series.
pub const ErrorBoundMargin: f32 = 1e-7;

/// `Point` with discrete `time` axis.
pub const DiscretePoint = Point(usize);

/// `Point` with continous `time` axis.
pub const ContinousPoint = Point(f64);

/// `Segment` models a straight line segment from `start_point` to `end_point`. All segments
/// have discrete points.
pub const Segment = struct {
    start_point: DiscretePoint,
    end_point: DiscretePoint,
};

/// Linear function of the form y = slope*x+intercept. It uses f80 for numerical stability.
pub const LinearFunction = struct {
    slope: f80,
    intercept: f80,
};

/// `Point` is a point represented by `time` and `value`. `time` is of datatype `time_type`.
fn Point(comptime time_type: type) type {
    return struct { time: time_type, value: f64 };
}

/// `SegmentMetadata` stores the information about an approximated segment during the execution
/// of Sim-Piece and Mix-Piece. It stores the starting time of the segment in `start_time`, the
/// `interception` point used to create the linear function approximation, and the slopes of
/// the upper and lower bounds that constraint the linear approximation in that segment.
pub const SegmentMetadata = struct {
    start_time: usize,
    intercept: f64,
    upper_bound_slope: f64,
    lower_bound_slope: f64,
};

/// `HashF64Context` provides context for hashing and comparing `f64` values for use in `HashMap`.
/// This context is essential when using `f64` as keys in a `HashMap`. It defines how the keys are
/// hashed and compared for equality.
pub const HashF64Context = struct {
    /// Hashes a `f64` value by bitcasting it to `u64`.
    pub fn hash(_: HashF64Context, value: f64) u64 {
        return @as(u64, @bitCast(value));
    }
    /// Compares two `f64` values for equality.
    pub fn eql(_: HashF64Context, value_one: f64, value_two: f64) bool {
        return value_one == value_two;
    }
};

/// Returns a HashMap with key type f64 and `value_type` given by the user.
pub fn HashMapf64(comptime value_type: type) type {
    return HashMap(f64, value_type, HashF64Context, std.hash_map.default_max_load_percentage);
}
