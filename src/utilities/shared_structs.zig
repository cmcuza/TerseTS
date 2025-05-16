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
    slope: f64,
    intercept: f64,
};

/// `Point` is a point represented by `time` and `value`. `time` is of datatype `time_type`.
fn Point(comptime time_type: type) type {
    return struct { time: time_type, value: f64 };
}
