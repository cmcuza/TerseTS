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

const pmc_mr = @import("functional/poor_mans_compression_midrange.zig");

/// The errors that can occur in TerseTS.
pub const Error = error{
    EmptyInput,
    IncorrectInput,
    NegativeErrorBound,
    OutOfMemory,
};

/// The compression methods in TerseTS.
pub const Method = enum {
    PoorMansCompressionMidrange,
};

/// Compress `uncompressed_values` within `error_bound` using `method` and write the result to
/// `compressed_values`. If an error occurs it is returned.
pub fn compress(
    uncompressed_values: []const f64,
    compressed_values: *ArrayList(u8),
    method: Method,
    error_bound: f32,
) Error!void {
    if (uncompressed_values.len == 0) return Error.EmptyInput;
    if (error_bound < 0) return Error.NegativeErrorBound;

    switch (method) {
        .PoorMansCompressionMidrange => {
            try pmc_mr.compress(uncompressed_values, compressed_values, error_bound);
        },
    }
}

/// Decompress `compressed_values` using `method` and write the result to `decompressed_values`. If
/// an error occurs it is returned.
pub fn decompress(
    compressed_values: []const u8,
    decompressed_values: *ArrayList(f64),
    method: Method,
) Error!void {
    if (compressed_values.len == 0) return Error.EmptyInput;

    switch (method) {
        .PoorMansCompressionMidrange => {
            try pmc_mr.decompress(compressed_values, decompressed_values);
        },
    }
}
