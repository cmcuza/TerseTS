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

//! Implementation of the SERF-QT algorithm from the paper
//! "Li, Ruiyuan, Zechao Chen, Ruyun Lu, Xiaolong Xu, Guangchao Yang, Chao Chen, Jie Bao, and Yu Zheng.
//! Serf: Streaming Error-Bounded Floating-Point Compression.
//! ACM SIGMOD 2025.
//! https://doi.org/10.1145/3725353.

const std = @import("std");
const math = std.math;
const mem = std.mem;
const io = std.io;
const ArrayList = std.ArrayList;
const testing = std.testing;
const Allocator = std.mem.Allocator;

const tersets = @import("../../tersets.zig");
const shared_functions = @import("../../utilities/shared_functions.zig");
const configuration = @import("../../configuration.zig");
const tester = @import("../../tester.zig");

const Error = tersets.Error;

/// Compress `uncompressed_values` within error_bound using "Serf-QT". The function writes the
/// result to `compressed_values`. The `compressed_values` includes an `EliasGamma` encoding of
/// the quantized elements. The `allocator` is used for memory management of intermediates containers
/// and the `method_configuration` parser. The function expects an `AbsoluteErrorBound` configuration.
/// If an error occurs it is returned.
pub fn compress(
    allocator: Allocator,
    uncompressed_values: []const f64,
    compressed_values: *ArrayList(u8),
    method_configuration: []const u8,
) Error!void {
    const parsed_configuration = try configuration.parse(
        allocator,
        configuration.AbsoluteErrorBound,
        method_configuration,
    );

    const error_bound: f32 = parsed_configuration.abs_error_bound;

    // Ensure the compressed values are not empty.
    if (uncompressed_values.len == 0) return Error.UnsupportedInput;

    // Avoid edge cases by defining bucket size slightly larger than 2 * error_bound.
    const bucket_size: f32 = 1.99 * error_bound;
    // Append the minimum value to the header of the compressed values.
    try shared_functions.appendValue(f32, bucket_size, compressed_values);

    //Intermediate quantized values.
    var quantized_values = ArrayList(usize).init(allocator);
    defer quantized_values.deinit();

    // Assume the first value is zero for simplicity (Section 4.1).
    var previous_value: f64 = 0.0;
    const usize_previous_value: usize = 0;

    // Quantize each value by mapping it to a discrete bucket index.
    // If the error_bound is zero, we compute the difference between the
    // value and the minimum value, ensuring all resulting integers are >= 0.
    // For non-zero error_bound, we apply fixed-width bucket quantization
    // using the defined bucket size (1.99 × error_bound).

    var encoded_value: u64 = 0;
    for (uncompressed_values) |value| {
        if (!math.isFinite(value) or @abs(value) > tester.max_test_value) return Error.UnsupportedInput;

        if (error_bound == 0.0) {
            // Quantization for the lossless case.
            // This case is not defined in the paper, but we implement it
            // as the difference from the previous value.
            const usize_value: u64 = shared_functions.floatBitsOrdered(value);
            encoded_value = usize_value - usize_previous_value;
        } else {
            // Fixed-width bucket quantization with rounding (Equation 2).
            const quantized_value: i64 = @intFromFloat(@floor((value - previous_value) / bucket_size + 0.5));
            // Apply zigzag encoding to handle negative values (Equation 4).
            encoded_value = shared_functions.zigzagEncoder(quantized_value + 1);
        }

        try quantized_values.append(encoded_value);
        previous_value = value;
    }

    // Encode the quantized values using Elias Gamma encoding.
    try shared_functions.eliasGammaEncode(allocator, quantized_values.items, compressed_values);
}
