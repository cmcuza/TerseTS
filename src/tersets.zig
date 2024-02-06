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

//! Provides a C-API for TerseTS.

const std = @import("std");

/// A pointer to the uncompressed values and the number of values.
pub const InputData = extern struct { values: [*]const f64, len: usize };

/// A pointer to the compressed values and the number of bytes.
pub const OutputData = extern struct { values: [*]const u8, len: usize };

/// Compress `input` to `output` using the specified method.
export fn compress(input_data: *const InputData, output_data: *OutputData) i32 {
    std.debug.print("{}\n", .{input_data.len});
    var i: usize = 0;
    while (i < input_data.len) : (i += 1) {
        std.debug.print("{} ", .{input_data.values[i]});
    }
    std.debug.print("\n", .{});

    output_data.len = input_data.len;

    return 0;
}

test "compress" {
    const input_values = [_]f64{ 0, 1, 2, 3, 4 };
    const input_slice = input_values[0..5];
    const input_data = InputData{
        .values = input_slice.ptr,
        .len = input_slice.len,
    };

    const output_values = [_]u8{};
    const output_slice = output_values[0..0];
    var output_data = OutputData{
        .values = output_slice.ptr,
        .len = output_slice.len,
    };

    try std.testing.expect(compress(&input_data, &output_data) == 0);
}
