// Copyright 2026 TerseTS Contributors
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

//! Provides comptime functions for generating test for testing TerseTS.

const std = @import("std");
const ArrayList = std.ArrayList;
const Allocator = std.mem.Allocator;
const Random = std.Random;
const tersets = @import("tersets.zig");
const generators = @import("generators.zig");

pub fn getGenerators(allocator: Allocator) !ArrayList(fn (Allocator, *ArrayList(f64), Random) void) {
    var list = ArrayList(fn (Allocator, *ArrayList(f64), Random) void).init(allocator);

    inline for (@typeInfo(generators).Struct.decls) |decl| {
        const fn_ptr = @field(generators, decl.name);
        if (@typeInfo(fn_ptr) == .Fn) {
            try list.append(fn_ptr);
        }
    }

    return list;
}
