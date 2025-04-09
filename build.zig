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

const std = @import("std");

pub fn build(b: *std.Build) void {

    // Configuration.
    const path = b.path("src/capi.zig");
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Task for compilation.
    const lib = b.addSharedLibrary(.{
        .name = "tersets",
        .version = .{ .major = 0, .minor = 0, .patch = 1 },
        .root_source_file = path,
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(lib);

    // Task for running tests.
    const tests = b.addTest(.{
        .root_source_file = path,
        .optimize = optimize,
    });
    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run library tests");
    test_step.dependOn(&run_tests.step);
}
