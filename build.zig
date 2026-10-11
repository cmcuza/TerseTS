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
    const target = b.standardTargetOptions(.{});

    // Define build options.
    const linking = b.option(
        std.builtin.LinkMode,
        "linking",
        "Build a static or dynamic (default) library",
    ) orelse .dynamic;

    const pic = b.option(
        bool,
        "pic",
        "Use Position Independent Code (PIC)",
    );

    const optimize = b.standardOptimizeOption(.{});

    // Create root module.
    const root_module = b.createModule(.{
        .root_source_file = b.path("src/capi.zig"),
        .target = target,
        .optimize = optimize,
        .pic = pic,
        .link_libc = true,
    });

    // Add PocketFFT directly to root_module, as it's a single C file with no dependencies.
    root_module.addIncludePath(b.path("lib/pocketfft"));
    root_module.addCSourceFile(.{
        .file = b.path("lib/pocketfft/pocketfft.c"),
    });

    // Task for translating pocketfft.h to Zig.
    // The official ZSF translate-c package is not used to avoid dependencies.
    const translate_pocketfft = b.addTranslateC(.{
        .root_source_file = b.path("lib/pocketfft/pocketfft.h"),
        .target = target,
        .optimize = optimize,
    });

    const translate_pocketfft_step = b.addUpdateSourceFiles();
    translate_pocketfft_step.addCopyFileToSource(translate_pocketfft.getOutput(), "src/pocketfft.zig");

    // Task for compilation.
    const library = b.addLibrary(.{
        .name = "tersets",
        .root_module = root_module,
        .linkage = linking,
        .version = .{ .major = 0, .minor = 0, .patch = 1 },
    });
    library.step.dependOn(&translate_pocketfft_step.step);

    if (linking == .static) {
        library.bundle_compiler_rt = true;
    }

    b.installArtifact(library);

    // Task for running tests.
    const tests = b.addTest(.{
        .root_module = root_module,
    });

    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run library tests");
    test_step.dependOn(&run_tests.step);
}
