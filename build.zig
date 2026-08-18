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
const Build = std.Build;
const Lang = std.Lang;
const LinkMode = std.builtin.LinkMode;

pub fn build(b: *Build) void {
    const target = b.standardTargetOptions(.{});

    // Define build options.
    const linking = b.option(
        LinkMode,
        "linking",
        "Build a static or dynamic (default) library",
    ) orelse LinkMode.dynamic;

    const pic = b.option(
        bool,
        "pic",
        "Use Position Independent Code (PIC)",
    ) orelse null;

    const optimize = b.standardOptimizeOption(.{});

    // Create root module.
    const root_module = create_module_using_tersets("src/capi.zig", target, optimize, pic);

    // Task for compilation.
    const library = b.addLibrary(.{
        .name = "tersets",
        .root_module = root_module,
        .linkage = linking,
        .version = .{ .major = 0, .minor = 0, .patch = 1 },
    });

    if (linking == LinkMode.static) {
        library.bundle_compiler_rt = true;
    }

    b.installArtifact(library);

    // Task for running unit tests.
    const unit_tests = b.addTest(.{
        .root_module = root_module,
    });
    const run_unit_tests = b.addRunArtifact(unit_tests);

    const unit_test_step = b.step("test", "Run unit tests");
    unit_test_step.dependOn(&run_unit_tests.step);

    // Task for creating and running integration tests.
    const integration_test_builder = b.addExecutable(.{
        .name = "integration_test_builder",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/codegen.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const build_integration_tests = b.addRunArtifact(integration_test_builder);

    const integration_test_module = create_module_using_tersets("src/integration_tests.zig", target, optimize, null);

    const integration_tests = b.addTest(.{ .root_module = integration_test_module });
    const run_integration_tests = b.addRunArtifact(integration_tests);

    const test_step = b.step("integration_test", "Run integration tests");
    test_step.dependOn(&build_integration_tests.step);
    test_step.dependOn(&run_integration_tests.step);
}

/// Create a module with the provided arguments and the additional configuration required to compile TerseTS.
fn create_module_using_tersets(
    b: *Build,
    root_source_file: []u8,
    target: Build.ResolvedTarget,
    optimize: Lang.OptimizeMode,
    pic: Build.Pic,
) Build.Module {
    // Paths to external libraries. Include the PocketFFT source file directly in the build,
    // as it's a single C file with no dependencies.
    const pocketfft_path = b.path("lib/pocketfft");
    const pocketfft_c_path = b.path("lib/pocketfft/pocketfft.c");

    const integration_test_module = b.createModule(.{
        .root_source_file = b.path(root_source_file),
        .target = target,
        .optimize = optimize,
        .pic = pic,
    });

    integration_test_module.addIncludePath(pocketfft_path);
    integration_test_module.addCSourceFile(.{ .file = pocketfft_c_path });
    integration_test_module.link_libc = true;
}
