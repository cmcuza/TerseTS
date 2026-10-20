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
const LinkMode = std.builtin.LinkMode;
const Module = std.Build.Module;
const OptimizeMode = std.builtin.OptimizeMode;
const ResolvedTarget = Build.ResolvedTarget;

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
    const root_module = create_module_using_tersets(b,
        "src/capi.zig",
        target,
        optimize,
        pic,
    );

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

    if (linking == LinkMode.static) {
        library.bundle_compiler_rt = true;
    }

    b.installArtifact(library);

    // Task for building and running tests.
    const unit_tests = b.addTest(.{ .root_module = root_module });
    const run_unit_tests = b.addRunArtifact(unit_tests);

    const integration_test_builder = b.addExecutable(.{
        .name = "integration_test_builder",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/codegen.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const build_integration_tests = b.addRunArtifact(integration_test_builder);
    const integration_test_paths_path = "src/integration_tests.zig";
    const format_integration_tests = b.addFmt(.{
        .paths = &.{b.path(integration_test_paths_path )},
    });
    format_integration_tests.step.dependOn(&build_integration_tests.step);

    const integration_test_module = create_module_using_tersets(
        b,
        integration_test_paths_path,
        target,
        optimize,
        null,
    );
    const integration_tests = b.addTest(.{ .root_module = integration_test_module });
    const run_integration_tests = b.addRunArtifact(integration_tests);

    const test_step = b.step("test", "Run unit and integration tests");
    test_step.dependOn(&run_unit_tests.step);
    test_step.dependOn(&format_integration_tests.step);
    test_step.dependOn(&run_integration_tests.step);
}

/// Create a module with the provided arguments and the additional configuration required to compile TerseTS.
fn create_module_using_tersets(
    b: *Build,
    root_source_file: []const u8,
    target: ResolvedTarget,
    optimize: OptimizeMode,
    pic: ?bool,
) *Module {
    const module = b.createModule(.{
        .root_source_file = b.path(root_source_file),
        .target = target,
        .optimize = optimize,
        .pic = pic,
    });

    // Add PocketFFT directly to root_module, as it's a single C file with no dependencies.
    module.addIncludePath(b.path("lib/pocketfft"));
    module.addCSourceFile(.{ .file = b.path("lib/pocketfft/pocketfft.c") });
    module.link_libc = true;
    return module;
}
