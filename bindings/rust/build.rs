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

use std::env;
use std::process::{self, Command};

/// Compile TerseTS into a statically linked library and link it. unwrap() is deliberately used, as
/// recovery is not possible if any of the unwrapped operations fail, so better to fail immediately.
fn main() {
    // Compute the repositories root for running zig build and linking.
    let current_directory = env::current_dir().unwrap();
    let repository_root = current_directory.parent().unwrap().parent().unwrap();

    // Cargo's target variables describe the artifact being built, while Rust `cfg`
    // values in a build script describe the host running the build.
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();

    // Make the optimization level of TerseTS and Rust bindings match.
    let build_profile = env::var("PROFILE").unwrap();
    let optimize = match build_profile.as_str() {
        // On Windows, both `Debug` and `ReleaseSafe` pull in Zig's stack-trace
        // panic handler (`std.debug.SelfInfo.Windows.findModule`), which calls
        // `LdrRegisterDllNotification`. That API exists in ntdll.dll but is absent
        // from the stripped ntdll.lib bundled in Rust's MSVC sysroot, causing a
        // link failure. `ReleaseFast` avoids this by omitting panic stack traces.
        // On other platforms, `-Doptimize=Debug` links correctly.
        "debug" if cfg!(windows) => "-Doptimize=ReleaseFast",
        "debug" => "-Doptimize=Debug",
        "release" => "-Doptimize=ReleaseFast",
        build_profile => {
            println!(
                "cargo::error=Profile must be debug (dev) or release, not {}.",
                build_profile
            );
            process::exit(1);
        }
    };

    // Build the TerseTS library into a statically linked library.
    let mut zig = Command::new("zig");
    zig.current_dir(repository_root)
        .args(["build", "-Dlinking=static", "-Dpic=true", optimize]);

    if target_os == "macos" && target_arch == "aarch64" {
        let deployment_target =
            env::var("MACOSX_DEPLOYMENT_TARGET").unwrap_or_else(|_| "11.0".to_owned());
        println!("cargo::rerun-if-env-changed=MACOSX_DEPLOYMENT_TARGET");
        zig.arg(format!("-Dtarget=aarch64-macos.{deployment_target}"));
    }

    let output = zig.output().unwrap();

    if !output.status.success() {
        // Output is captured by cargo and used as commands.
        println!(
            "cargo::error=Failed to build TerseTS as a static library due to {}.",
            String::from_utf8(output.stderr).unwrap()
        );
        process::exit(1);
    }

    // Specify the name and location of the TerseTS library for the linker.
    let mut library_path = repository_root.to_path_buf();
    library_path.push("zig-out");
    library_path.push("lib");

    // Re-packing the archive with Apple's `libtool` adds the required padding without modifying
    // the object contents.
    if target_os == "macos" && target_arch == "aarch64" {
        let lib = library_path.join("libtersets.a");
        let fixed = library_path.join("libtersets.fixed.a");
        let output = Command::new("libtool")
            .args(["-static", "-o"])
            .arg(&fixed)
            .arg(&lib)
            .output()
            .unwrap();

        if !output.status.success() {
            println!(
                "cargo::error=Failed to re-pack libtersets.a with libtool: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            process::exit(1);
        }

        // `libtool` preserves compiler_rt.o but does not index its soft-float symbols in this
        // archive. Rebuild the table of contents so ld can resolve symbols such as ___addxf3.
        let ranlib_output = Command::new("ranlib").arg(&fixed).output().unwrap();
        if !ranlib_output.status.success() {
            println!(
                "cargo::error=Failed to index libtersets.a with ranlib: {}",
                String::from_utf8_lossy(&ranlib_output.stderr)
            );
            process::exit(1);
        }

        std::fs::rename(&fixed, &lib).unwrap();
    }

    // On aarch64-macos, force every member of the archive into the link. ld's default
    // selective, index-driven extraction does not force-load `compiler_rt.o` for math
    // routines such as `_roundq` (referenced by `std.json`'s f128 integer validation):
    // Rust's own `compiler_builtins` satisfies the standard soft-float builtins, so ld
    // never pulls our `compiler_rt.o`, leaving `_roundq` (which is not a compiler-rt
    // builtin and has no Rust substitute) undefined. Whole-archive linking includes every
    // member unconditionally so all bundled soft-float and math symbols resolve. The
    // `libtool` re-pack above is still required so ld can parse the member at all.
    if target_os == "macos" && target_arch == "aarch64" {
        println!("cargo::rustc-link-lib=static:+whole-archive=tersets");
    } else {
        println!("cargo::rustc-link-lib=static=tersets");
    }
    println!("cargo::rustc-link-search=native={}", library_path.display());
}
