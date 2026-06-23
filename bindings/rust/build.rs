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
    let output = Command::new("zig")
        .current_dir(repository_root)
        .args(["build", "-Dlinking=static", "-Dpic=true", optimize])
        .output()
        .unwrap();

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

    // Zig's archiver emits Mach-O archive members that are not 8-byte aligned,
    // which Apple's linker rejects ("64-bit mach-o member ... not 8-byte aligned
    // in libtersets.a"). Re-pack the archive with cctools' `libtool`, which is
    // bundled with the Xcode command-line tools and writes correctly aligned
    // members. This runs automatically during `cargo build`, so macOS users need
    // no extra steps. See https://github.com/ziglang/zig/issues/1981.
    #[cfg(target_os = "macos")]
    {
        let lib = library_path.join("libtersets.a");
        let fixed = library_path.join("libtersets.fixed.a");
        let status = Command::new("libtool")
            .args(["-static", "-o"])
            .arg(&fixed)
            .arg(&lib)
            .status()
            .unwrap();
        if !status.success() {
            println!("cargo::error=Failed to re-pack libtersets.a with libtool to fix Mach-O member alignment.");
            process::exit(1);
        }
        std::fs::rename(&fixed, &lib).unwrap();
    }

    println!("cargo::rustc-link-lib=static=tersets");
    println!("cargo::rustc-link-search=native={}", library_path.display());
}
