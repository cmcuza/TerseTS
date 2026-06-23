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

    // On macOS/arm64 the static archive Zig produces needs two fix-ups before an
    // external (Rust-driven) linker can consume it reliably:
    //
    //   1. Alignment. Zig's archiver writes Mach-O members that are not 8-byte
    //      aligned, which Apple's linker rejects ("64-bit mach-o member ... not
    //      8-byte aligned in libtersets.a"). `libtool -static` re-packs the archive
    //      with correctly aligned members. See ziglang/zig#1981.
    //
    //   2. compiler_rt. The library uses `f80` (and `std.json` pulls in `f128`)
    //      software-float routines (`__addxf3`, `__mulxf3`, `__divxf3`, `roundq`,
    //      ...). On arm64 these have no hardware support and must come from Zig's
    //      compiler_rt, bundled into the archive as `compiler_rt.o` via
    //      `bundle_compiler_rt = true` in build.zig. On x86_64 `f80` is a native
    //      x87 type, so these routines were never needed there. We require that
    //      member to be present (failing loudly if not, so a missing runtime is a
    //      hard error instead of an intermittent "undefined symbols" link failure),
    //      and regenerate the archive symbol table with `ranlib` so the linker
    //      deterministically resolves those routines on demand. See ziglang/zig#15648.
    //
    // All of this lives in the archive, so the normal `static=tersets` link below
    // keeps working for downstream crates with no extra steps. `libtool`, `ar`, and
    // `ranlib` ship with the Xcode command-line tools.
    #[cfg(target_os = "macos")]
    {
        let lib = library_path.join("libtersets.a");

        // 1. Re-pack with libtool to fix 8-byte member alignment.
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

        // 2. Require compiler_rt.o so the f80/f128 soft-float routines are present.
        let listing = Command::new("ar").arg("t").arg(&lib).output().unwrap();
        if !String::from_utf8_lossy(&listing.stdout)
            .lines()
            .any(|member| member.contains("compiler_rt"))
        {
            println!(
                "cargo::error=compiler_rt.o is missing from libtersets.a; f80/f128 \
                 soft-float routines (e.g. __addxf3, roundq) would be undefined at \
                 link time. Keep `bundle_compiler_rt = true` for static linking in build.zig."
            );
            process::exit(1);
        }

        // 3. Rebuild the archive symbol table so the linker reliably pulls
        //compiler_rt.o on demand (the libtool re-pack can otherwise leave an
        //index that resolves only intermittently).
        let status = Command::new("ranlib").arg(&lib).status().unwrap();
        if !status.success() {
            println!("cargo::error=Failed to regenerate libtersets.a symbol table with ranlib.");
            process::exit(1);
        }
    }

    println!("cargo::rustc-link-lib=static=tersets");
    println!("cargo::rustc-link-search=native={}", library_path.display());
}
