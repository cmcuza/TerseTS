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

fn main() {
    // Re-run when build script or Zig sources change
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../../build.zig");
    println!("cargo:rerun-if-changed=../../src");
    // Re-run on every GitHub Actions run (prevents "Fresh" binaries missing the lib)
    println!("cargo:rerun-if-env-changed=GITHUB_RUN_ID");

    // Compute the repository root (bindings/rust -> bindings -> repo root)
    let current_directory = env::current_dir().unwrap();
    let repository_root = current_directory.parent().unwrap().parent().unwrap().to_path_buf();

    // Match Rust profile to Zig optimize
    let build_profile = env::var("PROFILE").unwrap();
    let optimize = match build_profile.as_str() {
        "debug" => "-Doptimize=Debug",
        "release" => "-Doptimize=ReleaseFast",
        other => {
            println!("cargo:warning=Unexpected PROFILE={other}, defaulting to Debug");
            "-Doptimize=Debug"
        }
    };

    // Build TerseTS as a static library

    let output = Command::new("zig")
        .current_dir(repository_root.as_path())
        .args(["build", "-Dlinking=static", optimize])
        .output()
        .unwrap();

    if !output.status.success() {
        println!(
            "cargo:warning=Failed to build TerseTS: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        process::exit(1);
    }

    // Zig output directory
    let library_path = repository_root.join("zig-out").join("lib");

    // Tell rustc where to find and how to link TerseTS
    println!("cargo:rustc-link-search=native={}", library_path.display());
    println!("cargo:rustc-link-lib=static=tersets");

    // If the binary ends up dynamically linked on macOS (e.g., due to stale artifacts or other link inputs),
    // ensure dyld can find the dylib in zig-out/lib without any env vars.
    if env::var("CARGO_CFG_TARGET_OS").unwrap_or_default() == "macos" {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", library_path.display());
    }
}
