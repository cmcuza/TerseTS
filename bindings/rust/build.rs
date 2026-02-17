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
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../../build.zig");
    println!("cargo:rerun-if-changed=../../src");
    println!("cargo:rerun-if-env-changed=GITHUB_RUN_ID");

    let current_directory = env::current_dir().unwrap();
    let repository_root = current_directory.parent().unwrap().parent().unwrap();

    let build_profile = env::var("PROFILE").unwrap();
    let optimize = match build_profile.as_str() {
        "debug" => "-Doptimize=Debug",
        "release" => "-Doptimize=ReleaseFast",
        _ => "-Doptimize=Debug",
    };

    let output = Command::new("zig")
        .current_dir(repository_root)
        .args(["build", "-Dlinking=static", optimize])
        .output()
        .unwrap();

    if !output.status.success() {
        println!(
            "cargo:warning=zig build failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        process::exit(1);
    }

    let library_path = repository_root.join("zig-out").join("lib");
    println!("cargo:rustc-link-search=native={}", library_path.display());
    println!("cargo:rustc-link-lib=static=tersets");
}

