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

/// Compile TerseTS into a statically linked library and link it.
fn main() {
    // Safe to unwrap as Rust bindings are stored two folders deep.
    let current_diroty = env::current_dir().unwrap();
    let repository_root = current_diroty.parent().unwrap().parent().unwrap();

    let output = Command::new("zig")
        .current_dir(repository_root)
        .args(["build", "-Dlinking=static"])
        .output()
        .unwrap();

    if !output.status.success() {
        // Output is captured by cargo and used as commands.
        println!(
            "cargo::error={} {}.",
            "Failed to build TerseTS as a static library due to",
            String::from_utf8(output.stderr).unwrap()
        );
        process::exit(1);
    }

    // TODO: Link the statically linked library.
}
