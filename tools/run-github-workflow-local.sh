#!/usr/bin/env sh

# Copyright 2026 TerseTS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Mirrors .github/workflows/ci.yml for local verification before pushing.

set -eu

repo_root="$(cd "$(dirname "$0")/.." && pwd)"

export CARGO_TERM_COLOR=always
export RUSTFLAGS="-D warnings"
export RUSTDOCFLAGS="-D warnings"

print_step() {
    printf "\n==> %s\n" "$1"
}

# Accumulates names of missing commands and reports them all at once.
assert_command_available() {
    local missing_commands=""
    local command_name

    for command_name in "$@"
    do
        if ! command -v "$command_name" >/dev/null 2>&1
        then
            missing_commands="${missing_commands:+$missing_commands }$command_name"
        fi
    done

    if [ -n "$missing_commands" ]
    then
        printf "Required commands not found in PATH: %s\n" "$missing_commands" >&2
        exit 1
    fi
}

assert_command_available zig gcc julia python rustup cargo

cd "$repo_root"

# Build in both modes first so binding tests can load the zig-out artifacts.
print_step "Zig fmt Check"
zig fmt --check .
print_step "Zig Build Debug"
zig build -Doptimize="Debug"
print_step "Zig Build ReleaseFast"
zig build -Doptimize="ReleaseFast"
print_step "Zig Build Test Debug"
zig build test -Doptimize="Debug"
print_step "Zig Build Test ReleaseFast"
zig build test -Doptimize="ReleaseFast"

cd "$repo_root/bindings/julia"
print_step "Julia Run Unittest"
julia Tests.jl

# Install from source, confirm the package resolves from site-packages, then test.
cd "$repo_root/bindings/python"
print_step "Python Install Binding"
python -m pip install . -v
print_step "Python Verify Installed Binding Import"
python - <<'PY'
import pathlib
import tersets

library_path = str(pathlib.Path(tersets.__file__)).lower()
print(library_path)
assert "site-packages" in library_path or "dist-packages" in library_path
PY
print_step "Python Run Unittest"
python -m unittest --verbose

# Zig outputs .dylib files into zig-out that break Rust static linking on macOS.
cd "$repo_root"
print_step "Clear Zig Cache and Zig Out"
rm -rf .zig-cache zig-out

cd "$repo_root/bindings/rust"
print_step "Cargo Build"
cargo build --verbose --release --all-targets
print_step "Cargo Clippy"
cargo clippy --verbose --release --all-targets
print_step "Cargo Doc"
cargo doc --verbose --release --no-deps
print_step "Cargo Test"
cargo test --verbose --release --all-targets -- --nocapture

printf "\nLocal GitHub workflow completed successfully.\n"
