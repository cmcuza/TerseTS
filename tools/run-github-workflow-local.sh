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

# Approximates .github/workflows/ci.yml for local verification before pushing.
# The C-binding only runs Clang for testing C-syntax. Therefore, it is skipped in this
# script to avoid requiring Clang to be installed locally.


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
    missing_commands=""

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

# Run Zig fmt to validate that the code is properly formatted.
print_step "Zig fmt Check"
if ! zig fmt --check .
then
    printf "Error: zig fmt check failed. Run 'zig fmt .' and re-run this script.\n" >&2
    exit 1
fi

# Build TereTS in both modes to validate that it works correctly.
print_step "Zig Build Debug"
zig build -Doptimize="Debug"
print_step "Zig Build ReleaseFast"
zig build -Doptimize="ReleaseFast"

# Build and test TerseTS in both modes to validate that they work correctly.
print_step "Zig Build Test Debug"
zig build test -Doptimize="Debug"
print_step "Zig Build Test ReleaseFast"
zig build test -Doptimize="ReleaseFast"

# Test C bindings.
print_step "Clang C-binding Syntax Check"
cd "$repo_root/bindings/c"
if command -v clang >/dev/null 2>&1; then
    command clang -Weverything tersets.h
else
    printf "Skipping C-binding syntax check because Clang is not available.\n" >&2
fi 

print_step "GCC C-binding Syntax Check"
gcc -Wall -Wextra tersets.h

# Test Julia binding.
cd "$repo_root/bindings/julia"
print_step "Julia Run Unittest"
julia Tests.jl

# Test Python binding.
# Install from source, confirm the package resolves from site-packages, then test.
cd "$repo_root/bindings/python"
print_step "Python Install Binding"
python -m pip install . -v
# We need to move out of the bindings/python directory to ensure
# that the installed package is imported instead of the local source code.
cd "$repo_root"
print_step "Python Verify Installed Binding Import"
python - <<'PY'
import pathlib
import tersets
library_path = str(pathlib.Path(tersets.__file__)).lower()
assert "site-packages" in library_path or "dist-packages" in library_path
PY
print_step "Python Run Unittest"
python -m unittest discover --verbose -s "$repo_root/bindings/python"

# Test Rust binding.
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
