# Contributing Guidelines for TerseTS

The repository is a Zig project with C and Python bindings. It uses the Apache‑2.0 license and includes a pre‑commit hook for building, formatting, and running tests. The following guidelines summarize how to contribute effectively.

## 1. Environment Setup 
* Install a recent Zig toolchain (v0.14.0 is used in the GitHub workflow).
* For Python bindings, install `setuptools` (and `build` for distribution) as shown in the Python README.
* Build the project in `Debug` mode with `zig build` and in Release modes with `zig build -Doptimize=ReleaseFast`, etc., as outlined in the README.

## 2. Running Checks
* Use the provided tools/pre-commit script before committing. It builds the library, runs tests, checks formatting, and compiles the C header across Debug and ReleaseFast profiles.
* `zig fmt --check .` is run to enforce formatting; ensure the code is formatted accordingly.

## 3. Code Style
* Every source file begins with the Apache‑2.0 license header. Copy the header from existing files, such as `src/functional/poor_mans_compression.zig`
* Module descriptions use `//!` comments and public functions use `///` comments. Follow this pattern in new Zig files.
* Keep functions concise and document parameters and returned values.
* Follow as close as possible [Zig coding style](https://ziglang.org/documentation/master/#Style-Guide).
* Every source file containing a new compression method must follow this convention:
  * `enum` and `struct` without methods are first.
  * The `compress` and `decompress` functions come right after, respectivelly.
  * `struct` containing methods after `decompress` function. 
  * Private functions follow in the same order in which they are called.
  * All test cases at the end of the file.

## 4. Tests
* Zig tests live in `src/tester.zig` and can be run with `zig build test src/capi.zig`.
* Python tests live in `bindings/python/tests` and run via `python3 -m unittest . `.
* Include tests for new features or bug fixes.

## 5. Commit Messages
* Use a short imperative summary.
* Provide a description of the code and contribution.
* If a new compression algorithm is proposed, be sure to reference the paper. 

## 6. Pull Requests
* Ensure `tools/pre-commit` passes locally before creating a PR.
* PRs trigger GitHub Actions to build, lint, and test on multiple platforms.
