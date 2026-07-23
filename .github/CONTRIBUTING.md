# Contributing to TerseTS

TerseTS accepts bug fixes, compression methods, binding changes, tests, and documentation updates. Keep each pull request focused on one change. For a large feature, a new compression category, or a public API change, open an issue before writing the implementation.

The [README](../README.md) explains how to build and use TerseTS. This guide covers repository conventions that contributors need when changing the implementation.

## Development setup

Use the Zig version selected by the [CI workflow](workflows/zig-fmt-build-and-test-on-pr-and-push.yml). Build and test the library from the repository root:

```sh
zig fmt --check .
zig build -Doptimize=Debug
zig build test -Doptimize=Debug
zig build -Doptimize=ReleaseFast
zig build test -Doptimize=ReleaseFast
```

Run [`tools/pre-commit`](../tools/pre-commit) before each commit:

```sh
sh tools/pre-commit
```

The script checks Zig formatting, builds and tests both optimization modes, compiles the C header with GCC and Clang, and runs the Python tests.

Run [`tools/run-github-workflow-local.sh`](../tools/run-github-workflow-local.sh) before requesting review when you have the complete toolchain available:

```sh
sh tools/run-github-workflow-local.sh
```

This script approximates the complete CI job, including Julia, an installed Python package, and Rust. It installs the Python binding into the active Python environment, so use a virtual environment when needed. Clang is optional for this script; Zig, GCC, Julia, Python, and Cargo are required.

The pre-commit hook runs Python tests with `python3`. The workflow-local script uses `python` (and `pip`) from PATH; use a virtual environment when you need to control which interpreter is used.

Run the checks for every binding that you change. The CI workflow is the source of truth for the commands and tool versions used by the project.

## Repository layout

| Path | Purpose |
| --- | --- |
| [`src/tersets.zig`](../src/tersets.zig) | Public Zig API, compression method registry, and method dispatch |
| [`src/configuration.zig`](../src/configuration.zig) | Configuration types, JSON parsing, validation, and test configurations |
| [`src/lossless_compression/`](../src/lossless_compression) | Lossless compression methods |
| [`src/lossy_compression/`](../src/lossy_compression) | Lossy methods grouped by compression approach |
| [`src/utilities/`](../src/utilities) | Shared data structures and algorithms |
| [`src/tester.zig`](../src/tester.zig) | Shared generators and compression test helpers |
| [`bindings/`](../bindings) | C, Julia, Python, and Rust APIs |

Place a new method in the existing category that describes how it compresses data. Add a new category only when none of the current directories fits.

## Zig style

Add the Apache-2.0 header from [`tools/LICENSE_HEADER`](../tools/LICENSE_HEADER) to new source files. Run `zig fmt` on every changed Zig file.

Follow [Zig's official style guide](https://ziglang.org/documentation/master/#Style-Guide):

- Use `camelCase` for ordinary callables.
- Use `TitleCase` for types, type aliases, and callables that return a type.
- Use `snake_case` for other values, including variables, constants, fields, and enum fields.
- Use `TitleCase.zig` for files with top-level fields. Use `snake_case.zig` for namespace files and `snake_case` for directories.
- Apply the same rules to acronyms and initialisms.

Some public TerseTS names predate these rules. Preserve a name when changing it would break compatibility. New members of an existing public type, such as `Method`, must remain consistent with that type until a planned API migration changes the complete API. Do not copy a compatibility exception into unrelated code.

Compression method files normally use this order:

1. Imports, aliases, constants, and types without methods.
2. Public compression and decompression functions.
3. Types with methods.
4. Private helpers in the order they are first used.
5. Tests.

## Compression method interface

Compression implementations receive JSON configuration text through `method_configuration`. Standard entry points are compatible with these function types:

```zig
const CompressFn = fn (
    allocator: std.mem.Allocator,
    uncompressed_values: []const f64,
    compressed_values: *std.ArrayList(u8),
    method_configuration: []const u8,
) tersets.Error!void;

const DecompressFn = fn (
    allocator: std.mem.Allocator,
    compressed_values: []const u8,
    decompressed_values: *std.ArrayList(f64),
) tersets.Error!void;
```

Expose these as `pub fn compress` and `pub fn decompress`. A module may use more specific function names when it implements multiple `Method` values. Keep parameter names consistent with nearby methods and with the public API.

Treat each method's encoded format as a contract between its compressor and decompressor. The decompressor must accept every stream produced by the compressor and reject malformed or truncated input with a TerseTS error. Document the byte or bit layout next to the code that writes it when the layout is not self-evident.

## Configuration

The public APIs pass configuration as JSON. Each compressor converts that JSON to a typed configuration defined in [`src/configuration.zig`](../src/configuration.zig). Bindings should pass the same configuration data to the Zig core instead of implementing their own validation rules.

Reuse an existing configuration type when its fields and validation match the method. Parse it near the start of the compressor:

```zig
const parsed_configuration = try configuration.parse(
    allocator,
    configuration.AbsoluteErrorBound,
    method_configuration,
);
const error_bound = parsed_configuration.abs_error_bound;
```

When a method needs a new configuration shape:

1. Add a specifically named configuration struct to `src/configuration.zig`.
2. Use `snake_case` JSON field names that match the Zig fields exactly.
3. Add validation to `configuration.parse`. Invalid JSON, missing fields, extra fields, and out-of-range values must return `Error.InvalidConfiguration`.
4. Add valid and invalid parsing tests next to the configuration tests.
5. Register a valid configuration in `defaultConfigurationBuilder` so the shared method tests can exercise the compressor.
6. Update user documentation only when the public configuration shape changes.

Method implementations without options still parse `EmptyConfiguration` and accept `{}`. Do not add a separate error-bound parameter or a binding-specific configuration format.

## Registering a method

Adding a compressor requires more than adding its implementation file:

1. Add the method to `Method` in `src/tersets.zig`.
2. Add compression and decompression dispatch in `src/tersets.zig`.
3. Decide whether `extract` and `rebuild` support the format. Add dispatch or keep the method on the explicit unsupported path.
4. Add the method's configuration to `defaultConfigurationBuilder`.
5. Mirror the method name and numeric value in every binding.
6. Add method-level and public API tests.

For non-empty input, the final byte of a compressed stream stores the `Method` value. Append new methods; never reorder or renumber existing entries. A changed value makes existing streams and language bindings incompatible.

Keep the method definitions synchronized in these files:

| API | Method definition |
| --- | --- |
| Zig | [`src/tersets.zig`](../src/tersets.zig) |
| C | [`bindings/c/tersets.h`](../bindings/c/tersets.h) |
| Rust | [`bindings/rust/src/lib.rs`](../bindings/rust/src/lib.rs) |
| Python | [`bindings/python/tersets/__init__.py`](../bindings/python/tersets/__init__.py) |
| Julia | [`bindings/julia/TerseTS.jl`](../bindings/julia/TerseTS.jl) |

If a change adds a TerseTS error, update `Error` in `src/tersets.zig`, the stable C error mapping in `src/capi.zig`, affected bindings, and their tests. Do not expose Zig's internal error integer values through the C API.

Keep bindings thin. Memory allocated by the C API must be released through the matching `free*` function. Avoid new runtime or build dependencies unless the change cannot be implemented reasonably with the existing toolchain.

## Tests

Place method-specific tests at the end of the method file. Use [`src/tester.zig`](../src/tester.zig) for generated data and public API coverage.

Tests for a lossy method should cover:

- The documented error bound across the supported data distributions.
- Zero error when the method claims to support it.
- Empty, single-value, and boundary-sized inputs where relevant.
- `NaN`, infinity, and unsupported values, with explicit expected behavior.
- Invalid configuration and malformed compressed input.
- State transitions at block, segment, or adaptive-window boundaries.

Tests for a lossless method should compare every decompressed bit pattern with the input. Include repeated values, special floating-point values, and inputs that switch between encoding modes.

Use `std.testing.allocator` in tests so leaks are reported. Run both Debug and ReleaseFast tests for bit-level code because integer casts, shifts, and safety checks can behave differently across optimization modes.

When a binding changes, test through that binding's public API. A Zig round trip does not verify FFI types, method values, ownership, or error translation.

## Documentation and comments

Write documentation for maintainers and users of the API:

- Use `//!` for a file or namespace and `///` for the declaration that follows it.
- Start with the behavior that is not already clear from the declaration name.
- Explain ownership, lifetimes, encoded layouts, invariants, supported input, and error behavior where they matter.
- Use the names found in TerseTS code. Introduce terminology from a paper or reference implementation only when it helps connect the implementation to its source.
- Cite a paper or upstream repository once near the module-level explanation. Do not reproduce a paper summary in code comments.
- Remove line-by-line narration, marketing claims, analogies, and repeated explanations.
- Use plain headings and text. Do not add icons or emoji.

Comments should explain why a choice or invariant exists. Code should explain the routine control flow.

## Utilities

Put shared code in `src/utilities` only when more than one method needs it or when it forms a distinct, tested algorithm. Use a specific name that describes its role. Keep method-specific helpers in the method file.

Document ownership for utilities that allocate or return slices. Add tests for boundary cases and errors, not only the expected path.

## Pull requests

Before requesting review:

- Keep the change focused and remove unrelated formatting or generated files.
- Explain what changed and why it belongs in TerseTS.
- Link the issue or paper that motivates the change when relevant.
- Describe public API, configuration, compressed format, and compatibility changes explicitly.
- List unsupported inputs or known limitations.
- Run the applicable local and binding checks.
- Update the README when users must call the API differently.

Use the [pull request template](PULL_REQUEST_TEMPLATE.md). Short, concrete descriptions are easier to review than a long account of the implementation process.

Commit summaries should be imperative and describe the change, for example, `Add Gorilla lossless compression`. Use the commit body when the reason or compatibility impact is not clear from the summary.

## Issues

A useful bug report includes the input or a small reproducer, the expected result, the actual result, the Zig version, the target platform, and the command used to build or test TerseTS. Do not include private time-series data; reduce it to the smallest input that still demonstrates the problem.
