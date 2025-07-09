# Contributing Guidelines for TerseTS

TerseTS is a high-performance compression library written in [Zig](https://ziglang.org/) with bindings for other languages through a C interface. It uses the Apache‑2.0 license and includes a pre‑commit hook for building, formatting, and running tests. We welcome contributions of all kinds: bug reports, feature requests, code, documentation, or tests. The following guidelines summarize how to contribute effectively.

## 1. Environment Setup

We recommend consulting the relevant README and source file for language-specific contribution setup.

| Language | Path |
|----------|------|
| Zig      | [README.md](../README.md) and [src/tersets.zig](../src/tersets.zig) |
| C        | [src/capi.zig](../src/capi.zig) and [c header](../bindings/c/tersets.h) comments |
| Python   | [bindings/python/README.md](../bindings/python/README.md) |

## 3. Development Tools and Checks
- Run [`tools/pre-commit`](../tools/pre-commit) before every commit. It:
  - Builds the library (`Debug` and `ReleaseFast`)
  - Runs Zig and Python tests
  - Formats code (`zig fmt`)
  - Compiles the C header
- To use it as a Git hook for automatic execution do:
  - On Unix/macOS:
    ```bash
    ln -s ../../tools/pre-commit .git/hooks/pre-commit
    ```
  - On Windows:
    ```powershell
    copy tools\pre-commit .git\hooks\pre-commit
    ```

## 3. Code Style
- Add the Apache-2.0 license header to new files. You can copy from [LICENSE_HEADER](../tools/LICENSE_HEADER).
- Follow [Zig's official style guide](https://ziglang.org/documentation/master/#Style-Guide).
- Format code with:
  ```bash
  zig fmt .
  ```

## 4. Language-Specific Contribution Guides
<a id="zig-specific"></a>
<details>
<summary><strong>Zig Contributing Steps</strong></summary>

### 🧩 Add a New Compression Method

In the `src/` directory, compression methods are organized by category. These categories are inspired by the following papers:

- [Time Series Compression Survey (2023)](https://dl.acm.org/doi/10.1145/3560814)  
- [CAMEO: Autocorrelation-Preserving Line Simplification (2024)](https://arxiv.org/abs/2501.14432)

Current categories:

- `src/functional_approximation/`  
  Methods using linear or non-linear functions to approximate segments.
  - e.g. [Poor Man’s Compression](../src/functional_approximation/poor_mans_compression.zig)
  - e.g. [Swing and Slide Filter](../src/functional_approximation/swing_slide_filter.zig)

- `src/histogram_representation/`  
  Methods that model the time series as a histogram.

- `src/line_simplification/`  
  Sampling-based or geometry-driven simplification techniques.
  - e.g. [Visvalingam-Whyatt](../src/line_simplification/visvalingam_whyatt.zig)
  - e.g. [Bottom-Up](../src/line_simplification/bottom_up.zig)

- `src/lossless_encoding/`  
  Techniques that preserve full precision of the input time series.
  - e.g. [Run Length Enconding](../src/lossless_enconding/run_length_encoding.zig)

- `src/value_quantization/`  
  Techniques that apply any kind of value quantization.
  - e.g. [Quantization](../src/value_quantization/bitpacked_quantization.zig)

🔧 If your method fits an existing category, add it under the respective folder using the filename pattern `[method_short_name].zig`.  If your method introduces a *new compression paradigm*, justify the new category in your commit message.


📌 **Note:** Add the new method to `tersets.Method` and expose the `compress` and `decompress` function in `src/tersets.zig`. 

### 🧱 File Structure Guidelines

Every source file containing a new compression method must follow this convention:
  - `enum` and `struct` without methods are first.
  - The `compress` and `decompress` functions are public and come right after, respectivelly.
  - struct containing methods after `decompress` function. 
  - Private functions follow in the same order in which they are called.
  - All test cases at the end of the file.

Example:

```c 
// 1. Enums and structs without methods
// 2. pub fn compress(...) Error!void { ... }
// 3. pub fn decompress(...) Error!void { ... }
// 4. structs with methods (if any)
// 5. private helper functions, in the order they are called
// 6. test cases at the end 
```

### 📏 Function Signature Guidelines
For consistency across TerseTS, follow this recommended signature:
```c
/// Compress `uncompressed_values` within `error_bound` using "[METHOD]". The function writes the 
/// result to `compressed_values`. The `allocator` is used for memory allocation of intermediate 
/// data structures. If an error occurs, it is returned.
pub fn compress(
    allocator: std.mem.Allocator,
    uncompressed_values: []const f64,
    compressed_values: *ArrayList(u8),
    error_bound: f32,
)  Error!void  { ... }

/// Decompress `compressed_values` produced by "[METHOD]". The function writes the result to
/// `decompressed_values`. The `allocator` is used for memory allocation of intermediate
/// data structures. If an error occurs it is returned.
pub fn decompress(
    allocator: std.mem.Allocator,
    compressed_values: []const u8,
    decompressed_values: *ArrayList(f64),
)  Error!void  { ... }
```

📌 If the allocator or error bound are not needed, they can be removed from the signature. 

### 🛠️ Adding Utility Functions

Shared logic that’s not specific to a single compression method should go in the `src/utilities/` folder. Current files include:


- `src/utilities/convex_hull.zig`  
  Contains a `ConvexHull` struct with functions for hull construction, point inclusion, and area computation.

- `src/utilities/hashed_priority_queue.zig`  
  Implements a priority queue with fast lookup and removal by key, useful for algorithms needing efficient dynamic heaps.

- `src/utilities/shared_structs.zig`  
  Defines general-purpose structs shared across modules (e.g., `DiscretePoint`, `Segment`, `LinearFunction`).

- `src/utilities/shared_functions.zig`  
  Contains common helper functions like appending values to a list and bit casting.

### ✅ Best Practices for Utilities

- **One responsibility per file:** Keep modules cohesive and focused.
- **Prefer reuse:** Import utilities rather than duplicating code.
- **Granularity:** If a utility becomes too large, consider splitting it into smaller modules.
- **Testing:** Add tests to illustrate and validate the behaviour when necessary.

📌 If you add a new utility that becomes broadly useful, mention it in the PR description.

### ⚠️ Zig Error Codes 

Errors in the Zig core are mapped to C-friendly integer codes in `capi.zig`:

| Code | Zig Error                 | Meaning                              |
|------|----------------------------|---------------------------------------|
| 0    | —                          | Success                               |
| 1    | `Error.UnknownMethod`      | Unsupported compression method        |
| 2    | `Error.UnsupportedInput`  | No/unexpected input data              |
| 3    | `Error.UnsupportedErrorBound` | Negative error bound              |
| 4    | `Error.OutOfMemory`       | Allocation failure                    |
| 5    | `Error.ItemNotFound`      | Missing data or lookup failure        |
| 6    | `Error.EmptyConvexHull`   | Convex hull utility received no data  |
| 7    | `Error.EmptyQueue`        | Priority queue utility empty          |


</details>

<a id="c-specific"></a>
<details>
<summary><strong>C Contributing Steps</strong></summary>

#### 📚 C API Overview
The C API in `bindings/c/tersets.h` mirrors Zig’s compression methods via:

- `enum Method { … }`: matching `tersets.Method`.
- `UncompressedValues`, `CompressedValues`, and `Configuration` structs
- `compress(...)` and `decompress(...)` functions

#### 🛠 Binding a New Method
1. If you add a new method in Zig, ensure the C `enum Method` gains the new entry (matching name & value).
2. If the method fits entirely in Zig without C exposure, you don’t need to touch C.

#### ✅ API Conventions

```c
int32_t compress(
    struct UncompressedValues uncompressed,
    struct CompressedValues *compressed_out,
    struct Configuration config
);
```

</details>

<a id="python-specific"></a>
<details>
<summary><strong>Python Contributing Steps</strong></summary>
sdsd
</details>

## 4. Tests
- Zig tests live in `src/tester.zig` and can be run with `zig build test src/capi.zig`.
- Python tests live in `bindings/python/tests` and run via `python3 -m unittest . `.
- Include tests for new features or bug fixes.

## 5. Git Commit Conventions

- Use short, imperative summaries (e.g., “Add swing filter compression”).
- Explain *why* the change is needed, not just *what* changed.
- If referencing a paper (e.g., for a new algorithm), include the citation or link.

See [cbea.ms/git-commit](https://cbea.ms/git-commit/) for a quick guide.


## 6. Pull Requests
- Ensure `tools/pre-commit` passes locally before creating a PR.
- PRs trigger GitHub Actions to build, lint, and test on multiple platforms.
