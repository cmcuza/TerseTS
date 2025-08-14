# Contributing Guidelines for TerseTS

TerseTS is a high-performance compression library written in [Zig](https://ziglang.org/) with bindings for other languages through a C interface. It uses the Apache‑2.0 license and includes a pre‑commit hook for building, formatting, and running tests. We welcome contributions of all kinds: bug reports, feature requests, code, documentation, or tests. The following guidelines summarize how to contribute effectively.

## 1. Environment Setup

Consult the relevant README and source file for language-specific contribution setup.

| Language | Path |
|----------|------|
| Zig      | [README.md](../README.md) and [src/tersets.zig](../src/tersets.zig) |
| C        | [src/capi.zig](../src/capi.zig) and [C header](../bindings/c/tersets.h) comments |
| Python   | [bindings/python/README.md](../bindings/python/README.md) |

## 2. Development Tools and Checks
- Run [`tools/pre-commit`](../tools/pre-commit) before every commit. It:
  - Builds the library (`Debug` and `ReleaseFast`)
  - Runs Zig and Python tests
  - Formats code (`zig fmt`)
  - Compiles the C header
- To use it as a Git hook for automatic execution do:
  - On Unix-like:
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

#### 🧩 Add a New Compression Method

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

#### 🧱 File Structure Guidelines

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

#### 📏 Function Signature Guidelines
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


<a id="zig-error-codes"></a>
#### ⚠️ Zig Error Codes 

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

📌 If a new `Error` code is needed, mention it in the PR description.

#### 🛠️ Adding Utility Functions

Shared logic that’s not specific to a single compression method should go in the `src/utilities/` folder. Current files include:


- `src/utilities/convex_hull.zig`  
  Contains a `ConvexHull` struct with functions for hull construction, point inclusion, and area computation.

- `src/utilities/hashed_priority_queue.zig`  
  Implements a priority queue with fast lookup and removal by key, useful for algorithms needing efficient dynamic heaps.

- `src/utilities/shared_structs.zig`  
  Defines general-purpose structs shared across modules (e.g., `DiscretePoint`, `Segment`, `LinearFunction`).

- `src/utilities/shared_functions.zig`  
  Contains common helper functions like appending values to a list and bit casting.

#### ✅ Best Practices for Utilities

- **One responsibility per file:** Keep modules cohesive and focused.
- **Prefer reuse:** Import utilities rather than duplicating code.
- **Granularity:** If a utility becomes too large, consider splitting it into smaller modules.
- **Testing:** Add tests to illustrate and validate the behaviour when necessary.

📌 If you add a new utility that becomes broadly useful, mention it in the PR description.
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
2. You do not need to modify anything else in C.

#### ✅ API Conventions

```c
int32_t compress(
    struct UncompressedValues uncompressed,
    struct CompressedValues *compressed_out,
    struct Configuration config
);
```
Error return codes are defined in the Zig layer and passed through the C API.

➡️ See [Zig Error Codes](#zig-error-codes) for a full list of meanings.

</details>

<a id="python-specific"></a>
<details>
<summary><strong>Python Contributing Steps</strong></summary>

#### 🐍 Python API Overview
Public functions:

```python
compress(values: List[float], method: Method, error_bound: float) -> bytes
decompress(data: bytes) -> List[float]
```

These wrap the `C API` via `ctypes`, loading the shared library built by Zig.

#### 🔍 Adding a Method in Python
After adding the new method in Zig:
* Add it to the Python’s `Method` enum.
* Add tests to confirm it works via compress/decompress.

</details>

## 5. Other Binding Contributions

We welcome improvements and new features in the C and Python bindings—including better error messages, usability improvements, or helper functions.

However, please note:

* 🧩 **Keep bindings thin:** Bindings should reflect the Zig core behavior as closely as possible.

* 📦 **No external dependencies:** Do not introduce new runtime or build-time dependencies (e.g., Python packages, C libraries) unless there is a very strong justification and it is essential to the functionality.

* 🧾 **Justify structural changes:** If you propose changes to how bindings are organized, loaded, or called, please clearly explain the motivation and benefit in the PR description.

## 6. Tests
All compression methods must be validated using the unified testing framework in `src/tester.zig`.

This file contains:
- Utility functions for generating test data (e.g., linear, sinusoidal, bounded random)
- High-level test runners like `testErrorBoundedCompressionMethod` and `testGenerateCompressAndDecompress`
- A set of error-bounded compression checks
- `NaN/infinity` injection for stress testing
- Tools for narrowing down minimal failing subsequences

#### 🧪 How to Add Tests

When implementing a new method (e.g., `Method.MyNewMethod`), you should:
1. Add `test` blocks in your compression method file (e.g., `my_new_method.zig`):
2. Use `testErrorBoundedCompressionMethod` with multiple data distributions. Example:
    ```c
    test "mynewmethod supports bounded error compression across distributions" {
       const allocator = testing.allocator;
       const data_distributions = &[_]tester.DataDistribution{
           .FiniteRandomValues,
           .LinearFunctions,
           .BoundedRandomValues,
           .SinusoidalFunction,
           .LinearFunctionsWithNansAndInfinities,
           .RandomValuesWithNansAndInfinities,
       };
       try tester.testErrorBoundedCompressionMethod(
           allocator,
           Method.MyNewMethod,
           data_distributions,
       );
    }
3. Use `testGenerateCompressAndDecompress` with `error_bound = 0`. Example:
    ```c
    test "mynewmethod can compress and decompress with zero error" {
       const allocator = testing.allocator;
       try tester.testGenerateCompressAndDecompress(
           allocator,
           tester.generateRandomValues,
           Method.MyNewMethod,
           0,
           tersets.isWithinErrorBound,
       );
    }
    ```

#### ⚠️ Methods with Input Constraints

Some methods may not support all data types (e.g., `NaN`, `inf`, high-magnitude floats). This is acceptable as long as:

* The test suite excludes unsupported distributions.
* You add explicit negative tests that check failure modes and confirm appropriate errors are raised.

Example:
```c
test "my-method cannot compress nan values" {
    const data = &[4]f64{ 1.0, math.nan(f64), 3.0, 4.0 };
    var buffer = ArrayList(u8).init(testing.allocator);
    defer buffer.deinit();

    compress(data, &buffer, testing.allocator, 0.1) catch |err| {
        try testing.expectEqual(Error.UnsupportedInput, err);
        return;
    };

    try testing.expectFmt("", "Expected error not triggered", .{});
}
```

#### 🔍 Specialized Behavior
Some methods may require additional tests beyond correctness:
* Compression efficiency (e.g. merging, segment length control).
* Internal structure properties (e.g. monotonicity, convexity).
* Stress tests with large or adversarial inputs.

You are encouraged to add dedicated test blocks for these in the same `.zig` file.

## 7. Git Commit Conventions

- Use short, imperative summaries (e.g., “Add swing filter compression”).
- Explain *why* the change is needed, not just *what* changed.
- If referencing a paper (e.g., for a new algorithm), include the citation or link.

See [cbea.ms/git-commit](https://cbea.ms/git-commit/) for a quick guide.


## 8. Pull Requests
We welcome PRs of all sizes, but please follow these guidelines:

- **Describe what the PR does**: include a summary of the feature or fix.
- **Explain why it’s needed**: especially if it introduces a new method, category, or utility.
- **Reference related issues** using `Fixes #123` or `Closes #123`.
- **Run `tools/pre-commit` locally** to catch issues early.
- **Link to papers** if you’re implementing an algorithm from a publication.
- **Avoid adding dependencies** unless justified (see note under “Bindings”).

📌 Need help structuring your PR? See [Pull Request Template](PULL_REQUEST_TEMPLATE.md).


## 9. Filing Issues

When opening an issue, please:

- Describe the problem clearly (what you expected vs. what happened).
- Include example data or steps to reproduce, if possible.
- Mention your system, Zig version, and how you built the project.

Use labels (bug, enhancement, question, etc.) to help triage faster.