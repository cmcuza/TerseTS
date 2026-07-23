---
applyTo: "src/**/*.zig,build.zig"
---

# Zig Style Review

The source of truth for Zig style is the official [Zig Style Guide](https://ziglang.org/documentation/master/#Style-Guide), as required by `.github/CONTRIBUTING.md`. Use nearby TerseTS code for behavior and compatibility, not to invent conflicting style rules.

## Official Zig Checks

- Use `camelCase` for ordinary callable declarations and callable values.
- Use `TitleCase` for types, type aliases, and callable declarations that return `type`.
- Use `snake_case` for other values, including variables, constants, fields, and enum fields. Namespace-like zero-field structs that are never instantiated also use `snake_case`.
- Use `TitleCase.zig` for files with top-level fields. Use `snake_case.zig` for namespace files and `snake_case` for directories.
- Apply normal casing to acronyms and initialisms, such as `readU32Be`, rather than preserving all-capital spelling.
- Avoid redundant words in type names, including `Value`, `Data`, `Context`, `Manager`, `State`, `utils`, `misc`, and contributor initials. Check the fully qualified name before repeating a namespace in a declaration name.
- Do not use underscore prefixes to imply privacy or avoid a collision. Choose a semantic name or use Zig's string identifier syntax for keywords.
- Use 4-space indentation and aim for 100 columns with judgment. Let `zig fmt` decide formatting where it has an opinion.

## TerseTS Checks

- For compression-method files, follow `.github/CONTRIBUTING.md`: declarations without methods first; then public `compress` and `decompress`; then types with methods, private helpers in call order, and tests.
- Treat established compatibility-sensitive public names as narrow exceptions. Keep additions consistent within an existing public type that is an exception, but do not request casing-only API breakage in an unrelated PR or copy the exception into unrelated code.
- Prefer explicit integer widths when data is serialized, bit-packed, or mirrored by bindings.
- Keep allocator ownership clear. New allocations should have a matching `defer`, `errdefer`, returned owned slice, or documented ownership transfer.
- Compression methods hold no state across calls: no `threadlocal` or global mutable variables, and no allocations from `std.heap.page_allocator`, which hides leaks from `testing.allocator`. Scratch structures are per-call; small fixed-size buffers, such as a 128-entry ring buffer, live on the stack.
- Move a helper function used by two or more method files to `src/utilities/shared_functions.zig`; when copies differ only in a value, share one function with that value as a parameter. Move a constant, type, or other value defined identically in two or more method files to `src/utilities/shared_structs.zig` together with its doc comment, and derive dependent constants instead of repeating literals (for example an index width from the shared table's length). Method-specific values stay local.
- Do not re-declare a shared value locally (`const x = shared_structs.x;`). Reference the shared namespace at the use site so the origin is visible; such aliases are compile-time bindings with no binary or runtime cost, so this is a provenance rule, not a memory one. Import and type aliases (`const math = std.math;`, `const Error = tersets.Error;`) stay.
- Do not re-check invariants already enforced at the public entry points (`tersets.compress`, `tersets.decompress`, the C API), such as rejecting empty input. Document the precondition in the method's doc comment instead. Methods validate only what the entry point cannot know: configuration shape, stream contents, and method-specific limits.
- Return existing `tersets.Error` values where possible instead of adding method-local error categories. Sibling codecs use `Error.UnsupportedInput` for input-shape problems and `Error.CorruptedCompressedData` for malformed streams.
- Keep compressed stream layouts documented next to the encoder and mirrored by decoder tests.
- For bit-level codecs: `@clz` and `@ctz` of a zero `u64` return 64, which does not fit in `u6`, so every path casting their result must handle a zero XOR first or use a wider type. Validate stream geometry with a TerseTS error before any `@intCast` that depends on it. Every decoder bit read handles `EndOfStream` with a TerseTS error; malformed input must never trap in Debug or ReleaseFast.
- When a file claims to follow a reference implementation, verify divergences against the reference source, not from memory, and document each deliberate divergence with its reason.
- Benchmark performance changes at the library's real call pattern — one `compress` call over the whole series — before optimizing for repeated small calls.
- For lossy codecs, tests should check the error bound, rejected unsupported inputs, malformed compressed data, and at least one boundary case for adaptive state.
- Generated-data tests run one round, without a local repeat loop or `test_rounds` constant: every CI run draws new random data, so coverage accumulates across runs.
- Avoid `std.debug.assert` in production code unless the surrounding codebase already uses it for the same kind of invariant. Prefer explicit error handling for data-dependent conditions.
- Run `zig fmt` on changed Zig files.
