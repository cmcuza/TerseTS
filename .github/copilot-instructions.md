# TerseTS Copilot Instructions

TerseTS is a Zig library for lossless and lossy time-series compression. It exposes a Zig API, a C API, and language bindings.

Use these sources in order when reviewing or editing the repository:

1. TerseTS policy in `.github/CONTRIBUTING.md`.
2. Zig's official [Style Guide](https://ziglang.org/documentation/master/#Style-Guide), including its naming and doc-comment guidance.
3. Nearby TerseTS code for domain behavior and established public API compatibility.

Do not infer a Zig style rule only from nearby code. Existing code can be inconsistent with the official guide.

When reviewing or editing TerseTS:

- Prioritize correctness, decompression safety, memory ownership, public API compatibility, and test coverage before style-only comments.
- Apply Zig naming rules by declaration semantics: ordinary callables use `camelCase`; types, type aliases, and callables returning a type use `TitleCase`; other values use `snake_case`.
- Use `snake_case` for namespace files and directories. A file with top-level fields represents a type and uses `TitleCase.zig`.
- Apply the same casing rules to acronyms and initialisms. Avoid underscore prefixes and redundant names, including repetition already supplied by the fully qualified namespace.
- Preserve compatibility-sensitive public names even when they predate the current style. Keep additions consistent within an existing public type that is a naming exception; for example, a new `Method` member follows that type's established convention until a planned API migration changes the type as a whole. Do not use such exceptions as precedent for unrelated declarations.
- When a compression method is added or renamed, check that `Method` values stay synchronized across `src/tersets.zig`, `src/capi.zig`, `bindings/c/tersets.h`, `bindings/rust/src/lib.rs`, `bindings/python/tersets/__init__.py`, and `bindings/julia/TerseTS.jl`.
- Keep method configuration names consistent with `src/configuration.zig`.
- Check the Apache-2.0 license header and copyright year.
- Prefer small, local changes. Do not introduce broad abstractions unless they remove concrete duplication or match an existing TerseTS pattern.
- Compression methods hold no state across calls: no `threadlocal` or global mutable variables, all allocations go through the caller's allocator (never `std.heap.page_allocator`, which hides leaks from `testing.allocator`), and everything allocated in a call is freed before it returns. Small fixed-size buffers belong on the stack.
- A helper function used by two or more method files moves to `src/utilities/shared_functions.zig`; when two copies differ only in a value, share one function with that value as a parameter. A constant, type, or other value defined identically in two or more method files moves to `src/utilities/shared_structs.zig` together with its doc comment. Reference shared declarations directly (`shared_structs.leading_zero_buckets`) instead of re-declaring them locally; such aliases cost nothing at runtime, but direct references keep the origin visible.
- Invariants already enforced at the public entry points (`tersets.compress`, `tersets.decompress`, the C API) are not re-checked inside compression methods. Document the precondition in the method's doc comment instead. Methods validate only what the entry point cannot know: configuration shape, stream contents, and method-specific limits.
- For bit-level codecs, inspect corrupted-input handling, sentinel/header layout, integer casts, shift widths, and Debug versus ReleaseFast behavior. `@clz` and `@ctz` of a zero `u64` return 64, which does not fit in `u6`; every path casting their result must handle a zero XOR first or use a wider type.
- When a file claims to follow a reference implementation, verify divergences against the reference source, not from memory, and require each deliberate divergence to be documented with its reason.
- Generated-data tests run one round, without a local repeat loop or `test_rounds` constant: every CI run draws new random data, so coverage accumulates across runs.
- Run or request `zig fmt` for changed Zig files, `zig build test --summary all`, and for bit-level codec changes `zig build test -Doptimize=ReleaseFast --summary all`.

Documentation and comments should be simple, precise, and maintainable:

- Follow Zig's doc-comment guidance: omit information already communicated by the declaration's name. Use `//!` for a file or namespace and `///` for the declaration that immediately follows it.
- A function doc comment's first job is its signature: name each parameter's role, the return value, and the error behavior, using the parameter names verbatim.
- Keep comments to one or two lines by default; treat a comment over three lines as a defect unless it documents a layout or invariant a maintainer needs. Use simple, common words rather than stacked technical qualifiers.
- State a rationale once, at the declaration that owns it; use sites do not re-explain it. Drop filler phrases such as "If an error occurs it is returned", "it is important to note", "deliberately", and "in order to".
- Stale comments are defects: when changing initialization, lifetime, or data-structure shape, update every comment that describes the old behavior.
- Use `assume` only for invariants whose violation causes unchecked illegal behavior, and `assert` only for invariants whose violation causes safety-checked illegal behavior.
- Document pointer and slice ownership and lifetime when they are not evident from the API.
- For TerseTS codecs, explain maintenance-sensitive input/output layouts, invariants, error cases, and non-obvious algorithm choices.
- Avoid broad paper summaries, marketing language, analogies, and line-by-line narration of obvious code.
- Use TerseTS domain names instead of copied names from papers or reference implementations when the TerseTS name is clearer.
- Reference papers or upstream repositories only when that reference helps future maintenance.
- Do not claim that prose was generated by an LLM. Review observable problems instead: redundancy, unsupported claims, terminology that does not match the code, or text that does not help maintain the implementation.

For Copilot code review, leave only actionable, line-specific comments. Name the violated Zig or TerseTS rule and suggest a concrete rewrite. Do not report a preference as a rule.
