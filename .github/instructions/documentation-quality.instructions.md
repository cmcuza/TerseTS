---
applyTo:
  - "**/*.md"
  - "src/**/*.zig"
  - "bindings/**/*.h"
  - "bindings/**/*.rs"
  - "bindings/**/*.py"
  - "bindings/**/*.jl"
---

# Documentation Quality Review

Ground Zig documentation review in the official [Doc Comment Guidance](https://ziglang.org/documentation/master/#Doc-Comment-Guidance). Apply the remaining checks below as TerseTS-specific documentation policy.

Review documentation, doc comments, and explanatory comments for TerseTS style:

- Omit information that is redundant with the declaration's name. Use `//!` for file or namespace documentation and `///` for the declaration that immediately follows it.
- Use `assume` and `assert` with Zig's precise illegal-behavior meanings.
- The first non-redundant sentence should state the declaration's purpose in TerseTS terms.
- Prefer short explanations of why the code exists, what layout or invariant it maintains, and what errors it returns.
- Flag vague or unsupported claims and long prose that does not add maintenance value. Do not infer or allege LLM authorship from writing style.
- Avoid mixing paper terminology, reference-implementation names, and TerseTS names unless the relationship is explicitly useful.
- Do not use analogies when direct compression, bit-layout, or zero-count terminology is clearer.
- Keep upstream references concise: cite the paper, DOI, or repository once near the module-level explanation when relevant.
- Flag comments that describe each line of code instead of the behavior or invariant a maintainer needs to preserve.
- Following Zig's [Lifetime and Ownership](https://ziglang.org/documentation/master/#Lifetime-and-Ownership) guidance, require ownership and lifetime documentation when a public API returns or retains a pointer or slice and the contract is not evident.
- Check that parameter names in documentation match the code exactly, especially `uncompressed_values`, `compressed_values`, `decompressed_values`, `method_configuration`, and configuration field names.

When commenting on a PR, cite the concrete Zig or TerseTS rule and suggest a rewrite direction rather than saying only that the text is unclear.
