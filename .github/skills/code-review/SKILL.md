---
name: tersets-code-review
description: Review TerseTS pull requests for documentation simplicity, naming consistency, style drift, and compression-code safety. Use this for Copilot code review on TerseTS PRs.
---

# TerseTS Code Review Skill

Use this workflow when reviewing TerseTS pull requests.

1. Read `.github/CONTRIBUTING.md` and apply Zig's official [Style Guide](https://ziglang.org/documentation/master/#Style-Guide) before using nearby code as a reference. Never derive a general style rule from one existing declaration.
2. Identify the touched area: core API, compression method, utility, binding, documentation, tests, or build tooling.
3. Check names by declaration semantics: ordinary callables are `camelCase`; types and type-returning callables are `TitleCase`; other values are `snake_case`. Preserve documented compatibility exceptions without extending them to unrelated code.
4. Compare terminology against nearby TerseTS code. Flag names copied from a paper or reference implementation when a TerseTS domain name would be clearer.
5. Review documentation using Zig's doc-comment guidance and TerseTS policy. A doc comment's first job is the signature: each parameter's role, the return value, and the error behavior. Comments are one or two lines by default, in simple common words; a rationale appears once, at the declaration that owns it. Flag comments that contradict the code — stale comments are defects. Report concrete defects without speculating about LLM authorship.
6. For new or modified compression methods, check encoder/decoder symmetry, malformed input behavior, allocator ownership, integer casts, shift widths, and Debug/ReleaseFast assumptions. Reject state that outlives a call (`threadlocal`, globals, `std.heap.page_allocator`) and re-checks of invariants the public entry points already enforce.
7. Check for duplication across method files: a helper function used by two or more of them belongs in `src/utilities/shared_functions.zig` (parameterize small differences), and a value defined identically in two or more belongs in `src/utilities/shared_structs.zig`. Shared values are referenced directly, not re-declared as local aliases.
8. For method enum changes, check synchronization across Zig, C, Rust, Python, and Julia bindings.
9. Check tests match the risk: round-trip behavior, error bounds for lossy methods, unsupported inputs tested at the level that owns the check, corrupted compressed data, and edge cases around window or block boundaries. Generated-data tests run one round; CI randomness accumulates coverage across runs.
10. Leave only actionable review comments. Each comment should identify the source rule, explain why the inconsistency matters for TerseTS, and suggest the direction of the fix.

Prefer targeted comments over broad summaries. If a PR has repeated documentation issues, comment on the clearest representative instance and ask the contributor to apply the same simplification throughout the touched files.
