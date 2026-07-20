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
5. Review documentation using Zig's doc-comment guidance and TerseTS policy. Good documentation omits facts already present in the name and explains relevant stream layouts, invariants, errors, ownership, and maintenance-sensitive choices. Report concrete defects without speculating about LLM authorship.
6. For new or modified compression methods, check encoder/decoder symmetry, malformed input behavior, allocator ownership, integer casts, shift widths, and Debug/ReleaseFast assumptions.
7. For method enum changes, check synchronization across Zig, C, Rust, Python, and Julia bindings.
8. Check tests match the risk: round-trip behavior, error bounds for lossy methods, unsupported inputs, corrupted compressed data, and edge cases around window or block boundaries.
9. Leave only actionable review comments. Each comment should identify the source rule, explain why the inconsistency matters for TerseTS, and suggest the direction of the fix.

Prefer targeted comments over broad summaries. If a PR has repeated documentation issues, comment on the clearest representative instance and ask the contributor to apply the same simplification throughout the touched files.
