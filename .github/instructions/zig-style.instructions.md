---
applyTo:
  - "src/**/*.zig"
  - "build.zig"
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
- Return existing `tersets.Error` values where possible instead of adding method-local error categories.
- Keep compressed stream layouts documented next to the encoder and mirrored by decoder tests.
- For lossy codecs, tests should check the error bound, rejected unsupported inputs, malformed compressed data, and at least one boundary case for adaptive state.
- Avoid `std.debug.assert` in production code unless the surrounding codebase already uses it for the same kind of invariant. Prefer explicit error handling for data-dependent conditions.
- Run `zig fmt` on changed Zig files.
