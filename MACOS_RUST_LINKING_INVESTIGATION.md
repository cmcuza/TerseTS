# macOS Rust Linking Investigation

Date: 2026-06-30  
Branch: `fix/mac-os-issue`  
Pull request: [#157 — Fix macOS CI link failure](https://github.com/cmcuza/TerseTS/pull/157)

## Purpose

This document records the investigation into the Apple Silicon Rust linking
failure, including the relevant background, attempted solutions, review
feedback, incorrect assumptions, observed CI results, and the final local
implementation.

## Build pipeline

The Rust crate does not compile the Zig source directly. Its build script runs
Zig first and then asks Rust to link the resulting static library:

```text
Zig source
   |
   | zig build -Dlinking=static
   v
libtersets.a
   |
   | cargo:rustc-link-lib=static=tersets
   v
Rust test or executable
   |
   | Apple linker extracts required archive members
   v
Final arm64 macOS binary
```

The static archive contains several Mach-O object files:

```text
libtersets.a
|- pocketfft.o
|- libtersets_zcu.o
`- compiler_rt.o
```

`compiler_rt.o` is included because `build.zig` sets:

```zig
library.bundle_compiler_rt = true;
```

It supplies software implementations of operations that Apple Silicon cannot
perform directly for Zig's `f80` type. These include symbols such as:

```text
___addxf3
___divxf3
___extenddfxf2
___mulxf3
___subxf3
___truncxfdf2
_roundq
```

The additional leading underscore seen in the linker diagnostics comes from
Mach-O symbol naming.

## Original failure: archive-member alignment

Apple's linker rejected `compiler_rt.o` because its payload did not begin at an
offset divisible by eight inside `libtersets.a`.

A local `aarch64-macos` cross-build produced these offsets:

| Archive member | Payload offset | Offset modulo 8 |
| --- | ---: | ---: |
| `pocketfft.o` | 9,848 | 0 |
| `libtersets_zcu.o` | 201,488 | 0 |
| `compiler_rt.o` | 2,427,636 | 4 |

The important distinction is that the Mach-O object itself was valid. The
problem was its position inside the archive container:

```text
Incorrect archive layout:

... previous member ... | four-byte boundary | compiler_rt.o
                                             ^
                                             offset % 8 == 4

Required archive layout:

... previous member ... | padding | compiler_rt.o
                                  ^
                                  offset % 8 == 0
```

This is why editing `compiler_rt.o` with `llvm-objcopy`, or changing alignment
inside the object, would address the wrong layer. The archive needs to be
reconstructed with different padding.

The related Zig issue also identifies `libtool -static` as a workaround:
[ziglang/zig#1981](https://github.com/ziglang/zig/issues/1981).

## Trial 1: repack with Apple's `libtool`

The first workaround ran:

```text
libtool -static -o libtersets.fixed.a libtersets.a
```

`libtool` reads the original archive and creates a new Apple-compatible static
archive. It can add padding between archive members without modifying their
machine code.

The warning printed by `libtool` about a misaligned member describes the input
archive it is reading. The warning does not by itself prove that its output
archive remains misaligned.

## Trial 2: check `compiler_rt.o` and run `ranlib`

The branch then added two steps:

1. Run `ar t` and require an archive member whose name contains
   `compiler_rt`.
2. Run `ranlib` after repacking.

This version passed the Apple Silicon CI job in workflow run
[28032316380](https://github.com/cmcuza/TerseTS/actions/runs/28032316380).

At this point, however, the purpose of `ranlib` had not been demonstrated
clearly. The comment claimed that symbol resolution could otherwise be
intermittent, but there was no recorded failure proving that claim.

## Pull-request review feedback

The unresolved review threads raised five points:

1. The workaround said Apple Silicon but ran on all macOS hosts.
2. The exit status from `ar t` was not checked.
3. A reviewer questioned whether repacking was safe and suggested
   `llvm-objcopy` or changing Zig object alignment.
4. A reviewer questioned why `ar` was required when `build.zig` already bundles
   `compiler_rt`.
5. A reviewer asked whether `libtool` actually leaves an invalid archive index
   and challenged the speculative wording around `ranlib`.

The first refactoring attempted to address all five:

- Cargo target variables replaced build-script host `cfg` checks.
- The workaround was limited to `aarch64-macos`.
- The unnecessary `ar t` check was removed.
- `ranlib` was removed because its necessity had not been proven.
- The comments explained archive-member placement rather than object-internal
  alignment.

## Incorrect conclusion: treating `ranlib` as redundant

Removing `ranlib` was incorrect.

The refactored version passed formatting and local Windows Rust tests, but the
Apple Silicon CI link failed in workflow run
[28475466349](https://github.com/cmcuza/TerseTS/actions/runs/28475466349).

The linker could find and load `libtersets_zcu.o`, but could not resolve the
software floating-point symbols supplied by `compiler_rt.o`:

```text
Undefined symbols for architecture arm64:
  "___addxf3"
  "___divxf3"
  "___extenddfxf2"
  "___extendsfxf2"
  "___fabsx"
  "___floatundixf"
  "___fmaxx"
  "___fminx"
  "___gtxf2"
  "___mulxf3"
  "___subxf3"
  "___truncxfdf2"
  "_roundq"
```

The earlier version differed behaviorally by running `ranlib`; the `ar t`
operation was read-only. The passing and failing CI runs therefore provide an
A/B test showing that symbol-index regeneration is required after this
`libtool` operation.

The corrected explanation is:

```text
libtool
  -> repairs archive-member placement
  -> preserves compiler_rt.o
  -> does not produce a table of contents that lets ld resolve the required
     compiler_rt soft-float symbols in this archive

ranlib
  -> rebuilds the archive table of contents
  -> lets ld locate compiler_rt.o for symbols such as ___addxf3
```

This directly answers the review question about whether `ranlib` is necessary.
It is not being retained as a defensive or speculative step; it is required by
an observed linker failure.

## Why the `ar t` check remains removed

`ar t` only lists member names. It does not prove:

- that the archive member is correctly aligned;
- that it contains the expected symbols;
- that the archive symbol index contains those symbols; or
- that Apple's linker can resolve them.

It also introduced another failure mode. If `ar` failed, empty standard output
was incorrectly reported as a missing `compiler_rt.o` member.

`build.zig` already requests `bundle_compiler_rt`, and the final linker is the
authoritative validation. The separate `ar` dependency is therefore
unnecessary.

## Deployment-target warning

The failed CI run also showed:

```text
object file was built for newer 'macOS' version (26.4)
than being linked (11.0)
```

Zig was building for the native runner version, macOS 26.4, while Rust was
linking for its default Apple Silicon deployment target, macOS 11.0.

This warning is separate from the missing-symbol failure, but leaving it in
place could produce binaries that do not run on the deployment versions Rust
claims to support.

The local fix now:

- uses `MACOSX_DEPLOYMENT_TARGET` when provided; and
- defaults to macOS 11.0 for `aarch64-macos`, matching Rust's default.

The resulting Zig target argument is:

```text
-Dtarget=aarch64-macos.11.0
```

Cargo is told to rerun the build script when `MACOSX_DEPLOYMENT_TARGET`
changes.

## Final local implementation

The current local implementation performs these operations:

1. Read Cargo's target OS and architecture.
2. Build Zig normally on non-Apple-Silicon targets.
3. On `aarch64-macos`, build Zig for the selected macOS deployment target.
4. Repack `libtersets.a` into `libtersets.fixed.a` with `libtool -static`.
5. Run `ranlib` on the fixed archive to rebuild its table of contents.
6. Replace the original archive only after both commands succeed.
7. Tell Rust to link the resulting static archive.

Running `ranlib` before replacing the original archive is safer than the
earlier ordering. If indexing fails, the original archive remains in place and
the build exits with the actual `ranlib` error.

## Rejected alternatives

### Remove `f80`

The project uses `f80` deliberately for additional precision in compression and
error-bound calculations. Replacing it with `f64` would be an algorithmic
change with possible correctness regressions. It might also fail to eliminate
all compiler-runtime requirements because Zig and `std.json` can require other
runtime helpers.

This is not an appropriate archive-packaging fix.

### Use `llvm-objcopy`

`llvm-objcopy` edits or copies object-file contents. The original failure
concerned the object's byte offset inside an archive, not the internal layout of
the object. Editing the object would be unnecessary and could introduce new
binary-format problems.

### Use an object or section `setAlignment()` API

Object-section alignment and archive-member placement are different layers.
Changing a section's alignment does not directly control where an archiver
places the entire object payload.

### Remove `compiler_rt`

Apple Silicon requires the software floating-point functions referenced by the
compiled Zig code. Without `compiler_rt`, the final Rust-driven link reports the
undefined symbols listed above.

### Keep the `ar t` presence check

Member-name presence is weaker than successful symbol resolution and adds an
unnecessary external command. It remains removed.

## Local validation performed

The following checks passed after the final local changes:

- Rust formatting check.
- `git diff --check`.
- Rust release test using the configured `rust-lld` linker.
- Zig cross-build using `aarch64-macos.11.0`.

The macOS-specific `libtool` and Apple `ranlib` behavior cannot be executed on
the current Windows development host. The next required validation is a new
GitHub Actions run on `macos-latest`.

## Unrelated issue discovered during testing

The Rust crate contains:

```toml
[target.x86_64-pc-windows-msvc]
linker = "rust-lld"
```

in `bindings/rust/.cargo/config.toml`.

Running Cargo from `bindings/rust` loads that configuration and passes.
Running Cargo from the repository root with `--manifest-path` does not load the
nested Cargo configuration, causing MSVC `link.exe` to reject Zig's
`compiler_rt.obj` with:

```text
LNK1143: invalid or corrupt file: no symbol for COMDAT section
```

Cargo configuration from a dependency is also not inherited by downstream
users. This is a separate Windows packaging problem and was not changed as part
of the macOS PR.

## Temporary local failures that were not code defects

Some initial local Rust and Zig commands failed with `AccessDenied` while
writing existing cache or incremental-compilation directories. Re-running the
checks with clean temporary target directories outside the restricted sandbox
succeeded. Those failures were environmental and unrelated to the macOS
linking changes.

## Remaining work

1. Commit and push the current `bindings/rust/build.rs` change.
2. Run PR CI on Apple Silicon.
3. Confirm that:
   - the archive-alignment error is absent;
   - the `___addxf3` family of undefined symbols is resolved;
   - the macOS 26.4 versus 11.0 deployment warnings are absent; and
   - Rust build, clippy, documentation, and tests all pass.
4. Update the PR discussion with the passing/failing A/B evidence.
5. Resolve review threads only after CI confirms the corrected implementation.

