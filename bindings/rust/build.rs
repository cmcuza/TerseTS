// Copyright 2026 TerseTS Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::env;
use std::process::{self, Command};

/// Compile TerseTS into a statically linked library and link it. unwrap() is deliberately used, as
/// recovery is not possible if any of the unwrapped operations fail, so better to fail immediately.
fn main() {
    // Compute the repositories root for running zig build and linking.
    let current_directory = env::current_dir().unwrap();
    let repository_root = current_directory.parent().unwrap().parent().unwrap();

    // Cargo's target variables describe the artifact being built, while Rust `cfg`
    // values in a build script describe the host running the build.
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();

    // Make the optimization level of TerseTS and Rust bindings match.
    let build_profile = env::var("PROFILE").unwrap();
    let optimize = match build_profile.as_str() {
        // On Windows, both `Debug` and `ReleaseSafe` pull in Zig's stack-trace
        // panic handler (`std.debug.SelfInfo.Windows.findModule`), which calls
        // `LdrRegisterDllNotification`. That API exists in ntdll.dll but is absent
        // from the stripped ntdll.lib bundled in Rust's MSVC sysroot, causing a
        // link failure. `ReleaseFast` avoids this by omitting panic stack traces.
        // On other platforms, `-Doptimize=Debug` links correctly.
        "debug" if cfg!(windows) => "-Doptimize=ReleaseFast",
        "debug" => "-Doptimize=Debug",
        "release" => "-Doptimize=ReleaseFast",
        build_profile => {
            println!(
                "cargo::error=Profile must be debug (dev) or release, not {}.",
                build_profile
            );
            process::exit(1);
        }
    };

    // Build the TerseTS library into a statically linked library.
    let mut zig = Command::new("zig");
    zig.current_dir(repository_root)
        .args(["build", "-Dlinking=static", "-Dpic=true", optimize]);

    if target_os == "macos" && target_arch == "aarch64" {
        let deployment_target =
            env::var("MACOSX_DEPLOYMENT_TARGET").unwrap_or_else(|_| "11.0".to_owned());
        println!("cargo::rerun-if-env-changed=MACOSX_DEPLOYMENT_TARGET");
        zig.arg(format!("-Dtarget=aarch64-macos.{deployment_target}"));
    }

    let output = zig.output().unwrap();

    if !output.status.success() {
        // Output is captured by cargo and used as commands.
        println!(
            "cargo::error=Failed to build TerseTS as a static library due to {}.",
            String::from_utf8(output.stderr).unwrap()
        );
        process::exit(1);
    }

    // Specify the name and location of the TerseTS library for the linker.
    let mut library_path = repository_root.to_path_buf();
    library_path.push("zig-out");
    library_path.push("lib");

    // Apple's `ld` requires 64-bit Mach-O archive members to start on an 8-byte boundary, but
    // Zig's archiver leaves `compiler_rt.o` at a 4-byte offset (ziglang/zig#1981), which `ld`
    // rejects — the soft-float and `_roundq` symbols it defines then go undefined at link time.
    //
    // We align the member in place ourselves rather than shelling out to `libtool`: newer macOS
    // `libtool` (26.x cctools) does not fix a misaligned member, it silently *drops* it, which
    // is worse. `compiler_rt.o` is the last archive member, so growing its extended name to push
    // the payload to an 8-byte boundary shifts only its own bytes; every other member header —
    // and the `__.SYMDEF` index that points at those headers — is left byte-identical, so no
    // index rebuild is needed and the object contents are untouched.
    if target_os == "macos" && target_arch == "aarch64" {
        let lib = library_path.join("libtersets.a");
        let archive = std::fs::read(&lib).unwrap();
        if let Some(aligned) = align_compiler_rt(&archive) {
            std::fs::write(&lib, &aligned).unwrap();
        }

        // Postcondition: confirm `compiler_rt.o` is now on an 8-byte boundary so `ld` can load
        // it. This turns any future change in the archive layout into a precise build-time
        // error instead of a confusing "undefined symbols" link failure. Only alignment is
        // checked, not presence: a static build bundles `compiler_rt` by default, so if the
        // member is absent there is nothing to align and the linker remains the authority.
        let check = std::fs::read(&lib).unwrap();
        if let Some(offset) = archive_member_payload_offset(&check, "compiler_rt")
            && offset % 8 != 0
        {
            println!(
                "cargo::error=compiler_rt.o is not 8-byte aligned in libtersets.a \
                 (payload offset {offset}, offset % 8 = {}); in-place alignment did not \
                 produce a linkable archive.",
                offset % 8
            );
            process::exit(1);
        }
    }

    // A plain static link resolves `compiler_rt.o` via normal archive extraction now that the
    // member is aligned. NOTE: do not switch this to `+whole-archive` — it was tried and made
    // things worse on aarch64-macos, because `-force_load` pulls in `compiler_rt.o`'s strong
    // soft-float defs, which then shadow Rust's own (working) `compiler_builtins` weak defs,
    // leaving the whole `xf` family plus `_roundq` undefined instead of resolving them.
    println!("cargo::rustc-link-lib=static=tersets");
    println!("cargo::rustc-link-search=native={}", library_path.display());
}

/// Return the byte offset of the Mach-O payload of the first `ar` archive member whose name
/// contains `needle`, or `None` if the archive cannot be parsed or no such member exists.
/// Used to verify archive-member alignment; see the postcondition check in `main`.
fn archive_member_payload_offset(archive: &[u8], needle: &str) -> Option<usize> {
    const MAGIC: &[u8] = b"!<arch>\n";
    if !archive.starts_with(MAGIC) {
        return None;
    }

    let mut offset = MAGIC.len();
    while offset + 60 <= archive.len() {
        let header = &archive[offset..offset + 60];
        let name = std::str::from_utf8(&header[0..16]).ok()?.trim_end();
        let size: usize = std::str::from_utf8(&header[48..58])
            .ok()?
            .trim()
            .parse()
            .ok()?;
        let data_start = offset + 60;

        // BSD/Mach-O archives store a name longer than 16 bytes as "#1/<len>", placing the
        // real name in the first <len> bytes of the member data; the object payload follows
        // it and `size` counts both the name and the payload.
        let (member_name, payload_offset) = match name.strip_prefix("#1/") {
            Some(len) => {
                let name_len: usize = len.parse().ok()?;
                let raw = archive.get(data_start..data_start + name_len)?;
                let real = std::str::from_utf8(raw).ok()?.trim_end_matches('\0');
                (real.to_owned(), data_start + name_len)
            }
            None => (name.trim_end_matches('/').to_owned(), data_start),
        };

        if member_name.contains(needle) {
            return Some(payload_offset);
        }

        // Member data is padded to an even boundary before the next header.
        offset = data_start + size + (size & 1);
    }

    None
}

/// Rewrite `archive` so the `compiler_rt.o` member's Mach-O payload starts on an 8-byte
/// boundary, by growing its extended name (`#1/<len>`) with NUL padding. Returns the rewritten
/// archive, or `None` if the member is absent, already aligned, or not the last member (Zig
/// places it last; only that case leaves every other member header — and the `__.SYMDEF` index
/// pointing at those headers — unshifted, so no index rebuild is required).
fn align_compiler_rt(archive: &[u8]) -> Option<Vec<u8>> {
    const MAGIC: &[u8] = b"!<arch>\n";
    if !archive.starts_with(MAGIC) {
        return None;
    }

    let mut offset = MAGIC.len();
    while offset + 60 <= archive.len() {
        let hdr_off = offset;
        let header = &archive[hdr_off..hdr_off + 60];
        let name = std::str::from_utf8(&header[0..16]).ok()?.trim_end();
        let size: usize = std::str::from_utf8(&header[48..58])
            .ok()?
            .trim()
            .parse()
            .ok()?;
        let data_start = hdr_off + 60;

        let (name_len, member_name, payload) = match name.strip_prefix("#1/") {
            Some(len) => {
                let name_len: usize = len.trim().parse().ok()?;
                let raw = archive.get(data_start..data_start + name_len)?;
                let real = std::str::from_utf8(raw).ok()?.trim_end_matches('\0');
                (name_len, real.to_owned(), data_start + name_len)
            }
            None => (0, name.trim_end_matches('/').to_owned(), data_start),
        };
        let next = data_start + size + (size & 1);

        if member_name.contains("compiler_rt") {
            let delta = (8 - (payload % 8)) % 8;
            // Bail out (leaving the archive untouched) unless this is a last, extended-name
            // member we can safely pad; the postcondition check in `main` then fails loudly.
            if delta == 0 || name_len == 0 || next < archive.len() {
                return None;
            }

            let member_end = data_start + size;
            let mut out = Vec::with_capacity(archive.len() + delta + 1);
            out.extend_from_slice(&archive[..hdr_off]);

            let mut new_header = header.to_vec();
            write_ar_field(&mut new_header[0..16], &format!("#1/{}", name_len + delta));
            write_ar_field(&mut new_header[48..58], &format!("{}", size + delta));
            out.extend_from_slice(&new_header);

            out.extend_from_slice(&archive[data_start..data_start + name_len]);
            out.extend(std::iter::repeat_n(0u8, delta));
            out.extend_from_slice(&archive[payload..member_end]);
            if out.len() % 2 == 1 {
                out.push(b'\n');
            }
            return Some(out);
        }

        offset = next;
    }

    None
}

/// Write `value` into an `ar` header field: left-justified and space-padded.
fn write_ar_field(field: &mut [u8], value: &str) {
    field.fill(b' ');
    field[..value.len()].copy_from_slice(value.as_bytes());
}
