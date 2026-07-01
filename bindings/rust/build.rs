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

    // On aarch64-macOS, Zig leaves the bundled `compiler_rt.o` member misaligned inside the
    // static archive, which Apple's linker cannot read. Nudge it onto an 8-byte boundary before
    // linking. The full rationale is documented above `align_compiler_rt` below.
    if target_os == "macos" && target_arch == "aarch64" {
        let archive_path = library_path.join("libtersets.a");
        let original_archive = std::fs::read(&archive_path).unwrap();
        if let Some(aligned_archive) = align_compiler_rt(&original_archive) {
            std::fs::write(&archive_path, &aligned_archive).unwrap();
        }

        // Confirm the postcondition: `compiler_rt.o` now starts on an 8-byte boundary. If a
        // future Zig version changes the archive layout so this no longer holds, fail here with
        // a clear message instead of letting the linker fail later with cryptic "undefined
        // symbols". A missing member is not an error: a static build bundles `compiler_rt` by
        // default, so if it is ever absent there is simply nothing to align.
        let final_archive = std::fs::read(&archive_path).unwrap();
        if let Some(payload_offset) = compiler_rt_payload_offset(&final_archive)
            && payload_offset % MACHO_MEMBER_ALIGNMENT != 0
        {
            println!(
                "cargo::error=compiler_rt.o is not 8-byte aligned in libtersets.a (payload \
                 offset {payload_offset}); in-place alignment did not produce a linkable archive."
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

// Apple's linker requires every 64-bit Mach-O member of a static archive to begin on an 8-byte
// boundary.
const MACHO_MEMBER_ALIGNMENT: usize = 8;

// A System V / BSD `ar` archive — the format Zig's archiver emits — is the 8-byte magic
// `!<arch>\n` followed by a sequence of members. Each member is a fixed 60-byte ASCII header
// followed by its bytes, and every member begins on an even offset.
const AR_MAGIC: &[u8] = b"!<arch>\n";
const AR_HEADER_LEN: usize = 60;
// Byte ranges of the two header fields we read: the member name, and its data length (a decimal
// string that counts the extended-name bytes plus the object payload).
const AR_NAME_FIELD: std::ops::Range<usize> = 0..16;
const AR_SIZE_FIELD: std::ops::Range<usize> = 48..58;

/// One member of an `ar` archive, located while walking the archive.
struct ArchiveMember {
    /// Byte offset of this member's 60-byte header within the archive.
    header_offset: usize,
    /// The member's file name, e.g. `"compiler_rt.o"`.
    name: String,
    /// For an extended (`#1/<len>`) name, the number of name bytes stored at the start of the
    /// member's data; `0` when the name is stored directly in the header.
    extended_name_len: usize,
    /// Value of the header's data-length field: the extended-name bytes plus the object payload.
    data_len: usize,
    /// Byte offset where the member's object payload (the Mach-O object) begins.
    payload_offset: usize,
}

/// Walk an `ar` archive and return its members in order, or `None` if `bytes` is not an `ar`
/// archive or a header cannot be parsed.
fn parse_archive_members(bytes: &[u8]) -> Option<Vec<ArchiveMember>> {
    if !bytes.starts_with(AR_MAGIC) {
        return None;
    }

    let mut members = Vec::new();
    let mut cursor = AR_MAGIC.len();
    while cursor + AR_HEADER_LEN <= bytes.len() {
        let header = &bytes[cursor..cursor + AR_HEADER_LEN];
        let name_field = std::str::from_utf8(&header[AR_NAME_FIELD]).ok()?.trim_end();
        let data_len: usize = std::str::from_utf8(&header[AR_SIZE_FIELD])
            .ok()?
            .trim()
            .parse()
            .ok()?;
        let data_offset = cursor + AR_HEADER_LEN;

        // A name longer than 16 bytes uses the extended form `#1/<len>`, storing the real name in
        // the first `<len>` bytes of the data with the object payload right after it. A short name
        // sits in the header itself, padded with a trailing '/'.
        let (extended_name_len, name, payload_offset) = match name_field.strip_prefix("#1/") {
            Some(len) => {
                let name_len: usize = len.trim().parse().ok()?;
                let raw_name = bytes.get(data_offset..data_offset + name_len)?;
                let name = std::str::from_utf8(raw_name)
                    .ok()?
                    .trim_end_matches('\0')
                    .to_owned();
                (name_len, name, data_offset + name_len)
            }
            None => (0, name_field.trim_end_matches('/').to_owned(), data_offset),
        };

        members.push(ArchiveMember {
            header_offset: cursor,
            name,
            extended_name_len,
            data_len,
            payload_offset,
        });

        // Member data is padded to an even length before the next header begins.
        cursor = data_offset + data_len + (data_len % 2);
    }

    Some(members)
}

/// Byte offset of the `compiler_rt.o` payload in `bytes`, or `None` if the archive cannot be
/// parsed or the member is absent. Used to verify alignment after the rewrite.
fn compiler_rt_payload_offset(bytes: &[u8]) -> Option<usize> {
    let members = parse_archive_members(bytes)?;
    let member = members
        .iter()
        .find(|member| member.name.contains("compiler_rt"))?;
    Some(member.payload_offset)
}


/// Rewrite `bytes` so the `compiler_rt.o` member's payload starts on an 8-byte boundary, and
/// return the new archive. Returns `None` — leaving the caller to keep the original archive
/// unchanged — if the member is absent, already aligned, or not in the expected last /
/// extended-name form that makes this rewrite safe.
fn align_compiler_rt(bytes: &[u8]) -> Option<Vec<u8>> {
    let members = parse_archive_members(bytes)?;
    let index = members
        .iter()
        .position(|member| member.name.contains("compiler_rt"))?;
    let member = &members[index];

    let misalignment = member.payload_offset % MACHO_MEMBER_ALIGNMENT;
    if misalignment == 0 {
        return None; // Already aligned; nothing to do.
    }
    let padding = MACHO_MEMBER_ALIGNMENT - misalignment;

    // The rewrite is only safe when this is the last member and uses an extended name: only then
    // does growing the name leave every other header — and the `__.SYMDEF` index — unmoved.
    let is_last_member = index == members.len() - 1;
    if member.extended_name_len == 0 || !is_last_member {
        return None;
    }

    let name_offset = member.header_offset + AR_HEADER_LEN;
    let payload_end = member.payload_offset + (member.data_len - member.extended_name_len);

    let mut aligned = Vec::with_capacity(bytes.len() + padding + 1);
    // Everything before this member is copied unchanged.
    aligned.extend_from_slice(&bytes[..member.header_offset]);

    // Copy the 60-byte header, then overwrite the name and data-length fields to account for the
    // extra padding. The other header fields (timestamp, uid/gid, mode) pass through as-is.
    let mut header = bytes[member.header_offset..name_offset].to_vec();
    let padded_name = format!("#1/{}", member.extended_name_len + padding);
    let padded_data_len = format!("{}", member.data_len + padding);
    write_ar_header_field(&mut header[AR_NAME_FIELD], &padded_name);
    write_ar_header_field(&mut header[AR_SIZE_FIELD], &padded_data_len);
    aligned.extend_from_slice(&header);

    // The original name bytes, then the extra NUL padding, then the object payload copied verbatim.
    aligned.extend_from_slice(&bytes[name_offset..name_offset + member.extended_name_len]);
    aligned.extend(std::iter::repeat_n(0u8, padding));
    aligned.extend_from_slice(&bytes[member.payload_offset..payload_end]);

    // Members are padded to an even length.
    if aligned.len() % 2 == 1 {
        aligned.push(b'\n');
    }
    Some(aligned)
}

/// Overwrite an `ar` header field in place with `value`, left-justified and space-padded — the
/// fixed-width ASCII convention `ar` uses for its header fields.
fn write_ar_header_field(field: &mut [u8], value: &str) {
    field.fill(b' ');
    field[..value.len()].copy_from_slice(value.as_bytes());
}
