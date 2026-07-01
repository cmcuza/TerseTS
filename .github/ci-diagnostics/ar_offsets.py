#!/usr/bin/env python3
"""TEMPORARY CI diagnostic for the aarch64-macos compiler_rt.o linking issue.

Prints the payload offset (and offset modulo 8) of every member in one or more
`ar` static archives. Apple's `ld` requires 64-bit Mach-O archive members to
start on an 8-byte boundary; this makes the offsets observable on the real Apple
toolchain so the `libtool`/`ranlib` re-pack and the whole-archive workaround can
be validated in CI.

Remove this file together with the "macOS compiler_rt archive diagnostic"
workflow step before merging.
"""

import sys


def print_members(path: str) -> None:
    with open(path, "rb") as handle:
        data = handle.read()

    if data[:8] != b"!<arch>\n":
        print(f"  {path}: not an ar archive")
        return

    offset = 8
    while offset + 60 <= len(data):
        header = data[offset : offset + 60]
        name = header[0:16].decode("latin1").rstrip()
        try:
            size = int(header[48:58].decode("latin1").strip())
        except ValueError:
            break

        data_start = offset + 60
        # BSD/Mach-O archives store a name longer than 16 bytes as "#1/<len>",
        # placing the real name in the first <len> bytes of the member data; the
        # object payload follows it and `size` counts both.
        if name.startswith("#1/"):
            name_len = int(name[3:])
            real = data[data_start : data_start + name_len].split(b"\x00")[0].decode("latin1")
            payload = data_start + name_len
        else:
            real = name.rstrip("/")
            payload = data_start

        flag = "" if payload % 8 == 0 else "  <-- NOT 8-byte aligned"
        print(f"  {real:24} payload_offset={payload:>10}  %8={payload % 8}{flag}")

        nxt = data_start + size
        if nxt % 2:
            nxt += 1
        offset = nxt


if __name__ == "__main__":
    for archive in sys.argv[1:]:
        print(f"[{archive}]")
        print_members(archive)
