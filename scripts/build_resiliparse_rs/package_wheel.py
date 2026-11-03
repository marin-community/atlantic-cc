# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Package a resiliparse-rs build as a wheel that provides ``resiliparse._extract_rs``.

The wheel installs the extension and the shared libraries it needs into the ``resiliparse``
package directory, byte for byte, and nothing else:

    resiliparse/_extract_rs.cpython-312-x86_64-linux-gnu.so   (macOS: ..._extract_rs.cpython-312-darwin.so)
    resiliparse/liblexbor.so.2                               (macOS: liblexbor.2.dylib)

The extension finds lexbor through its ``$ORIGIN`` (``@loader_path``) rpath, the same layout production
used. There is no ``resiliparse/__init__.py``: without the upstream ``resiliparse`` distribution,
``resiliparse`` is a namespace package; with it, the files sit inside upstream's package directory and
neither distribution owns a file of the other.

The dist-info carries ``atlantic_cc_build.json`` (kind, fork commit, per-file sha256/md5, build facts) and
the licenses. For ELF builds the platform tag is derived from the highest ``GLIBC_2.x`` symbol version the
binaries require, and the extension's non-system ``DT_NEEDED`` entries must all be shipped.

    python package_wheel.py --kind source --extension target/release/libresiliparse_extract_rs.so \\
        --library liblexbor.so.2=vcpkg_installed/x64-linux/lib/liblexbor.so.2 \\
        --license LICENSE=LICENSE --out-dir dist
"""

import argparse
import base64
import hashlib
import io
import json
import re
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path

DIST_NAME = "resiliparse-rs-atlantic-cc"
FORK_REPOSITORY = "https://github.com/marin-community/atlantic-cc-resiliparse"
FORK_COMMIT = "850891b277e919f671ec19f12aafdce451b327f4"
VERSION = f"0+{FORK_COMMIT[:7]}"
PYTHON_TAG = "cp312"
PACKAGE_DIR = "resiliparse"
MODULE_NAME = "_extract_rs"
BUILD_INFO_NAME = "atlantic_cc_build.json"
KINDS = ("as-run", "source")

# Libraries every manylinux platform provides; anything else the extension needs must be in the wheel.
SYSTEM_LIBRARIES = frozenset(
    {"libc.so.6", "libm.so.6", "libgcc_s.so.1", "ld-linux-x86-64.so.2", "libpthread.so.0", "libdl.so.2", "librt.so.1"}
)
ELF_MAGIC = b"\x7fELF"
EM_X86_64 = 62
SHT_DYNAMIC = 6
SHT_GNU_VERNEED = 0x6FFFFFFE
DT_NULL = 0
DT_NEEDED = 1
MACHO_MAGICS = (b"\xcf\xfa\xed\xfe", b"\xca\xfe\xba\xbe")
GLIBC_VERSION = re.compile(r"^GLIBC_2\.(\d+)$")
ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)

SUMMARY = {
    "as-run": "The exact resiliparse-rs binaries that extracted Atlantic-CC v1.0 "
    "(marin-community/atlantic-cc-resiliparse @ 850891b), repackaged as a wheel.",
    "source": "resiliparse-rs (marin-community/atlantic-cc-resiliparse @ 850891b) built from pinned sources, the HTML "
    "extractor of Atlantic-CC v1.0.",
}
DESCRIPTION = """\
Provides `resiliparse._extract_rs`, the Rust main-content extractor of the resiliparse-rs fork
`marin-community/atlantic-cc-resiliparse` (tag `atlantic-cc-v1.0`) at commit
850891b277e919f671ec19f12aafdce451b327f4, plus the lexbor shared library it links
(phoerious/lexbor @ 647a0b5). Atlantic-CC v1.0 called

    from resiliparse._extract_rs import extract_plain_text
    extract_plain_text(html, main_content=True, preserve_formatting="markdown")

on every page. `atlantic_cc_build.json` in this distribution's metadata says whether the binaries are the
as-run ones or a source build, and records their hashes. See docs/extractor.md in the Atlantic-CC repository.
"""


@dataclass(frozen=True)
class ElfDeps:
    needed: tuple[str, ...]
    symbol_versions: tuple[str, ...]


def read_elf_deps(data: bytes) -> ElfDeps:
    """``DT_NEEDED`` libraries and required symbol versions (``.gnu.version_r``) of an x86_64 ELF file."""
    if data[:4] != ELF_MAGIC or data[4] != 2 or data[5] != 1:
        raise ValueError("not a little-endian ELF64 file")
    (machine,) = struct.unpack_from("<H", data, 18)
    if machine != EM_X86_64:
        raise ValueError(f"ELF machine {machine} is not x86_64")
    (section_offset,) = struct.unpack_from("<Q", data, 40)
    entry_size, n_sections = struct.unpack_from("<HH", data, 58)
    # (sh_name, sh_type, sh_flags, sh_addr, sh_offset, sh_size, sh_link, sh_info, sh_addralign, sh_entsize)
    sections = [struct.unpack_from("<IIQQQQIIQQ", data, section_offset + i * entry_size) for i in range(n_sections)]

    def string(table: int, offset: int) -> str:
        start = sections[table][4] + offset
        return data[start : data.index(b"\0", start)].decode()

    needed: list[str] = []
    versions: set[str] = set()
    for _, kind, _, _, offset, size, link, info, _, _ in sections:
        if kind == SHT_DYNAMIC:
            for pos in range(offset, offset + size, 16):
                tag, value = struct.unpack_from("<qQ", data, pos)
                if tag == DT_NULL:
                    break
                if tag == DT_NEEDED:
                    needed.append(string(link, value))
        elif kind == SHT_GNU_VERNEED:
            pos = offset
            for _ in range(info):
                _, n_aux, _, aux_offset, next_offset = struct.unpack_from("<HHIII", data, pos)
                aux = pos + aux_offset
                for _ in range(n_aux):
                    _, _, _, name, next_aux = struct.unpack_from("<IHHII", data, aux)
                    versions.add(string(link, name))
                    aux += next_aux
                pos += next_offset
    return ElfDeps(tuple(needed), tuple(sorted(versions)))


def is_macho(data: bytes) -> bool:
    return data[:4] in MACHO_MAGICS


def manylinux_tag(binaries: list[bytes]) -> str:
    """``manylinux_2_<N>_x86_64`` for the highest ``GLIBC_2.N`` any of ``binaries`` requires."""
    minors = [
        int(match.group(1))
        for data in binaries
        for version in read_elf_deps(data).symbol_versions
        if (match := GLIBC_VERSION.match(version))
    ]
    return f"manylinux_2_{max(minors)}_x86_64"


def extension_filename(platform_tag: str) -> str:
    if platform_tag.startswith("manylinux_") and platform_tag.endswith("_x86_64"):
        return f"{MODULE_NAME}.cpython-312-x86_64-linux-gnu.so"
    if platform_tag.startswith("macosx_"):
        return f"{MODULE_NAME}.cpython-312-darwin.so"
    raise ValueError(f"unsupported platform tag {platform_tag}")


def check_libraries_shipped(extension: bytes, shipped: set[str]) -> None:
    """Raise unless every non-system library the ELF extension links is part of the wheel."""
    missing = [lib for lib in read_elf_deps(extension).needed if lib not in SYSTEM_LIBRARIES and lib not in shipped]
    if missing:
        raise ValueError(f"the extension links {missing}, which the wheel does not ship")


def md5_b64(data: bytes) -> str:
    return base64.b64encode(hashlib.md5(data).digest()).decode()


def record_hash(data: bytes) -> str:
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


def metadata_text(kind: str, license_files: list[str]) -> str:
    lines = [
        "Metadata-Version: 2.1",
        f"Name: {DIST_NAME}",
        f"Version: {VERSION}",
        f"Summary: {SUMMARY[kind]}",
        f"Home-page: {FORK_REPOSITORY}",
        "License: Apache-2.0",
        "Requires-Python: ==3.12.*",
        "Classifier: License :: OSI Approved :: Apache Software License",
        "Classifier: Programming Language :: Python :: 3.12",
        "Classifier: Programming Language :: Rust",
        f"Project-URL: Source, {FORK_REPOSITORY}/tree/{FORK_COMMIT}",
        *(f"License-File: {name}" for name in license_files),
        "Description-Content-Type: text/markdown",
    ]
    return "\n".join(lines) + "\n\n" + DESCRIPTION


def wheel_text(platform_tag: str) -> str:
    return (
        "Wheel-Version: 1.0\n"
        "Generator: atlantic-cc scripts/build_resiliparse_rs/package_wheel.py\n"
        "Root-Is-Purelib: false\n"
        f"Tag: {PYTHON_TAG}-{PYTHON_TAG}-{platform_tag}\n"
    )


def zip_bytes(entries: list[tuple[str, bytes]]) -> bytes:
    """A deterministic zip: fixed timestamps and modes, entries in the given order."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for path, data in entries:
            info = zipfile.ZipInfo(path, date_time=ZIP_EPOCH)
            info.compress_type = zipfile.ZIP_DEFLATED
            mode = 0o755 if path.startswith(f"{PACKAGE_DIR}/") else 0o644
            info.external_attr = (0o100000 | mode) << 16
            archive.writestr(info, data)
    return buffer.getvalue()


def build_wheel(
    kind: str,
    extension: bytes,
    libraries: dict[str, bytes],
    licenses: dict[str, bytes],
    build_facts: dict,
    platform_tag: str,
) -> tuple[str, bytes, dict]:
    """Return ``(wheel filename, wheel bytes, build info)``."""
    package_files = {f"{PACKAGE_DIR}/{extension_filename(platform_tag)}": extension}
    package_files.update({f"{PACKAGE_DIR}/{name}": data for name, data in libraries.items()})
    build_info = {
        "kind": kind,
        "fork_repository": FORK_REPOSITORY,
        "fork_commit": FORK_COMMIT,
        "platform_tag": platform_tag,
        "files": {
            path: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "md5_b64": md5_b64(data)}
            for path, data in sorted(package_files.items())
        },
        "build": build_facts,
    }
    dist_info = f"{DIST_NAME.replace('-', '_')}-{VERSION}.dist-info"
    meta_files = {
        "METADATA": metadata_text(kind, sorted(licenses)).encode(),
        "WHEEL": wheel_text(platform_tag).encode(),
        BUILD_INFO_NAME: (json.dumps(build_info, indent=2, sort_keys=True) + "\n").encode(),
        **{f"licenses/{name}": data for name, data in licenses.items()},
    }
    entries = sorted(package_files.items()) + sorted((f"{dist_info}/{name}", data) for name, data in meta_files.items())
    record = "".join(f"{path},{record_hash(data)},{len(data)}\n" for path, data in entries)
    record += f"{dist_info}/RECORD,,\n"
    entries.append((f"{dist_info}/RECORD", record.encode()))
    filename = f"{DIST_NAME.replace('-', '_')}-{VERSION}-{PYTHON_TAG}-{PYTHON_TAG}-{platform_tag}.whl"
    return filename, zip_bytes(entries), build_info


def parse_named(values: list[str]) -> dict[str, str]:
    """``["NAME=VALUE", ...]`` -> ``{NAME: VALUE}``; the value may itself contain ``=``."""
    named: dict[str, str] = {}
    for item in values:
        name, sep, value = item.partition("=")
        if not sep or not name or not value:
            raise ValueError(f"expected NAME=VALUE, got {item!r}")
        named[name] = value
    return named


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kind", choices=KINDS, required=True)
    parser.add_argument("--extension", type=Path, required=True, help="the built pyo3 extension (any file name)")
    parser.add_argument(
        "--library", action="append", default=[], help="NAME=PATH of a shared library shipped next to the extension"
    )
    parser.add_argument("--license", action="append", default=[], help="NAME=PATH placed under dist-info/licenses/")
    parser.add_argument(
        "--expect-md5", action="append", default=[], help="NAME=MD5_B64: fail unless that shipped file has this md5"
    )
    parser.add_argument("--build-facts", type=Path, help="JSON object recorded under 'build' in the build info")
    parser.add_argument("--platform-tag", default="auto", help="'auto' derives manylinux_2_N_x86_64 from ELF files")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    extension = args.extension.read_bytes()
    libraries = {name: Path(path).read_bytes() for name, path in parse_named(args.library).items()}
    licenses = {name: Path(path).read_bytes() for name, path in parse_named(args.license).items()}
    build_facts = json.loads(args.build_facts.read_text()) if args.build_facts else {}

    if is_macho(extension):
        if args.platform_tag == "auto":
            raise ValueError("pass --platform-tag (e.g. macosx_11_0_arm64) for a Mach-O build")
        platform_tag = args.platform_tag
    else:
        check_libraries_shipped(extension, set(libraries))
        derived = manylinux_tag([extension, *libraries.values()])
        if args.platform_tag not in ("auto", derived):
            raise ValueError(f"--platform-tag {args.platform_tag} disagrees with the binaries' glibc needs ({derived})")
        platform_tag = derived

    shipped = {MODULE_NAME + ".so": extension, **libraries}
    for name, want in parse_named(args.expect_md5).items():
        have = md5_b64(shipped[name])
        if have != want:
            raise ValueError(f"{name} has md5 {have}, expected {want}")

    filename, wheel, build_info = build_wheel(args.kind, extension, libraries, licenses, build_facts, platform_tag)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / filename).write_bytes(wheel)
    (args.out_dir / BUILD_INFO_NAME).write_text(json.dumps(build_info, indent=2, sort_keys=True) + "\n")
    print(args.out_dir / filename)


if __name__ == "__main__":
    main()
