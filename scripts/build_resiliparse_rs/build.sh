#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
#
# Build resiliparse._extract_rs, the HTML main-content extractor Atlantic-CC v1.0 was built with (the
# resiliparse-rs fork marin-community/atlantic-cc-resiliparse, tag atlantic-cc-v1.0 = 850891b), from
# pinned sources; run the fork's golden tests; package it as a wheel (package_wheel.py) and as a
# production-layout artifact directory; check that the installed wheel imports and extracts with no
# build tree around. See docs/extractor.md.
#
#   build.sh --out-dir DIR [--work-dir DIR] [--python PYTHON] [--skip-tests]
#
# Supported hosts: Linux x86_64 (glibc) and macOS arm64. Needs git, curl, tar, a C compiler (gcc on
# Linux, the Xcode command line tools on macOS), CPython 3.12 and rustup. The pinned Rust toolchain
# is installed through rustup; cmake, ninja and (Linux) libclang are pip-installed into a venv under
# the work dir; vcpkg is fetched at a pinned commit. Nothing is written outside --work-dir,
# --out-dir and the rustup/cargo homes.
set -euo pipefail

# --- Pins --------------------------------------------------------------------------------------------
FORK_URL="https://github.com/marin-community/atlantic-cc-resiliparse.git"
FORK_COMMIT="850891b277e919f671ec19f12aafdce451b327f4"
# rustc that built production's _extract_rs.so (its .comment section: "rustc version 1.98.0 (88d9e12ae 2026-08-18)").
RUST_TOOLCHAIN="1.98.0"
# microsoft/vcpkg master when production built lexbor (2026-08-11). Port versions still come from the
# fork's vcpkg-configuration.json baseline (71f123418c6cdc00258964629b0ede4317ba1a36); lexbor itself is
# the fork's overlay port (phoerious/lexbor @ 647a0b5, SHA512-pinned, -O3, shared library).
VCPKG_URL="https://github.com/microsoft/vcpkg.git"
VCPKG_COMMIT="dc31f86a441067d6c2e9e4ad52988eb41ef4f442"
VCPKG_TOOL_URL="https://github.com/microsoft/vcpkg-tool/releases/download/2026-07-27"
VCPKG_TOOL_SHA512_GLIBC="be59d1fdf3725d2fb4bd1c0b435266726aaff2f87cb2503b775f44fb9b392ab474e6370d7e90e9d07b2c584b7deacc0670f98b477905c2e0d5cba5e01cee93dc"
VCPKG_TOOL_SHA512_MACOS="33a8a949e2de5f1880ef02bfd447454ee149368b8d1179d8c597abba518ab12133b29b8ee4fe830de69857772c937d5097afc68b379c9d315ab72a725ed4d43b"
# Production pip-installed these (latest releases on 2026-08-11) because apt was unusable on its worker.
PIP_BUILD_TOOLS=("cmake==4.4.2" "ninja==1.13.0")
PIP_LIBCLANG="libclang==18.1.1"
MACOS_DEPLOYMENT_TARGET="11.0"
CARGO_PACKAGE="resiliparse-extract-rs"

die() { echo "build.sh: $*" >&2; exit 1; }

OUT_DIR=""
WORK_DIR="$PWD/.cache/resiliparse_rs_build"  # .cache/ is gitignored in this repository
PYTHON="python3.12"
RUN_TESTS=1
while [ $# -gt 0 ]; do
  case "$1" in
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --work-dir) WORK_DIR="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --skip-tests) RUN_TESTS=0; shift ;;
    *) die "unknown argument: $1" ;;
  esac
done
[ -n "$OUT_DIR" ] || die "--out-dir is required"
mkdir -p "$OUT_DIR" "$WORK_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
WORK_DIR="$(cd "$WORK_DIR" && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$(command -v "$PYTHON")" || die "python not found"
"$PYTHON" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12))' || die "$PYTHON is not CPython 3.12 (pyo3 builds are not abi3)"

case "$(uname -s)-$(uname -m)" in
  Linux-x86_64)
    TRIPLET="x64-linux"
    VCPKG_TOOL_ASSET="vcpkg-glibc"
    VCPKG_TOOL_SHA512="$VCPKG_TOOL_SHA512_GLIBC"
    BUILT_LIB="libresiliparse_extract_rs.so"
    RPATH_ORIGIN='$ORIGIN'
    PLATFORM_TAG="auto"
    ;;
  Darwin-arm64)
    TRIPLET="arm64-osx"
    VCPKG_TOOL_ASSET="vcpkg-macos"
    VCPKG_TOOL_SHA512="$VCPKG_TOOL_SHA512_MACOS"
    BUILT_LIB="libresiliparse_extract_rs.dylib"
    RPATH_ORIGIN="@loader_path"
    PLATFORM_TAG="macosx_${MACOS_DEPLOYMENT_TARGET/./_}_arm64"
    ;;
  *) die "unsupported host $(uname -s)-$(uname -m)" ;;
esac

sha512_matches() {  # file expected_hex
  "$PYTHON" -c 'import hashlib, sys; sys.exit(hashlib.sha512(open(sys.argv[1], "rb").read()).hexdigest() != sys.argv[2])' "$1" "$2"
}

fetch_commit() {  # url commit dir
  rm -rf "$3"
  git init -q "$3"
  git -C "$3" fetch -q --depth 1 "$1" "$2"
  git -C "$3" -c advice.detachedHead=false checkout -q FETCH_HEAD
  [ "$(git -C "$3" rev-parse HEAD)" = "$2" ] || die "$1 did not check out at $2"
}

# --- Toolchain ----------------------------------------------------------------------------------------
TOOLVENV="$WORK_DIR/toolvenv"
if [ ! -x "$TOOLVENV/bin/cmake" ]; then
  "$PYTHON" -m venv "$TOOLVENV"
  tools=("${PIP_BUILD_TOOLS[@]}")
  [ "$TRIPLET" = "x64-linux" ] && tools+=("$PIP_LIBCLANG")
  "$TOOLVENV/bin/pip" install --disable-pip-version-check --no-cache-dir --quiet "${tools[@]}"
fi
if [ "$TRIPLET" = "x64-linux" ]; then
  LIBCLANG_PATH="$("$TOOLVENV/bin/python" -c 'import clang, os; print(os.path.join(os.path.dirname(clang.__file__), "native"))')"
  # The libclang wheel has no compiler builtin headers; lexbor.h includes stddef.h/stdarg.h.
  export BINDGEN_EXTRA_CLANG_ARGS="-I$(gcc -print-file-name=include)"
else
  LIBCLANG_PATH="$(xcode-select -p)/usr/lib"
  [ -f "$LIBCLANG_PATH/libclang.dylib" ] || LIBCLANG_PATH="$(xcode-select -p)/Toolchains/XcodeDefault.xctoolchain/usr/lib"
  [ -f "$LIBCLANG_PATH/libclang.dylib" ] || die "libclang.dylib not found; install the Xcode command line tools"
  export MACOSX_DEPLOYMENT_TARGET="$MACOS_DEPLOYMENT_TARGET"
  export VCPKG_KEEP_ENV_VARS="MACOSX_DEPLOYMENT_TARGET"
fi

command -v rustup >/dev/null || die "rustup not found (https://rustup.rs)"
rustup toolchain install "$RUST_TOOLCHAIN" --profile minimal --no-self-update
export RUSTUP_TOOLCHAIN="$RUST_TOOLCHAIN"
rustc --version | grep -q "^rustc $RUST_TOOLCHAIN " || die "rustc is $(rustc --version), need $RUST_TOOLCHAIN"

# --- Sources ------------------------------------------------------------------------------------------
REPO="$WORK_DIR/chatnoir-resiliparse"
fetch_commit "$FORK_URL" "$FORK_COMMIT" "$REPO"
# As production: the extractor links only lexbor; the other manifest entries are fastwarc dependencies.
"$PYTHON" - "$REPO/vcpkg.json" <<'PY'
import json, sys
path = sys.argv[1]
with open(path) as f:
    manifest = json.load(f)
manifest["dependencies"] = ["lexbor"]
with open(path, "w") as f:
    f.write(json.dumps(manifest, indent=2) + "\n")
PY

VCPKG_ROOT="$WORK_DIR/vcpkg"
if [ "$(git -C "$VCPKG_ROOT" rev-parse HEAD 2>/dev/null || true)" != "$VCPKG_COMMIT" ]; then
  fetch_commit "$VCPKG_URL" "$VCPKG_COMMIT" "$VCPKG_ROOT"
fi
if [ ! -f "$VCPKG_ROOT/vcpkg" ] || ! sha512_matches "$VCPKG_ROOT/vcpkg" "$VCPKG_TOOL_SHA512"; then
  curl -fsSL --retry 3 -o "$VCPKG_ROOT/vcpkg" "$VCPKG_TOOL_URL/$VCPKG_TOOL_ASSET"
  sha512_matches "$VCPKG_ROOT/vcpkg" "$VCPKG_TOOL_SHA512" || die "vcpkg tool checksum mismatch"
fi
chmod +x "$VCPKG_ROOT/vcpkg"

# --- Build --------------------------------------------------------------------------------------------
export VCPKG_ROOT
export VCPKG_DEFAULT_TRIPLET="$TRIPLET"
export VCPKG_DISABLE_METRICS=1
export VCPKG_BINARY_SOURCES="clear"
export X_VCPKG_REGISTRIES_CACHE="$WORK_DIR/vcpkg-registries"
mkdir -p "$X_VCPKG_REGISTRIES_CACHE"
export PATH="$TOOLVENV/bin:$VCPKG_ROOT:$PATH"
export LIBCLANG_PATH
export CMAKE_POLICY_VERSION_MINIMUM="3.5"
export PYO3_PYTHON="$PYTHON"
# Some CI images export a shared CARGO_TARGET_DIR; keep the build inside the checkout.
export CARGO_TARGET_DIR="$REPO/target"
# As production: RUSTFLAGS replaces .cargo/config.toml's build.rustflags (--cfg pyo3_disable_reference_pool),
# so that cfg is off. It only changes how pyo3 defers reference-count updates, not extracted text.
export RUSTFLAGS="-C link-arg=-Wl,-rpath,$RPATH_ORIGIN"

cd "$REPO"
vcpkg install --triplet "$TRIPLET" --x-install-root "$REPO/vcpkg_installed"
cargo build --release --locked -p "$CARGO_PACKAGE"
if [ "$RUN_TESTS" = 1 ]; then
  cargo test --release --locked -p resiliparse --test extract_golden 2>&1 | tee "$OUT_DIR/golden_tests.log"
fi

BUILT="$REPO/target/release/$BUILT_LIB"
LIB_DIR="$REPO/vcpkg_installed/$TRIPLET/lib"
if [ "$TRIPLET" = "x64-linux" ]; then
  LEXBOR_NAME="liblexbor.so.2"
else
  LEXBOR_NAME="$(otool -L "$BUILT" | awk '$1 ~ /^@rpath\/liblexbor/ {sub("@rpath/", "", $1); print $1}')"
  [ -n "$LEXBOR_NAME" ] || die "the extension does not load lexbor through @rpath"
fi
[ -f "$LIB_DIR/$LEXBOR_NAME" ] || die "$LIB_DIR/$LEXBOR_NAME not found"

{
  echo "Rust crates statically linked into _extract_rs (cargo tree; license as declared by each crate)."
  echo "The Rust standard library (MIT OR Apache-2.0) is linked as well."
  cargo tree --locked -p "$CARGO_PACKAGE" -e normal,no-proc-macro --prefix none --format '{p} | {l}' \
    | sed -e 's/ (\*)$//' -e 's/ (\/[^)]*)//' | sort -u
} > "$WORK_DIR/THIRD_PARTY_RUST_CRATES.txt"
licenses=(--license "LICENSE=$REPO/LICENSE" --license "lexbor/LICENSE=$REPO/vcpkg_installed/$TRIPLET/share/lexbor/copyright")
LEXBOR_SRC="$(ls -d "$VCPKG_ROOT"/buildtrees/lexbor/src/*.clean 2>/dev/null | head -n 1 || true)"
[ -n "$LEXBOR_SRC" ] && [ -f "$LEXBOR_SRC/NOTICE" ] && licenses+=(--license "lexbor/NOTICE=$LEXBOR_SRC/NOTICE")
licenses+=(--license "THIRD_PARTY_RUST_CRATES.txt=$WORK_DIR/THIRD_PARTY_RUST_CRATES.txt")

# --- Record what built it -----------------------------------------------------------------------------
first_line() { "$@" 2>&1 | head -n 1; }
export FACT_RUSTC="$(rustc -vV | tr '\n' ';')"
export FACT_CC="$(first_line cc --version)"
export FACT_CMAKE="$(first_line cmake --version)"
export FACT_NINJA="$(first_line ninja --version)"
export FACT_VCPKG="$(first_line vcpkg version)"
export FACT_LIBC="$( (ldd --version 2>/dev/null || sw_vers -productVersion) | head -n 1)"
export FACT_OS="$( (. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME") || echo "macOS $(sw_vers -productVersion)")"
export FACT_VCPKG_COMMIT="$VCPKG_COMMIT"
export FACT_LIBCLANG="$LIBCLANG_PATH"
export FACT_RUSTFLAGS="$RUSTFLAGS"
"$PYTHON" - "$WORK_DIR/build_facts.json" <<'PY'
import datetime, json, os, platform, sys
facts = {
    "built_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
    "host": f"{platform.system()}-{platform.machine()}",
    "os": os.environ["FACT_OS"],
    "libc": os.environ["FACT_LIBC"],
    "python": platform.python_version(),
    "rustc": os.environ["FACT_RUSTC"],
    "rustflags": os.environ["FACT_RUSTFLAGS"],
    "cc": os.environ["FACT_CC"],
    "cmake": os.environ["FACT_CMAKE"],
    "ninja": os.environ["FACT_NINJA"],
    "vcpkg_tool": os.environ["FACT_VCPKG"],
    "vcpkg_commit": os.environ["FACT_VCPKG_COMMIT"],
    "libclang": os.environ["FACT_LIBCLANG"],
}
with open(sys.argv[1], "w") as f:
    json.dump(facts, f, indent=2)
PY

# --- Package ------------------------------------------------------------------------------------------
WHEEL="$("$PYTHON" "$SCRIPT_DIR/package_wheel.py" --kind source --extension "$BUILT" \
  --library "$LEXBOR_NAME=$LIB_DIR/$LEXBOR_NAME" "${licenses[@]}" \
  --build-facts "$WORK_DIR/build_facts.json" --platform-tag "$PLATFORM_TAG" --out-dir "$OUT_DIR" | tail -n 1)"

# Production-layout artifact (what Marin's workers downloaded): the fork's Python tree plus both libraries.
ARTIFACT="$OUT_DIR/artifact"
rm -rf "$ARTIFACT"
mkdir -p "$ARTIFACT"
cp "$BUILT" "$ARTIFACT/_extract_rs.so"
cp "$LIB_DIR/$LEXBOR_NAME" "$ARTIFACT/$LEXBOR_NAME"
tar -C "$REPO" -czf "$ARTIFACT/resiliparse_py.tar.gz" resiliparse-py
"$PYTHON" - "$ARTIFACT/manifest.json" "$FORK_COMMIT" "$FORK_URL" "$LEXBOR_NAME" <<'PY'
import json, platform, sys
path, commit, repository, lexbor = sys.argv[1:]
manifest = {
    "commit_sha": commit,
    "repository": repository,
    "platform": f"{platform.system()}-{platform.machine()} {' '.join(platform.libc_ver())}".strip(),
    "python_version": platform.python_version(),
    "extension": "_extract_rs.so",
    "extra_shared_libraries": [lexbor],
}
with open(path, "w") as f:
    json.dump(manifest, f, indent=2)
PY

# --- Contract check -----------------------------------------------------------------------------------
# Install the wheel into a fresh venv and hide the build tree, so the import can only succeed through
# the extension's own rpath; the fork's golden pages must come out exactly as the Rust tests expect.
CHECK_VENV="$WORK_DIR/checkvenv"
rm -rf "$CHECK_VENV"
"$PYTHON" -m venv "$CHECK_VENV"
"$CHECK_VENV/bin/pip" install --disable-pip-version-check --no-cache-dir --quiet --no-index --no-deps "$WHEEL"
mv "$REPO/vcpkg_installed" "$REPO/vcpkg_installed.hidden"
trap 'mv "$REPO/vcpkg_installed.hidden" "$REPO/vcpkg_installed" 2>/dev/null || true' EXIT
"$CHECK_VENV/bin/python" -I - "$REPO/resiliparse-rs/tests/extract_golden" "$OUT_DIR/contract_check.json" <<'PY'
import hashlib, json, pathlib, sys
from resiliparse._extract_rs import extract_plain_text
import resiliparse._extract_rs as module

golden_dir, out_path = pathlib.Path(sys.argv[1]), sys.argv[2]
smoke = extract_plain_text(
    "<html><body><article><h1>Smoke</h1><p>This paragraph exists so the startup smoke test can assert that "
    "the Rust extractor returns real main content rather than an empty string.</p></article></body></html>",
    main_content=True,
    preserve_formatting="markdown",
)
assert smoke.startswith("# Smoke\n"), smoke
report = {"module": module.__file__, "golden_basic": {}, "golden_markdown_sha256": {}}
for page in sorted(golden_dir.glob("*.html")):
    html = page.read_text(encoding="utf-8")
    expected = page.with_name(page.stem + ".expected.txt").read_text(encoding="utf-8")
    report["golden_basic"][page.stem] = extract_plain_text(html, main_content=True) == expected
    markdown = extract_plain_text(html, main_content=True, preserve_formatting="markdown")
    report["golden_markdown_sha256"][page.stem] = hashlib.sha256(markdown.encode()).hexdigest()
assert all(report["golden_basic"].values()), report
with open(out_path, "w") as f:
    json.dump(report, f, indent=2)
print(json.dumps(report, indent=2))
PY
echo "built $WHEEL"
