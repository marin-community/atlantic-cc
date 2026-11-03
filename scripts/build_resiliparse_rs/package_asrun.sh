#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
#
# Repackage the resiliparse-rs binaries production loaded (the artifact Marin published at
# gs://marin-<region>/artifacts/resiliparse_rs/latest/) as a wheel, byte for byte. Maintainers only.
#
#   package_asrun.sh ARTIFACT_DIR OUT_DIR
#
# ARTIFACT_DIR holds _extract_rs.so and liblexbor.so.2; their md5s must equal
# atlantic_cc.spec.ASRUN_EXTRACTOR_ARTIFACT. Needs git, curl, cargo (only for `cargo tree`, to list the
# Rust crates linked into the extension) and uv (runs package_wheel.py in this repository's environment).
set -euo pipefail

ARTIFACT_DIR="$(cd "$1" && pwd)"
mkdir -p "$2"
OUT_DIR="$(cd "$2" && pwd)"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
FORK_URL="https://github.com/marin-community/atlantic-cc-resiliparse.git"
FORK_COMMIT="850891b277e919f671ec19f12aafdce451b327f4"
LEXBOR_URL="https://github.com/phoerious/lexbor/archive/647a0b5.tar.gz"
LEXBOR_SHA512="a8d7c71a197edab5446bf757bfda4653cb839d69c59e0e6dc80ddf5512c94103e3c54c6a52ffc4de882dea3c7901b1103e7b5c835b80403feee48c0c9ef48d83"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/package_asrun.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

git init -q "$WORK/fork"
git -C "$WORK/fork" fetch -q --depth 1 "$FORK_URL" "$FORK_COMMIT"
git -C "$WORK/fork" -c advice.detachedHead=false checkout -q FETCH_HEAD
{
  echo "Rust crates statically linked into _extract_rs (cargo tree; license as declared by each crate)."
  echo "The Rust standard library (MIT OR Apache-2.0) is linked as well."
  (cd "$WORK/fork" && cargo tree --locked -p resiliparse-extract-rs -e normal,no-proc-macro --prefix none \
    --format '{p} | {l}' --target x86_64-unknown-linux-gnu) | sed -e 's/ (\*)$//' -e 's/ (\/[^)]*)//' | sort -u
} > "$WORK/THIRD_PARTY_RUST_CRATES.txt"

curl -fsSL --retry 3 -o "$WORK/lexbor.tar.gz" "$LEXBOR_URL"
uv run --project "$REPO_ROOT" python -c 'import hashlib, sys; sys.exit(hashlib.sha512(open(sys.argv[1], "rb").read()).hexdigest() != sys.argv[2])' \
  "$WORK/lexbor.tar.gz" "$LEXBOR_SHA512"
mkdir "$WORK/lexbor"
tar -xzf "$WORK/lexbor.tar.gz" -C "$WORK/lexbor" --strip-components 1

# Build facts read from the binaries' .comment sections and from the artifact's manifest.json.
cat > "$WORK/build_facts.json" <<'JSON'
{
  "source": "gs://marin-us-central2/artifacts/resiliparse_rs/latest/ (same md5s in every Marin region)",
  "built_at": "2026-08-26T22:25:01Z (_extract_rs.so); liblexbor.so.2 is byte-identical to the 2026-08-11 build",
  "host": "Iris CPU task, python:3.12-slim (Debian 13), Linux-x86_64 glibc 2.41",
  "python": "3.12.14",
  "rustc": "1.98.0 (88d9e12ae 2026-08-18), linker rust-lld (LLD 22.1.8)",
  "rustflags": "-C link-arg=-Wl,-rpath,$ORIGIN",
  "cc": "gcc (Debian 14.2.0-19) 14.2.0",
  "lexbor": "phoerious/lexbor@647a0b5 via the fork's vcpkg overlay port, -O3, shared",
  "runpath": "$ORIGIN:/root/resiliparse_rs_build/chatnoir-resiliparse/vcpkg_installed/x64-linux/lib"
}
JSON

md5s="$(uv run --project "$REPO_ROOT" python - <<'PY'
from atlantic_cc import spec

artifact = spec.ASRUN_EXTRACTOR_ARTIFACT
print(" ".join(f"--expect-md5 {name}={artifact[name]['md5_b64']}" for name in ("_extract_rs.so", "liblexbor.so.2")))
PY
)"
# shellcheck disable=SC2086  # md5s is a list of flags
uv run --project "$REPO_ROOT" python "$HERE/package_wheel.py" --kind as-run \
  --extension "$ARTIFACT_DIR/_extract_rs.so" --library "liblexbor.so.2=$ARTIFACT_DIR/liblexbor.so.2" $md5s \
  --license "LICENSE=$WORK/fork/LICENSE" --license "lexbor/LICENSE=$WORK/lexbor/LICENSE" \
  --license "lexbor/NOTICE=$WORK/lexbor/NOTICE" --license "THIRD_PARTY_RUST_CRATES.txt=$WORK/THIRD_PARTY_RUST_CRATES.txt" \
  --build-facts "$WORK/build_facts.json" --out-dir "$OUT_DIR"
