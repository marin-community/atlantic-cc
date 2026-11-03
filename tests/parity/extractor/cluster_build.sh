#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
#
# Iris job body: build resiliparse-rs from source with scripts/build_resiliparse_rs/build.sh and
# upload the outputs (wheel, production-layout artifact, build info, logs) to GCS.
#
#   cluster_build.sh debian    GCS_PREFIX   # default Iris task image (python:3.12-slim, Debian 13, glibc 2.41)
#   cluster_build.sh manylinux GCS_PREFIX   # --task-image quay.io/pypa/manylinux_2_28_x86_64 --no-sync
#
# debian reproduces production's environment: same image family, the image's rustup, Iris's
# CARGO_HOME (/cargo) and production's work dir (/root/resiliparse_rs_build), so paths embedded in
# the binaries match the as-run build. manylinux works under /app instead: Iris drops all
# capabilities and AlmaLinux's /root is mode 0550, so root cannot write there. /tmp is noexec.
set -euo pipefail

FLAVOR="$1"
GCS_PREFIX="$2"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"

case "$FLAVOR" in
  debian)
    BASE="/root"
    PYTHON="$(command -v python)"  # the task venv's CPython 3.12, as production
    ;;
  manylinux)
    BASE="/app/a7"
    export HOME="$BASE/home" CARGO_HOME="$BASE/home/.cargo" RUSTUP_HOME="$BASE/home/.rustup"
    mkdir -p "$HOME"
    bash "$REPO_ROOT/scripts/build_resiliparse_rs/install_manylinux_deps.sh"
    export PATH="$CARGO_HOME/bin:$PATH"
    PYTHON="/opt/python/cp312-cp312/bin/python3.12"
    ;;
  *) echo "unknown flavor $FLAVOR" >&2; exit 2 ;;
esac

OUT_DIR="$BASE/a7_out/$FLAVOR"
LOG="$BASE/a7_out/build-$FLAVOR.log"
mkdir -p "$OUT_DIR"
status=0
bash "$REPO_ROOT/scripts/build_resiliparse_rs/build.sh" --python "$PYTHON" --work-dir "$BASE/resiliparse_rs_build" \
  --out-dir "$OUT_DIR" 2>&1 | tee "$LOG" || status=$?
cp "$LOG" "$OUT_DIR/"
"$PYTHON" "$HERE/gcs_upload.py" "$OUT_DIR" "$GCS_PREFIX"
exit "$status"
