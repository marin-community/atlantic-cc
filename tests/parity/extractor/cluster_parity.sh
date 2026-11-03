#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
#
# Iris job body (default task image, --extra gcs): extractor parity on fixtures F1 and F3 and install
# checks of the wheels next to upstream resiliparse.
#
#   cluster_parity.sh GCS_ROOT JOBS
#
# Reads GCS_ROOT/wheels/asrun/*.whl, GCS_ROOT/build/{debian,manylinux}/*.whl and production's artifact
# (gs://marin-us-central2/artifacts/resiliparse_rs/latest/); writes report.json, per-page results, the F1
# golden, install checks and auditwheel output to GCS_ROOT/parity/. Variants:
#   asrun            as-run wheel in a fresh venv (the reference)
#   prod-layout      production's unpacked artifact first on sys.path, as Marin's workers loaded it
#   src-debian       source build in the default task image (production's environment)
#   src-manylinux    source build in quay.io/pypa/manylinux_2_28_x86_64
#   asrun-up109      as-run wheel installed before resiliparse 1.0.9, whose Cython html2text is imported first
#   asrun-up200      resiliparse 2.0.0 installed before the as-run wheel (its native module loads on import)
set -euo pipefail

GCS_ROOT="$1"
JOBS="$2"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
BASE="/root/a7_parity"  # not /tmp: noexec
IN="$BASE/in"
RUN="$BASE/run"
UPLOAD="$BASE/upload"
mkdir -p "$IN" "$RUN" "$UPLOAD"
PYTHON="$(command -v python)"

"$PYTHON" - "$GCS_ROOT" "$IN" <<'PY'
import sys
import fsspec
root, dest = sys.argv[1], sys.argv[2]
fs = fsspec.filesystem("gs")
sources = {
    "asrun": f"{root}/wheels/asrun/*.whl",
    "debian": f"{root}/build/debian/*.whl",
    "manylinux": f"{root}/build/manylinux/*.whl",
    "prod": "gs://marin-us-central2/artifacts/resiliparse_rs/latest/*",
}
for name, pattern in sources.items():
    for path in fs.glob(pattern):
        fs.get(path, f"{dest}/{name}/{path.rsplit('/', 1)[-1]}")
        print("fetched", path)
PY

make_venv() {  # name, then pip install argument lists separated by "--"
  local venv="$BASE/venvs/$1"
  shift
  rm -rf "$venv"
  "$PYTHON" -m venv "$venv"
  local args=()
  for arg in "$@" --; do
    if [ "$arg" = "--" ]; then
      [ "${#args[@]}" -gt 0 ] && "$venv/bin/pip" install --quiet --disable-pip-version-check "${args[@]}"
      args=()
    else
      args+=("$arg")
    fi
  done
}
ASRUN_WHEEL="$(ls "$IN"/asrun/*.whl)"
make_venv asrun "$ASRUN_WHEEL"
make_venv src-debian "$(ls "$IN"/debian/*.whl)"
make_venv src-manylinux "$(ls "$IN"/manylinux/*.whl)"
make_venv asrun-up109 "$ASRUN_WHEEL" -- "resiliparse==1.0.9"
make_venv asrun-up200 "resiliparse==2.0.0" -- "$ASRUN_WHEEL"

rm -rf "$BASE/prod"
mkdir -p "$BASE/prod"
tar -xzf "$IN/prod/resiliparse_py.tar.gz" -C "$BASE/prod"
cp "$IN/prod/liblexbor.so.2" "$IN/prod/_extract_rs.so" "$BASE/prod/resiliparse-py/resiliparse/"

# --- Installing next to upstream resiliparse, in both orders, and uninstalling either one -----------------
probe() {  # step label -> one JSON line describing what imports in the coexist venv
  "$BASE/venvs/coexist/bin/python" -I - "$1" <<'PY'
import hashlib, importlib, importlib.metadata, json, sys
html = "<html><body><article><h1>Title</h1><p>Some text that is long enough to be main content.</p></article></body></html>"
out = {"step": sys.argv[1], "dists": sorted(d.metadata["Name"] for d in importlib.metadata.distributions())}
for name in ("resiliparse.extract.html2text", "resiliparse._extract_rs"):
    try:
        module = importlib.import_module(name)
        text = module.extract_plain_text(html, main_content=True)
        # resiliparse 2.x registers html2text from its native module, without a __file__.
        out[name] = {"file": getattr(module, "__file__", None), "text_sha256": hashlib.sha256(text.encode()).hexdigest()[:16]}
    except Exception as e:
        out[name] = f"{type(e).__name__}: {e}"
print(json.dumps(out))
PY
}
COEXIST="$UPLOAD/coexistence.jsonl"
: > "$COEXIST"
for upstream in 1.0.9 2.0.0; do
  PIP="$BASE/venvs/coexist/bin/pip"
  make_venv coexist "$ASRUN_WHEEL" -- "resiliparse==$upstream"
  probe "upstream $upstream installed after ours" >> "$COEXIST"
  "$PIP" uninstall --quiet -y resiliparse
  probe "upstream $upstream uninstalled" >> "$COEXIST"
  "$PIP" install --quiet --disable-pip-version-check "resiliparse==$upstream"
  "$PIP" uninstall --quiet -y resiliparse-rs-atlantic-cc
  probe "ours uninstalled, upstream $upstream kept" >> "$COEXIST"
  "$PIP" install --quiet --disable-pip-version-check "$ASRUN_WHEEL"
  probe "ours installed after upstream $upstream" >> "$COEXIST"
done
cat "$COEXIST"

make_venv auditwheel "auditwheel==6.8.2"
for wheel in "$ASRUN_WHEEL" "$IN"/debian/*.whl "$IN"/manylinux/*.whl; do
  echo "== $(basename "$wheel")"
  "$BASE/venvs/auditwheel/bin/auditwheel" show "$wheel" || true
done > "$UPLOAD/auditwheel.txt" 2>&1
cat "$UPLOAD/auditwheel.txt"

# --- Parity -----------------------------------------------------------------------------------------------
status=0
(cd "$REPO_ROOT" && "$PYTHON" -m tests.parity.extractor.run_parity --work-dir "$RUN" --fixture F1 --fixture F3 \
  --variant "asrun=$BASE/venvs/asrun/bin/python" \
  --variant "prod-layout=$PYTHON" --package-dir "prod-layout=$BASE/prod/resiliparse-py" \
  --variant "src-debian=$BASE/venvs/src-debian/bin/python" \
  --variant "src-manylinux=$BASE/venvs/src-manylinux/bin/python" \
  --variant "asrun-up109=$BASE/venvs/asrun-up109/bin/python" --preload "asrun-up109=resiliparse.extract.html2text" \
  --variant "asrun-up200=$BASE/venvs/asrun-up200/bin/python" \
  --reference asrun --jobs "$JOBS" --write-golden "F1=$UPLOAD/golden_f1_asrun.tsv.gz") 2>&1 | tee "$UPLOAD/run_parity.log" \
  || status=$?
cp "$RUN"/report.json "$RUN"/results_*.jsonl.gz "$RUN"/index_*.json.gz "$UPLOAD/"
(cd "$RUN" && tar -czf "$UPLOAD/worker_stderr.tar.gz" ./*.stderr)
"$PYTHON" "$HERE/gcs_upload.py" "$UPLOAD" "$GCS_ROOT/parity"
exit "$status"
