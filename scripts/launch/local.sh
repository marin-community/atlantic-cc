#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
#
# Extract and classify a WARC list on one machine (a CPU box, one GPU, or one TPU VM).
#
#   OUTPUT=out WARC_LIST=my.warc.paths EXTRACTOR=/path/to/resiliparse_rs BACKEND=torch scripts/launch/local.sh
#
# Runs every shard of NUM_SHARDS in turn and restarts a shard when phase A stalls (exit 70); a
# rerun of the script resumes from the markers under OUTPUT. A shard where some WARC raised (exit 1)
# is reported and the loop moves on: run the script again to retry those WARCs (each WARC gets three
# starts before it is recorded as abandoned).
set -uo pipefail

: "${OUTPUT:?output root (local directory, gs:// or s3://)}"
: "${WARC_LIST:?paths file}"
: "${EXTRACTOR:?resiliparse-rs artifact prefix, or install the extension as a package and drop --extractor}"
BACKEND="${BACKEND:-torch}"                 # torch (CPU or GPU) or jax (TPU, GPU or CPU)
NUM_SHARDS="${NUM_SHARDS:-1}"
PHASE_A_PROCS="${PHASE_A_PROCS:-2}"
EXTRA="${EXTRA:-${BACKEND}}"                # uv extra: torch, jax, or tpu on a TPU VM

failed=0
for ((shard = 0; shard < NUM_SHARDS; shard++)); do
  while true; do
    uv run --extra "${EXTRA}" atlantic-cc extract-classify \
      --warc-list "${WARC_LIST}" --output "${OUTPUT}" \
      --shard-index "${shard}" --num-shards "${NUM_SHARDS}" \
      --backend "${BACKEND}" --phase-a-procs "${PHASE_A_PROCS}" --extractor "${EXTRACTOR}"
    status=$?
    [[ ${status} -eq 70 ]] && { echo "shard ${shard}: phase A stalled; restarting" >&2; continue; }
    [[ ${status} -ne 0 ]] && { echo "shard ${shard}: exit ${status}; rerun to retry" >&2; failed=1; }
    break
  done
done
exit "${failed}"
