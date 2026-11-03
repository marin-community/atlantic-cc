#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
#
# Extract and classify a range of shards on Marin's Iris cluster, one TPU job per block of shards.
#
# Each job runs `atlantic-cc extract-classify` over its shards one after another. Iris retries a
# job that exits non-zero (a stalled phase A exits 70, a WARC that raised exits 1), and markers make
# every rerun resume where the last one stopped, so retries never redo finished WARCs.
#
# Usage (from the repository root, so the job bundles this checkout):
#   OUTPUT=gs://my-bucket/atlantic-cc/extract WARC_LIST=gs://my-bucket/v1.0.input.warc.paths.gz \
#   FIRST_SHARD=0 LAST_SHARD=99 scripts/launch/iris_tpu.sh
#
# Production's settings on a TPU v4-8 host (240 vCPUs): 14 phase-A processes x 6 extraction
# processes, queue depth 6, 98 CPUs and 350 GB. The defaults below are a smaller footprint; phase B
# on the TPU is the bottleneck either way (see docs/running.md).
set -euo pipefail

: "${OUTPUT:?output root, e.g. gs://bucket/prefix}"
: "${WARC_LIST:?paths file, e.g. v1.0.input.warc.paths.gz}"
: "${FIRST_SHARD:?first shard index}"
: "${LAST_SHARD:?last shard index (inclusive)}"
NUM_SHARDS="${NUM_SHARDS:-32000}"          # 32000 = production's base shards
SHARDS_PER_JOB="${SHARDS_PER_JOB:-10}"
TPU="${TPU:-v4-8}"                         # production used v4-8, v5p-8, v5litepod-8/16/32 and v6e-4/8/16/32
REGION="${REGION:-us-central2}"            # keep the job in the region of OUTPUT
PHASE_A_PROCS="${PHASE_A_PROCS:-8}"
CPUS="${CPUS:-56}"                         # PHASE_A_PROCS * (1 + 6 extraction processes)
MEMORY="${MEMORY:-200GB}"
EXTRACTOR="${EXTRACTOR:?resiliparse-rs artifact prefix (see docs/extraction.md)}"
IRIS="${IRIS:-iris}"
MODEL_ARGS=()
[[ -n "${FASTTEXT_MODEL:-}" ]] && MODEL_ARGS+=(--fasttext-model "${FASTTEXT_MODEL}")
[[ -n "${POOLED_MODEL:-}" ]] && MODEL_ARGS+=(--pooled-model "${POOLED_MODEL}")
[[ -n "${TERMINAL_MODEL:-}" ]] && MODEL_ARGS+=(--terminal-model "${TERMINAL_MODEL}")

for ((first = FIRST_SHARD; first <= LAST_SHARD; first += SHARDS_PER_JOB)); do
  last=$((first + SHARDS_PER_JOB - 1))
  ((last > LAST_SHARD)) && last=${LAST_SHARD}
  # The job's own loop; ${s} is expanded on the worker.
  script="for s in \$(seq ${first} ${last}); do
    python -m atlantic_cc.cli extract-classify --warc-list '${WARC_LIST}' --output '${OUTPUT}' \
      --shard-index \${s} --num-shards ${NUM_SHARDS} --backend jax \
      --phase-a-procs ${PHASE_A_PROCS} --extractor '${EXTRACTOR}' --cache-dir /root/.cache/atlantic_cc \
      ${MODEL_ARGS[*]} || exit \$?
  done"
  "${IRIS}" --cluster=marin job run \
    --tpu "${TPU}" --region "${REGION}" --cpu "${CPUS}" --memory "${MEMORY}" --enable-extra-resources \
    --extra tpu --extra gcs \
    --job-name "atlcc-extract-${TPU}-$(printf '%05d' "${first}")" --no-wait \
    -- bash -c "${script}"
done
