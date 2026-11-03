#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
#
# Extract and classify on a Slurm GPU cluster: one array task per shard, PyTorch backend.
#
#   sbatch --array=0-999%64 scripts/launch/slurm_gpu.sbatch
#
# with OUTPUT, WARC_LIST and EXTRACTOR exported (and optionally NUM_SHARDS, the model sources and
# PHASE_A_PROCS). Every task resumes from its markers, so a requeued or rerun task skips the WARCs it
# finished. Exit status 70 means phase A stalled (a hung extraction); the task requeues itself.
# GPU scores agree with production statistically, not bit for bit (docs/reproducibility.md).
#SBATCH --job-name=atlcc-extract
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail

: "${OUTPUT:?output root (local, gs:// or s3://)}"
: "${WARC_LIST:?paths file, e.g. v1.0.input.warc.paths.gz}"
: "${EXTRACTOR:?resiliparse-rs artifact prefix, or install the extension as a package and drop --extractor}"
NUM_SHARDS="${NUM_SHARDS:-1000}"
PHASE_A_PROCS="${PHASE_A_PROCS:-4}"      # each uses 1 + 6 CPUs and ~6 GB of RAM
MODEL_ARGS=()
[[ -n "${FASTTEXT_MODEL:-}" ]] && MODEL_ARGS+=(--fasttext-model "${FASTTEXT_MODEL}")
[[ -n "${POOLED_MODEL:-}" ]] && MODEL_ARGS+=(--pooled-model "${POOLED_MODEL}")
[[ -n "${TERMINAL_MODEL:-}" ]] && MODEL_ARGS+=(--terminal-model "${TERMINAL_MODEL}")

status=0
uv run --extra torch atlantic-cc extract-classify \
  --warc-list "${WARC_LIST}" --output "${OUTPUT}" \
  --shard-index "${SLURM_ARRAY_TASK_ID}" --num-shards "${NUM_SHARDS}" \
  --backend torch --torch-device cuda \
  --phase-a-procs "${PHASE_A_PROCS}" --extractor "${EXTRACTOR}" \
  "${MODEL_ARGS[@]}" || status=$?
if [[ ${status} -eq 70 ]]; then
  scontrol requeue "${SLURM_JOB_ID}"
fi
exit "${status}"
