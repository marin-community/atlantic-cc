#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
#
# Extract and classify on a Slurm GPU cluster: one array task per shard, PyTorch backend.
#
# Submit from the repository root, after creating the log directory (older Slurm versions do not):
#
#   mkdir -p logs
#   export OUTPUT=/scratch/atlantic-cc WARC_LIST=v1.0.input.warc.paths.gz NUM_SHARDS=1000
#   sbatch --array=0-999%64 scripts/launch/slurm_gpu.sbatch
#
# Required: OUTPUT and WARC_LIST. Optional: NUM_SHARDS (default 1000; the array covers 0 to
# NUM_SHARDS-1), PHASE_A_PROCS, EXTRACTOR (a production-layout resiliparse-rs artifact; unset uses
# the wheel that the extractor extra installs), CACHE_DIR (passed as --cache-dir), FASTTEXT_MODEL, POOLED_MODEL and
# TERMINAL_MODEL. With a hand-installed torch (Pascal or Volta GPUs), also export UV_NO_SYNC=1 so
# that uv keeps it (docs/running.md#gpus).
#
# HOME: by default the cache directory (~/.cache/atlantic_cc: the unpacked EXTRACTOR and copies of
# gs:// or s3:// model folders), the Hugging Face cache (~/.cache/huggingface: 1.4 GB of models) and
# Triton's kernel cache (~/.triton) live under $HOME. Where home is small or a network volume, export
# CACHE_DIR, HF_HOME and TRITON_CACHE_DIR on scratch space; CACHE_DIR must allow executable mappings.
#
# Every task resumes from its markers, so a requeued or rerun task skips the WARCs it finished.
# Exit status 70 means phase A stalled (a hung extraction); the task requeues itself. Exit status 1
# means a WARC failed or the setup is broken; the end of the log says which. GPU scores agree with
# production statistically, not bit for bit (docs/reproducibility.md).
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
NUM_SHARDS="${NUM_SHARDS:-1000}"
PHASE_A_PROCS="${PHASE_A_PROCS:-4}"      # each uses 1 + 6 CPUs and ~6 GB of RAM

status=0
uv run --extra torch --extra extractor atlantic-cc extract-classify \
  --warc-list "${WARC_LIST}" --output "${OUTPUT}" \
  --shard-index "${SLURM_ARRAY_TASK_ID}" --num-shards "${NUM_SHARDS}" \
  --backend torch --torch-device cuda --phase-a-procs "${PHASE_A_PROCS}" \
  ${EXTRACTOR:+--extractor "${EXTRACTOR}"} ${CACHE_DIR:+--cache-dir "${CACHE_DIR}"} \
  ${FASTTEXT_MODEL:+--fasttext-model "${FASTTEXT_MODEL}"} ${POOLED_MODEL:+--pooled-model "${POOLED_MODEL}"} \
  ${TERMINAL_MODEL:+--terminal-model "${TERMINAL_MODEL}"} || status=$?
if [[ ${status} -eq 70 ]]; then
  scontrol requeue "${SLURM_JOB_ID}"
fi
exit "${status}"
