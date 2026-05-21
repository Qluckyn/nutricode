#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 优先使用 myclassify 环境进行训练
# PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda_envs/myclassify/bin/python}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/miniconda3/envs/myclassify/bin/python}"

MODEL_TYPE="${MODEL_TYPE:-clip}"
SYNTH_VARIANT="${SYNTH_VARIANT:-raw}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/runs/ablation/classify_outputs}"

cd "${ROOT_DIR}/classify"

# 运行原始训练脚本（不改动源码）
MODEL_TYPE="${MODEL_TYPE}" \
SYNTH_VARIANT="${SYNTH_VARIANT}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
PYTHON_BIN="${PYTHON_BIN}" \
  bash run_both_real_and_synth.sh
