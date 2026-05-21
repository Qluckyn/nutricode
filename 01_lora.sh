#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU="${1:-0}"
SPLIT_IDX="${2:-0}"

# 优先使用 datadream 环境进行 LoRA 训练
export PATH="/root/autodl-tmp/conda_envs/datadream/bin:${PATH}"

cd "${ROOT_DIR}/sd_lora"

# 运行原始训练脚本（不改动源码）
bash bash_run.sh "${GPU}" "${SPLIT_IDX}"
