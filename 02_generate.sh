#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU="${1:-0}"
SPLIT_IDX="${2:-0}"

# 优先使用 datadream 环境进行生图
export PATH="/root/autodl-tmp/conda_envs/datadream/bin:${PATH}"

cd "${ROOT_DIR}/generate"

# 运行原始生成脚本（不改动源码）
bash bash_run.sh "${GPU}" "${SPLIT_IDX}"
