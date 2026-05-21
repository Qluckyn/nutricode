#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 优先使用 datadream 环境进行过滤
# PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda_envs/datadream/bin/python}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/miniconda3/envs/datadream/bin/python}"

# 默认路径与 qc_filter.py 常量一致
# REAL_ROOTS_DEFAULT=(
#   "/root/autodl-tmp/datadream/data/malnutrition/real_train_fewshot/seed0"
#   "/root/autodl-tmp/datadream/data/normal_train_fewshot/seed0"
# )
# REAL_ROOT_DEFAULT="/root/autodl-tmp/datadream/data/my_dataset_binary/seed0"
REAL_ROOT_DEFAULT="/root/autodl-tmp/runs/cv/fold_4/real_train_groups/seed0"

SYNTH_ROOT_DEFAULT="/root/autodl-tmp/datadream_outputs/generated_images/my_dataset/sd2.1/gs3.5_nis50/shot20_seed0_template1_lr0.0001_ep240/train"
# SYNTH_ROOT_DEFAULT="/root/autodl-tmp/datadream_outputs/generated_images/my_dataset_binary/sd2.1/gs3.5_nis50/shot20_seed0_template1_lr0.0001_ep240/train"
OUT_DIR_DEFAULT="/root/autodl-tmp/datadream_outputs/generated_images/my_dataset/sd2.1/gs3.5_nis50/shot20_seed0_template1_lr0.0001_ep240/filtered_train"
# OUT_DIR_DEFAULT="/root/autodl-tmp/datadream_outputs/generated_images/my_dataset_binary/sd2.1/gs3.5_nis50/shot20_seed0_template1_lr0.0001_ep240/filtered_train"

# REAL_ROOTS=("${REAL_ROOTS_DEFAULT[@]}")
# 注意 --real-roots 只传一个路径，而不是两个
REAL_ROOT="${REAL_ROOT:-${REAL_ROOT_DEFAULT}}"
SYNTH_ROOT="${SYNTH_ROOT:-${SYNTH_ROOT_DEFAULT}}"
OUT_DIR="${OUT_DIR:-${OUT_DIR_DEFAULT}}"
TAU_QUANTILE="${TAU_QUANTILE:-95}"
COV_SHRINKAGE="${COV_SHRINKAGE:-ledoit_wolf}"
COV_REG_EPS="${COV_REG_EPS:-1e-6}"
K_BETA="${K_BETA:-8.0}"
K_ABS="${K_ABS:-}"
ONLY_GROUPS=(
  "normal_front_face"
  "normal_left_three-quarter_face"
  "normal_right_three-quarter_face"
  "malnourished_front_face"
  "malnourished_left_three-quarter_face"
  "malnourished_right_three-quarter_face"
)
# ONLY_GROUPS=(
#   "normal_face"
#   "malnourished_face"
# )

cd "${ROOT_DIR}/passing"

FILTER_ARGS=(
  # --real-roots "${REAL_ROOTS[@]}"
  # 注意 --real-roots 只传一个路径，而不是两个
  --real-roots "${REAL_ROOT}"
  --synthetic-root "${SYNTH_ROOT}"
  --output-dir "${OUT_DIR}"
  --tau-quantile "${TAU_QUANTILE}"
  --cov-shrinkage "${COV_SHRINKAGE}"
  --cov-reg-eps "${COV_REG_EPS}"
  --k-beta "${K_BETA}"
  --only-groups "${ONLY_GROUPS[@]}"
)

if [[ -n "${K_ABS}" ]]; then
  FILTER_ARGS+=(--k-abs "${K_ABS}")
fi

# 运行原始过滤脚本（不改动源码）
"${PYTHON_BIN}" qc_filter.py "${FILTER_ARGS[@]}"
