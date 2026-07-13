#!/usr/bin/env bash
set -euo pipefail

# 手部生成独立入口：不修改面部 bash_run.sh，不复用面部输出目录。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/miniconda3/envs/datadream/bin/python}"
GPU="${GPU:-0}"
MODEL_DIR="${MODEL_DIR:-/root/autodl-tmp/models/AI-ModelScope/stable-diffusion-2-1}"
LORA_ROOT="${LORA_ROOT:-/root/autodl-tmp/runs/hand_sd_lora/fold_0/seed22}"
SAVE_ROOT="${SAVE_ROOT:-/root/autodl-tmp/runs/hand_generated/fold_0/seed22}"

RUN_GENERATION="${RUN_GENERATION:-false}"
NIPC="${NIPC:-25}"
BS="${BS:-1}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-3.5}"
INFERENCE_STEPS="${INFERENCE_STEPS:-30}"
SEED="${SEED:-22000}"
COUNT_START="${COUNT_START:-0}"

N_SHOT=12
N_TEMPLATE=1
DD_LR=1e-4
DD_EPOCH=40
FEWSHOT_SEED=seed22

CLASS_NAMES=(
  "malnourished_hand_pose01"
  "malnourished_hand_pose02"
  "normal_hand_pose01"
  "normal_hand_pose02"
)
LORA_MID="hand_nutrition/shot12_seed22_tpl1_notextlora/lr0.0001_epoch40"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[ERROR] Python 环境不可执行：${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "[ERROR] Stable Diffusion 模型不存在：${MODEL_DIR}" >&2
  exit 2
fi
for class_name in "${CLASS_NAMES[@]}"; do
  weight="${LORA_ROOT}/${LORA_MID}/${class_name}/pytorch_lora_weights.safetensors"
  if [[ ! -f "${weight}" ]]; then
    echo "[ERROR] 缺少手部 LoRA 权重：${weight}" >&2
    exit 2
  fi
done

echo "[INFO] 手部生成配置"
echo "[INFO] LORA_ROOT=${LORA_ROOT}"
echo "[INFO] SAVE_ROOT=${SAVE_ROOT}"
echo "[INFO] NIPC=${NIPC}, BS=${BS}, GS=${GUIDANCE_SCALE}, STEPS=${INFERENCE_STEPS}"
echo "[INFO] SEED=${SEED}, COUNT_START=${COUNT_START}"

if [[ "${RUN_GENERATION,,}" != "true" ]]; then
  echo "[INFO] 四套权重检查通过，当前未启动生成。"
  echo "[INFO] 少量验收示例：NIPC=2 RUN_GENERATION=true bash run_hand_generate.sh"
  exit 0
fi

mkdir -p "${SAVE_ROOT}"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" generate.py \
  --bs="${BS}" \
  --n_img_per_class="${NIPC}" \
  --sd_version=sd2.1 \
  --mode=datadream \
  --guidance_scale="${GUIDANCE_SCALE}" \
  --num_inference_steps="${INFERENCE_STEPS}" \
  --seed="${SEED}" \
  --count_start="${COUNT_START}" \
  --n_shot="${N_SHOT}" \
  --n_template="${N_TEMPLATE}" \
  --dataset=hand_nutrition \
  --n_set_split=1 \
  --split_idx=0 \
  --fewshot_seed="${FEWSHOT_SEED}" \
  --datadream_lr="${DD_LR}" \
  --datadream_epoch="${DD_EPOCH}" \
  --datadream_train_text_encoder=False \
  --is_dataset_wise_model=False \
  --save_dir="${SAVE_ROOT}" \
  --datadream_dir="${LORA_ROOT}" \
  --model_dir="${MODEL_DIR}" \
  --is_tqdm=True \
  "$@"

echo "[OK] 手部少量生成完成：${SAVE_ROOT}"
