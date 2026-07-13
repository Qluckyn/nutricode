#!/usr/bin/env bash
set -euo pipefail

# 手部 LoRA 独立入口：不复用、不修改面部 bash_run.sh 的类别、数据和输出目录。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# 手部入口默认使用已安装 Diffusers 的独立 datadream 环境，仍允许显式覆盖。
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/miniconda3/envs/datadream/bin/python}"
GPU="${GPU:-0}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/data_hand/lora_train_audited/fold_0}"
MANIFEST="${MANIFEST:-${DATA_ROOT}/manifest.json}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-/root/autodl-tmp/models/AI-ModelScope/stable-diffusion-2-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/runs/hand_sd_lora/fold_0/seed22}"
PREVIEW_ROOT="${PREVIEW_ROOT:-${OUTPUT_ROOT}/preprocessed_previews}"

# CLASS_IDX=all 训练四类；也可指定 0、1、2、3 做单类训练或冒烟测试。
CLASS_IDX="${CLASS_IDX:-all}"
RUN_TRAINING="${RUN_TRAINING:-false}"
N_SHOT="${N_SHOT:-12}"
EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-10}"
SEED="${SEED:-22}"
TRAIN_TEXT_ENCODER="${TRAIN_TEXT_ENCODER:-false}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-0}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-500}"

CLASS_NAMES=(
  "malnourished_hand_pose01"
  "malnourished_hand_pose02"
  "normal_hand_pose01"
  "normal_hand_pose02"
)

if [[ "${CLASS_IDX}" == "all" ]]; then
  CLASS_IDXS=(0 1 2 3)
elif [[ "${CLASS_IDX}" =~ ^[0-3]$ ]]; then
  CLASS_IDXS=("${CLASS_IDX}")
else
  echo "[ERROR] CLASS_IDX 仅支持 all、0、1、2、3" >&2
  exit 2
fi

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "[ERROR] 缺少手部四类数据目录：${DATA_ROOT}" >&2
  exit 2
fi
if [[ ! -f "${MANIFEST}" ]]; then
  echo "[ERROR] 缺少阶段 A manifest：${MANIFEST}" >&2
  exit 2
fi
if [[ ! -d "${PRETRAINED_MODEL}" ]]; then
  echo "[ERROR] 缺少 Stable Diffusion 模型：${PRETRAINED_MODEL}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}/logs" "${PREVIEW_ROOT}"

echo "[INFO] 手部 LoRA 配置"
echo "[INFO] DATA_ROOT=${DATA_ROOT}"
echo "[INFO] MANIFEST=${MANIFEST}"
echo "[INFO] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "[INFO] CLASS_IDX=${CLASS_IDX}"
echo "[INFO] EPOCHS=${EPOCHS}, N_SHOT=${N_SHOT}, SEED=${SEED}"
echo "[INFO] TRAIN_TEXT_ENCODER=${TRAIN_TEXT_ENCODER}"

if [[ "${RUN_TRAINING,,}" != "true" ]]; then
  echo "[INFO] 当前仅检查配置，未启动 GPU 训练。"
  echo "[INFO] 设置 RUN_TRAINING=true 后训练；单类冒烟示例：CLASS_IDX=0 EPOCHS=1 RUN_TRAINING=true bash run_hand_lora.sh"
  exit 0
fi

for class_idx in "${CLASS_IDXS[@]}"; do
  class_name="${CLASS_NAMES[$class_idx]}"
  class_preview_dir="${PREVIEW_ROOT}/${class_name}"
  log_path="${OUTPUT_ROOT}/logs/${class_name}.log"

  echo "[INFO] 开始训练 [${class_idx}] ${class_name}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -m accelerate.commands.launch \
    --num_processes=1 \
    --num_machines=1 \
    --mixed_precision=fp16 \
    --dynamo_backend=no \
    datadream.py \
      --pretrained_model_name_or_path="${PRETRAINED_MODEL}" \
      --dataset=hand_nutrition \
      --fewshot_data_dir_override="${DATA_ROOT}" \
      --hand_dataset_manifest="${MANIFEST}" \
      --preview_preprocessed_dir="${class_preview_dir}" \
      --preview_preprocessed_count=4 \
      --n_template=1 \
      --fewshot_seed="seed${SEED}" \
      --train_batch_size="${BATCH_SIZE}" \
      --gradient_accumulation_steps="${GRAD_ACCUM}" \
      --learning_rate="${LEARNING_RATE}" \
      --lr_scheduler=cosine \
      --lr_warmup_steps="${WARMUP_STEPS}" \
      --num_train_epochs="${EPOCHS}" \
      --report_to=tensorboard \
      --train_text_encoder="${TRAIN_TEXT_ENCODER}" \
      --is_tqdm=True \
      --output_dir="${OUTPUT_ROOT}" \
      --n_shot="${N_SHOT}" \
      --target_class_idx="${class_idx}" \
      --resume_from_checkpoint=None \
      --resolution=512 \
      --mixed_precision=fp16 \
      --checkpointing_steps="${CHECKPOINTING_STEPS}" \
      --dataloader_num_workers="${DATALOADER_WORKERS}" \
      --seed="${SEED}" \
      "${@}" 2>&1 | tee "${log_path}"
done

echo "[OK] 手部 LoRA 训练完成：${OUTPUT_ROOT}"
