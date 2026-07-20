#!/usr/bin/env bash
# 训练 V4 提示词对齐版 pose02 类别 LoRA；输出目录与历史 LoRA 权重隔离。
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/miniconda3/envs/datadream/bin/python}"
GPU="${GPU:-0}"
DATA_ROOT="/root/autodl-tmp/runs/hand_synth_expansion_qc/fold_0_pose02/lora_data"
OUTPUT_ROOT="/root/autodl-tmp/runs/hand_synth_expansion_qc/fold_0_pose02/lora_weights_v4_prompt_aligned"
MODEL_ROOT="/root/autodl-tmp/models/AI-ModelScope/stable-diffusion-2-1"

cd "${ROOT_DIR}/sd_lora"
for class_idx in 0 1; do
  if [[ "${class_idx}" == "0" ]]; then
    class_name="malnourished_hand_pose02"
  else
    class_name="normal_hand_pose02"
  fi
  run_root="${OUTPUT_ROOT}/${class_name}"
  mkdir -p "${run_root}/logs" "${run_root}/previews"
  echo "[pose02-LoRA] class=${class_name}, seed=22"

  # 文本编码器冻结，仅训练 UNet LoRA；训练文字取自已更新的 HAND_LORA_PROMPTS。
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -m accelerate.commands.launch \
    --num_processes=1 --num_machines=1 --mixed_precision=fp16 --dynamo_backend=no \
    datadream.py \
      --pretrained_model_name_or_path="${MODEL_ROOT}" \
      --dataset=hand_nutrition_pose02 \
      --fewshot_data_dir_override="${DATA_ROOT}" \
      --hand_dataset_manifest="${DATA_ROOT}/hand_lora_manifest.json" \
      --preview_preprocessed_dir="${run_root}/previews" \
      --preview_preprocessed_count=4 \
      --n_template=1 --fewshot_seed=seed22 \
      --train_batch_size=2 --gradient_accumulation_steps=4 \
      --learning_rate=1e-4 --lr_scheduler=cosine --lr_warmup_steps=10 \
      --num_train_epochs=40 --report_to=tensorboard --train_text_encoder=false \
      --is_tqdm=True --output_dir="${run_root}" --n_shot=12 \
      --target_class_idx="${class_idx}" --resume_from_checkpoint=None \
      --resolution=512 --rank=16 --mixed_precision=fp16 \
      --checkpointing_steps=500 --dataloader_num_workers=0 --seed=22 \
      2>&1 | tee "${run_root}/logs/train.log"
done
