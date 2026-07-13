#!/usr/bin/env bash
set -euo pipefail

# 阶段 D 手部专用入口：与面部 QC/分类产物完全隔离。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/miniconda3/envs/myclassify/bin/python}"
INPUT_ROOT="${INPUT_ROOT:-/root/autodl-tmp/runs/hand_generated/stage_c_batch_nipc25/fold_0/seed22/hand_nutrition/sd2.1/gs3.5_nis30/shot12_seed22_template1_lr0.0001_ep40_notextlora}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/runs/hand_synth_qc/stage_d_nipc25/fold_0/seed22}"
REAL_MANIFEST="${REAL_MANIFEST:-/root/autodl-tmp/data_hand/lora_train_audited/fold_0/manifest.json}"
HAND_MODEL="${HAND_MODEL:-/root/autodl-tmp/models/mediapipe/hand_landmarker.task}"

"${PYTHON_BIN}" hand_quality_control.py \
  --input-root "${INPUT_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --real-manifest "${REAL_MANIFEST}" \
  --hand-model "${HAND_MODEL}" \
  --clip-model "ViT-B/16" \
  --seed 22 \
  --blind-sample-per-class 10 \
  "$@"

echo "[OK] 手部阶段 D 自动 QC 与盲审材料已输出：${OUTPUT_ROOT}"
