#!/bin/bash
# 方向 A：ROI 描述符辅助监督训练脚本
# 用法示例：ALPHA_ROI=0.1 SYNTH_VARIANT=qc bash run_roi_aux.sh


GPU="0"

# Classifier backbone
MODEL_TYPE="${MODEL_TYPE:-clip}"  # clip | resnet50

# Optional: override python executable (e.g. /root/autodl-tmp/conda_envs/myclassify/bin/python)
PYTHON_BIN="${PYTHON_BIN:-python}"

# Output root
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/runs/ablation/roi_aux_outputs}"

# MODE: both (pooled real+synth) | synth (only-synth)
MODE="${MODE:-both}"

# SYNTH_VARIANT: raw -> use .../train ; qc -> use .../filtered_train
SYNTH_VARIANT="${SYNTH_VARIANT:-raw}"

# Experiment name under OUTPUT_ROOT
ALPHA_ROI="${ALPHA_ROI:-0.1}"
EXP_NAME="${EXP_NAME:-${MODEL_TYPE}_roi_aux_alpha${ALPHA_ROI}_${SYNTH_VARIANT}_pool0.7_nipc500_lr1e-5_nomix}"
OUTPUT_DIR="${OUTPUT_ROOT}/${EXP_NAME}"

DATASET="my_dataset"

# -----------------------------
# Real data (fewshot pooled)
# -----------------------------
FEWSHOT_SEED="seed0"
# ?????
# REAL_MAL_GROUP_DIR="${REAL_MAL_GROUP_DIR:-/root/autodl-tmp/datadream/data/malnutrition/real_train_fewshot/${FEWSHOT_SEED}}"
# REAL_NOR_GROUP_DIR="${REAL_NOR_GROUP_DIR:-/root/autodl-tmp/datadream/data/normal_train_fewshot/${FEWSHOT_SEED}}"
REAL_MAL_GROUP_DIR="${REAL_MAL_GROUP_DIR:-/root/autodl-tmp/runs/cv/fold_4/real_train_groups/${FEWSHOT_SEED}}"
REAL_NOR_GROUP_DIR="${REAL_NOR_GROUP_DIR:-/root/autodl-tmp/runs/cv/fold_4/real_train_groups/${FEWSHOT_SEED}}"
# REAL_BINARY_ROOT="${REAL_BINARY_ROOT:-/root/autodl-tmp/datadream/data/my_dataset_binary}"
REAL_BINARY_ROOT="${REAL_BINARY_ROOT:-/root/autodl-tmp/runs/cv/fold_4/my_dataset_binary}"
REAL_BINARY_DIR="${REAL_BINARY_DIR:-${REAL_BINARY_ROOT}/${FEWSHOT_SEED}}"
# "/root/autodl-tmp/datadream/data/my_dataset_binary/seed0"

prepare_real_binary_dir() {
	mkdir -p "${REAL_BINARY_DIR}"
	for d in malnourished_front_face malnourished_left_three-quarter_face malnourished_right_three-quarter_face; do
		if [[ -e "${REAL_BINARY_DIR}/${d}" ]]; then rm -rf "${REAL_BINARY_DIR:?}/${d}"; fi
		ln -s "${REAL_MAL_GROUP_DIR}/${d}" "${REAL_BINARY_DIR}/${d}"
	done
	for d in normal_front_face normal_left_three-quarter_face normal_right_three-quarter_face; do
		if [[ -e "${REAL_BINARY_DIR}/${d}" ]]; then rm -rf "${REAL_BINARY_DIR:?}/${d}"; fi
		ln -s "${REAL_NOR_GROUP_DIR}/${d}" "${REAL_BINARY_DIR}/${d}"
	done

	TRAIN_DIR="${REAL_BINARY_DIR}" python - <<'PY'
import os
from pathlib import Path

train_dir = Path(os.environ['TRAIN_DIR']).resolve()
mal_groups = [
	'malnourished_front_face',
	'malnourished_left_three-quarter_face',
	'malnourished_right_three-quarter_face',
]
nor_groups = [
	'normal_front_face',
	'normal_left_three-quarter_face',
	'normal_right_three-quarter_face',
]
ext_ok = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

def reset_dir(p: Path):
	if p.is_symlink() or (p.exists() and not p.is_dir()):
		p.unlink()
	p.mkdir(parents=True, exist_ok=True)

def link_group(group: str, dst_dir: Path):
	src_dir = train_dir / group
	if not src_dir.is_dir():
		raise FileNotFoundError(f"Missing group dir: {src_dir}")
	for fp in src_dir.iterdir():
		if not fp.is_file() or fp.suffix.lower() not in ext_ok:
			continue
		dst = dst_dir / f"{group}__{fp.name}"
		if not dst.exists():
			dst.symlink_to(fp)

mal_dst = train_dir / 'malnourished_face'
nor_dst = train_dir / 'normal_face'
reset_dir(mal_dst)
reset_dir(nor_dst)
for g in mal_groups:
	link_group(g, mal_dst)
for g in nor_groups:
	link_group(g, nor_dst)

print(f"[OK] Prepared pooled real fewshot dir: {train_dir}")
print(f"  malnourished_face={len(list(mal_dst.iterdir()))}")
print(f"  normal_face={len(list(nor_dst.iterdir()))}")
PY
}

# -----------------------------
# Synth data (raw vs qc)
# -----------------------------
SYNTH_RAW_DIR="${SYNTH_RAW_DIR:-/root/autodl-tmp/datadream_outputs/generated_images/my_dataset/sd2.1/gs3.5_nis50/shot20_seed0_template1_lr0.0001_ep240/train}"
# SYNTH_RAW_DIR="${SYNTH_RAW_DIR:-/root/autodl-tmp/datadream_outputs/generated_images/my_dataset_binary/sd2.1/gs3.5_nis50/shot20_seed0_template1_lr0.0001_ep240/train}"
# SYNTH_RAW_DIR="${SYNTH_RAW_DIR:-/root/autodl-tmp/runs/cv/fold_4/synth_raw/my_dataset/sd2.1/gs3.5_nis50/shot20_seed0_template1_lr0.0001_ep240/train}"
SYNTH_QC_DIR="${SYNTH_QC_DIR:-/root/autodl-tmp/datadream_outputs/generated_images/my_dataset/sd2.1/gs3.5_nis50/shot20_seed0_template1_lr0.0001_ep240/filtered_train}"
# SYNTH_QC_DIR="${SYNTH_QC_DIR:-/root/autodl-tmp/datadream_outputs/generated_images/my_dataset_binary/sd2.1/gs3.5_nis50/shot20_seed0_template1_lr0.0001_ep240/filtered_train}"
# SYNTH_QC_DIR="${SYNTH_QC_DIR:-/root/autodl-tmp/runs/cv/fold_4/synth_raw/my_dataset/sd2.1/gs3.5_nis50/shot20_seed0_template1_lr0.0001_ep240/filtered_train}"

if [[ "${SYNTH_VARIANT}" == "qc" ]]; then
	SYNTH_TRAIN_DIR="${SYNTH_QC_DIR}"
else
	SYNTH_TRAIN_DIR="${SYNTH_RAW_DIR}"
fi

ensure_binary_links() {
	local train_dir="$1"
	TRAIN_DIR="${train_dir}" python - <<'PY'
import os
from pathlib import Path

train_dir = Path(os.environ['TRAIN_DIR']).resolve()

mal_groups = [
	'malnourished_front_face',
	'malnourished_left_three-quarter_face',
	'malnourished_right_three-quarter_face',
]
nor_groups = [
	'normal_front_face',
	'normal_left_three-quarter_face',
	'normal_right_three-quarter_face',
]

ext_ok = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

def reset_dir(p: Path):
	if p.is_symlink() or (p.exists() and not p.is_dir()):
		p.unlink()
	p.mkdir(parents=True, exist_ok=True)

def link_group(group: str, dst_dir: Path):
	src_dir = train_dir / group
	if not src_dir.is_dir():
		raise FileNotFoundError(f"Missing group dir: {src_dir}")
	for fp in src_dir.iterdir():
		if not fp.is_file() or fp.suffix.lower() not in ext_ok:
			continue
		dst = dst_dir / f"{group}__{fp.name}"
		if not dst.exists():
			dst.symlink_to(fp)

mal_dst = train_dir / 'malnourished_face'
nor_dst = train_dir / 'normal_face'
reset_dir(mal_dst)
reset_dir(nor_dst)
for g in mal_groups:
	link_group(g, mal_dst)
for g in nor_groups:
	link_group(g, nor_dst)

print(f"[OK] Prepared synth binary dirs in: {train_dir}")
print(f"  malnourished_face={len(list(mal_dst.iterdir()))}")
print(f"  normal_face={len(list(nor_dst.iterdir()))}")
PY
}

NIPC=${NIPC:-500}
LR=${LR:-1e-5}
MIN_LR=1e-5
WD=1e-4
EPOCH=40
WARMUP_EPOCH=4
IS_MIX_AUG=${IS_MIX_AUG:-TRUE}

N_SHOT=20
N_TEMPLATE=1

IS_SYNTH_TRAIN=True
IS_DATASET_WISE=False
DD_LR=1e-4
DD_EP=240
DD_TTE=True

if [[ "${MODE}" == "synth" ]]; then
	IS_POOLED=FALSE
else
	IS_POOLED=TRUE
fi

LAMBDA_1=${LAMBDA_1:-0.8}

# Prepare directories needed by dataloaders
prepare_real_binary_dir
ensure_binary_links "${SYNTH_TRAIN_DIR}"

echo "[INFO] MODEL_TYPE=${MODEL_TYPE}"
echo "[INFO] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[INFO] ALPHA_ROI=${ALPHA_ROI}"

#export EVAL_ONLY=1 SKIP_CKPT=1
SYNTH_RAW_DIR="${SYNTH_RAW_DIR}" SYNTH_QC_DIR="${SYNTH_QC_DIR}" CUDA_VISIBLE_DEVICES=$GPU WANDB_MODE=disabled "$PYTHON_BIN" main.py \
--model_type=$MODEL_TYPE \
--output_dir=$OUTPUT_DIR \
--n_img_per_cls=$NIPC \
--is_lora_image=True \
--is_lora_text=True \
--is_synth_train=True \
--synth_train_data_dir_override="${SYNTH_TRAIN_DIR}" \
--sd_version="sd2.1" \
--n_template=$N_TEMPLATE \
--guidance_scale=3.5 \
--is_pooled_fewshot=$IS_POOLED \
--lambda_1=$LAMBDA_1 \
--epochs=$EPOCH \
--warmup_epochs=$WARMUP_EPOCH \
--wandb_project=datadream \
--dataset=$DATASET \
--n_shot=$N_SHOT \
--lr=$LR \
--wd=$WD \
--min_lr=$MIN_LR \
--fewshot_seed=$FEWSHOT_SEED \
--is_mix_aug=$IS_MIX_AUG \
--is_dataset_wise=$IS_DATASET_WISE \
--datadream_lr=$DD_LR \
--datadream_epoch=$DD_EP \
--datadream_train_text_encoder=$DD_TTE \
--use_roi_aux_head=True \
--alpha_roi=$ALPHA_ROI \
--roi_descriptor_cache_path=/root/autodl-tmp/runs/roi_descriptor_cache.json \
--mediapipe_model_path=/root/autodl-tmp/face_landmarker.task \
${PARAM:-}
