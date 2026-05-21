#!/bin/bash
# Generate images using DataDream LoRA saved under datadream_dir
# Usage: bash_run.sh <GPU_IDX> <SPLIT_IDX>

GPU="${1:-0}"
N_SET_SPLIT="${N_SET_SPLIT:-1}"
SPLIT_IDX="${2:-0}"

BS=8
NIPC=1000
# 如果你改成 SD="sdxl"，则会自动使用 SDXL 模型和对应的路径
SD="sd2.1"
GS=3.5

N_SHOT=20
N_TEMPLATE=1

MODE="datadream"
DD_LR=1e-4
DD_EP=240

DATASET="my_dataset"
# DATASET="my_dataset_binary"
FEWSHOT_SEED="seed0"
IS_DATASETWISE=False

echo "Generate config:"
echo "- dataset: $DATASET"
echo "- per-class images: $NIPC"
echo "- bs: $BS"
echo "- guidance_scale: $GS"
echo "- n_shot: $N_SHOT"
echo "- datadream_lr: $DD_LR"
echo "- datadream_epoch: $DD_EP"
echo ""

CUDA_VISIBLE_DEVICES=$GPU python generate.py \
  --bs=$BS \
  --n_img_per_class=$NIPC \
  --sd_version=$SD \
  --mode=$MODE \
  --guidance_scale=$GS \
  --n_shot=$N_SHOT \
  --n_template=$N_TEMPLATE \
  --dataset=$DATASET \
  --n_set_split=$N_SET_SPLIT \
  --split_idx=$SPLIT_IDX \
  --fewshot_seed=$FEWSHOT_SEED \
  --datadream_lr=$DD_LR \
  --datadream_epoch=$DD_EP \
  --is_dataset_wise_model=$IS_DATASETWISE \
  --is_tqdm=True