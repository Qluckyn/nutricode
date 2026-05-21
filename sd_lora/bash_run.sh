#!/bin/bash
# Train LoRA per-class for my_dataset (normal_* in SUBSET_NAMES['my_dataset'])
# Usage: bash bash_run.sh <GPU_IDX> <SPLIT_IDX>
# Example: bash bash_run.sh 0 0

GPU="${1:-0}"
SET_SPLIT=1
SPLIT_IDX="${2:-0}"

### ------------------
### Parameters
### ------------------
DATASET="my_dataset"
# DATASET="my_dataset_binary"
N_CLS=6
# N_CLS=2
FEWSHOT_SEED="seed0"
N_SHOT=20
NUM_TRAIN_EPOCH=240

OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/datadream_outputs/models}"

# Make sure this matches SUBSET_NAMES['my_dataset'] in util_data.py
CLASS_NAMES=(
  'normal_front_face'
  'normal_left_three-quarter_face'
  'normal_right_three-quarter_face'
  'malnourished_front_face'
  'malnourished_left_three-quarter_face'
  'malnourished_right_three-quarter_face'
)
# CLASS_NAMES=(
#   'normal_face'
#   'malnourished_face'
# )
    
# Local model path (explicitly set to your downloaded SD2.1)
PRETRAINED_MODEL="/root/autodl-tmp/models/AI-ModelScope/stable-diffusion-2-1"

### ------------------
### Calculate CLASS_IDXS
### ------------------
START_RANGE=$(( (N_CLS / SET_SPLIT) * SPLIT_IDX ))
END_RANGE=$(( (N_CLS / SET_SPLIT) * (SPLIT_IDX + 1) - 1 ))
if [ $SPLIT_IDX -eq $((SET_SPLIT - 1)) ]; then
    FINAL_END_RANGE=$((N_CLS - 1))
else
    FINAL_END_RANGE=$END_RANGE
fi
CLASS_IDXS=($(seq $START_RANGE $FINAL_END_RANGE))

echo "Training class indices: ${CLASS_IDXS[@]}"
echo "Output root: $OUTPUT_ROOT"
echo "Few-shot root (seed will be appended by datadream): $(awk '/fewshot_data_dir/{p=1;next}p && /my_dataset/{print; exit}' /root/autodl-tmp/NutriPro/Datapro/sd_lora/local.yaml 2>/dev/null || true)"
echo ""

### ------------------
### Run (per-class)
### ------------------
for CLASS_IDX in "${CLASS_IDXS[@]}"; do
  CLASSNAME="${CLASS_NAMES[$CLASS_IDX]}"
  echo "=========================================="
  echo "Start training [${CLASS_IDX}]: ${CLASSNAME}"
  echo "Output root (base): ${OUTPUT_ROOT}"
  echo "=========================================="

  mkdir -p "$OUTPUT_ROOT"

  # CUDA_VISIBLE_DEVICES=$GPU accelerate launch \
  CUDA_VISIBLE_DEVICES=$GPU python -m accelerate.commands.launch \
    --num_processes=1 \
    --num_machines=1 \
    --mixed_precision=fp16 \
    datadream.py \
      --pretrained_model_name_or_path="$PRETRAINED_MODEL" \
      --dataset="$DATASET" \
      --n_template=1 \
      --fewshot_seed="$FEWSHOT_SEED" \
      --train_batch_size=8 \
      --gradient_accumulation_steps=1 \
      --learning_rate=1e-4 \
      --lr_scheduler="cosine" \
      --lr_warmup_steps=100 \
      --num_train_epochs="$NUM_TRAIN_EPOCH" \
      --report_to="tensorboard" \
      --train_text_encoder=True \
      --is_tqdm=True \
      --output_dir="$OUTPUT_ROOT" \
      --n_shot="$N_SHOT" \
      --target_class_idx="$CLASS_IDX" \
      --resume_from_checkpoint=None \
      --resolution=512 \
      --mixed_precision="fp16" \
      --checkpointing_steps=500 \
      --dataloader_num_workers=4 \
      --seed=42

  if [ $? -ne 0 ]; then
    echo "❌ Training failed for ${CLASSNAME}"
    exit 1
  else
    echo "✅ Done: ${CLASSNAME}"
  fi
done

echo "All training finished. Example LoRA path will be:"
echo "${OUTPUT_ROOT}/<dataset>/shot${N_SHOT}_${FEWSHOT_SEED}_tpl1/lr0.0001_epoch${NUM_TRAIN_EPOCH}/<classname>/pytorch_lora_weights.safetensors"