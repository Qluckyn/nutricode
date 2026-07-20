#!/usr/bin/env bash
# fold_0 软类别可分性 C2-MD 重跑：保留旧硬门结果，输出至独立 soft 目录。
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GEN_PYTHON="${GEN_PYTHON:-/root/autodl-tmp/miniconda3/envs/datadream/bin/python}"
CLASSIFY_PYTHON="${CLASSIFY_PYTHON:-/root/autodl-tmp/miniconda3/envs/myclassify/bin/python}"
ROOT="/root/autodl-tmp/runs/hand_synth_expansion_qc/c2_qc_scaleup/fold_0"
DATA_ROOT="/root/autodl-tmp/data_hand/pose01_pose02_5fold_seed22"
GPU="${GPU:-0}"

run_classifier() {
  local route_root="$1" condition="$2"
  local train_dir="${route_root}/classifier_data_c2_md_soft/${condition}/train"
  local output_dir="${route_root}/classifier_runs_c2_md_soft_textlora/${condition}/seed22"
  mkdir -p "${output_dir}"
  (
    cd "${ROOT_DIR}/classify"
    CUDA_VISIBLE_DEVICES="${GPU}" WANDB_MODE=disabled "${CLASSIFY_PYTHON}" main.py \
      --model_type=clip --dataset=hand_nutrition \
      --real_train_data_dir_override="${train_dir}" --real_test_data_dir_override="${DATA_ROOT}/fold_0/test" \
      --output_dir="${output_dir}" --n_img_per_cls=None --is_synth_train=False --is_pooled_fewshot=False \
      --is_lora_image=True --is_lora_text=True --use_roi_aux_head=False --is_mix_aug=False \
      --use_hand_transforms=True --hand_transform_profile=v2c_deterministic --hand_pose=02 \
      --select_best_on_test=False --batch_size=8 --batch_size_eval=8 --epochs=40 --warmup_epochs=4 \
      --lr=1e-5 --min_lr=1e-6 --wd=1e-4 --log=none --seed=22 \
      --is_hand_subject_balanced=False --is_hand_pose02_v3_mixed_sampling=True \
      --hand_pose02_v3_real_per_class=12 --hand_pose02_v3_synth_per_class=36 \
      --hand_pose02_v3_synth_pool_per_class=90 --hand_pose02_max_synth_per_parent_per_epoch=4
  ) 2>&1 | tee "${output_dir}/train.log"
}

for route in datadream_i2i_no_cn foundhand_datadream_i2i_no_cn; do
  route_root="${ROOT}/${route}"
  config="${route_root}/config_c2_md_soft.yaml"
  score="${route_root}/qc/roi_feature_scores_v10.jsonl"
  [[ "${route}" == "foundhand_datadream_i2i_no_cn" ]] && score="${route_root}/qc/roi_feature_scores_v10_separable.jsonl"
  log="${route_root}/c2_md_soft_fold0.log"
  CUDA_VISIBLE_DEVICES="${GPU}" "${GEN_PYTHON}" "${ROOT_DIR}/hand_synth_expansion_qc/run_c2_md_filter.py" \
    --config "${config}" --feature-manifest "${route_root}/qc/feature_input_manifest.jsonl" \
    --structure-scores "${score}" --output "${route_root}/qc/qc_manifest_c2_md_soft.jsonl" \
    --model-output "${route_root}/qc/c2_md_soft_model.json" \
    --class-separability-mode soft --soft-separability-weight 0.25 2>&1 | tee -a "${log}"
  "${GEN_PYTHON}" "${ROOT_DIR}/hand_synth_expansion_qc/select_matched_c1_c2_v2.py" \
    --config "${config}" --qc-manifest "${route_root}/qc/qc_manifest_c2_md_soft.jsonl" 2>&1 | tee -a "${log}"
  selection="${route_root}/selection_c1_c2_md_soft/matched_c1_c2_selection.json"
  for condition in c1_raw c2_qc; do
    "${GEN_PYTHON}" "${ROOT_DIR}/hand_synth_expansion_qc/build_classifier_data.py" \
      --config "${config}" --condition "${condition}" --selection-manifest "${selection}" 2>&1 | tee -a "${log}"
    run_classifier "${route_root}" "${condition}" 2>&1 | tee -a "${log}"
  done
done
