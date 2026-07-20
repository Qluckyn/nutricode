#!/usr/bin/env bash
# fold_1--fold_4 的 C2-MD 可恢复流水线：两条无 ControlNet I2I 路线依次执行。
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# 生图/ROI-DINO 使用 DataDream 环境；分类继续使用既有 myclassify 环境。
GENERATION_PYTHON="${GENERATION_PYTHON:-/root/autodl-tmp/miniconda3/envs/datadream/bin/python}"
CLASSIFY_PYTHON="${CLASSIFY_PYTHON:-/root/autodl-tmp/miniconda3/envs/myclassify/bin/python}"
WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp/runs/hand_synth_expansion_qc/c2_md_5fold_seed22}"
DATA_ROOT="/root/autodl-tmp/data_hand/pose01_pose02_5fold_seed22"
LANDMARKER="/root/autodl-tmp/models/mediapipe/hand_landmarker.task"
DINO="/root/autodl-tmp/models/hand_synthesis_v2/dinov2_small"
GPU="${GPU:-0}"

run_classifier() {
  local fold="$1" route_root="$2" condition="$3"
  local train_dir="${route_root}/classifier_data_c2_md_separable/${condition}/train"
  local output_dir="${route_root}/classifier_runs_c2_md_separable_textlora/${condition}/seed22"
  mkdir -p "${output_dir}"
  # C1/C2 每轮各使用真实 12 张、合成 36 张；合成池均固定为每类 90 张。
  (
    cd "${ROOT_DIR}/classify"
    CUDA_VISIBLE_DEVICES="${GPU}" WANDB_MODE=disabled "${CLASSIFY_PYTHON}" main.py \
      --model_type=clip --dataset=hand_nutrition \
      --real_train_data_dir_override="${train_dir}" \
      --real_test_data_dir_override="${DATA_ROOT}/fold_${fold}/test" \
      --output_dir="${output_dir}" --n_img_per_cls=None \
      --is_synth_train=False --is_pooled_fewshot=False \
      --is_lora_image=True --is_lora_text=True \
      --use_roi_aux_head=False --is_mix_aug=False \
      --use_hand_transforms=True --hand_transform_profile=v2c_deterministic \
      --hand_pose=02 --select_best_on_test=False \
      --batch_size=8 --batch_size_eval=8 --epochs=40 --warmup_epochs=4 \
      --lr=1e-5 --min_lr=1e-6 --wd=1e-4 --log=none --seed=22 \
      --is_hand_subject_balanced=False --is_hand_pose02_v3_mixed_sampling=True \
      --hand_pose02_v3_real_per_class=12 --hand_pose02_v3_synth_per_class=36 \
      --hand_pose02_v3_synth_pool_per_class=90 \
      --hand_pose02_max_synth_per_parent_per_epoch=4
  ) 2>&1 | tee "${output_dir}/train.log"
}

for fold in 1 2 3 4; do
  for route in datadream_i2i_no_cn foundhand_datadream_i2i_no_cn; do
    route_root="${WORK_ROOT}/fold_${fold}/${route}"
    config="${route_root}/config_c2_md_separable.yaml"
    mode="op_i2i"
    [[ "${route}" == "foundhand_datadream_i2i_no_cn" ]] && mode="foundhand_i2i"
    log="${route_root}/pipeline.log"

    # 原 90 张候选由软链接复用；生成器自动跳过它们，仅生成新增候选。
    CUDA_VISIBLE_DEVICES="${GPU}" "${GENERATION_PYTHON}" "${ROOT_DIR}/hand_synth_expansion_qc/generate_no_controlnet_i2i.py" \
      --config "${config}" --mode "${mode}" --plan "${route_root}/candidate_plan_c2_md.jsonl" 2>&1 | tee -a "${log}"
    "${GENERATION_PYTHON}" "${ROOT_DIR}/hand_synth_expansion_qc/audit_small_ablation.py" \
      --config "${config}" --mode "${mode}" --compute-dhash 2>&1 | tee -a "${log}"
    "${GENERATION_PYTHON}" "${ROOT_DIR}/hand_synth_expansion_qc/prepare_qc_feature_manifest_v2.py" \
      --config "${config}" 2>&1 | tee -a "${log}"

    # ROI、PCA 和马氏距离模型仅由本折真实训练图拟合，筛选阶段不读取测试集。
    CUDA_VISIBLE_DEVICES="${GPU}" "${GENERATION_PYTHON}" "${ROOT_DIR}/hand_synth_expansion_qc/run_c2_roi_feature_scoring.py" \
      --manifest "${route_root}/qc/feature_input_manifest.jsonl" --landmarker "${LANDMARKER}" \
      --dinov2 "${DINO}" --output "${route_root}/qc/roi_feature_scores_v10_separable.jsonl" 2>&1 | tee -a "${log}"
    CUDA_VISIBLE_DEVICES="${GPU}" "${GENERATION_PYTHON}" "${ROOT_DIR}/hand_synth_expansion_qc/run_c2_md_filter.py" \
      --config "${config}" --feature-manifest "${route_root}/qc/feature_input_manifest.jsonl" \
      --structure-scores "${route_root}/qc/roi_feature_scores_v10_separable.jsonl" \
      --output "${route_root}/qc/qc_manifest_c2_md_separable.jsonl" \
      --model-output "${route_root}/qc/c2_md_separable_model.json" \
      --class-separability-mode soft --soft-separability-weight 0.25 2>&1 | tee -a "${log}"
    "${GENERATION_PYTHON}" "${ROOT_DIR}/hand_synth_expansion_qc/select_matched_c1_c2_v2.py" \
      --config "${config}" --qc-manifest "${route_root}/qc/qc_manifest_c2_md_separable.jsonl" 2>&1 | tee -a "${log}"

    selection="${route_root}/selection_c1_c2_md_separable/matched_c1_c2_selection.json"
    for condition in c1_raw c2_qc; do
      "${GENERATION_PYTHON}" "${ROOT_DIR}/hand_synth_expansion_qc/build_classifier_data.py" \
        --config "${config}" --condition "${condition}" --selection-manifest "${selection}" 2>&1 | tee -a "${log}"
      run_classifier "${fold}" "${route_root}" "${condition}" 2>&1 | tee -a "${log}"
    done
  done
done
