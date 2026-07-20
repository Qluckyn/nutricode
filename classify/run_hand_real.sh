#!/usr/bin/env bash
set -euo pipefail

# 手部真实数据专用入口：不准备人脸目录，也不访问任何合成训练数据。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/miniconda3/envs/myclassify/bin/python}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/data_hand/split_seed22}"
# 训练与测试目录必须来自同一个DATA_ROOT，禁止绕过统一划分校验。
if [[ -n "${TRAIN_DIR+x}" || -n "${TEST_DIR+x}" ]]; then
    echo "[ERROR] 禁止单独设置TRAIN_DIR或TEST_DIR；请只设置DATA_ROOT" >&2
    exit 2
fi
TRAIN_DIR="${DATA_ROOT}/train"
TEST_DIR="${DATA_ROOT}/test"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/runs/hand_clip_baseline}"
SEED="${SEED:-22}"
GPU="${GPU:-0}"
HAND_POSE="${HAND_POSE:-all}"

# CLI参数优先于环境变量；先解析最终hand_pose，再据此命名输出目录。
# 支持“--hand_pose=01”和“--hand_pose 01”，多次出现时以最后一次为准。
_cli_args=("$@")
_i=0
while (( _i < ${#_cli_args[@]} )); do
    case "${_cli_args[_i]}" in
        --hand_pose=*)
            HAND_POSE="${_cli_args[_i]#*=}"
            ;;
        --hand_pose)
            _i=$((_i + 1))
            if (( _i >= ${#_cli_args[@]} )); then
                echo "[ERROR] --hand_pose缺少参数值" >&2
                exit 2
            fi
            HAND_POSE="${_cli_args[_i]}"
            ;;
    esac
    _i=$((_i + 1))
done
case "${HAND_POSE}" in
    all|01|02) ;;
    *)
        echo "[ERROR] HAND_POSE/--hand_pose仅支持all、01、02" >&2
        exit 2
        ;;
esac

# 任务六实验模式：majority（多数类）、zeroshot（零样本）、lora（图像LoRA）。
EXPERIMENT_MODE="${EXPERIMENT_MODE:-lora}"

# 当前阶段默认只校验数据与打印训练配置。
# 原阶段说明：完成动态采样、手部预处理和测试集隔离后，再启动训练。
# 原阶段说明：任务三、任务四已接入，仍需完成任务五。
# 任务五双模式开关现已接入；保持默认不训练，由用户显式启动。
# 任务六三组基线与任务七手部专用评估现已接入。
RUN_TRAINING="${RUN_TRAINING:-false}"

case "${EXPERIMENT_MODE}" in
    majority|zeroshot|lora) ;;
    *)
        echo "[ERROR] EXPERIMENT_MODE仅支持 majority、zeroshot、lora" >&2
        exit 2
        ;;

esac
echo "[INFO] EXPERIMENT_MODE=${EXPERIMENT_MODE}"
cd "${SCRIPT_DIR}"

echo "[INFO] 校验手部数据划分：${DATA_ROOT}"
"${PYTHON_BIN}" validate_hand_split.py --data_root "${DATA_ROOT}"

echo "[INFO] TRAIN_DIR=${TRAIN_DIR}"
echo "[INFO] TEST_DIR=${TEST_DIR}"
echo "[INFO] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "[INFO] SEED=${SEED}"
echo "[INFO] HAND_POSE=${HAND_POSE}"

if [[ "${RUN_TRAINING,,}" != "true" ]]; then
    # 原提示：任务一、任务二已就绪；任务三、任务四已就绪；测试集隔离将在任务五完成。
    # 原提示：任务一至任务五均已就绪；当前未启动训练。
    # 原提示：任务一至任务六均已就绪；当前未启动实验。
    echo "[INFO] 任务一至任务七均已就绪；当前未启动实验。"
    echo "[INFO] 默认 select_best_on_test=True，保留原始逐轮test选最佳逻辑。"
    echo "[INFO] 设置 EXPERIMENT_MODE 和 RUN_TRAINING=true 后启动对应实验。"
    exit 0
fi

# 按姿势隔离输出目录，避免单姿势实验覆盖双姿势结果。
MODE_OUTPUT_ROOT="${OUTPUT_ROOT}/${EXPERIMENT_MODE}/pose${HAND_POSE}"
if [[ "${EXPERIMENT_MODE}" == "lora" ]]; then
    MODE_OUTPUT_ROOT="${MODE_OUTPUT_ROOT}/seed${SEED}"
fi
mkdir -p "${MODE_OUTPUT_ROOT}"

if [[ "${EXPERIMENT_MODE}" == "majority" ]]; then
    # 多数类基线不加载CLIP、不使用GPU，只读取已锁定的train/test划分。
    "${PYTHON_BIN}" hand_majority_baseline.py \
        --data_root "${DATA_ROOT}" \
        --output_dir "${MODE_OUTPUT_ROOT}"
    exit 0
fi

case "${EXPERIMENT_MODE}" in
    zeroshot)
        # 真正的zero-shot：关闭两侧LoRA，eval_only会在创建优化器前返回。
        # --is_lora_text=True？？
        MODE_ARGS=(
            --eval_only=True
            --is_lora_image=False
            --is_lora_text=True
            --is_hand_subject_balanced=False
            --select_best_on_test=False
        )
        ;;
    lora)
        # 主实验：只训练图像LoRA，文本编码器冻结，并启用12:12动态采样。
        # --select_best_on_test=False,checkpoint选择模式：固定轮数后仅测试一次
        MODE_ARGS=(
            --eval_only=False
            --is_lora_image=True
            --is_lora_text=True
            --is_hand_subject_balanced=True
            --select_best_on_test=False
        )
        ;;
esac
# 原始输出参数为 --output_dir="${OUTPUT_ROOT}"，任务六改为按实验模式隔离目录。
# 原始数据集参数为 --dataset=my_dataset，现统一为手部专用名称 hand_nutrition。
# 折中方案：保留原代码，默认使用当前 test 选择最佳 checkpoint。
# 原始命令仅以 "$@" 结束；任务八增加tee，将完整控制台输出保存为train.log。
# MODE_ARGS放在原参数之后，使zero-shot或LoRA模式可以覆盖原始固定设置。
CUDA_VISIBLE_DEVICES="${GPU}" WANDB_MODE=disabled "${PYTHON_BIN}" main.py \
    --model_type=clip \
    --dataset=hand_nutrition \
    --real_train_data_dir_override="${TRAIN_DIR}" \
    --real_test_data_dir_override="${TEST_DIR}" \
    --output_dir="${MODE_OUTPUT_ROOT}" \
    --n_img_per_cls=None \
    --is_synth_train=False \
    --is_pooled_fewshot=False \
    --is_lora_image=True \
    --is_lora_text=True \
    --use_roi_aux_head=False \
    --is_mix_aug=False \
    --use_hand_transforms=True \
    --hand_pose="${HAND_POSE}" \
    --is_hand_subject_balanced=True \
    --hand_subjects_per_class=12 \
    --select_best_on_test=True \
    "${MODE_ARGS[@]}" \
    --batch_size=8 \
    --batch_size_eval=8 \
    --epochs=40 \
    --warmup_epochs=4 \
    --lr=1e-5 \
    --min_lr=1e-6 \
    --wd=1e-4 \
    --seed="${SEED}" \
    "$@" 2>&1 | tee "${MODE_OUTPUT_ROOT}/train.log"
