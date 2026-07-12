#!/usr/bin/env bash
set -euo pipefail

# 手部真实数据实验顶层入口，与原 04_classify.sh 的人脸/合成流程相互独立。
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "${ROOT_DIR}/classify"
bash run_hand_real.sh "$@"
