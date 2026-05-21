# NutriCode
<!-- 再次修改测试 -->
目录：
1) LoRA 训练
2) 生图
3) 过滤（QC）
4) 分类训练

一般的执行流程是:
1、sd_lora
先针对不同的数据集,自己划分类,并在util_data文件下进行类的添加，如
    'my_dataset': [
         'normal_front_face',
         'normal_left_three-quarter_face',
         'normal_right_three-quarter_face',
         'malnourished_front_face',
         'malnourished_left_three-quarter_face',
         'malnourished_right_three-quarter_face',
     ],
同时修改对应脚本 在bash_run文件中  进行如下修改
DATASET="my_dataset"
N_CLS=6

# Make sure this matches SUBSET_NAMES['my_dataset'] in util_data.py
CLASS_NAMES=(
  'normal_front_face'
  'normal_left_three-quarter_face'
  'normal_right_three-quarter_face'
  'malnourished_front_face'
  'malnourished_left_three-quarter_face'
  'malnourished_right_three-quarter_face'
)
2、训练完成lora后，进行sd2.1生图
注意 在进行生图前，也需要修改generate目录下的util.py与bash_run中
你所需要的数据类（与lora中的修改一致）
3、接下来 进行passing
事实上这个passing是硬编码 里面涉及的landmark是根据mediapipe进行手动框选后一个一个写上去的
所以如果不进行营养不良面部相关研究 请忽略
4、进行classify
注意 在进行生图前，也需要修改classify目录下的util.py与run_both_real_and_synth.sh中
你所需要的数据类（按需修改 可以不一致）
以及根据需要，选择是否开启pool（真实+合成）等
learning rate一般在1e-6 - 1e-5之间比较好
Lamda 一般在0.65-0.8之间比较好
 

## 目录结构
- sd_lora/     LoRA 训练（DataDream）
- generate/    合成图像生成（DataDream）
- passing/     过滤（Mahalanobis + top-k）
- classify/    分类训练（CLIP/ResNet）
- requirements.txt（从当前环境合并生成）
- requirements-lora-gen.txt（datadream 环境）
- requirements-classify.txt（myclassify 环境）
- 01_lora.sh / 02_generate.sh / 03_filter.sh / 04_classify.sh

## 快速开始

### 1) LoRA 训练
```bash
# bash 01_lora.sh 0 0
cd sd_lora
bash bash_run.sh
```

### 2) 生图
```bash
# bash 02_generate.sh 0 0
cd generate
bash bash_run.sh
```

### 3) 过滤
```bash
bash 03_filter.sh
```

### 4) 分类训练
```bash
bash 04_classify.sh
```

## 环境说明
- LoRA / 生图 / 过滤：datadream 环境
  - /root/autodl-tmp/conda_envs/datadream
- 分类训练：myclassify 环境
  - /root/autodl-tmp/conda_envs/myclassify



### 环境用途说明
- datadream：LoRA 训练、Stable Diffusion 生图、QC 过滤（依赖 diffusers、mediapipe、insightface 等）
- myclassify：分类训练（CLIP / ResNet）

### requirements 文件
- requirements.txt：把两个环境用到的 Python 包合并在一起，方便你一次性了解“整体依赖”。
- requirements-lora-gen.txt：只包含 LoRA 训练 + 生图 + 过滤需要的包，对应 datadream 环境。
- requirements-classify.txt：只包含分类训练需要的包，对应 myclassify 环境。

如果你不确定装哪个：
- 只跑 LoRA/生图/过滤：装 requirements-lora-gen.txt
- 只跑分类：装 requirements-classify.txt
- 想全流程跑：装 requirements.txt

安装示例（在对应环境里执行）：
```bash
pip install -r requirements-lora-gen.txt
```

## 说明
- sd_lora/、generate/、passing/、classify/ 下的代码为 NutriPro/Datapro 的原始拷贝。
- 不包含数据与模型权重，路径默认沿用你当前工程结构（datadream、datadream_outputs、models）。
- 过滤步骤已包含 face_landmarker.task。

## YHQ的开发环境
- GPU：NVIDIA GeForce RTX 5090（显存 32607 MiB）
- 驱动与 CUDA（nvidia-smi）：Driver 595.58.03 / CUDA 13.2
- CUDA Toolkit（nvcc）：未安装（本机 nvcc 不存在）
- Torch 栈锁定版本：
  - torch==2.10.0.dev20251026+cu128
  - torchvision==0.25.0.dev20251026+cu128
  - torchaudio==2.10.0.dev20251026+cu128

如果你在别的机器上跑，请以你自己的 GPU/驱动/CUDA 版本为准，必要时重建环境并调整依赖。

## Wrapper 脚本参数
各脚本支持常见环境变量覆盖：

- 01_lora.sh
  - 调用 sd_lora/bash_run.sh
  - SD2.1 按类训练 LoRA
  - 常见参数：
    - OUTPUT_ROOT：LoRA 权重输出根目录
    - PRETRAINED_MODEL：SD 2.1 模型路径
    - N_SHOT：每类 few-shot 数量
    - NUM_TRAIN_EPOCH：训练轮数

- 02_generate.sh
  - 调用 generate/bash_run.sh
  - DataDream 生图（启用 LoRA）
  - 常见参数：
    - NIPC：每类生成数量
    - GS：guidance scale
    - N_SHOT / N_TEMPLATE：few-shot 配置
    - DD_LR / DD_EP：LoRA 学习率与 epoch

- 03_filter.sh
  - 调用 passing/qc_filter.py
  - 默认路径与 qc_filter.py 常量一致
  - 常见参数：
    - REAL_ROOTS / SYNTH_ROOT / OUT_DIR：数据路径
    - TAU_QUANTILE：阈值分位数（默认 95）
    - COV_SHRINKAGE：协方差收缩方式（ledoit_wolf / oas / ridge）
    - K_BETA 或 K_ABS：Top-K 策略
    - ONLY_GROUPS：运行的组名列表

- 04_classify.sh
  - 调用 classify/run_both_real_and_synth.sh
  - 默认模型为 CLIP
  - 常见参数：
    - MODEL_TYPE：clip 或 resnet50
    - SYNTH_VARIANT：raw 或 qc
    - OUTPUT_ROOT：输出根目录

## 数据与模型路径约定
此目录不包含数据和权重，请确保以下路径存在或按需修改脚本：
- 真实数据（few-shot）：
  - /root/autodl-tmp/datadream/data/malnutrition/real_train_fewshot/seed0
  - /root/autodl-tmp/datadream/data/normal_train_fewshot/seed0
- 合成数据（raw / filtered）：
  - /root/autodl-tmp/datadream_outputs/generated_images/...
- SD 模型权重：
  - /root/autodl-tmp/models/AI-ModelScope/stable-diffusion-2-1

## face_landmarker.task 说明
- 已拷贝到 passing/face_landmarker.task
- qc_filter.py 默认使用该模型文件进行 landmark 计算
- 如果你替换模型文件，请保持路径一致或修改脚本中的 MODEL_PATH

