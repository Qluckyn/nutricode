# NutriCode 项目说明文档

> 本文档基于当前 `nutricode/` 目录下的 README、Shell 脚本、Python 源码、YAML 配置和依赖文件整理。项目依赖的数据集、模型权重、运行输出多数位于 `/root/autodl-tmp/...`，当前仓库不包含这些外部数据与权重。

## 1. 项目整体概述

NutriCode 是一个面向营养不良相关面部图像研究的图像生成、质量过滤与分类训练项目。项目的数据生成部分以 DataDream 为原型，围绕少样本真实图像训练 Stable Diffusion 2.1 的 LoRA，再用训练好的 LoRA 按类别生成合成图像，随后可通过面部关键点与临床启发式特征对合成图像进行质量过滤，最后用真实数据与合成数据训练分类模型。

项目主要解决的问题是：在真实营养不良/正常面部图像样本较少的情况下，借助 DataDream/LoRA 生成可用于训练的合成图像，并通过过滤与真实+合成联合训练来提升分类模型的数据覆盖能力。

核心功能包括：

- 少样本 LoRA 训练：对 `my_dataset` 的多个面部视角/营养状态类别分别训练 Stable Diffusion LoRA。
- 合成图像生成：加载对应类别的 LoRA 权重，用 Stable Diffusion 2.1 生成每类合成图像。
- 合成图像过滤：使用 MediaPipe FaceLandmarker、InsightFace fallback、手工定义的面部 ROI 指标和 Mahalanobis 距离筛选合成图像。
- 分类训练：支持 CLIP LoRA 微调和 ResNet50 训练；当前主脚本默认使用 CLIP，并支持真实 few-shot 与合成数据联合训练。
- 根目录 Wrapper：`01_lora.sh`、`02_generate.sh`、`03_filter.sh`、`04_classify.sh` 串起完整流程。

## 2. 整体架构分析

### 2.1 架构模式

项目是一个脚本驱动的离线机器学习流水线，整体属于“多阶段实验管线”架构：

```text
真实 few-shot 数据
    |
    v
[sd_lora] DataDream LoRA 训练
    |
    v
LoRA 权重
    |
    v
[generate] Stable Diffusion + LoRA 生图
    |
    v
原始合成图像
    |
    v
[passing] 面部特征/Mahalanobis/Top-K 过滤
    |
    v
过滤后合成图像
    |
    v
[classify] CLIP/ResNet 分类训练与评估
```

根目录脚本负责环境选择、路径切换和参数传递；各子目录保留相对独立的 DataDream/分类代码。

### 2.2 模块划分

- `sd_lora/`：LoRA 训练模块。读取 few-shot 真实图像和类别模板，使用 diffusers/accelerate/peft 训练每类 LoRA 权重。
- `generate/`：合成图像生成模块。读取 Stable Diffusion 模型和 LoRA 权重，按类别生成图片，保存 prompts 与生成图像。
- `passing/`：质量控制过滤模块。读取真实图像建立每个组的 4 维特征分布，再筛选合成图像。
- `classify/`：分类训练模块。构建真实/合成数据 DataLoader，训练 CLIP 或 ResNet50，保存最优 checkpoint 和预测分析 JSON。
- 根目录：统一入口脚本、依赖文件、总 README。

### 2.3 数据流完整链路

1. 准备真实 few-shot 数据。
   - LoRA 默认从 `sd_lora/local.yaml` 的 `fewshot_data_dir.my_dataset` 读取，并追加 `seed0`。
   - 期望结构类似：

```text
real_train_groups/seed0/
  normal_front_face/
  normal_left_three-quarter_face/
  normal_right_three-quarter_face/
  malnourished_front_face/
  malnourished_left_three-quarter_face/
  malnourished_right_three-quarter_face/
```

2. 训练 LoRA。
   - 入口：`bash 01_lora.sh 0 0`
   - 实际调用：`sd_lora/bash_run.sh -> accelerate launch datadream.py`
   - 输出示例：

```text
/root/autodl-tmp/datadream_outputs/models/my_dataset/
  shot20_seed0_tpl1/
    lr0.0001_epoch240/
      normal_front_face/pytorch_lora_weights.safetensors
      ...
```

3. 生成合成图像。
   - 入口：`bash 02_generate.sh 0 0`
   - 实际调用：`generate/bash_run.sh -> python generate.py`
   - 生成代码会按类别加载对应 LoRA 权重，并输出：

```text
<save_dir>/my_dataset/sd2.1/
  gs3.5_nis50/
    shot20_seed0_template1_lr0.0001_ep240/
      train/<class_name>/*.png
      train/<class_name>/prompts.json
```

4. 过滤合成图像。
   - 入口：`bash 03_filter.sh`
   - 实际调用：`passing/qc_filter.py`
   - 对真实图像每个组学习 Mahalanobis 分布，对合成图像计算同样指标，先按距离阈值筛选，再执行 Top-K。
   - 输出：

```text
filtered_train/<class_name>/*.png
filtered_train/<class_name>/filter_stats.json
```

5. 分类训练。
   - 入口：`bash 04_classify.sh`
   - 实际调用：`classify/run_both_real_and_synth.sh -> classify/main.py`
   - 当前分类代码的 `my_dataset` 是二分类：

```text
malnourished_face
normal_face
```

   - `run_both_real_and_synth.sh` 会将 6 个视角组软链接汇聚成二分类目录，再训练 CLIP/ResNet。

## 3. 文件与目录详解

### 3.1 根目录

| 路径 | 作用与职责 | 依赖关系 |
|---|---|---|
| `./` | 项目根目录，承载完整实验管线入口与各模块源码。 | 子目录 `sd_lora/`、`generate/`、`passing/`、`classify/`。 |
| `./.ipynb_checkpoints/` | Jupyter 自动生成的检查点目录。当前未发现被源码引用。 | 无直接依赖；可忽略。 |
| `README.md` | 当前已有的简要项目说明，描述流程、环境、路径和 Wrapper 参数。 | 被本文档参考。 |
| `README_PROJEXT.md` | 本文档，项目理解与运行说明。 | 无运行时依赖。 |
| `01_lora.sh` | LoRA 训练统一入口。设置 datadream 环境 PATH，切换到 `sd_lora/`，执行 `bash_run.sh`。 | 依赖 `/root/autodl-tmp/conda_envs/datadream/bin`、`sd_lora/bash_run.sh`。 |
| `02_generate.sh` | 生图统一入口。设置 datadream 环境 PATH，切换到 `generate/`，执行 `bash_run.sh`。 | 依赖 datadream 环境、`generate/bash_run.sh`。 |
| `03_filter.sh` | QC 过滤统一入口。默认使用 datadream 环境 Python，组织真实/合成/输出路径参数并调用 `passing/qc_filter.py`。 | 依赖 `passing/qc_filter.py`、MediaPipe 模型、真实与合成图像路径。 |
| `04_classify.sh` | 分类训练统一入口。默认使用 myclassify 环境 Python，设置模型类型、合成数据版本和输出目录，调用分类训练脚本。 | 依赖 `classify/run_both_real_and_synth.sh`、myclassify 环境。 |
| `requirements.txt` | 合并依赖文件，覆盖 LoRA/生成/过滤/分类相关包。 | 可用于整体环境参考；版本冲突需进一步确认。 |
| `requirements-lora-gen.txt` | datadream 环境依赖，面向 LoRA、Stable Diffusion 生图、QC 过滤。 | 依赖 diffusers、accelerate、mediapipe、insightface、opencv 等。 |
| `requirements-classify.txt` | myclassify 环境依赖，面向分类训练。 | 依赖 torch、clip、loralib、timm、wandb 等；完整版本以文件为准。 |

### 3.2 `sd_lora/`：DataDream LoRA 训练

| 路径 | 作用与职责 | 依赖关系 |
|---|---|---|
| `sd_lora/` | LoRA 训练模块目录。 | 由 `01_lora.sh` 进入并执行。 |
| `sd_lora/readme.txt` | 简短说明该目录用于 Stable Diffusion LoRA 微调。 | 文档参考。 |
| `sd_lora/bash_run.sh` | 按类别循环训练 LoRA。默认 `DATASET=my_dataset`、`N_CLS=6`、`N_SHOT=20`、`NUM_TRAIN_EPOCH=240`。 | 调用 `accelerate launch datadream.py`；依赖 `local.yaml`、`util_data.py`、SD2.1 模型。 |
| `sd_lora/local.yaml` | LoRA 本地路径配置。包含 SD2.1 模型路径和 few-shot 数据根路径。 | 被 `sd_lora/config.py` 读取。 |
| `sd_lora/config.py` | 解析 `datadream.py` 命令行参数，读取 `local.yaml`，拼接 LoRA 输出目录与 few-shot 数据目录。 | 依赖 `yaml`、`torch`、`util_data.SUBSET_NAMES`。 |
| `sd_lora/datadream.py` | 核心 LoRA 训练代码，基于 diffusers DreamBooth/LoRA 流程改造。负责构造数据集、加载 Stable Diffusion、训练 UNet/Text Encoder LoRA、保存权重。 | 依赖 `config.parse_args`、`util.natural_keys`、`util_data.SUBSET_NAMES/TEMPLATES_SMALL`、diffusers、accelerate、peft、safetensors。 |
| `sd_lora/util.py` | 工具函数：随机种子设置、自然排序 key。 | 被 `datadream.py` 使用。 |
| `sd_lora/util_data.py` | 数据集类别名与 prompt 模板定义。当前 `my_dataset` 为 6 类视角/标签组合，另含 `my_dataset_binary`。 | 被 `config.py`、`datadream.py` 使用；类别顺序必须与 `bash_run.sh` 的 `CLASS_NAMES` 保持一致。 |

`sd_lora/util_data.py` 中当前关键类别：

```python
my_dataset = [
    "normal_front_face",
    "normal_left_three-quarter_face",
    "normal_right_three-quarter_face",
    "malnourished_front_face",
    "malnourished_left_three-quarter_face",
    "malnourished_right_three-quarter_face",
]
```

### 3.3 `generate/`：合成图像生成

| 路径 | 作用与职责 | 依赖关系 |
|---|---|---|
| `generate/` | 生图模块目录。 | 由 `02_generate.sh` 进入并执行。 |
| `generate/readme.txt` | 简短说明 `bash_run.sh` 是生成脚本，`generate.py` 是 DataDream 生图代码。 | 文档参考。 |
| `generate/bash_run.sh` | 生图运行脚本。默认每类生成 `NIPC=1000`，SD 版本 `sd2.1`，guidance scale `3.5`，加载 `shot20_seed0_template1_lr0.0001_ep240` 对应 LoRA。 | 调用 `python generate.py`；依赖 datadream 环境、`generate/local.yaml`、LoRA 权重。 |
| `generate/local.yaml` | 生图本地路径配置。包含合成图输出根目录 `save_dir`、LoRA 权重根目录 `datadream_dir`、SD 模型路径。 | 被 `generate.py` 读取。 |
| `generate/generate.py` | 核心生图代码。加载 Stable Diffusion Pipeline，按类别加载 LoRA 权重，构造 prompt，批量生成并保存图像与 `prompts.json`。 | 依赖 `fire`、`diffusers`、`torch`、`safetensors`、`yaml`、`util.py`。 |
| `generate/util.py` | 生图工具与类别定义。提供随机种子、批量迭代、目录创建、`SUBSET_NAMES`、`TEMPLATES_SMALL`。 | 被 `generate.py` 使用；类别必须与 LoRA 阶段匹配。 |

### 3.4 `passing/`：合成图质量过滤

| 路径 | 作用与职责 | 依赖关系 |
|---|---|---|
| `passing/` | 过滤模块目录。 | 由 `03_filter.sh` 进入并执行。 |
| `passing/face_landmarker.task` | MediaPipe FaceLandmarker 模型文件。仓库内提供一份，但 `qc_filter.py` 默认常量指向 `/root/autodl-tmp/face_landmarker.task`。需进一步确认实际运行是否使用仓库内文件或外部文件。 | 被 `qc_filter.py` 的 `MODEL_PATH` 使用。 |
| `passing/qc_filter.py` | QC 过滤主程序。定义面部关键点 ROI、提取 4 维临床启发式特征、对真实数据学习 Mahalanobis 分布、筛选合成图像并保存统计。 | 依赖 `cv2`、`numpy`、`mediapipe`、`insightface`、`sklearn.covariance`、真实/合成图像目录。 |

过滤指标 `s(x)` 的固定顺序为：

```text
temporal_ratio
orbital_ratio
cheek_texture
jawline_sharpness
```

过滤流程：

1. 从真实数据各组目录提取 4 维指标。
2. 为每个组计算均值、协方差、协方差逆矩阵。
3. 用真实样本 Mahalanobis 距离的指定分位数作为阈值，默认 P95。
4. 对合成图像计算同样指标，保留距离不超过阈值的候选。
5. 在候选内按距离从小到大执行 Top-K，默认 `K = ceil(8.0 * n_real)`。
6. 复制保留图片到 `filtered_train/<group>/`，保存 `filter_stats.json`。

### 3.5 `classify/`：分类训练

| 路径 | 作用与职责 | 依赖关系 |
|---|---|---|
| `classify/` | 分类训练模块目录。 | 由 `04_classify.sh` 进入并执行。 |
| `classify/readme.txt` | 简短说明模型、数据文件夹和训练脚本用途。 | 文档参考。 |
| `classify/run_both_real_and_synth.sh` | 分类训练主运行脚本。准备真实/合成二分类软链接目录，设置训练参数，调用 `main.py`。 | 依赖真实数据目录、合成数据目录、`main.py`、myclassify 环境。 |
| `classify/local.yaml` | 分类本地路径配置，包含合成训练根目录、真实训练/测试目录、metadata、CLIP 下载目录、wandb key。 | 被 `classify/config.py` 读取。 |
| `classify/fold_local.yaml` | 交叉验证 fold 风格的分类路径配置。当前代码默认读取 `local.yaml`，该文件未在默认脚本中直接使用，需进一步确认是否手动替换/改名使用。 | 无默认直接依赖。 |
| `classify/config.py` | 解析分类训练参数，读取 `local.yaml`，构造输出目录、合成数据路径、日志和 wandb group。 | 依赖 `util_data.SUBSET_NAMES`、`yaml`。 |
| `classify/data.py` | 数据加载与增强。定义 `FixedLabelImageFolder`、真实数据 DataLoader、合成数据 DataLoader、few-shot pooling 逻辑和若干通用数据集 split 方法。 | 依赖 `utils.make_dirs`、`util_data`、torch/torchvision/PIL。 |
| `classify/main.py` | 分类训练、评估、checkpoint 保存和预测分析入口。支持 CLIP 与 ResNet50，默认训练后加载最佳模型输出详细预测 JSON。 | 依赖 `config.get_args`、`data.py`、`models/clip.py`、`models/resnet50.py`、`utils.py`、`util_data.py`。 |
| `classify/util_data.py` | 分类任务的类别、模板、metadata、图像增强工具定义。注意这里 `my_dataset` 当前是二分类，而不是 LoRA/生成阶段的 6 类。 | 被 `config.py`、`data.py`、`main.py`、`models/clip.py` 使用。 |
| `classify/utils.py` | 训练通用工具：随机种子、余弦学习率、日志指标、平滑统计、目录创建等。 | 被 `main.py`、`data.py` 使用。 |
| `classify/models/` | 分类模型实现目录。 | 被 `classify/main.py` 使用。 |
| `classify/models/clip.py` | CLIP 分类模型包装。加载 OpenAI CLIP，按类别 prompt 编码文本，将图像/文本特征相似度作为 logits，并可对视觉/文本 Transformer 注入 LoRA。 | 依赖 `models/lora.py`、`util_data.SUBSET_NAMES/TEMPLATES_SMALL`、`clip`、`timm`。 |
| `classify/models/lora.py` | CLIP Transformer attention 的 LoRA 替换实现。 | 被 `models/clip.py` 使用；依赖 `loralib`、torch。 |
| `classify/models/resnet50.py` | 从头定义 ResNet50，用于分类对照。 | 被 `main.py` 可选使用。 |

分类阶段当前 `my_dataset` 类别：

```python
my_dataset = [
    "malnourished_face",
    "normal_face",
]
```

注意：根 README 和 `03_filter.sh` 使用 6 个视角组；`classify/run_both_real_and_synth.sh` 会把这 6 个组汇聚为二分类目录。因此分类输入目录需要最终包含 `malnourished_face/` 与 `normal_face/`，或由脚本创建软链接聚合目录。

## 4. 完整运行步骤

### 4.1 datadream 环境

#### 可以做什么

datadream 环境用于：

- Stable Diffusion 2.1 LoRA 训练；
- DataDream/LoRA 生图；
- 合成图像 QC 过滤。

项目中默认环境路径：

```bash
/root/autodl-tmp/conda_envs/datadream
```

#### 启动环境

方式一：使用根目录 Wrapper，脚本会优先把 datadream 环境加入 `PATH`：

```bash
cd /root/nutricode
bash 01_lora.sh 0 0
```

方式二：手动进入环境，需进一步确认本机 conda 初始化方式：

```bash
conda activate /root/autodl-tmp/conda_envs/datadream
```

如果需要重建环境，可参考：

```bash
pip install -r requirements-lora-gen.txt
```

#### 执行 LoRA 训练

```bash
cd /root/nutricode
bash 01_lora.sh 0 0
```

参数含义：

- 第一个参数：GPU 编号，默认 `0`。
- 第二个参数：类别切分编号，默认 `0`。

常见覆盖变量：

```bash
OUTPUT_ROOT=/root/autodl-tmp/datadream_outputs/models \
bash 01_lora.sh 0 0
```

训练前需要确认：

- `sd_lora/local.yaml` 中 `fewshot_data_dir.my_dataset` 存在，并包含 `seed0/<class_name>/`。
- `sd_lora/bash_run.sh` 的 `CLASS_NAMES` 与 `sd_lora/util_data.py` 的 `SUBSET_NAMES["my_dataset"]` 一致。
- SD2.1 模型路径 `/root/autodl-tmp/models/AI-ModelScope/stable-diffusion-2-1` 存在。

#### 执行生图

```bash
cd /root/nutricode
bash 02_generate.sh 0 0
```

生成前需要确认：

- `generate/local.yaml` 中 `datadream_dir` 指向 LoRA 权重根目录。
- `generate/local.yaml` 中 `save_dir` 是期望的合成图输出根目录。
- `generate/util.py` 的 `SUBSET_NAMES["my_dataset"]` 与 LoRA 训练类别一致。
- `generate/bash_run.sh` 的 `N_SHOT/N_TEMPLATE/DD_LR/DD_EP` 与 LoRA 训练输出目录匹配。

#### 执行 QC 过滤

```bash
cd /root/nutricode
bash 03_filter.sh
```

常见覆盖变量：

```bash
SYNTH_ROOT=/path/to/generated/train \
OUT_DIR=/path/to/filtered_train \
TAU_QUANTILE=95 \
K_BETA=8.0 \
bash 03_filter.sh
```

过滤前需要确认：

- `passing/qc_filter.py` 的 `MODEL_PATH` 默认是 `/root/autodl-tmp/face_landmarker.task`，而仓库内模型在 `passing/face_landmarker.task`。若外部路径不存在，需要复制模型或修改路径。
- `REAL_ROOTS` 与 `SYNTH_ROOT` 下有同名组目录。
- 真实组内有效样本数量足够，否则该组会被跳过。

### 4.2 myclassify 环境

#### 可以做什么

myclassify 环境用于：

- 使用 CLIP 或 ResNet50 训练分类器；
- 支持仅合成数据训练或真实 few-shot + 合成数据联合训练；
- 输出训练日志、最优 checkpoint、混淆矩阵与详细预测 JSON。

项目中默认环境路径：

```bash
/root/autodl-tmp/conda_envs/myclassify
```

#### 启动环境

方式一：使用根目录 Wrapper，脚本会直接指定 myclassify Python：

```bash
cd /root/nutricode
bash 04_classify.sh
```

方式二：手动进入环境，需进一步确认本机 conda 初始化方式：

```bash
conda activate /root/autodl-tmp/conda_envs/myclassify
```

如果需要重建环境，可参考：

```bash
pip install -r requirements-classify.txt
```

#### 执行分类训练

默认使用 CLIP、原始合成图、真实+合成联合训练：

```bash
cd /root/nutricode
bash 04_classify.sh
```

使用过滤后的合成图：

```bash
SYNTH_VARIANT=qc bash 04_classify.sh
```

使用 ResNet50：

```bash
MODEL_TYPE=resnet50 bash 04_classify.sh
```

常见覆盖变量：

```bash
MODEL_TYPE=clip \
SYNTH_VARIANT=qc \
OUTPUT_ROOT=/root/autodl-tmp/runs/ablation/classify_outputs \
NIPC=500 \
LR=1e-5 \
LAMBDA_1=0.8 \
bash 04_classify.sh
```

分类前需要确认：

- `REAL_MAL_GROUP_DIR`、`REAL_NOR_GROUP_DIR` 指向真实 6 组 few-shot 数据。
- `SYNTH_RAW_DIR` 或 `SYNTH_QC_DIR` 指向生图/过滤输出的 `train` 或 `filtered_train`。
- `classify/util_data.py` 的 `my_dataset` 是二分类，脚本会尝试创建 `malnourished_face/normal_face` 聚合目录。
- 测试集默认来自 `classify/local.yaml` 的 `/root/autodl-tmp/test_data`，需要包含二分类目录；如果仍是 6 组目录，需进一步确认评估目标与目录结构。

### 4.3 联合运行完整流程

推荐顺序：

```bash
cd /root/nutricode

# 1. 训练每个 6 组类别的 LoRA
bash 01_lora.sh 0 0

# 2. 使用 LoRA 生成每组图像
bash 02_generate.sh 0 0

# 3. 可选：过滤合成图像
bash 03_filter.sh

# 4A. 使用原始合成图训练分类器
SYNTH_VARIANT=raw bash 04_classify.sh

# 4B. 或使用过滤后合成图训练分类器
SYNTH_VARIANT=qc bash 04_classify.sh
```

联合运行时需要特别保持一致的参数：

- `DATASET`：LoRA、生图、过滤、分类脚本中都默认是 `my_dataset`。
- 类别名：LoRA/生图/过滤使用 6 组视角类别；分类阶段汇聚成 2 类。
- `N_SHOT`、`FEWSHOT_SEED`、`N_TEMPLATE`、`DD_LR`、`DD_EP`：这些参数会影响 LoRA 权重目录和生图目录命名，必须互相匹配。
- 路径：当前 `sd_lora/local.yaml`、`generate/local.yaml` 与根 README 中部分示例路径不完全一致，实际运行前要统一。

## 5. 接口说明

### 5.1 Shell 入口

```bash
bash 01_lora.sh [GPU] [SPLIT_IDX]
bash 02_generate.sh [GPU] [SPLIT_IDX]
bash 03_filter.sh
bash 04_classify.sh
```

### 5.2 `sd_lora/datadream.py` 关键参数

常用参数由 `sd_lora/bash_run.sh` 传入：

- `--pretrained_model_name_or_path`：SD2.1 模型路径。
- `--dataset`：数据集名，默认 `my_dataset`。
- `--fewshot_seed`：few-shot seed，默认 `seed0`。
- `--n_shot`：每类少样本数量，默认 `20`。
- `--target_class_idx`：当前训练类别索引。
- `--output_dir`：LoRA 输出根目录。
- `--num_train_epochs`：训练 epoch。
- `--learning_rate`：LoRA 学习率。
- `--train_text_encoder`：是否训练 text encoder LoRA。

### 5.3 `generate/generate.py` 关键参数

由 `fire.Fire(main)` 暴露，常用参数：

- `--sd_version`：如 `sd2.1`。
- `--mode`：`datadream` 或 `zeroshot`。
- `--guidance_scale`：扩散模型 guidance scale。
- `--num_inference_steps`：推理步数。
- `--n_img_per_class`：每类生成数量。
- `--n_set_split`、`--split_idx`：类别切分生成。
- `--dataset`：数据集名。
- `--datadream_lr`、`--datadream_epoch`：用于定位 LoRA 权重目录。
- `--is_dataset_wise_model`：是否加载 dataset-wise LoRA；当前脚本默认逐类 LoRA。

### 5.4 `passing/qc_filter.py` 关键参数

```bash
python passing/qc_filter.py \
  --real-roots <real_root_1> <real_root_2> \
  --synthetic-root <generated_train_dir> \
  --output-dir <filtered_train_dir> \
  --tau-quantile 95 \
  --cov-shrinkage ledoit_wolf \
  --k-beta 8.0 \
  --only-groups normal_front_face ...
```

主要参数：

- `--real-roots`：一个或多个真实数据根目录，内部需包含组目录。
- `--synthetic-root`：合成图 `train` 根目录。
- `--output-dir`：过滤后输出目录。
- `--tau-quantile`：Mahalanobis 阈值分位数。
- `--cov-shrinkage`：协方差估计方式。
- `--k-beta`/`--k-abs`：Top-K 策略。
- `--learn-only`：只学习真实分布。
- `--max-synth-images`：限制每组处理数量，用于 smoke test。

### 5.5 `classify/main.py` 关键参数

常用参数由 `classify/run_both_real_and_synth.sh` 传入：

- `--model_type`：`clip` 或 `resnet50`。
- `--dataset`：默认 `my_dataset`。
- `--is_synth_train`：是否使用合成数据训练。
- `--synth_train_data_dir_override`：直接指定合成训练目录。
- `--is_pooled_fewshot`：是否将真实 few-shot 与合成数据混合。
- `--lambda_1`：真实/合成 loss 加权中真实数据权重。
- `--n_img_per_cls`：每类合成图使用数量。
- `--lr`、`--wd`、`--epochs`、`--warmup_epochs`：训练超参数。
- `--is_lora_image`、`--is_lora_text`：CLIP 图像/文本分支是否启用 LoRA。
- `--eval_only`、`--eval_ckpt`：仅评估模式相关参数。

## 6. 技术栈

- Python 深度学习：PyTorch、torchvision、torch.cuda.amp。
- 文生图与 LoRA：diffusers、accelerate、peft、safetensors、Stable Diffusion 2.1。
- Prompt/CLIP：OpenAI CLIP、TEMPLATES_SMALL。
- 分类模型：CLIP + LoRA、ResNet50。
- 图像处理与过滤：OpenCV、MediaPipe FaceLandmarker、InsightFace、scikit-learn covariance。
- 实验日志：TensorBoard、可选 Weights & Biases。
- 配置与入口：Bash、YAML、Fire、argparse。

## 7. 注意事项与需进一步确认

- `README_PROJEXT.md` 文件名按任务要求保留为 `PROJEXT`，疑似应为 `PROJECT`，需进一步确认是否要另存为 `README_PROJECT.md`。
- LoRA/生图阶段的 `my_dataset` 是 6 类，分类阶段的 `my_dataset` 是 2 类。这是有意汇聚还是历史遗留，需结合实验设计进一步确认。
- `passing/qc_filter.py` 默认 `MODEL_PATH=/root/autodl-tmp/face_landmarker.task`，但仓库内存在 `passing/face_landmarker.task`。实际运行前需要统一。
- `sd_lora/bash_run.sh` 中打印 few-shot root 的 `awk` 路径指向 `/root/autodl-tmp/NutriPro/Datapro/...`，与当前仓库路径不一致；这只是提示输出，不影响 `datadream.py` 从当前 `sd_lora/local.yaml` 读取配置。
- `generate/local.yaml` 的 `datadream_dir` 默认是 `/root/autodl-tmp/runs/cv/fold_4/lora_models`，而 `sd_lora/bash_run.sh` 默认 LoRA 输出是 `/root/autodl-tmp/datadream_outputs/models`。若直接按默认脚本运行，需进一步确认并统一这两个路径。
- `classify/fold_local.yaml` 当前未被默认代码读取。如需使用 fold 配置，可能需要复制为 `classify/local.yaml` 或修改 `config.py`，需进一步确认。
- `requirements.txt` 是合并环境依赖，可能同时包含 datadream 与 myclassify 的不同版本依赖。建议按任务分别使用 `requirements-lora-gen.txt` 和 `requirements-classify.txt`。
- 仓库不包含真实数据、Stable Diffusion 模型、CLIP 缓存、LoRA 输出和生成图像。所有外部路径需在运行前检查。
- QC 过滤是针对营养不良面部研究硬编码的，包含手工定义的 MediaPipe landmark ROI。若换成其他研究对象，应跳过或重写该模块。
- 当前分类训练脚本会创建/重建软链接目录，涉及 `rm -rf` 已存在的聚合目录项。运行前请确认 `REAL_BINARY_DIR` 与 `SYNTH_TRAIN_DIR` 中的 `malnourished_face/normal_face` 可以被重建。

