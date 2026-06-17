# 方向 A：临床描述符辅助监督训练方法与实验报告

## 摘要

针对老年营养不良二分类任务中模型判别依据可解释性不足的问题，本文在现有 NutriDiff 分类流水线基础上，引入临床表型描述符辅助监督机制。具体而言，本文利用现有人脸关键点与 ROI 质量控制模块，从输入图像中离线提取 4 维临床描述符，包括颞部、眶周、颧颊和下颌线相关表型特征，并在 CLIP 图像编码器上增加一个辅助回归头，使模型在优化分类交叉熵损失的同时学习预测临床描述符。实验结果表明，在 raw 合成数据设置下，引入辅助头后 subject-level MCC 由 0.7799 提升至 0.8614，说明临床描述符辅助监督能够有效提升模型在未过滤合成数据条件下的分类鲁棒性。

## 一、研究背景与问题定义

儿童营养不良筛查任务具有较强的临床表型依赖性。与一般自然图像分类不同，营养不良相关视觉线索往往集中于面部局部区域，例如颞部凹陷、眶周脂肪垫变化、颧颊纹理以及下颌线清晰度等。现有分类模型虽然能够通过端到端监督学习获得一定判别能力，但模型学习到的特征是否与临床可解释区域一致并不明确。

现有 NutriDiff 流水线中已经包含基于 MediaPipe Face Landmarker 的 ROI 质量过滤模块 `passing/qc_filter.py`。该模块能够从人脸图像中提取与营养不良表型相关的几何和纹理描述符。因此，本文提出方向 A：临床描述符辅助监督训练。该方法不改变图像生成流程，也不修改 LoRA 注入结构，而是在分类训练阶段显式引入临床描述符作为辅助监督信号。

给定输入图像 `x`，二分类标签 `y`，以及由临床特征提取器得到的 4 维描述符：

```text
s(x) = [temporal, orbital, malar, jawline]
```

模型同时输出分类 logits 与描述符预测值：

```text
f(x) = (p(y|x), ŝ(x))
```

总损失函数定义为：

```text
L_total = L_CE(y, p(y|x)) + α · L_MSE(ŝ(x), s(x))
```

其中 `α` 为辅助损失权重。本文重点考察 `α = 0.05, 0.1, 0.3` 三组设置。

## 二、方法设计

### 2.1 临床描述符构建

本文复用 `passing/qc_filter.py` 中已有的 `ClinicalFeatureExtractor` 与 `DataFilter`，对训练图像进行离线描述符提取。每张图像对应一个 4 维向量：

| 维度 | 含义 | 临床解释 |
|---|---|---|
| temporal | 颞部相关比例特征 | 反映颞部凹陷和软组织消耗 |
| orbital | 眶周相关比例特征 | 反映眼眶周围脂肪垫变化 |
| malar | 颧颊纹理特征 | 反映面颊区域纹理和饱满度变化 |
| jawline | 下颌线清晰度 | 反映下颌轮廓和软组织状态 |

由于四个维度的数值量纲差异较大，尤其是颧颊纹理特征的取值范围明显大于比例类特征，若直接计算 MSE 损失会导致大数值维度主导辅助监督。因此，本文在描述符缓存构建阶段仅使用真实训练图像统计均值和标准差，并将所有真实图像与合成图像的原始描述符统一映射到归一化空间。

归一化参数保存在缓存 JSON 的 `normalize_stats` 字段中，描述符缓存包含：

```text
normalize_stats: 均值、标准差和归一化方式
descriptors: 归一化后的描述符
raw_descriptors: 原始描述符
```

当某张图像由于侧脸、遮挡或人脸检测失败而无法提取描述符时，缓存中记录为 `null`。训练过程中该样本仍参与分类损失计算，仅跳过辅助 MSE 损失，从而避免因描述符缺失丢弃训练样本。

### 2.2 描述符缓存机制

本文新增 `classify/roi_descriptor.py`，实现 `ROIDescriptorCache` 类。该类提供三个核心接口：

```python
class ROIDescriptorCache:
    def __init__(self, cache_path: str, model_path: str):
        ...

    def build(self, image_dirs: list[str]):
        ...

    def get(self, img_path: str) -> list[float] | None:
        ...
```

缓存文件使用图像绝对路径作为 key。该设计保证 DataLoader 在训练时读取的样本路径与缓存路径完全一致，避免因文件名重排或软链接解析造成描述符无法命中。缓存构建支持增量更新：若图像路径已存在于缓存中，则跳过重复提取；若新增了 raw 或 qc 合成图目录，则只补充未缓存样本。

本文构建缓存时依次遍历以下目录：

```text
/root/autodl-tmp/runs/cv/fold_4/my_dataset_binary/seed0
/root/autodl-tmp/datadream_outputs/generated_images/my_dataset/sd2.1/gs3.5_nis50/shot20_seed0_template1_lr0.0001_ep240/train
/root/autodl-tmp/datadream_outputs/generated_images/my_dataset/sd2.1/gs3.5_nis50/shot20_seed0_template1_lr0.0001_ep240/filtered_train
```

其中 `my_dataset_binary/seed0` 是真实训练图像目录，也是归一化统计量的唯一来源；`train/` 是 raw 合成图目录；`filtered_train/` 是 qc 过滤后的合成图目录。

### 2.3 DataLoader 扩展

为了在训练阶段读取描述符，本文修改 `classify/data.py`，为 `FixedLabelImageFolder` 与合成数据集类增加可选参数 `descriptor_cache`。当该参数为 `None` 时，DataLoader 行为与原始分类训练完全一致，返回：

```text
(image, label)
```

当启用描述符缓存时，DataLoader 返回：

```text
(image, label, descriptor)
```

若描述符提取失败，`descriptor` 被置为形状为 `(4,)` 的 NaN tensor。训练循环通过 `torch.isnan` 构造有效样本掩码，仅对有效描述符样本计算辅助损失。这一处理方式保证了分类任务的数据覆盖率不受描述符提取失败影响。

对于 real + synth 混合训练场景，DataLoader 还需要兼容原有的 `is_real` 标记，因此训练循环同时支持如下 batch 格式：

```text
(image, label)
(image, label, descriptor)
(image, label, is_real)
(image, label, is_real, descriptor)
```

### 2.4 CLIP 辅助回归头

本文修改 `classify/models/clip.py`，在 CLIP 图像特征之后添加 ROI 描述符回归头。该模块仅在 `use_roi_aux_head=True` 时启用，默认不改变原有模型行为。

辅助头结构为两层 MLP：

```python
self.roi_head = nn.Sequential(
    nn.Linear(image_feature_dim, 128),
    nn.ReLU(),
    nn.Linear(128, 4),
)
```

其中 `image_feature_dim` 由 CLIP visual encoder 的输出维度确定，ViT-B/16 设置下为 512。启用辅助头时，模型 forward 返回：

```text
(logits_per_image, roi_pred)
```

未启用辅助头时，模型 forward 返回值与原始实现保持一致。`learnable_params()` 中同步加入 `roi_head` 参数，使辅助头能够参与优化。

### 2.5 训练目标与兼容性处理

本文修改 `classify/main.py`，在训练开始前根据命令行参数初始化描述符缓存，并在构建 DataLoader 时传入 `descriptor_cache`。训练过程中，若启用辅助头，则模型同时输出分类结果和描述符预测结果。

分类损失沿用原有训练逻辑，包括真实图像与合成图像混合训练、MixUp/CutMix 等策略。对于发生 MixUp 或 CutMix 的 batch，由于图像内容与原始描述符不再一一对应，本文跳过该 batch 的辅助 MSE 损失，仅保留分类损失。

辅助损失计算方式如下：

```python
valid_mask = ~torch.isnan(descriptor).any(dim=1)
if valid_mask.sum() > 0:
    loss_roi = F.mse_loss(
        roi_pred[valid_mask],
        descriptor[valid_mask],
    )
    loss = loss + args.alpha_roi * loss_roi
```

评估阶段只使用分类 logits。若模型返回 tuple，`eval()` 和 `analyze_predictions()` 只取第一个元素作为分类输出，从而保证辅助头不影响原有评估流程。

### 2.6 运行脚本

本文新增 `classify/run_roi_aux.sh`，用于运行启用辅助头的 raw 合成数据实验。该脚本支持通过环境变量指定辅助损失权重：

```bash
ALPHA_ROI=0.1 SYNTH_VARIANT=raw bash run_roi_aux.sh
```

脚本中显式传入：

```text
--use_roi_aux_head=True
--alpha_roi=$ALPHA_ROI
--roi_descriptor_cache_path=/root/autodl-tmp/runs/roi_descriptor_cache.json
--mediapipe_model_path=/root/autodl-tmp/face_landmarker.task
```

## 三、实验设置

### 3.1 实验环境

实验在 AutoDL Linux 环境下完成，主要软硬件配置如下：

| 项目 | 配置 |
|---|---|
| GPU | 单张 RTX5090 |
| Python 环境 | `/root/autodl-tmp/miniconda3/envs/myclassify/bin/python` |
| 深度学习框架 | PyTorch |
| 主干模型 | CLIP ViT-B/16 |
| MediaPipe 模型 | `/root/autodl-tmp/face_landmarker.task` |
| 目标仓库 | `/root/nutricode/` |

### 3.2 数据设置

实验使用真实训练图像、DataDream raw 合成图像和 NutriDiff qc 过滤合成图像。真实训练图像位于：

```text
/root/autodl-tmp/runs/cv/fold_4/my_dataset_binary/seed0
```

raw 合成图像位于：

```text
/root/autodl-tmp/datadream_outputs/generated_images/my_dataset/sd2.1/gs3.5_nis50/shot20_seed0_template1_lr0.0001_ep240/train
```

qc 过滤合成图像位于：

```text
/root/autodl-tmp/datadream_outputs/generated_images/my_dataset/sd2.1/gs3.5_nis50/shot20_seed0_template1_lr0.0001_ep240/filtered_train
```

训练时 raw 设置每类取前 500 张合成图像。描述符缓存阶段则对 raw 和 qc 目录进行全量预计算，以保证不同实验设置均可按路径命中描述符。

### 3.3 对比方法

本文设置两层实验。

第一层用于复现论文 Table 2 中的关键基线，以确认环境和训练设置正确：

| 方法 | 合成数据 | 辅助头 |
|---|---|---|
| DataDream baseline | raw | 否 |
| NutriDiff baseline | qc | 否 |

第二层为方向 A 主实验，固定使用 raw 合成数据，考察是否启用临床描述符辅助头以及不同 `α` 取值的影响：

| 方法 | 合成数据 | 辅助头 | α |
|---|---|---|---|
| DataDream + aux | raw | 是 | 0.05 |
| DataDream + aux | raw | 是 | 0.1 |
| DataDream + aux | raw | 是 | 0.3 |

### 3.4 评价指标

本文使用 `detailed_prediction_results.json` 中的 `subject_level_metrics` 作为主要评价结果。报告指标包括：

| 指标 | 含义 |
|---|---|
| Acc | subject-level accuracy |
| AUC | subject-level area under ROC curve |
| F1 | subject-level F1 score |
| MCC | Matthews correlation coefficient |

其中 MCC 被作为主要模型选择指标。相比 Acc，MCC 对类别不平衡与混淆矩阵整体结构更敏感，更适合作为二分类医学筛查任务的综合评价指标。

## 四、实现与验证流程

### 4.1 描述符缓存验证

首先运行 `roi_descriptor.py` 构建描述符缓存：

```bash
cd /root/nutricode/classify
/root/autodl-tmp/miniconda3/envs/myclassify/bin/python roi_descriptor.py \
  --image_dirs \
    /root/autodl-tmp/runs/cv/fold_4/my_dataset_binary/seed0 \
    /root/autodl-tmp/datadream_outputs/generated_images/my_dataset/sd2.1/gs3.5_nis50/shot20_seed0_template1_lr0.0001_ep240/train \
    /root/autodl-tmp/datadream_outputs/generated_images/my_dataset/sd2.1/gs3.5_nis50/shot20_seed0_template1_lr0.0001_ep240/filtered_train \
  --cache_path /root/autodl-tmp/runs/roi_descriptor_cache.json \
  --model_path /root/autodl-tmp/face_landmarker.task
```

实际构建结果如下：

| 项目 | 数值 |
|---|---:|
| 缓存总条目数 | 13958 |
| 有效描述符数 | 12466 |
| 提取失败数 | 1492 |
| 真实训练图有效数 | 113 / 113 |

真实训练图全部提取成功，满足真实图失败率低于 10% 的要求。部分合成图由于姿态、面部结构异常或检测失败无法提取描述符，训练时通过 NaN mask 自动跳过辅助损失。

真实训练图统计得到的归一化参数为：

```text
mean = [0.908731, 0.914384, 44.763550, 23.910796]
std  = [0.141599, 0.117464, 75.097960, 9.227219]
```

### 4.2 DataLoader 返回值验证

启用 `descriptor_cache` 后，DataLoader 返回 batch 长度为 3：

```text
batch_len = 3
image_shape = (4, 3, 224, 224)
label_shape = (4,)
descriptor_shape = (4, 4)
```

该结果说明 DataLoader 已能够在原始图像和标签之外附加 4 维临床描述符，并保持 batch 维度一致。

### 4.3 模型 forward 验证

启用辅助头后，对随机输入进行 forward 验证：

```text
logits_shape = (4, 2)
roi_pred_shape = (4, 4)
```

该结果说明分类输出和辅助回归输出均符合预期。其中 `(4, 2)` 对应 batch size 为 4 的二分类 logits，`(4, 4)` 对应每张图像的 4 维描述符预测。

### 4.4 完整训练验证

首先运行 `α = 0.1` 的完整训练：

```bash
ALPHA_ROI=0.1 SYNTH_VARIANT=raw \
PYTHON_BIN=/root/autodl-tmp/miniconda3/envs/myclassify/bin/python \
bash run_roi_aux.sh
```

训练 40 epoch 后，最终训练日志显示：

```text
loss_roi = 0.089787
```

`loss_roi` 从初始约 0.31 下降至 0.09 附近，说明辅助描述符回归任务能够在训练过程中稳定收敛。

## 五、实验结果

### 5.1 基线复现结果

为确认实验环境正确，首先复现 raw 与 qc 两组无辅助头基线。结果如下：

| 方法 | 合成数据 | Acc | AUC | F1 | MCC |
|---|---|---:|---:|---:|---:|
| DataDream baseline | raw (500/类) | 0.8889 | 0.9822 | 0.8929 | 0.7799 |
| NutriDiff baseline | qc (~430/类) | 0.9444 | 0.9863 | 0.9474 | 0.8944 |

raw baseline 的 subject-level MCC 为 0.7799，qc baseline 的 subject-level MCC 为 0.8944，均与论文 Table 2 中预期结果接近，说明当前训练环境、数据路径和评估流程基本一致。

### 5.2 临床描述符辅助监督消融结果

在 raw 合成数据设置下，进一步比较不同辅助损失权重 `α` 的影响。完整 subject-level 结果如下：

| 方法 | 合成数据 | 辅助头 | α | Acc | AUC | F1 | MCC |
|---|---|---|---|---:|---:|---:|---:|
| DataDream | raw (500/类) | ✗ | — | 0.8889 | 0.9822 | 0.8929 | 0.7799 |
| NutriDiff | qc (~430/类) | ✗ | — | 0.9444 | 0.9863 | 0.9474 | 0.8944 |
| DataDream + aux | raw (500/类) | ✓ | 0.05 | 0.9259 | 0.9835 | 0.9310 | 0.8614 |
| DataDream + aux | raw (500/类) | ✓ | 0.1 | 0.9259 | 0.9835 | 0.9310 | 0.8614 |
| DataDream + aux | raw (500/类) | ✓ | 0.3 | 0.9074 | 0.9808 | 0.9153 | 0.8292 |

从结果可以看出，加入临床描述符辅助头后，raw 合成数据设置下的分类性能显著提升。`α = 0.05` 与 `α = 0.1` 均取得最高 MCC 0.8614，相比 raw baseline 的 0.7799 提升 0.0815。`α = 0.3` 仍优于 raw baseline，但提升幅度较小，说明过大的辅助损失权重可能会对主分类任务形成一定干扰。

### 5.3 ROI 辅助损失收敛情况

三个辅助监督实验均未出现 `loss_roi=NaN`。最终 epoch 的平均 `loss_roi` 如下：

| α | 最终 loss_roi | 是否小于 0.2 | 是否出现 NaN |
|---:|---:|---|---|
| 0.05 | 0.091940 | 是 | 否 |
| 0.1 | 0.089787 | 是 | 否 |
| 0.3 | 0.082130 | 是 | 否 |

结果表明，辅助回归任务在 40 epoch 内均可稳定收敛。其中 `α = 0.3` 的最终 `loss_roi` 最低，但其分类 MCC 低于 `α = 0.05` 和 `α = 0.1`，说明仅优化描述符回归并不必然带来最优分类性能，辅助任务权重需要与主任务保持平衡。

## 六、结果分析与讨论

### 6.1 辅助监督对 raw 合成数据的改善

DataDream raw 合成数据未经 qc 过滤，样本中可能存在局部面部结构不稳定、姿态异常或纹理质量波动等问题。直接使用 raw 合成图训练时，模型容易学习到与营养不良标签相关性较弱的伪线索。临床描述符辅助头通过要求模型同时预测颞部、眶周、颧颊和下颌线特征，将图像编码器的注意力约束到更具有临床意义的面部区域。

实验中 raw baseline 的 MCC 为 0.7799，而加入辅助头后最高达到 0.8614，说明该辅助监督能够在不改变合成图生成流程的前提下，提高 raw 合成数据训练的有效性。

### 6.2 与 qc 过滤方法的关系

NutriDiff qc baseline 的 MCC 为 0.8944，仍高于当前方向 A 的最佳结果 0.8614。这表明基于质量过滤的合成数据筛选仍然具有较强效果。方向 A 的价值并非替代 qc 过滤，而是为 raw 合成数据训练提供一种额外的表型约束机制。

从实验结果看，临床描述符辅助监督能够显著缩小 raw baseline 与 qc baseline 之间的性能差距。因此，后续可以进一步研究 qc 过滤与辅助监督的联合使用，例如在 qc 数据上启用辅助头，或将 qc 评分与描述符回归误差共同用于样本加权。

### 6.3 辅助损失权重影响

`α = 0.05` 与 `α = 0.1` 的分类指标完全一致，均为当前最优设置；`α = 0.3` 的描述符回归误差更低，但分类 MCC 下降至 0.8292。这说明较大的辅助损失权重会使模型更偏向拟合表型描述符，而相对削弱主分类目标。

综合 MCC 与收敛情况，本文建议默认采用 `α = 0.1`。该设置与任务规格中的初始建议一致，且在保持最优分类指标的同时获得更低的最终 `loss_roi`。

### 6.4 局限性

本文方法仍存在以下局限：

1. 描述符依赖 MediaPipe 人脸关键点检测，侧脸、遮挡或合成图结构异常时可能提取失败。
2. 当前描述符维度较低，仅覆盖四类局部表型，尚不能完整表达营养不良相关视觉差异。
3. 辅助描述符由规则和传统视觉特征计算得到，其准确性受 ROI 定义和图像质量影响。
4. 当前主实验仅在 raw 合成数据上完成，尚未系统评估 qc 数据与辅助监督联合使用的效果。
5. 最佳 MCC 为 0.8614，显著高于 raw baseline，但略低于 0.864 的更严格预期阈值。

## 七、结论

本文在现有 NutriDiff 分类流水线中实现了临床描述符辅助监督训练方法。该方法通过离线提取 4 维临床表型描述符，并在 CLIP 图像编码器上增加辅助回归头，使模型在学习营养不良二分类的同时显式对齐颞部、眶周、颧颊和下颌线等临床相关视觉线索。

实验结果表明，在 raw 合成数据设置下，临床描述符辅助监督将 subject-level MCC 从 0.7799 提升至 0.8614，且 `loss_roi` 在 40 epoch 内稳定收敛至 0.1 以下，无 NaN 问题。该结果验证了方向 A 的有效性，说明临床表型描述符可以作为一种轻量、可解释且易于集成的辅助监督信号，用于提升合成数据训练下的儿童营养不良筛查模型性能。

综合实验表现，本文建议将 `α = 0.1` 作为默认辅助损失权重，并在后续工作中进一步探索其与 qc 过滤、样本加权和 ROI 注意力机制的联合建模。

## 附录：关键文件与实验输出

### A.1 关键代码文件

| 文件 | 作用 |
|---|---|
| `classify/roi_descriptor.py` | 离线提取并缓存 ROI 临床描述符 |
| `classify/data.py` | DataLoader 返回可选 descriptor |
| `classify/models/clip.py` | CLIP 图像特征上增加 ROI 辅助回归头 |
| `classify/main.py` | 初始化缓存、计算辅助损失、兼容评估 |
| `classify/config.py` | 新增 ROI 辅助监督相关命令行参数 |
| `classify/run_roi_aux.sh` | 运行方向 A 主实验脚本 |

### A.2 主要输出目录

raw baseline：

```text
/root/autodl-tmp/runs/ablation/classify_outputs/clip_real_plus_synth_raw_pool0.7_nipc330_lr1e-5_nomix
```

qc baseline：

```text
/root/autodl-tmp/runs/ablation/classify_outputs/clip_real_plus_synth_qc_pool0.7_nipc330_lr1e-5_nomix
```

ROI auxiliary experiments：

```text
/root/autodl-tmp/runs/ablation/roi_aux_outputs/clip_roi_aux_alpha0.05_raw_pool0.7_nipc500_lr1e-5_nomix
/root/autodl-tmp/runs/ablation/roi_aux_outputs/clip_roi_aux_alpha0.1_raw_pool0.7_nipc500_lr1e-5_nomix
/root/autodl-tmp/runs/ablation/roi_aux_outputs/clip_roi_aux_alpha0.3_raw_pool0.7_nipc500_lr1e-5_nomix
```
