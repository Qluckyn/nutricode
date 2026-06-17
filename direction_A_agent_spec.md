# 任务规格书：方向 A — 临床描述符辅助监督训练

> 本文档是写给 AI Coding Agent 的完整任务说明。
> 目标仓库：`/root/nutricode/`，工作环境：AutoDL Linux，单张 RTX5090，Python 3.10，PyTorch + CLIP。

---

## 一、任务目标

在现有 NutriDiff 分类流水线基础上，为 CLIP 分类器添加一个**辅助 ROI 回归头**，使模型在学习营养不良二分类的同时，显式预测 4 维临床表型描述符 `s(x) = [temporal, orbital, malar, jawline]`。

**核心公式：**

```
L_total = L_CE(分类) + α · L_MSE(ŝ(x), s(x))
```

其中 `s(x)` 由 `passing/qc_filter.py` 中已有的 `ClinicalFeatureExtractor` 计算，`α` 为超参数（建议初始值 0.1）。

**不允许改动的文件：**
- `passing/qc_filter.py`（只读，用于计算描述符）
- `sd_lora/`、`generate/`（生成阶段，与本任务无关）
- `classify/models/lora.py`（LoRA 注入层，不修改）

---

## 二、需要新建或修改的文件

### 2.1 新建：`classify/roi_descriptor.py`

**功能：** 对一个 batch 的图像路径，离线预计算 `s(x)`，缓存到磁盘，训练时按路径查表读取。

**实现要求：**

```python
# 核心接口
class ROIDescriptorCache:
    """
    预计算并缓存所有训练图像的 4 维描述符。
    缓存文件为 JSON，key = 图像绝对路径，value = [s1, s2, s3, s4]。
    """
    def __init__(self, cache_path: str, model_path: str):
        # cache_path: 缓存 JSON 文件路径，如 /root/autodl-tmp/runs/roi_descriptor_cache.json
        # model_path: MediaPipe face_landmarker.task 模型路径
        ...

    def build(self, image_dirs: list[str]):
        """
        遍历 image_dirs 下所有图像，调用 ClinicalFeatureExtractor 提取描述符并保存。
        已存在缓存的图像跳过（增量更新）。
        提取失败的图像记录 null，训练时跳过辅助损失。
        """
        ...

    def get(self, img_path: str) -> list[float] | None:
        """返回 [s1, s2, s3, s4] 或 None（提取失败时）"""
        ...
```

**复用现有代码的方式：**

```python
# 从 passing/qc_filter.py 中导入（注意：classify/ 和 passing/ 是同级目录）
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'passing'))
from qc_filter import ClinicalFeatureExtractor, DataFilter
```

`ClinicalFeatureExtractor` 的初始化需要 `model_path`（MediaPipe `.task` 文件路径），已知在服务器上位于：
```
/root/autodl-tmp/face_landmarker.task
```

提取单张图像描述符的调用方式参考 `qc_filter.py` 的 `_extract_descriptor` 方法，返回一个 `np.ndarray` shape `(4,)`，对应 `[temporal_ratio, orbital_ratio, cheek_texture, jawline_sharpness]`。

**归一化策略（重要）：** 4 个维度量纲不同，MSE 损失会被大值维度主导。建议在 `build()` 阶段统计训练集的均值和标准差，将描述符归一化到 `[0, 1]` 区间，并将归一化参数保存在缓存 JSON 中供训练时使用。

---

### 2.2 修改：`classify/data.py`

**目标：** 让 DataLoader 在返回 `(image, label)` 的同时，可选地返回 `(image, label, descriptor)`。

**修改位置：** `FixedLabelImageFolder` 类。

```python
class FixedLabelImageFolder(Dataset):
    def __init__(self, root, transform, class_names,
                 descriptor_cache=None):  # ← 新增参数
        ...
        self.descriptor_cache = descriptor_cache  # ROIDescriptorCache 实例或 None

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = default_loader(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        if self.descriptor_cache is not None:
            desc = self.descriptor_cache.get(img_path)
            if desc is not None:
                desc_tensor = torch.tensor(desc, dtype=torch.float32)
            else:
                desc_tensor = torch.full((4,), float('nan'))  # 失败时用 NaN 标记
            return image, label, desc_tensor

        return image, label
```

**注意：** `get_data_loader()` 和 `get_synth_train_data_loader()` 函数也需要接受 `descriptor_cache` 参数并向下传递，但默认值为 `None`，保持向后兼容（原来的训练流程不受影响）。

---

### 2.3 修改：`classify/models/clip.py`

**目标：** 在 CLIP 图像特征之上添加一个 4 维回归头。

**修改位置：** `CLIP` 类。

```python
class CLIP(nn.Module):
    def __init__(self, ..., use_roi_aux_head: bool = False):  # ← 新增参数
        super().__init__()
        ...
        # 在现有初始化代码末尾添加：
        self.use_roi_aux_head = use_roi_aux_head
        if use_roi_aux_head:
            # CLIP ViT-B/16 的图像特征维度为 512
            self.roi_head = nn.Sequential(
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, 4),
            )
            # roi_head 参数需要加入可训练参数列表

    def learnable_params(self):
        params = [p for p in self.clip.parameters() if p.requires_grad]
        if self.use_roi_aux_head:
            params += list(self.roi_head.parameters())
        return params

    def forward(self, x, tokenized_text=None, output_features=False, **kwargs):
        ...
        image_feats = self.forward_image(x)  # shape: (B, 512)
        text_feats = self.forward_text(tokenized_text)
        logit_scale = self.clip.logit_scale.exp()
        logits_per_image = logit_scale * image_feats @ text_feats.t()

        if self.use_roi_aux_head:
            # 从归一化前的特征计算回归输出
            roi_pred = self.roi_head(image_feats)  # shape: (B, 4)
            if output_features:
                return {"logits": logits_per_image, "roi_pred": roi_pred,
                        "image_feats": image_feats, "text_feats": text_feats}
            return logits_per_image, roi_pred  # 训练时返回 tuple

        if output_features:
            return {"logits": logits_per_image, "image_feats": image_feats, "text_feats": text_feats}
        return logits_per_image
```

**向后兼容：** `use_roi_aux_head=False`（默认）时，`forward()` 返回值与原来完全一致，现有训练脚本不需要改动。

---

### 2.4 修改：`classify/main.py`

**目标：** 在训练循环中计算并加入辅助损失。

#### (a) 在 `main()` 中初始化描述符缓存和辅助头

```python
def main(args):
    ...
    # 在 Model and optimizer 部分，CLIP 初始化之前：
    descriptor_cache = None
    if getattr(args, 'use_roi_aux_head', False):
        from roi_descriptor import ROIDescriptorCache
        descriptor_cache = ROIDescriptorCache(
            cache_path=args.roi_descriptor_cache_path,
            model_path=args.mediapipe_model_path,
        )
        # 收集所有训练图像目录，构建缓存（如已存在则跳过）
        descriptor_cache.build([
            args.real_train_data_dir,   # my_dataset_binary/seed0（真实图，113张）
            SYNTH_RAW_DIR,              # raw 合成图全量目录（9000张/类，全部建缓存）
            SYNTH_QC_DIR,               # qc 过滤图目录（~864张，增量补充）
        ])
        # 说明：raw 有 9000 张/类但训练时 NIPC=500 只取前500张。
        # 全量建缓存不影响训练逻辑：DataLoader 按实际路径查表，命中就用辅助损失，
        # 命中不到（路径不在缓存里）就跳过辅助损失，分类损失照常计算。
        # SYNTH_RAW_DIR / SYNTH_QC_DIR 从 local.yaml 或环境变量读取，
        # 与 run_roi_aux.sh 中的路径保持一致。

    # CLIP 初始化时传入 use_roi_aux_head
    if args.model_type == "clip":
        model = CLIP(
            ...,
            use_roi_aux_head=getattr(args, 'use_roi_aux_head', False),
        )
    ...
    # DataLoader 初始化时传入 descriptor_cache
    train_loader, val_loader = load_data_loader(args, descriptor_cache=descriptor_cache)
```

#### (b) 修改 `train_one_epoch()` 中的损失计算

```python
def train_one_epoch(...):
    ...
    for it, batch in enumerate(...):
        # 解包 batch（兼容有无描述符两种情况）
        if len(batch) == 3:
            image, label, descriptor = batch
            has_descriptor = True
        else:
            image, label = batch
            descriptor = None
            has_descriptor = False

        ...

        with torch.cuda.amp.autocast(...):
            # 根据模型是否有辅助头选择 forward 模式
            if args.use_roi_aux_head:
                logit, roi_pred = model(image)  # roi_pred: (B, 4)
            else:
                logit = model(image)

            # 原有分类损失（保持原逻辑不变）
            loss = compute_classification_loss(logit, label, is_real, args)  # 封装原有 real/synth 分支

            # 辅助 ROI 回归损失
            if args.use_roi_aux_head and has_descriptor and descriptor is not None:
                descriptor = descriptor.cuda(non_blocking=True)
                # 过滤掉提取失败的样本（NaN）
                valid_mask = ~torch.isnan(descriptor).any(dim=1)
                if valid_mask.sum() > 0:
                    loss_roi = F.mse_loss(
                        roi_pred[valid_mask],
                        descriptor[valid_mask],
                    )
                    loss = loss + args.alpha_roi * loss_roi
                    metric_logger.update(loss_roi=loss_roi.item())
```

---

### 2.5 修改：`classify/config.py`

在 `argparse` 参数列表中新增以下参数：

```python
# ROI 辅助监督相关
parser.add_argument("--use_roi_aux_head", type=str2bool, default=False,
                    help="是否启用 ROI 描述符辅助回归头")
parser.add_argument("--alpha_roi", type=float, default=0.1,
                    help="ROI 辅助损失权重")
parser.add_argument("--roi_descriptor_cache_path", type=str,
                    default="/root/autodl-tmp/runs/roi_descriptor_cache.json",
                    help="描述符缓存文件路径")
parser.add_argument("--mediapipe_model_path", type=str,
                    default="/root/autodl-tmp/face_landmarker.task",
                    help="MediaPipe face_landmarker.task 文件路径")
```

---

### 2.6 新建：`classify/run_roi_aux.sh`

基于 `run_both_real_and_synth.sh` 复制并添加新参数：

```bash
#!/bin/bash
# 方向 A：ROI 描述符辅助监督训练脚本
# 用法示例：
#   ALPHA_ROI=0.1 SYNTH_VARIANT=qc bash run_roi_aux.sh

GPU="0"
MODEL_TYPE="${MODEL_TYPE:-clip}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/runs/ablation/roi_aux_outputs}"
SYNTH_VARIANT="${SYNTH_VARIANT:-qc}"
ALPHA_ROI="${ALPHA_ROI:-0.1}"
EXP_NAME="${EXP_NAME:-clip_roi_aux_alpha${ALPHA_ROI}_${SYNTH_VARIANT}}"
OUTPUT_DIR="${OUTPUT_ROOT}/${EXP_NAME}"

# ... （其余变量与 run_both_real_and_synth.sh 相同）...

CUDA_VISIBLE_DEVICES=$GPU WANDB_MODE=disabled "$PYTHON_BIN" main.py \
  # ... （与原脚本相同的参数）... \
  --use_roi_aux_head=True \
  --alpha_roi=$ALPHA_ROI \
  --roi_descriptor_cache_path=/root/autodl-tmp/runs/roi_descriptor_cache.json \
  --mediapipe_model_path=/root/autodl-tmp/face_landmarker.task \
  ${PARAM:-}
```

---

## 三、实现顺序（严格按此执行）

**第一步：构建描述符缓存（独立测试）**

先实现 `roi_descriptor.py`，然后单独运行以验证：

```bash
cd /root/nutricode/classify
python roi_descriptor.py \
  --image_dirs \
    /root/autodl-tmp/runs/cv/fold_4/my_dataset_binary/seed0 \
    /root/autodl-tmp/datadream_outputs/generated_images/my_dataset/sd2.1/gs3.5_nis50/shot20_seed0_template1_lr0.0001_ep240/train \
    /root/autodl-tmp/datadream_outputs/generated_images/my_dataset/sd2.1/gs3.5_nis50/shot20_seed0_template1_lr0.0001_ep240/filtered_train \
  --cache_path /root/autodl-tmp/runs/roi_descriptor_cache.json \
  --model_path /root/autodl-tmp/face_landmarker.task
```

三个目录的作用：
- `my_dataset_binary/seed0`：真实训练图（113张），归一化统计量只从这里计算
- `train/`（raw）：全量合成图（18000张），训练时只取前500张/类，但全部建缓存，增量写入
- `filtered_train/`（qc）：过滤后合成图（~864张），增量补充到同一个缓存文件

**为什么用 `my_dataset_binary/seed0` 而不是 `real_train_groups/seed0`：**
DataLoader（`FixedLabelImageFolder`）读取的是 `my_dataset_binary/seed0`，其中图像文件名带子组前缀（如 `malnourished_front_face__166_01.png`）。描述符缓存的 key 必须与 DataLoader 的 `sample_paths` 完全一致，否则 `cache.get(img_path)` 永远命中不到，辅助损失全部被跳过。

预期输出：分别打印三个目录的提取样本数、失败数，最终缓存总条目约 19000 条。真实图失败率 < 10%，合成图由于姿态多样失败率可能达到 20-30%，属于正常现象（命中不到的样本训练时自动跳过辅助损失）。

**第二步：验证 DataLoader 返回值**

在 Python 交互环境中：

```python
from classify.data import get_data_loader
from classify.roi_descriptor import ROIDescriptorCache

cache = ROIDescriptorCache(cache_path=..., model_path=...)
train_loader, _ = get_data_loader(..., descriptor_cache=cache)
batch = next(iter(train_loader))
assert len(batch) == 3, "应返回 (image, label, descriptor)"
image, label, descriptor = batch
assert descriptor.shape == (batch_size, 4)
print("descriptor sample:", descriptor[:3])  # 检查有无 NaN
```

**第三步：验证模型 forward 返回值**

```python
from classify.models.clip import CLIP
model = CLIP(dataset='my_dataset', is_lora_image=True, is_lora_text=True,
             clip_download_dir=..., use_roi_aux_head=True).cuda()
dummy = torch.randn(4, 3, 224, 224).cuda()
logits, roi_pred = model(dummy)
assert logits.shape == (4, 2)   # 二分类
assert roi_pred.shape == (4, 4)  # 4 维描述符
print("forward OK")
```

**第四步：跑完整训练（alpha=0.1，raw 合成数据）**

```bash
ALPHA_ROI=0.1 SYNTH_VARIANT=raw bash run_roi_aux.sh

```

观察 loss 曲线，`loss_roi` 应在前几个 epoch 快速下降到 < 0.5（归一化后）。如果 `loss_roi` 不下降，检查描述符是否正确归一化。

**第五步：消融对比实验**

实验分两层，对应论文中不同的对比目的：

**第一层：复现论文 Table 2 基线（验证环境正确）**

```bash
# DataDream baseline：raw 合成数据，无过滤，无辅助头
# 对应论文 Table 2 的 DataDream 行
# NIPC=500 限制每类取前500张
SYNTH_VARIANT=raw bash run_both_real_and_synth.sh

# NutriDiff baseline：qc 过滤数据，无辅助头
# 对应论文 Table 2 的 NutriDiff 行
SYNTH_VARIANT=qc bash run_both_real_and_synth.sh
```

两个结果应与论文 Table 2 接近（subject-level MCC，raw: MCC≈0.779，qc: MCC≈0.894）。
如果数值偏差 > 0.03，先排查环境问题，不要继续下一层。

**第二层：方向 A 主实验（alpha 消融）**

baseline 是 `SYNTH_VARIANT=raw`，主实验也用 raw，控制变量为"有无辅助头"：

```bash
# alpha=0.05
ALPHA_ROI=0.05 SYNTH_VARIANT=raw bash run_roi_aux.sh

# alpha=0.1（主实验，建议优先跑）
ALPHA_ROI=0.1 SYNTH_VARIANT=raw bash run_roi_aux.sh

# alpha=0.3
ALPHA_ROI=0.3 SYNTH_VARIANT=raw bash run_roi_aux.sh
```

**完整对比表（实验结束后按此格式整理结果）：**

| 方法 | 合成数据 | 辅助头 | α | Acc | AUC | F1 | MCC |
|---|---|---|---|---|---|---|---|
| DataDream | raw (500/类) | ✗ | — | | | | |
| NutriDiff | qc (~430/类) | ✗ | — | | | | |
| DataDream + aux | raw (500/类) | ✓ | 0.05 | | | | |
| DataDream + aux | raw (500/类) | ✓ | 0.1 | | | | |
| DataDream + aux | raw (500/类) | ✓ | 0.3 | | | | |

所有指标读取各实验 `output_dir/detailed_prediction_results.json` 中的 `subject_level_metrics`。

**alpha 最优值判断标准：** 以 subject-level MCC 为主指标选择最优 alpha，要求同时满足：
1. MCC 高于 raw baseline（> 0.820）
2. `loss_roi` 在 40 epoch 内收敛（不震荡）
3. 分类 loss 收敛速度不明显慢于 baseline

---

## 四、关键约束和边界条件

**描述符计算失败的处理：**
- 侧脸（left/right 45°）MediaPipe 检测失败率较高，当 `descriptor = None` 时，该样本只计算分类损失，不计算 `loss_roi`，通过 `valid_mask` 过滤实现。
- 不要因描述符缺失而丢弃样本，分类损失必须对所有样本计算。

**合成图像的描述符：**
- raw 合成图（train/）和 qc 过滤图（filtered_train/）都需要预计算描述符，写入同一个缓存文件（增量更新，不重复计算）。
- `descriptor_cache.build()` 接受多个目录，按顺序遍历，已缓存的路径自动跳过。
- 归一化统计量（均值、标准差）**只用真实训练图**计算，合成图的描述符用相同统计量归一化，不重新统计。
- 训练时 DataLoader 按实际加载的图像路径查缓存，命中就加辅助损失，命中不到就只算分类损失，不报错也不丢样本。

**归一化：**
- 4 维描述符中 `cheek_texture`（颧颊拉普拉斯方差）的数值范围远大于其他三维，必须归一化。
- 归一化统计量只在真实训练图像上计算，合成图的归一化使用相同的统计量。
- 归一化参数保存在缓存 JSON 中，字段名 `normalize_stats: {mean: [...], std: [...]}`。

**辅助头不参与评估：**
- `eval()` 函数和 `analyze_predictions()` 不需要修改，它们只用 `logit`（分类输出）。
- 在 `eval()` 中，如果模型返回 tuple，只取第一个元素：
  ```python
  output = model(image)
  if isinstance(output, tuple):
      output = output[0]  # 只取 logit
  ```

**checkpoint 兼容性：**
- `save_model()` 不需要修改，`model.state_dict()` 会自动包含 `roi_head` 的权重。
- `analyze_predictions()` 中加载 checkpoint 时，`strict=False` 已经设置，roi_head 的权重会被正确加载。

---

## 五、预期结果和判断标准

**实验成功的判断标准：**

1. `loss_roi` 在 40 epoch 内收敛到 < 0.2（归一化后的 MSE）
2. subject-level MCC 相比 baseline（无辅助头）提升 ≥ 0.01（即从 0.854 提升到 ≥ 0.864）
3. 不出现 `loss_roi=NaN`（如出现，检查描述符归一化是否异常）

**alpha 选择建议：**
- 如果 `loss_roi` 收敛太慢（> 10 epoch 才开始下降），说明 alpha 太小，尝试 0.3
- 如果分类 loss 收敛明显变慢，说明 alpha 太大，回退到 0.05
- 最优 alpha 通过 subject-level MCC 选择，不是 loss_roi

**实验失败的诊断：**

| 现象 | 原因 | 解决 |
|---|---|---|
| `loss_roi = NaN` 从第 1 步开始 | 描述符未归一化，数值爆炸 | 检查 cache JSON 的 normalize_stats |
| descriptor 全为 NaN | MediaPipe 初始化失败 | 检查 face_landmarker.task 路径 |
| 分类指标反而下降 | alpha 太大 | 将 alpha 从 0.1 降至 0.05 |
| `loss_roi` 不下降 | roi_head 梯度未传到 | 检查 learnable_params() 是否包含 roi_head |

---

## 六、代码质量要求

- `roi_descriptor.py` 需要有 `if __name__ == '__main__':` 入口，接受 `--image_dirs`、`--cache_path`、`--model_path` 三个命令行参数，支持独立运行测试
- 所有新增函数加中文注释，风格与现有代码保持一致
- 不修改现有代码逻辑，只在扩展点（`__init__` 新参数、`forward` 新返回值）上追加
- 新增参数全部有 `default` 值且不影响 `use_roi_aux_head=False` 时的行为
