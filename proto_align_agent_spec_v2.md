# 任务规格书：原型引导的双空间一致性约束（Proto Align）

> 工作环境：AutoDL Linux，单张 RTX5090，Python 3.10，PyTorch + CLIP。
> 本任务在现有 NutriDiff 流水线基础上叠加，**不依赖方向 A（use_roi_aux_head）**，可独立运行。

---

## 一、任务目标

在分类器训练阶段，对合成图添加**原型对齐损失（Prototype Alignment Loss）**：
用真实样本在 CLIP 特征空间中的类均值向量（原型）作为锚点，
显式约束合成图的 CLIP 特征向量向本类原型靠近、远离异类原型。

**核心公式：**

```
L_total = λ_1 · L_CE_real + (1-λ_1) · L_CE_synth    ← 现有逻辑（λ_1=0.8）
        + β · L_proto                                  ← 新增（只对合成图计算）

L_proto = mean over synth samples of:
    max(0,  d(f_syn, p_y)  -  d(f_syn, p_ȳ)  +  margin)

其中：
  f_syn  = CLIP 图像特征（已 L2 归一化，shape: B×512）
  p_y    = 本类原型（归一化，shape: 512）
  p_ȳ   = 异类原型（归一化，shape: 512）
  d(a,b) = 1 - cosine_similarity(a, b)  （余弦距离）
  margin = 超参数，建议初始值 0.2
  β      = 超参数，建议初始值 0.1
```

**不允许改动的文件：**
- `passing/qc_filter.py`
- `sd_lora/`、`generate/`
- `classify/models/lora.py`
- `classify/data.py`
- `classify/roi_descriptor.py`

---

## 二、现有代码关键约束（必须先读懂再动手）

### 2.1 最新 batch 的四种解包方式

`train_one_epoch()` 里已有如下解包逻辑（**不要修改这段**）：

```python
if args.is_synth_train and args.is_pooled_fewshot:
    if len(batch) == 4:
        image, label, is_real, descriptor = batch   # 方向A开启时
    else:
        image, label, is_real = batch               # 方向A未开启时
else:
    if len(batch) == 3:
        image, label, descriptor = batch
    else:
        image, label = batch
```

原型对齐只在 `args.is_synth_train and args.is_pooled_fewshot` 为 True 时才有意义，
因为 `is_real` 标志位只在这个条件下存在。
`real_mask = (is_real == 1)`，`synth_mask = (is_real == 0)` 在这段解包逻辑之后已经可用。

### 2.2 CLIP forward 已支持 `output_features=True`

`classify/models/clip.py` 的 `forward()` 已经实现：

```python
# use_roi_aux_head=False 时（本任务默认情况）：
model(image)                          # 返回 logits_per_image，shape: B×2
model(image, output_features=True)    # 返回字典：
                                      # {"logits": ..., "image_feats": ..., "text_feats": ...}

# use_roi_aux_head=True 时（方向A）：
model(image)                          # 返回 (logits, roi_pred) tuple
model(image, output_features=True)    # 返回字典，多一个 "roi_pred" 键
```

`image_feats` 在 `forward_image()` 里已经做了 L2 归一化（`image_feats / image_feats.norm(dim=1, keepdim=True)`），**不需要再次归一化**。

### 2.3 `label_origin` 与 `label` 的区别

```python
label_origin = label                        # 解包后立即保存原始整数标签
label_origin = label_origin.cuda(...)

# 之后 label 可能被 CutMix/MixUp 变成 float one-hot
# PrototypeManager 需要整数标签做索引，必须用 label_origin，不能用 label
```

### 2.4 当前 `train_one_epoch()` 的 forward 块

位置在函数内 `with torch.cuda.amp.autocast(fp16_scaler is not None):` 块：

```python
roi_pred = None
if getattr(args, "use_roi_aux_head", False):
    logit, roi_pred = model(image)
else:
    logit = model(image)
```

原型对齐需要修改这个块，在获取 `logit` 的同时也获取 `image_feats`。

### 2.5 `train_one_epoch()` 函数签名（当前）

```python
def train_one_epoch(
    model, criterion, data_loader, optimizer, scheduler, epoch,
    fp16_scaler, cutmix_or_mixup, args,
    val_loader, best_stats, best_top1,
):
```

需要新增 `proto_manager=None` 参数，同时在 `main()` 的调用处也要追加。

---

## 三、需要新建或修改的文件

### 3.1 新建：`classify/proto_align.py`

完整实现如下，Agent 按此实现，不要改变类名和方法签名：

```python
"""
原型引导对齐模块（Proto Align）
维护每个类别在 CLIP 特征空间的原型向量，计算合成图的原型对齐损失。
只依赖 torch，不依赖项目其他模块。
"""
import torch
import torch.nn.functional as F


class PrototypeManager:
    """
    管理 CLIP 特征空间中的类原型向量。
    原型 = 真实样本特征的指数移动平均（EMA）。
    """

    def __init__(
        self,
        n_classes: int,
        feat_dim: int = 512,
        momentum: float = 0.999,
        device: str = "cuda",
    ):
        """
        Args:
            n_classes: 类别数（本项目为 2）
            feat_dim:  CLIP ViT-B/16 图像特征维度（固定 512）
            momentum:  EMA 动量，越大原型越稳定，建议 0.999
            device:    计算设备
        """
        self.n_classes = n_classes
        self.momentum = momentum
        self.device = device
        # 原型初始化为全零，第一个 batch 后被真实样本覆盖
        self.prototypes = torch.zeros(n_classes, feat_dim).to(device)
        self._initialized = [False] * n_classes

    @torch.no_grad()
    def update(
        self,
        feats: torch.Tensor,
        labels: torch.Tensor,
        real_mask: torch.Tensor,
    ):
        """
        用当前 batch 的真实样本特征更新原型（EMA）。

        Args:
            feats:      图像特征，shape (B, 512)，已 L2 归一化，float32
            labels:     整数标签，shape (B,)，值 0 或 1（必须是 label_origin）
            real_mask:  布尔掩码，shape (B,)，True = 真实样本
        """
        real_feats = feats[real_mask]
        real_labels = labels[real_mask]

        for c in range(self.n_classes):
            cls_mask = (real_labels == c)
            if cls_mask.sum() == 0:
                continue
            cls_mean = real_feats[cls_mask].mean(dim=0)
            cls_mean = F.normalize(cls_mean, dim=0)

            if not self._initialized[c]:
                self.prototypes[c] = cls_mean
                self._initialized[c] = True
            else:
                self.prototypes[c] = (
                    self.momentum * self.prototypes[c]
                    + (1.0 - self.momentum) * cls_mean
                )
                self.prototypes[c] = F.normalize(self.prototypes[c], dim=0)

    def compute_loss(
        self,
        feats: torch.Tensor,
        labels: torch.Tensor,
        synth_mask: torch.Tensor,
        margin: float = 0.2,
    ) -> torch.Tensor:
        """
        计算合成样本的原型对齐损失（Triplet-style）。

        对每张合成图：
            d_pos = 1 - cos(f_syn, p_y)    与本类原型的余弦距离
            d_neg = 1 - cos(f_syn, p_ȳ)   与异类原型的余弦距离
            loss_i = max(0, d_pos - d_neg + margin)

        Args:
            feats:      图像特征，shape (B, 512)，已 L2 归一化
            labels:     整数标签，shape (B,)（必须是 label_origin）
            synth_mask: 布尔掩码，shape (B,)，True = 合成样本
            margin:     triplet margin

        Returns:
            标量 loss（若无合成样本或原型未初始化则返回 0）
        """
        if not synth_mask.any():
            return torch.tensor(0.0, device=self.device)

        if not all(self._initialized):
            # 原型未完成初始化，跳过（训练开始几步内）
            return torch.tensor(0.0, device=self.device)

        # 确保 feats 是 float32（FP16 训练时可能是 float16）
        feats = feats.float()

        syn_feats = feats[synth_mask]       # (N_syn, 512)
        syn_labels = labels[synth_mask]     # (N_syn,)

        proto_pos = self.prototypes[syn_labels]       # (N_syn, 512)
        proto_neg = self.prototypes[1 - syn_labels]   # (N_syn, 512)，只有两类才成立

        # feats 和 prototypes 均已归一化，点积 = 余弦相似度
        sim_pos = (syn_feats * proto_pos).sum(dim=1)   # (N_syn,)
        sim_neg = (syn_feats * proto_neg).sum(dim=1)   # (N_syn,)

        d_pos = 1.0 - sim_pos
        d_neg = 1.0 - sim_neg

        loss = F.relu(d_pos - d_neg + margin)
        return loss.mean()


if __name__ == "__main__":
    """独立单元测试，运行：python proto_align.py"""
    pm = PrototypeManager(n_classes=2, feat_dim=512, momentum=0.999)

    B = 8
    feats = torch.randn(B, 512).cuda()
    feats = F.normalize(feats, dim=1)
    labels = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1]).cuda()
    real_mask  = torch.tensor([True,  True,  True,  True,
                                False, False, False, False]).cuda()
    synth_mask = ~real_mask

    pm.update(feats, labels, real_mask)

    assert all(pm._initialized), "原型应已初始化"
    assert abs(pm.prototypes[0].norm().item() - 1.0) < 1e-4, "原型0未归一化"
    assert abs(pm.prototypes[1].norm().item() - 1.0) < 1e-4, "原型1未归一化"
    print(f"prototypes initialized: {pm._initialized}")
    print(f"prototype[0] norm: {pm.prototypes[0].norm().item():.6f}")
    print(f"prototype[1] norm: {pm.prototypes[1].norm().item():.6f}")

    loss = pm.compute_loss(feats, labels, synth_mask, margin=0.2)
    print(f"proto loss: {loss.item():.6f}")
    assert loss.item() >= 0, "loss 应为非负数"
    loss.backward()
    print("backward OK")
    print("所有断言通过 ✓")
```

---

### 3.2 修改：`classify/main.py`

**共三处修改，不改动其他任何逻辑。**

#### 修改 A：顶部导入（在 `from models.resnet50 import ResNet50` 之后追加）

```python
from proto_align import PrototypeManager
```

#### 修改 B：`main()` 中初始化 PrototypeManager

位置：`model = model.cuda()` 之后，`criterion = nn.CrossEntropyLoss().cuda()` 之前，追加：

```python
# 原型管理器（只在启用原型对齐且使用合成数据时初始化）
proto_manager = None
if getattr(args, "use_proto_align", False) and args.is_synth_train:
    proto_manager = PrototypeManager(
        n_classes=args.n_classes,
        feat_dim=512,
        momentum=getattr(args, "proto_momentum", 0.999),
        device="cuda",
    )
```

然后修改训练循环中对 `train_one_epoch` 的调用，追加 `proto_manager` 参数：

```python
# 原调用（main() 中 for epoch 循环里）：
train_stats, best_stats, best_top1 = train_one_epoch(
    model, criterion, train_loader, optimizer, scheduler, epoch,
    fp16_scaler, cutmix_or_mixup, args,
    val_loader, best_stats, best_top1,
)

# 改为：
train_stats, best_stats, best_top1 = train_one_epoch(
    model, criterion, train_loader, optimizer, scheduler, epoch,
    fp16_scaler, cutmix_or_mixup, args,
    val_loader, best_stats, best_top1,
    proto_manager=proto_manager,    # 新增
)
```

#### 修改 C：`train_one_epoch()` 函数签名和 forward 块

**函数签名** 末尾追加参数：

```python
def train_one_epoch(
    model, criterion, data_loader, optimizer, scheduler, epoch,
    fp16_scaler, cutmix_or_mixup, args,
    val_loader, best_stats, best_top1,
    proto_manager=None,    # ← 新增，默认 None 保持向后兼容
):
```

**forward 块**，将现有这段：

```python
with torch.cuda.amp.autocast(fp16_scaler is not None):
    roi_pred = None
    if getattr(args, "use_roi_aux_head", False):
        logit, roi_pred = model(image)
    else:
        logit = model(image)
```

替换为：

```python
with torch.cuda.amp.autocast(fp16_scaler is not None):
    roi_pred = None
    image_feats = None
    need_proto = (
        proto_manager is not None
        and args.is_synth_train
        and args.is_pooled_fewshot
    )

    if getattr(args, "use_roi_aux_head", False):
        if need_proto:
            # 方向A + 原型对齐同时开启
            out = model(image, output_features=True)
            logit = out["logits"]
            roi_pred = out["roi_pred"]
            image_feats = out["image_feats"]
        else:
            logit, roi_pred = model(image)
    else:
        if need_proto:
            # 只开启原型对齐
            out = model(image, output_features=True)
            logit = out["logits"]
            image_feats = out["image_feats"]
        else:
            logit = model(image)
```

然后在 ROI 辅助损失代码块之后（`metric_logger.update(loss_roi=...)` 之后），追加原型对齐损失块：

```python
            # ── 原型对齐损失（新增）──────────────────────────────
            if need_proto and image_feats is not None:
                # 1. 用本 batch 的真实样本更新原型（不参与梯度）
                with torch.no_grad():
                    proto_manager.update(
                        image_feats.detach().float(),
                        label_origin,           # 整数标签，非 one-hot
                        real_mask.bool(),
                    )
                # 2. 计算合成图的原型对齐损失（参与梯度）
                loss_proto = proto_manager.compute_loss(
                    image_feats,
                    label_origin,
                    synth_mask.bool(),
                    margin=getattr(args, "proto_margin", 0.2),
                )
                loss = loss + args.beta_proto * loss_proto
                metric_logger.update(loss_proto=loss_proto.item())
            # ─────────────────────────────────────────────────────
```

**注意**：上述代码块必须放在 `with torch.cuda.amp.autocast(...)` 块内部，在分类损失和 ROI 辅助损失计算之后，在 `if not math.isfinite(loss.item()):` 检查之前。

---

### 3.3 修改：`classify/config.py`

在现有 `--alpha_roi` 参数之后追加：

```python
# 原型对齐相关
parser.add_argument("--use_proto_align", type=str2bool, default=False,
                    help="是否启用原型对齐损失")
parser.add_argument("--beta_proto", type=float, default=0.1,
                    help="原型对齐损失权重 β")
parser.add_argument("--proto_margin", type=float, default=0.2,
                    help="原型对齐 triplet margin")
parser.add_argument("--proto_momentum", type=float, default=0.999,
                    help="原型 EMA 动量")
```

---

### 3.4 新建：`classify/run_proto_align.sh`

复制 `run_both_real_and_synth.sh` 的全部内容，在最后的 `main.py` 调用命令里追加以下参数：

```bash
#!/bin/bash
# 原型对齐实验脚本
# 用法：BETA_PROTO=0.1 SYNTH_VARIANT=raw bash run_proto_align.sh

BETA_PROTO="${BETA_PROTO:-0.1}"
PROTO_MARGIN="${PROTO_MARGIN:-0.2}"

# ... 其余变量从 run_both_real_and_synth.sh 原样复制 ...

# 在 EXP_NAME 中体现超参：
EXP_NAME="${EXP_NAME:-clip_proto_align_beta${BETA_PROTO}_margin${PROTO_MARGIN}_${SYNTH_VARIANT:-raw}}"

# main.py 调用追加：
#   --use_proto_align=True \
#   --beta_proto=$BETA_PROTO \
#   --proto_margin=$PROTO_MARGIN \
#   --proto_momentum=0.999 \
```

---

## 四、实现顺序（严格按此执行，不要跳步骤）

**第一步：实现并单独验证 `proto_align.py`**

```bash
cd /root/nutricode/classify
python proto_align.py
```

必须看到：
```
prototypes initialized: [True, True]
prototype[0] norm: 1.000000（误差 < 1e-4）
prototype[1] norm: 1.000000（误差 < 1e-4）
proto loss: （正数）
backward OK
所有断言通过 ✓
```

如果 `loss=0.0`：检查 `synth_mask` 是否全 False。
如果 backward 报错：检查 `compute_loss` 的 `feats` 参数是否在计算图中（不要在外部 `detach()`）。

**第二步：验证修改不破坏 baseline**

```bash
SYNTH_VARIANT=raw bash run_both_real_and_synth.sh
```

结果与修改前的 raw baseline 完全一致（MCC 偏差 < 0.005）。
如果有偏差，说明修改 C 的 `need_proto=False` 路径影响了原有逻辑，回查。

**第三步：跑主实验（beta=0.1，raw 合成数据）**

```bash
BETA_PROTO=0.1 SYNTH_VARIANT=raw bash run_proto_align.sh
```

观察 `loss_proto` 变化趋势（在训练日志 `Averaged train stats:` 行查看）：
- 前 5 epoch：从较大值（>0.05）逐步下降
- 10 epoch 后：稳定在 0.005–0.05 之间
- 始终为 0.0：`proto_manager` 未正确传入，检查修改 B
- 始终不下降：margin 可能过小（合成图已经比 margin 更靠近本类），尝试增大到 0.3

**第四步：消融实验**

```bash
# baseline（无原型对齐，用于对比）
SYNTH_VARIANT=raw bash run_both_real_and_synth.sh

# beta 消融
BETA_PROTO=0.05  SYNTH_VARIANT=raw bash run_proto_align.sh
BETA_PROTO=0.1   SYNTH_VARIANT=raw bash run_proto_align.sh   # 主实验
BETA_PROTO=0.3   SYNTH_VARIANT=raw bash run_proto_align.sh

# margin 消融（固定 beta=0.1）
BETA_PROTO=0.1 PROTO_MARGIN=0.1 SYNTH_VARIANT=raw bash run_proto_align.sh
BETA_PROTO=0.1 PROTO_MARGIN=0.3 SYNTH_VARIANT=raw bash run_proto_align.sh
```

**完整对比表（填写 `detailed_prediction_results.json` 中的 `subject_level_metrics`）：**

| 方法 | 合成数据 | 原型对齐 | β | margin | Acc | AUC | F1 | MCC |
|---|---|---|---|---|---|---|---|---|
| DataDream baseline | raw | ✗ | — | — | | | | |
| + 原型对齐 | raw | ✓ | 0.05 | 0.2 | | | | |
| + 原型对齐 | raw | ✓ | 0.1 | 0.2 | | | | |
| + 原型对齐 | raw | ✓ | 0.3 | 0.2 | | | | |
| + 原型对齐 | raw | ✓ | 0.1 | 0.1 | | | | |
| + 原型对齐 | raw | ✓ | 0.1 | 0.3 | | | | |

---

## 五、关键约束和边界条件

**`label_origin` 必须用整数标签：**
解包后的 `label` 在 CutMix/MixUp 激活后会变成 float one-hot，
`PrototypeManager.update()` 和 `compute_loss()` 都用 `label_origin`（整数），
不能传 `label`，否则 `self.prototypes[syn_labels]` 会报 index 类型错误。

**FP16 兼容：**
`image_feats` 在 autocast 块内可能是 float16，
`compute_loss()` 开头已有 `feats = feats.float()` 转换，
`update()` 调用时已有 `.float()` 转换，两处都覆盖到了。

**`need_proto` 的位置：**
必须在 batch 解包之后（`real_mask`、`synth_mask` 已定义之后）才能正确使用，
但 `need_proto` 的条件判断不依赖 `real_mask`，可以在 forward 块开头计算。

**`use_proto_align=False` 时零开销：**
`need_proto=False` → 走原有 `logit = model(image)` 路径 → 无任何额外计算，
`proto_manager=None` → `metric_logger` 不会有 `loss_proto` 条目 → 日志格式不变。

**原型冷启动：**
训练最初几个 batch 若某类真实样本未出现，`_initialized[c]=False`，
`compute_loss()` 里 `all(self._initialized)` 为 False 时返回 0，安全跳过。
通常第一个 batch 就同时包含两类真实图，冷启动影响极小。

---

## 六、诊断表

| 现象 | 最可能原因 | 排查方法 |
|---|---|---|
| `loss_proto` 不出现在日志 | `proto_manager` 未传入 `train_one_epoch` | 检查修改 B 的函数调用处 |
| `loss_proto` 始终 0.0 | `synth_mask` 全为 False | 打印 `synth_mask.sum()` 确认 |
| `loss_proto` 始终 0.0 | `use_proto_align` 未传入脚本 | 检查 `run_proto_align.sh` 是否有 `--use_proto_align=True` |
| backward 报错 dtype | FP16 float16 与 float32 混合 | 确认 `compute_loss` 有 `feats = feats.float()` |
| index 类型错误 | 用了 `label` 而非 `label_origin` | 全局替换为 `label_origin` |
| baseline MCC 变了 | 修改 C 影响了原有分支 | 确认 `need_proto=False` 时走原有路径 |
| MCC 比 baseline 低 | β 过大抑制分类损失 | 将 β 从 0.1 降至 0.05 |
| `loss_proto` 不收敛 | margin 过小 | 打印 `(d_pos - d_neg).mean()`，margin 应略大于此值 |

---

## 七、代码质量要求

- `proto_align.py` 只依赖 `torch`，不导入项目任何其他模块
- `__main__` 入口包含断言，运行即可验证正确性
- 所有新增代码加中文注释，风格与现有代码一致
- `use_proto_align=False`（默认）时与现有代码行为完全相同，不增加任何计算开销
- 不修改 `data.py`、`models/lora.py`、`roi_descriptor.py` 等无关文件
