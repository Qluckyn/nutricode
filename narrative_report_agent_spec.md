# 任务规格书：结构化临床诊断依据自动生成模块（路线A：规则模板）

> v2.0 —— 本版本已根据实际环境走查结果重新整理，所有路径、阈值逻辑、过滤规则均已在真实缓存文件上验证过，可直接交付 Agent 实现。
> 目标仓库：`/root/nutricode/`（或本地开发环境对应路径），工作环境：Python 3.10。
> 本任务**不依赖 GPU**，可在 CPU 环境下独立开发和调试，只依赖已缓存的 JSON 结果文件。

---

## 一、任务目标

在现有可解释性分析（`classify/roi_attention_analysis.py`）与临床描述符提取（`classify/roi_descriptor.py`）的输出基础上，新增一个**规则驱动的自然语言诊断依据生成模块**，将每个受试者的四维ROI临床描述符数值与四个ROI的注意力归因分数，转换为结构化的中文诊断依据文本，例如：

> "颞部相对亮度偏低（模型对该区域的关注度显著高于全局均值），提示颞肌萎缩，符合营养不良表现；下颌轮廓梯度偏高，提示皮下脂肪流失后骨性标志突出。"

**核心原则：本模块是确定性规则系统，不训练任何生成模型，不引入语言模型幻觉风险。** 所有文本内容必须可追溯到具体的数值判断依据，禁止生成任何无法从输入数值反推出来的临床结论。

**不允许改动的文件：**
- `passing/qc_filter.py`（只读，四维描述符计算逻辑）
- `classify/roi_descriptor.py`（只读，描述符缓存逻辑）
- `classify/roi_attention_analysis.py`（只读，注意力归因分数计算逻辑；本任务只消费其输出的 JSON，不修改其内部逻辑）
- `classify/models/`、`sd_lora/`、`generate/`（与本任务无关）

---

## 二、输入数据（已在实际环境中确认存在、格式匹配、内容有效）

### 2.1 ROI 临床描述符缓存

**文件路径：** `/root/autodl-tmp/runs/roi_descriptor_cache_with_test.json`

这份缓存文件的构建历史（供 Agent 理解数据来源，无需重新操作）：
1. 最初由以下命令构建，覆盖真实训练集 + 未过滤合成图 + 过滤后合成图三类数据：
   ```bash
   python roi_descriptor.py \
     --image_dirs \
       /root/autodl-tmp/runs/cv/fold_4/my_dataset_binary/seed0 \
       /root/autodl-tmp/datadream_outputs/generated_images/my_dataset/sd2.1/gs3.5_nis50/shot20_seed0_template1_lr0.0001_ep240/train \
       /root/autodl-tmp/datadream_outputs/generated_images/my_dataset/sd2.1/gs3.5_nis50/shot20_seed0_template1_lr0.0001_ep240/filtered_train \
     --cache_path /root/autodl-tmp/runs/roi_descriptor_cache.json \
     --model_path /root/autodl-tmp/face_landmarker.task
   ```
2. 由于该缓存缺少测试集（`roi_attention_analysis.py` 实际分析的54人），已复制为新文件并增量补充测试集，**保持第一个目录不变**以确保 `normalize_stats` 不被重新计算：
   ```bash
   cp roi_descriptor_cache.json roi_descriptor_cache_with_test.json
   python roi_descriptor.py \
     --image_dirs \
       /root/autodl-tmp/runs/cv/fold_4/my_dataset_binary/seed0 \
       /root/autodl-tmp/test_data \
     --cache_path /root/autodl-tmp/runs/roi_descriptor_cache_with_test.json \
     --model_path /root/autodl-tmp/face_landmarker.task
   ```

已核实的缓存内容（供 Agent 自查断言使用）：

| 数据来源 | 条目数 | 说明 |
|---|---|---|
| `fold_4/my_dataset_binary/seed0`（真实训练集） | 113 | 38人×3视角=114，减1张 landmark 检测失败 |
| `datadream_outputs/.../train`（未过滤合成候选） | 12098 | 不参与本模块任何计算，忽略 |
| `.../filtered_train`（过滤后合成图） | 1747 | 不参与本模块任何计算，忽略 |
| `test_data`（真实测试集） | 163 | 54人×3视角=162，多出1条属于路径匹配误差或额外文件，其中1条提取失败（见下方已知个例） |

`normalize_stats` 内容（已确认补建测试集前后完全一致，未被污染）：
```json
{
  "mean": [0.9087313115740592, 0.9143844458708342, 44.76355005012703, 23.910795792412934],
  "std":  [0.14159902800847937, 0.1174638251228731, 75.09795952753632, 9.227219380415926],
  "normalization": "sigmoid_zscore",
  "source": "first_image_dir",
  "n_real_valid": 113
}
```

**已知个例（Agent 无需处理，`aggregate_subject_descriptors`/`aggregate_subject_views` 的聚合逻辑已覆盖此类情况）：** 受试者 `171`（`malnourished_face` 组）的右45°视角 `171_03.png` landmark 提取失败（`descriptors` 中对应值为 `null`），正面 `171_01.png` 和左45° `171_02.png` 均正常。该受试者报告应仅用可用的两个视角聚合生成，并在输出中标注 `"views_used": ["front", "left_45"]`。

缓存 JSON 结构：
```json
{
  "normalize_stats": { "mean": [...], "std": [...], "normalization": "sigmoid_zscore", "n_real_valid": 113 },
  "descriptors": {
    "/root/autodl-tmp/test_data/malnourished_face/171_01.png": [0.354, 0.244, 0.384, 0.235],
    "/root/autodl-tmp/test_data/malnourished_face/171_02.png": [0.636, 0.400, 0.435, 0.279],
    "/root/autodl-tmp/test_data/malnourished_face/171_03.png": null,
    ...
  },
  "raw_descriptors": { "...同 key，原始未归一化数值..." }
}
```

四维顺序固定为 `[temporal_ratio, orbital_ratio, cheek_texture, jawline_sharpness]`，与 `passing/qc_filter.py` 中 `DataFilter._metrics_to_vector` 的顺序严格一致。**归一化值域为 [0, 1]，由训练集真实样本（113条）的 sigmoid-zscore 统计量决定，0.5 附近对应训练集均值。**

**临床方向语义（务必与代码方向核对一致，不可反向）：**

| 维度 | 名称 | 数值越低 | 数值越高 |
|---|---|---|---|
| s1 | temporal_ratio（颞部相对亮度） | 越暗，提示颞肌萎缩越明显 | 越接近正常 |
| s2 | orbital_ratio（眶周相对亮度） | 越暗，提示眶周凹陷越明显 | 越接近正常 |
| s3 | cheek_texture（颧颊纹理方差） | 越接近正常 | 越大，提示颧颊皮下脂肪流失、皮肤纹理起伏增加 |
| s4 | jawline_sharpness（下颌轮廓梯度） | 越接近正常 | 越大，提示下颌皮下脂肪流失、骨性轮廓锐化 |

### 2.2 ROI 注意力归因分数

**文件路径：** `/root/autodl-tmp/runs/vis/roi_validation_full/roi_attention_records.json`

由 `classify/roi_attention_analysis.py` 的 `analyze_image_target` 生成，JSON 结构为一个 list，每条记录对应"一张图像 × 一个 target_class"的组合。

**已确认：该文件由 `--targets both` 模式生成，`target_class` 字段同时包含 `"malnourished_face"` 和 `"normal_face"` 两种取值**，即同一张图像在文件中出现两条记录，分别对应对两个类别的梯度归因结果，`attr_signed_roi_{roi}_balance` 的符号含义随 `target_class` 完全相反。**本模块所有逻辑必须固定只使用 `target_class == "malnourished_face"` 的记录**（详见 3.3 节），这是强制项。

记录结构：
```json
{
  "true_class": "malnourished_face",
  "image_path": "/abs/path/to/subjectXX_01.png",
  "subject_id": "subjectXX",
  "view": "front",
  "target_class": "malnourished_face",
  "predicted_class": "malnourished_face",
  "malnourished_probability": 0.87,
  "attr_pos_roi_temporal_enrichment": 1.42,
  "attr_pos_roi_orbital_enrichment": 0.95,
  "attr_pos_roi_malar_enrichment": 1.18,
  "attr_pos_roi_jawline_enrichment": 1.05,
  "attr_signed_roi_temporal_balance": 0.21,
  "attr_signed_roi_orbital_balance": -0.03,
  "attr_signed_roi_malar_balance": 0.08,
  "attr_signed_roi_jawline_balance": 0.02,
  ...
}
```

关键字段说明：
- `attr_pos_roi_{roi}_enrichment`：该ROI区域的正向注意力密度 / 全图平均注意力密度。**> 1 表示模型对该区域的关注显著高于随机水平，是"模型是否在看这个区域"的核心判据。**
- `attr_signed_roi_{roi}_balance`：带符号平衡度量，**在 `target_class=="malnourished_face"` 的记录中，正值表示该区域的注意力偏向支持"营养不良"判断，负值表示偏向支持"正常"判断。**（该数值来自对 `score = logit(malnourished) - logit(normal)` 的反向传播，target_class 不同则梯度方向不同，故必须固定筛选。）

---

## 三、需要新建的文件

### 3.1 新建：`classify/narrative_report.py`

**核心接口：**

```python
from dataclasses import dataclass
from typing import Optional

ROI_NAMES = ["temporal", "orbital", "malar", "jawline"]
ROI_CN_NAMES = {
    "temporal": "颞部", "orbital": "眶周",
    "malar": "颧颊", "jawline": "下颌缘",
}
# 与 roi_descriptor.py 中四维向量顺序一一对应
ROI_TO_DESCRIPTOR_INDEX = {"temporal": 0, "orbital": 1, "malar": 2, "jawline": 3}
# 方向：low_is_concerning=True 表示数值越低越异常（temporal/orbital）
#      False 表示数值越高越异常（malar/jawline）
ROI_DIRECTION = {
    "temporal": True, "orbital": True,
    "malar": False, "jawline": False,
}
# 固定筛选目标类别，见 2.2 节说明；不允许改为其他值
REQUIRED_TARGET_CLASS = "malnourished_face"


@dataclass
class ROIFinding:
    roi: str                       # "temporal" / "orbital" / "malar" / "jawline"
    descriptor_value: float        # 归一化后的 [0,1] 数值，三视角聚合后的中位数
    severity_level: str            # "normal" / "mild" / "severe"，固定三级
    attention_enrichment: float    # 该 ROI 的注意力富集度（仅取自 target_class=="malnourished_face"）
    attention_attended: bool       # 是否判定为"模型关注该区域"（enrichment > attended_threshold）
    attention_balance: float       # 带符号平衡度量（同上，仅取自 malnourished_face 目标类）
    sentence: str                  # 该 ROI 对应生成的中文短句（可能为空字符串）


def classify_severity(descriptor_value: float, direction_low_is_concerning: bool,
                       thresholds: tuple) -> str:
    """
    将归一化描述符值（[0,1] 空间，来自 roi_descriptor_cache 的 "descriptors" 字段，
    不是 "raw_descriptors"）映射为三级严重程度：{"normal", "mild", "severe"}。

    thresholds = (low_or_high_q, mid_q)，来自 build_thresholds()，
    基于真实训练集（113条有效样本）的分位数计算，禁止使用任意猜测的固定值。

    若 direction_low_is_concerning=True（temporal / orbital，数值越低越异常）：
        descriptor_value <= low_or_high_q         -> "severe"   （low_or_high_q = 10th percentile）
        low_or_high_q < descriptor_value <= mid_q -> "mild"     （mid_q = 35th percentile）
        descriptor_value > mid_q                  -> "normal"

    若 direction_low_is_concerning=False（malar / jawline，数值越高越异常）：
        descriptor_value >= low_or_high_q         -> "severe"   （low_or_high_q = 90th percentile）
        mid_q <= descriptor_value < low_or_high_q -> "mild"     （mid_q = 65th percentile）
        descriptor_value < mid_q                  -> "normal"
    """
    low_or_high_q, mid_q = thresholds
    if direction_low_is_concerning:
        if descriptor_value <= low_or_high_q:
            return "severe"
        elif descriptor_value <= mid_q:
            return "mild"
        else:
            return "normal"
    else:
        if descriptor_value >= low_or_high_q:
            return "severe"
        elif descriptor_value >= mid_q:
            return "mild"
        else:
            return "normal"


def build_thresholds(descriptor_cache_path: str, roi: str,
                      real_train_image_paths: list) -> tuple:
    """
    计算该 ROI 维度的分位数阈值，用于 classify_severity。

    实现要求：
    1. 从 roi_descriptor_cache_with_test.json 的 "descriptors" 字段（归一化 [0,1] 数值）
       读取，不能用 "raw_descriptors"——两者数值空间不同（raw 是原始尺度，
       如 cheek_texture 的 raw 均值约44.7，与归一化后的 [0,1] 完全不在一个空间）。
    2. real_train_image_paths 必须显式传入，固定指向
       `/root/autodl-tmp/runs/cv/fold_4/my_dataset_binary/seed0` 目录下的图像路径
       （通过 glob 枚举，不要依赖缓存文件自身区分真实/合成——缓存 key
       只是绝对路径，没有任何标记真实或合成的字段）。
    3. 分位数按方向选取：
       - direction_low_is_concerning=True 的 ROI（temporal, orbital）：
         取 10th percentile 作为 low_or_high_q，35th percentile 作为 mid_q
       - direction_low_is_concerning=False 的 ROI（malar, jawline）：
         取 90th percentile 作为 low_or_high_q，65th percentile 作为 mid_q
       返回顺序统一为 (low_or_high_q, mid_q)，与 classify_severity 的
       thresholds 参数顺序一致。
    4. 自查断言：真实训练集有效样本数应在 113 ± 5 范围内
       （与已确认的 normalize_stats.n_real_valid=113 对齐）；
       超出范围应直接抛异常并打印实际读取到的路径列表长度，
       防止误将测试集或合成图像路径传入 real_train_image_paths。
    """
    ...


def aggregate_subject_descriptors(descriptor_cache: dict, subject_id: str,
                                   image_dir: str = "/root/autodl-tmp/test_data") -> dict:
    """
    从 roi_descriptor_cache_with_test.json 的 "descriptors" 字段中，
    按 subject_id 匹配该受试者的三个视角图像路径
    （命名约定：{subject_id}_01.png=正面, _02.png=左45°, _03.png=右45°，
    与 classify/main.py 的 _parse_subject_id_from_path 保持一致的解析规则，
    建议直接复用该函数而不是重新实现一套解析逻辑）。

    对每个 ROI 维度，取三视角中非 null 的归一化描述符值的中位数。
    若某视角提取失败（值为 null，如已知的 171_03.png 个例），
    仅用剩余可用视角计算中位数，并在返回结果中记录：
        "views_used": ["front", "left_45"]   # 示例：右45°缺失时
    若三个视角全部为 null，应抛出异常（不应出现这种情况，
    出现即说明该受试者数据整体存在问题，需人工核查，
    不应静默生成空报告）。

    返回：{roi_name: 聚合后的归一化描述符值, "views_used": [...]}
    """
    ...


def generate_roi_sentence(roi: str, severity: str, attended: bool, balance: float) -> str:
    """
    从模板库中按 (roi, severity, attended) 组合选取一条中文短句模板并返回。
    severity 取值固定为 {"normal", "mild", "severe"}。

    若 severity == "normal"，通常不生成句子（返回空字符串），
    除非 attended=True 且 balance > 0（模型对一个"数值测量正常"的区域
    给出了偏向营养不良判断的高关注度，属于需要单独报告的反常注意力情况，
    对应 TEMPLATE_BANK 中每个 ROI 的 "anomalous_attend" 分支）。

    注意：调用方必须保证传入的 balance 值来自
    target_class=="malnourished_face" 的记录（见 3.3 节的强制过滤要求），
    本函数自身不做二次校验，由 aggregate_subject_views 在上游保证。
    """
    ...


def generate_subject_report(
    subject_id: str,
    descriptor_values: dict,      # {roi: 归一化值}，来自 aggregate_subject_descriptors
    attention_scores: dict,       # {roi: {"enrichment": ..., "balance": ...}}，来自 aggregate_subject_views
    thresholds: dict,             # {roi: (low_or_high_q, mid_q)}，来自 build_thresholds，按 ROI 分别计算
    predicted_class: str,
    malnourished_probability: float,
    attended_threshold: float = 1.15,
) -> dict:
    """
    生成单个受试者的完整诊断依据报告。

    返回结构：
    {
        "subject_id": subject_id,
        "predicted_class": predicted_class,
        "malnourished_probability": malnourished_probability,
        "views_used": [...],                # 透传自 aggregate 步骤
        "roi_findings": [ROIFinding, ...],   # 按 attention_enrichment 降序排列
        "narrative": "拼接后的完整中文段落",
        "narrative_sentence_count": int,
    }

    拼接逻辑：
    1. 对四个 ROI 分别调用 classify_severity(descriptor_values[roi], ROI_DIRECTION[roi], thresholds[roi])
       + generate_roi_sentence(...)
    2. 按 attention_enrichment 从高到低排序（模型最关注的区域优先陈述）
    3. 过滤掉 severity=="normal" 且 attended=False 的 ROI（不生成冗余的"正常"陈述）
    4. 用分号拼接非空句子，若全部为空则输出兜底句：
       "模型未检测到显著异常的面部区域特征，各ROI区域测量值均在正常范围内。"
    5. 段首统一加一句总括：
       f"该受试者预测为{predicted_class_cn}（置信度{prob:.1%}）。模型关注的关键面部区域证据如下：" + narrative正文
    """
    ...
```

### 3.2 模板库设计（写在同一文件内，作为模块级常量）

**要求：每个 (roi, severity) 组合至少准备 2 条同义句式，生成时随机选取一条，避免所有报告文字完全雷同。每个 ROI 固定需要 `severe`、`mild`、`anomalous_attend` 三组模板，`normal` 不需要模板。**

```python
TEMPLATE_BANK = {
    "temporal": {
        "severe": [
            "颞部相对亮度明显偏低，提示颞肌萎缩较为显著",
            "颞部区域亮度显著低于全脸均值，符合中重度颞肌萎缩表现",
        ],
        "mild": [
            "颞部相对亮度轻度偏低，提示可能存在颞肌轻度萎缩",
            "颞部区域亮度略低于正常范围，建议关注颞肌状态",
        ],
        "anomalous_attend": [
            "模型对颞部区域给予较高关注，但该区域测量值处于正常范围，建议结合其他证据综合判断",
        ],
    },
    "orbital": {
        "severe": [
            "眶周相对亮度明显偏低，提示眶周脂肪垫萎缩、凹陷较为明显",
            "眶周区域亮度显著低于全脸均值，符合中重度眶周凹陷表现",
        ],
        "mild": [
            "眶周相对亮度轻度偏低，提示可能存在轻度眶周凹陷",
            "眶周区域亮度略低于正常范围，建议关注眶周脂肪状态",
        ],
        "anomalous_attend": [
            "模型对眶周区域给予较高关注，但该区域测量值处于正常范围，建议结合其他证据综合判断",
        ],
    },
    "malar": {
        "severe": [
            "颧颊纹理方差显著升高，提示颧颊皮下脂肪明显流失、皮肤纹理起伏增大",
            "颧颊区域纹理复杂度明显高于正常范围，符合中重度脂肪流失表现",
        ],
        "mild": [
            "颧颊纹理方差轻度升高，提示可能存在轻度皮下脂肪流失",
            "颧颊区域纹理略高于正常范围，建议关注该区域软组织状态",
        ],
        "anomalous_attend": [
            "模型对颧颊区域给予较高关注，但该区域测量值处于正常范围，建议结合其他证据综合判断",
        ],
    },
    "jawline": {
        "severe": [
            "下颌轮廓梯度显著升高，提示皮下脂肪流失后骨性下颌轮廓明显锐化",
            "下颌缘区域轮廓锐利度明显高于正常范围，符合中重度脂肪流失表现",
        ],
        "mild": [
            "下颌轮廓梯度轻度升高，提示可能存在轻度皮下脂肪流失",
            "下颌缘区域轮廓略偏锐利，建议关注该区域软组织状态",
        ],
        "anomalous_attend": [
            "模型对下颌缘区域给予较高关注，但该区域测量值处于正常范围，建议结合其他证据综合判断",
        ],
    },
}
```

**注意：** 上述模板文字为初版草案，Agent 在实现时需要与项目负责人核对每条模板的临床表述是否准确（必要时请教营养科医师核实措辞），**不得擅自编造模板未覆盖的临床结论**。

### 3.3 三视角注意力分数聚合逻辑（强制 target_class 过滤）

```python
def aggregate_subject_views(records: list, subject_id: str) -> dict:
    """
    从 roi_attention_records.json 中筛选出该 subject_id 的注意力归因记录。

    第一步（强制，不可省略）：
        records = [r for r in records
                   if r["subject_id"] == subject_id
                   and r["target_class"] == "malnourished_face"]

    只保留 target_class=="malnourished_face" 的记录再做后续聚合。

    原因：attr_signed_roi_{roi}_balance 来自对
    "score = logit(malnourished) - logit(normal)" 反向传播得到的归因，
    target_class 不同，balance 符号的临床含义完全相反。叙事报告统一以
    "支持营养不良判断的证据强度"为叙述基准，因此必须固定
    target_class="malnourished_face"，不随该受试者的 true_class 或
    predicted_class 变化而改变取哪条记录——即便是 normal_face 受试者，
    也读取其 target_class="malnourished_face" 的那条记录，此时 balance
    若为负，就表示"该区域证据支持正常判断"，这正是模板逻辑期望的统一语义。

    已确认该文件由 --targets both 生成，即每张图像在文件中对应两条记录
    （malnourished_face 和 normal_face 各一条），此过滤为强制项，
    不是可选的防御性代码。

    过滤之后，对每个 ROI 的 enrichment / balance 取三视角中位数
    （不是均值，避免单一视角因人脸检测异常产生的离群值影响结果）。
    若某视角缺失（如已知的 171_03.png 个例），仅用可用视角计算中位数，
    并在返回结果中标注 "views_used": [...]。

    若过滤后记录数为 0，应抛出异常而不是静默返回空结果，
    防止下游生成一份内容全部缺失的"假报告"。

    返回：{roi_name: {"enrichment": 中位数, "balance": 中位数}, "views_used": [...]}
    """
    ...
```

### 3.4 新建 CLI 脚本：`classify/generate_narrative_reports.py`

与 `roi_attention_analysis.py` 的命令行风格保持一致：

```python
"""
用法示例：

python generate_narrative_reports.py \
    --descriptor_cache /root/autodl-tmp/runs/roi_descriptor_cache_with_test.json \
    --attention_records /root/autodl-tmp/runs/vis/roi_validation_full/roi_attention_records.json \
    --real_train_dirs /root/autodl-tmp/runs/cv/fold_4/my_dataset_binary/seed0 \
    --target_class malnourished_face \
    --output_dir /root/autodl-tmp/runs/narrative_reports \
    --attended_threshold 1.15

参数说明：
--real_train_dirs：显式指定真实训练集图像所在目录，用于枚举
    real_train_image_paths 传给 build_thresholds，不依赖缓存文件自身
    区分真实/合成。脚本启动时应打印实际枚举到的图像数量，
    供人工核对是否在 113~114 范围内（超出应报警并中止）。

--target_class：固定默认值 "malnourished_face"，不建议改动。
    若传入其他值，脚本应打印警告说明这会改变整个报告的叙述基准方向
    （见 3.3 节），并要求二次确认后才继续执行。
"""
```

**输出要求：**
1. `narrative_reports.json`：list，每个元素是 `generate_subject_report` 的完整返回结构
2. `narrative_reports.csv`：扁平化表格，每行一个受试者，列包括 `subject_id, predicted_class, malnourished_probability, views_used, narrative`，供快速人工浏览
3. `narrative_reports_readable.txt`：纯文本，逐受试者打印完整报告，方便直接拿给医生看

---

## 四、验证与测试要求

1. **单元测试**：对 `classify_severity`，构造边界值（low_or_high_q / mid_q 边界 ±0.001）测试三级分级是否正确、无越界，两个方向（`direction_low_is_concerning=True/False`）都要覆盖。
2. **数值空间一致性检查（必测）**：断言 `build_thresholds` 计算阈值时读取的是 `descriptors`（归一化 [0,1]）字段，不是 `raw_descriptors`。可用简单检查：从 cache 中随机取 10 条真实训练样本，其归一化值应落在 [0,1] 区间内；若发现传入 `classify_severity` 的值超出 [0,1]，说明误用了 raw 值，应立即报错。
3. **真实样本过滤检查（必测）**：断言 `build_thresholds` 内部用于计算分位数的样本数量在 113 ± 5 范围内（与 `normalize_stats.n_real_valid=113` 对齐），超出范围直接抛异常。
4. **target_class 过滤检查（必测，最容易被漏掉的一项）**：在 `aggregate_subject_views` 的单元测试中，构造一个包含同一 `subject_id` 但 `target_class` 分别为 `malnourished_face` 和 `normal_face`（balance 符号相反）的假记录列表，断言函数返回结果中的 balance 值只能来自 `malnourished_face` 那条记录。这是唯一会在真实数据上"悄无声息产生错误结果而不报错"的一类 bug，必须有专门测试覆盖，不能只靠人工抽查发现。
5. **缺失视角处理检查（必测，已有真实个例可直接用作测试用例）**：用受试者 `171` 的真实数据（`171_03.png` 提取失败）作为回归测试样本，验证 `aggregate_subject_descriptors` 和 `aggregate_subject_views` 均能在缺失一个视角的情况下正常生成报告，且 `views_used` 字段正确标注为 `["front", "left_45"]`。
6. **回归测试**：随机抽取测试集中 5 个真实受试者（`malnourished_face` 和 `normal_face` 各 2-3 个），人工核对生成的 `narrative` 字段：
   - 数值方向与文字表述是否一致（不能出现"数值偏高但文字说偏低"这类矛盾）
   - `malnourished_face` 组的报告是否显著比 `normal_face` 组包含更多 severity 非 normal 的发现
7. **禁止项检查**：跑一遍全部输出文本，确认没有任何句子包含模板库之外的临床术语或数值（规则系统本身应天然满足此项，写一个简单的断言测试即可）。

---

## 五、分阶段交付建议

| 阶段 | 交付物 | 预计工作量 |
|---|---|---|
| 阶段1 | `build_thresholds` + `classify_severity`，跑通阈值分箱，人工检查113例真实训练集分布是否合理 | 0.5天 |
| 阶段2 | `TEMPLATE_BANK` 初版 + `generate_roi_sentence` | 0.5天 |
| 阶段3 | `aggregate_subject_descriptors` + `aggregate_subject_views`（含 target_class 强制过滤）+ `generate_subject_report`，用受试者171跑通含缺失视角的端到端报告生成 | 0.5天 |
| 阶段4 | CLI脚本 + 全测试集（54人，163条描述符记录）批量生成，人工抽查报告质量 | 0.5天 |
| 阶段5（可选） | 请营养科医师对50份报告做"符合/部分符合/不符合"评审，统计一致率，作为该模块的验证指标写入论文 | 视医师配合时间而定 |

---

## 六、与论文的对接建议

该模块运行完成后，建议在毕业论文里新增一节（可放在"可解释性分析"章节之后），内容包括：
- 规则系统的设计逻辑（阈值分箱依据、模板库构建方式）
- 若完成阶段5，报告医师评审的一致率统计结果
- 2-3个典型受试者的报告案例展示（1个营养不良组、1个正常组，附对应的注意力热图截图）
