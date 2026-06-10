# NutriDiff 项目学习笔记

---

## 一、项目概览

**任务**：基于人脸图像的老年营养不良二分类筛查（malnourished_face vs normal_face）

**核心方法**：NutriDiff = LoRA微调SD2.1生成合成图像 + 马氏距离过滤 + CLIP分类器

**数据集**：
- 训练集：18例营养不良 + 20例正常，共38人（可自主配合采集）
- 测试集：27例营养不良 + 27例正常，共54人（卧床或接受肠内营养）
- 图像规格：每人3个视角（正面、左45°、右45°），共276张

---

## 二、目录结构

```
/root/autodl-tmp/runs/cv/fold_4/
├── real_train_groups/seed0/          # 原始6分类真实数据
│   ├── malnourished_front_face/
│   ├── malnourished_left_three-quarter_face/
│   ├── malnourished_right_three-quarter_face/
│   ├── normal_front_face/
│   ├── normal_left_three-quarter_face/
│   └── normal_right_three-quarter_face/
└── my_dataset_binary/seed0/          # 脚本自动创建的2分类软链接目录
    ├── malnourished_face/            # 三个malnourished子组合并
    └── normal_face/                  # 三个normal子组合并

/root/autodl-tmp/datadream_outputs/generated_images/
└── my_dataset/sd2.1/gs3.5_nis50/shot20_seed0_template1_lr0.0001_ep240/
    ├── train/                        # 原始合成图（每类3000张）
    │   ├── malnourished_face/        # 三个malnourished子组合并
    │   └── normal_face/              # 三个malnourished子组合并
    └── filtered_train/               # 过滤后合成图
        ├── malnourished_face/        # 408张,三个malnourished子组合并
        └── normal_face/              # 456张,三个malnourished子组合并

/root/autodl-tmp/test_data/           # 测试集(每类下都有27个人，每人对应3个视角的图片(01表示正脸,02表示左45度脸,03表示右45度脸))
|____malnourished_face/
|    |____166_01.png
|    |____166_02.png
|    |____166_03.png
|    |____169_01.png
|    |____169_02.png
|    |____169_03.png
|____normal_face/
     |____06_01.png
     |____06_02.png
     |____06_03.png
     |____100_01.png
     |____100_02.png
     |____100_03.png

```

---

## 三、核心文件说明

| 文件 | 作用 |
|---|---|
| `04_classify.sh` | 入口脚本，透传环境变量调用下层脚本 |
| `run_both_real_and_synth.sh` | 核心脚本，准备数据目录、拼接参数、调用main.py |
| `local.yaml` | 本地路径配置 |
| `config.py` | 解析命令行参数+读取local.yaml+自动推导路径 |
| `data.py` | 数据集加载，新增FixedLabelImageFolder和my_dataset分支 |
| `main.py` | 训练+评估+预测分析 |
| `clip.py` | CLIP模型定义，含LoRA注入逻辑 |
| `lora.py` | LoRA多头注意力实现 |
| `qc_filter.py` | 马氏距离过滤合成图像 |
| `03_filter.sh` | 过滤脚本入口 |

---

## 四、local.yaml正确配置

```yaml
wandb_key: null
clip_download_dir:

synth_train_data_dir: /root/autodl-tmp/datadream_outputs/generated_images

real_train_data_dir:
  my_dataset: /root/autodl-tmp/runs/cv/fold_4/my_dataset_binary/seed0  # 含seed0，2分类目录

real_test_data_dir:
  my_dataset: /root/autodl-tmp/test_data

real_train_fewshot_data_dir:
  my_dataset: /root/autodl-tmp/runs/cv/fold_4/my_dataset_binary  # 不含seed0，由代码自动拼接
```

**两个路径的区别：**
- `real_train_data_dir`：`is_synth_train=False`时构建训练集，直接读取2分类目录
- `real_train_fewshot_data_dir`：`is_synth_train=True`时混入真实few-shot样本，代码自动拼接seed0

---

## 五、三种执行命令

```bash
# 1. 只用真实数据（Baseline）
PARAM="--is_synth_train=False" bash 04_classify.sh

# 2. 真实+合成不过滤（DataDream）
SYNTH_VARIANT=raw bash 04_classify.sh

# 3. 真实+合成过滤（NutriDiff）
SYNTH_VARIANT=qc bash 04_classify.sh

# ResNet backbone版本（在上述命令中加入）
MODEL_TYPE=resnet50

# 不使用LoRA版本
PARAM="--is_synth_train=False --is_lora_image=False --is_lora_text=False" bash 04_classify.sh
```

---

## 六、数据加载逻辑

### `is_synth_train=False`（Baseline）
```
FixedLabelImageFolder(real_train_data_dir)
→ 读取 my_dataset_binary/seed0/malnourished_face/ 和 normal_face/
→ 训练数据：约53+60张真实图
→ 损失：loss = CE(logit, label)
```

### `is_synth_train=True` + `SYNTH_VARIANT=raw`（DataDream）
```
DatasetSynthImage(synth_train_data_dir=.../train)
→ 合成图：每类取前500张（实际共3000张）is_real=0
→ 真实图：20张×重复25次=500张 is_real=1
→ 总计：每类约1000张
→ 损失：0.8×loss_real + 0.2×loss_synth
```

### `is_synth_train=True` + `SYNTH_VARIANT=qc`（NutriDiff）
```
DatasetSynthImage(synth_train_data_dir=.../filtered_train)
→ 合成图：malnourished=408张，normal=456张（全部取完，不足500）is_real=0
→ 真实图：20张×重复25次=500张 is_real=1
→ 损失：0.8×loss_real + 0.2×loss_synth
```

---

## 七、标签体系

```python
SUBSET_NAMES["my_dataset"] = ['malnourished_face', 'normal_face']
# 模型内部标签：malnourished_face=0, normal_face=1

# analyze_predictions中重新编码（医学惯例）
pos_idx = class_names.index("malnourished_face")  # =0
y_true_img = (all_targets == 0).astype(int)
# malnourished → y_true=1（阳性）
# normal       → y_true=0（阴性）

# json中的true_label含义：
# 1 = 营养不良阳性患者（正确）
# 0 = 正常阴性
```

---

## 八、两个LoRA不要混淆

| | 生成阶段（SD2.1） | 分类阶段（CLIP） |
|---|---|---|
| 作用对象 | 文本编码器 + UNet | 图像编码器 + 文本编码器 |
| 目的 | 生成符合临床特征的合成图 | 微调分类器适应营养不良任务 |
| 参数设置 | rank=8，lr=1e-4，240epoch | rank=16，alpha=32，dropout=0.1 |
| Baseline用吗 | ❌ | ✅ |

---

## 九、过滤机制（03_filter.sh + qc_filter.py）

**两阶段过滤：**

1. **马氏距离阈值（P95）**：计算合成图像与真实图像分布的马氏距离，超过真实样本距离分布P95的图像被淘汰

2. **Top-K二次筛选**：在通过阈值的候选集中，按马氏距离从小到大取前K张
   ```
   K = ceil(k_beta × n_real) = ceil(8.0 × 20) = 160张/子组
   6个子组合并后：malnourished=408张，normal=456张
   ```

**4维表型描述符 s(x)：**
- 颞部/眶周：相对亮度（软组织流失→区域变暗）
- 颧颊：拉普拉斯响应方差（脂肪流失→纹理变化）
- 下颌缘：梯度幅值均值（脂肪流失→轮廓锐利）

---

## 十、评估指标体系

**两个评估级别：**

| 级别 | 样本数 | 含义 |
|---|---|---|
| image级别 | 162张 | 每张图独立评估 |
| subject级别 | 54人 | 同一人3张图预测概率取平均 |

**论文报告的是subject级别**（临床上对一个人做判断）

**各指标含义：**

| 指标 | 含义 | 临床意义 |
|---|---|---|
| Acc | 总体准确率 | 直观分类性能 |
| AUC | ROC曲线下面积 | 不受类别不平衡影响的区分能力 |
| F1 | 精确率和召回率的调和平均 | 平衡假阳性和假阴性 |
| sen | 灵敏度=TP÷(TP+FN) | 漏诊率相关，营养不良被正确识别的比例 |
| spe | 特异度=TN÷(TN+FP) | 误诊率相关，正常人被正确识别的比例 |
| MCC | Matthews相关系数 | 最综合的单一指标，适合类别不平衡 |

---

## 十一、重要细节与坑

1. **image级别 vs subject级别**：之前误用image级别与论文对比导致看起来差距大，正确应用subject级别后结果高度一致
2. **两个seed不同**：`utils.py`的`fix_random_seeds(seed=12)`是函数默认值不生效，实际用`config.py`的`--seed 22`
