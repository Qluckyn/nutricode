# 手部合成数据 V2

本目录只承载手部 V2 的环境审计、条件准备、结构控制生成和后续质量控制代码。所有入口、模型条件
和输出均与面部实验隔离，不调用面部生成脚本，也不向面部实验目录写入产物。

当前已完成 V2-A、V2-B、V2-C、V2-D 中当前依赖允许的 P1/P3 原始候选生成、V2-E 自动
质量控制，以及 V2-F 盲审包和汇总程序准备。V2-F 正在等待两名真实审阅者独立填写表格，
尚未产生可用于训练的最终接收集。
总体任务与验收规则见：

- `../HAND_SYNTHETIC_DATA_V2_TASK_PLAN.md`；
- `../HAND_SYNTHETIC_DATA_V2_REVIEW_CORRECTION.md`。

## 阶段 V2-A：环境与结构控制可行性

V2-A 冻结 V1 产物，并分别验证两条结构控制路线：

1. SD2.1 OpenPose ControlNet：后备路线，完成一张双手展开冒烟图；
2. FoundHand：当前主路线，完成 `pose01` 握拳和 `pose02` 展开各一张冒烟图。

FoundHand 和 ControlNet 都不提供可靠的原生营养不良语义控制。营养标签只能继承自当前折同类真实
外观父图；结构模型只负责姿势、关键点和手部形态约束。

正式入口：

```bash
# ControlNet 冒烟与 V1 冻结审计
bash hand_synthesis_v2/run_stage_a.sh

# FoundHand 双姿势补充验收
bash hand_synthesis_v2/run_stage_a_foundhand.sh
```

主要实现：

- `smoke_controlnet_sd21.py`：ControlNet 基础冒烟实现；
- `smoke_controlnet_sd21_local.py`：显式 SD2.1 配置兼容层；
- `audit_v2a.py`：V1 冻结及环境/许可证审计；
- `smoke_foundhand.py`：FoundHand 基础推理与完整元数据；
- `smoke_foundhand_xorder.py`：固定俯拍场景的左右手横坐标槽位适配；
- `smoke_foundhand_xorder_v3.py`：RTX 5090、旧 checkpoint 与 FP32 最终兼容入口；
- `audit_foundhand_supplement.py`：FoundHand 补充纠正审计。

输出：

```text
/root/autodl-tmp/runs/hand_synthesis_v2/v2_audit/fold_0/
├── environment_report.json
├── v1_freeze_manifest.json
├── smoke_controlnet_sd21/
└── smoke_foundhand/
```

执行记录：

- `V2A_EXECUTION_REPORT.md`：初始 ControlNet 审计记录；
- `V2A_FOUNDHAND_SUPPLEMENT_REPORT.md`：FoundHand 补充记录，路线结论优先于初始报告。

## 阶段 V2-B：条件准备与人工确认

V2-B 读取 fold_0 已审计的 48 张真实训练手部图，生成：

- 256×256 保持比例白色补边图和 3×3 坐标变换矩阵；
- 左右手各 21 个关键点、置信度和包围框；
- SAM 手部掩码；
- 匿名条件预览、逐图元数据和最终条件 manifest。

`pose01` 已全量人工审核；`pose02` 的低置信候选及其余候选均已检查。最终接受 25 套条件：

| 复合类别 | 接受 | 拒绝 |
|---|---:|---:|
| 营养不良 pose01 | 1 | 11 |
| 营养不良 pose02 | 11 | 1 |
| 正常 pose01 | 2 | 10 |
| 正常 pose02 | 11 | 1 |

完整复现入口：

```bash
bash hand_synthesis_v2/run_stage_b.sh
```

该入口依次调用：

```text
run_stage_b_prepare.sh
    -> prepare_conditions.py
run_stage_b_finalize.sh
    -> finalize_condition_review.py
run_stage_b_validate.sh
    -> validate_conditions.py
```

人工审核决定固定在 `configs/v2b_fold0_manual_review.json`。重新准备数据后，不得在未重新人工检查
预览的情况下沿用该文件处理不同 fold 或不同图像版本。

输出：

```text
/root/autodl-tmp/runs/hand_synthesis_v2/conditions/fold_0/v2_b/
├── condition_manifest_draft.jsonl
├── condition_manifest.jsonl
├── condition_summary_draft.json
├── condition_summary.json
├── images/
├── keypoints/
├── masks/
├── metadata/
├── previews/
└── review/
    └── validation_report.json
```

阶段报告见 `V2B_EXECUTION_REPORT.md`。

## 阶段 V2-C：低风险非生成增强基线

V2-C 已完成 fold_0、seed=22 的 P0 基线：

- 每张真实训练图生成 1 张可追溯增强后代，真实/增强各 108 张；
- 禁止水平翻转和裁切，仿射前增加安全边界；
- 逐图记录父受试者、姿势、类别、哈希、派生 seed 和全部增强参数；
- 每轮按父受试者执行 12:12 类别平衡采样；
- 固定训练 40 epoch，不按测试集选择 checkpoint；
- 测试 loader 仅含 13 名真实测试受试者的 26 张图片。

复现入口：bash hand_synthesis_v2/run_stage_c.sh

完整指标、隔离审计和限制见 V2C_EXECUTION_REPORT.md。后续生成方法必须与本 P0 使用相同真实
划分和评估逻辑。

## 阶段 V2-D：首折小规模生成方法筛选

V2-D 已按 fold_0、seed=22 的冻结计划生成当前可执行的两组原始候选：

- P1 SDXL 低强度 img2img：四个复合类别各 10 张，共 40 张；
- P3 FoundHand 关键点 + 同类外观参考：四个复合类别各 10 张，共 40 张；
- P2 因缺少合法 MANO 和 SDXL 手部 ControlNet 明确阻塞；
- P4 因缺少 HandRefiner 且必须等待结构 QC 选择局部修复对象而阻塞。

完整入口：

```bash
bash hand_synthesis_v2/run_stage_d.sh
```

主要实现：

- `prepare_v2d_plan.py`：冻结 P1/P2/P3 候选、父样本、参数和派生 seed；
- `generate_p1_sdxl_img2img.py`：离线加载固定 revision 的 SDXL 并执行低强度 img2img；
- `generate_p3_foundhand.py`：复用已审计 FoundHand 权重执行同类外观与关键点控制；
- `validate_v2d_candidates.py`：校验文件、哈希、计数、追溯字段和手/脸隔离标记。

输出：

```text
/root/autodl-tmp/runs/hand_synthesis_v2/pilot/fold_0/v2_d/
├── candidate_plan.jsonl
├── p1_sdxl_img2img/
├── p3_foundhand/
├── generated_manifest.jsonl
└── generation_validation_report.json
```

本阶段只完成原始候选生成与追溯校验，不代表任何方法已经晋级，也不能直接把候选加入分类训练。
详细参数、运行数据、阻塞原因和结论边界见 `V2D_EXECUTION_REPORT.md`。

## 阶段 V2-E：自动质量控制与人工复核分流

V2-E 已对 80 张 P1/P3 候选执行：

- 文件、元数据、父图、条件和模型权重哈希复核；
- MediaPipe 双手数量和姿势启发式分流；
- DINOv2 中层/末层及 V2-C 任务模型的全图/ROI 特征比较；
- 每类真实图留一标定的 PCA + Ledoit–Wolf shrinkage Mahalanobis；
- ROI dHash、LPIPS、DINOv2 cosine 三指标近重复检查；
- 营养类别、姿势和方法间的背景、曝光、颜色、构图与配置捷径检查；
- 非破坏性 raw 软链接和最近邻复核网格生成。

完整入口：

```bash
bash hand_synthesis_v2/run_stage_e.sh

# 已有结果时只重新执行验收
VALIDATE_ONLY=true bash hand_synthesis_v2/run_stage_e.sh
```

输出：

```text
/root/autodl-tmp/runs/hand_synthesis_v2/qc/fold_0/v2_e/
├── quality_report.json
├── validation_report.json
├── automatic_qc_manifest.jsonl
├── manual_review_queue.jsonl
├── distribution_calibration.json
├── duplicate_calibration.json
├── shortcut_report.json
├── real_structure_calibration.json
├── raw/
└── nearest_neighbor_grids/
```

80 张候选全部保持 `manual_review_required`，最终训练可用数量为 0。自动结果只用于 V2-F
复核排序；详细数字、捷径警报和结论边界见 `V2E_EXECUTION_REPORT.md`。

## 阶段 V2-F：标签保持与双人独立盲审

V2-F 已为 V2-E 的 80 张候选建立可复现盲审包：

- 原图—候选图随机左右排列，公共材料不显示营养类别、生成方法、父受试者或候选侧；
- 候选图—最近真实图另行随机左右排列，用于人工判断近重复风险；
- 两名审阅者使用完全分离的 CSV，记录手数、解剖、姿势、表型保持、人工痕迹、近重复和接收意见；
- 汇总器计算逐字段一致率与 Cohen's kappa，分歧样本必须完成仲裁；
- 只有最终人工结论和整体一致性门槛同时通过时才创建 `accepted/` 软链接；
- 当前没有按父受试者交叉拟合的内部分类模型，因此不运行分类器辅助诊断，更不会用分类器替代人工判断。

准备与结构验收入口：

```bash
# 仅首次准备；为保护随机化私钥，非空输出目录会拒绝覆盖
bash hand_synthesis_v2/run_stage_f_prepare.sh
```

当前盲审材料位于：

```text
/root/autodl-tmp/runs/hand_synthesis_v2/blind_review/fold_0/v2_f/
├── public/
│   ├── pairs/                 # 原图—候选图高分辨率配对
│   ├── pages/                 # 上述配对的导航页
│   ├── privacy_pairs/         # 候选图—最近真实图近重复审查
│   ├── privacy_pages/         # 近重复配对导航页
│   ├── public_manifest.json
│   └── REVIEW_GUIDE.md
├── review_forms/
│   ├── reviewer_1/review.csv
│   └── reviewer_2/review.csv
├── private/                   # 只供数据管理员保管，不交给审阅者
├── preparation_report.json
└── package_validation_report.json
```

审阅者只能取得 `public/` 和自己的 CSV，不能查看 `private/` 或另一人的表格。两份表格填写完成后运行：

```bash
bash hand_synthesis_v2/run_stage_f_finalize.sh
```

若产生 `pending_adjudication`，由第三方填写自动生成的 `review_forms/adjudication.csv`，再执行同一
汇总命令。用户随后明确要求以其中一份表为准，当前已按 `reviewer_1` 完成单审覆盖汇总：接收
38/80（P1 为 27，P3 为 11），输出位于 `accepted_single_reviewer_override/`。该结果没有双审一致率
或 Cohen's kappa，不能报告为正式 V2-F 通过；P1、P3 也都没有达到四类均 7/10 等正式晋级条件。
实现边界见 `V2F_PREPARATION_REPORT.md`，最终执行数字见 `V2F_EXECUTION_REPORT.md`。

显式复现单审覆盖命令：

```bash
bash hand_synthesis_v2/run_stage_f_finalize.sh \
  --single-reviewer-override reviewer_1
```

用户进一步要求完全只看 `pair_accept`，将 `yes` 和 `uncertain` 都接收。当前最终覆盖状态为
`complete_pair_accept_only_override`，接收 60/80（P1 为 36，P3 为 24），输出位于
`accepted_pair_accept_only_override/`。该集合包含解剖或表型字段不通过的图片，只能用于高风险探索，
不能视为正式 QC 合格集。复现命令：

```bash
bash hand_synthesis_v2/run_stage_f_finalize.sh \
  --single-reviewer-override reviewer_1 \
  --pair-accept-only
```

## V2-G 首折分类效用验证

已完成 C0（仅真实图）、C1（真实图 + V2-C 常规增强）、C2（仅 P1）、C3（真实图 + P1）和
C4（真实图 + P3）的 3 个随机种子实验，共 15 个运行。
每组固定训练 40 epoch，不使用测试集选模。C3 每轮为 48 张真实图和 12 张合成图，真实占比 80%。
结果分别统计 `pose01`、`pose02` 和双姿势概率平均后的受试者级指标。C4 因 P3 四类数量为
10/6/1/7，每轮只按四类各抽 1 张，是低剂量探索，不能与 C3 作等剂量优劣结论。

执行或复现：

```bash
bash hand_synthesis_v2/run_stage_g.sh
```

输出根目录为
`/root/autodl-tmp/runs/hand_synthesis_v2/classifier/fold_0/v2_g`，核心汇总是
`aggregate_metrics.json` 和 `metrics_long.csv`，完整结论见 `V2G_EXECUTION_REPORT.md`。

当前 V2-G 仅为 `pair_accept-only` 单审覆盖下的探索性实验：正式 V2-F 没有通过，且真实/合成域
分类 ROC-AUC 为 0.935，不能据此进入正式规模扩展或声明临床有效。

## P1-T 可见形态提示词诊断

新增独立 P1-T pose02 反事实实验：20张训练父图分别生成 neutral、reduced_fullness 和
preserved_fullness 三联图，共60张。三联共享同一seed和strength=0.25，两个营养类别都生成全部
提示词方向，避免把提示词方向直接等同于标签。

```bash
bash hand_synthesis_v2/run_stage_d_p1t.sh
```

输出位于 `/root/autodl-tmp/runs/hand_synthesis_v2/pilot/fold_0/v2_d_p1t`。当前全部图片均为
`diagnostic_only`。用户已确认60张图的整体视觉质量均可通过，但未提供饱满度方向盲排，因此不获得正式
训练资格。为检验提示词对分类效果的影响，已另建显式诊断覆盖，完成中性、标签对齐、标签反向三组各
3个seed的固定40轮训练：

```bash
bash hand_synthesis_v2/run_stage_g_p1t.sh
```

结果根目录为 `/root/autodl-tmp/runs/hand_synthesis_v2/classifier/fold_0/v2_g_p1t`。标签对齐组相对中性组
在图片汇总、pose01和pose02的Balanced Accuracy及Macro F1上均无提升；联合指标的小幅变化也被标签
反向组复现，当前不支持稳定的提示词分类增益。详细设计和生成结果见 `P1T_EXECUTION_REPORT.md`，分类
结果见 `P1T_CLASSIFIER_EXECUTION_REPORT.md`。

## 环境说明

- FoundHand 原始环境的 PyTorch 2.3/CUDA 12.1 不支持 RTX 5090 的 `sm_120`；
- 当前 FoundHand 推理通过 `datadream` 环境的 PyTorch 2.10/CUDA 12.8 执行；
- FoundHand 官方实现使用 FP32 时间嵌入，最终入口保持 FP32；
- FoundHand 代码通过 `PYTHONPATH` 引用，未修改 `/root/autodl-tmp/FoundHand`；
- 当前没有合法 MANO 模型，`pose01` 只能使用人工确认的关键点模板，不能声明完成 MANO/深度拟合。

## 当前使用规则

- 分类训练固定使用 `/root/autodl-tmp/data_hand/split_seed22/train` 的全部54名受试者；
- 后续分类评估固定只使用 `/root/autodl-tmp/data_hand/split_seed22/test` 的13名受试者、26张真实图；
- 测试集不进入训练、生成父图池、生成审核或数据增强，但其分类结果用于 C3-H10→H100 的探索性比较；
- 营养标签只继承自真实外观父图；
- 跨受试者结构迁移只能发生在同一营养类别；
- `pose01` 模板数量少，后续必须按营养类别均衡复用；
- V2-A 冒烟图和 V2-B 条件图均不直接进入分类训练集；
- 正式分类对照为 C0（仅真实）、C1（常规增强）和 C3（生成增强）；
- C3-H10、H25、H50、H100 均需完成，并分别报告 pose01、pose02 和双姿势概率融合结果；
- H10 内层分类结果已删除；正式 H10 已用完整54名训练受试者和固定测试集完成重跑；
- C3-MIX-H10（0.15/0.25/0.35）敏感性对照已完成；双Pose BA/Macro F1 与 S015-H10 持平、AUC更低，H25主方案继续使用 strength=0.15；
- C3-S015-H25 已完成：100张（四类各25）主阶梯图；双Pose BA为0.5333，低于H10的0.6278，H10暂为已完成主阶梯中的最佳规模；
- 当前结果属于固定测试集上的探索性比较，不表述为未参与方案选择的独立盲测；
- 用户授权的新批次审核只以匿名配对图的 `pair_accept` 为准，`yes` 和 `uncertain` 均接收。
