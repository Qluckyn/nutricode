后续目标应控制得很明确：在当前固定的 train/test 划分上，完成一个无数据泄漏、可复现的 CLIP 基线实验，判断测试集上是否存在初步区分信号，而不是追求临床可用性能。

## 总体实验路线

```text
数据与泄漏检查
→ 建立手部专用训练入口
→ 修正预处理与类别提示
→ 实现12:12受试者动态采样
→ 跑 zero-shot 与 LoRA 基线
→ 测试集只评估一次
→ 图片级 + 受试者级结果分析
→ 多随机种子稳定性验证
```

## 任务一：锁定数据划分

使用现有目录：

```text
train: /root/autodl-tmp/data_hand/split_seed22/train
test:  /root/autodl-tmp/data_hand/split_seed22/test
```

训练启动前自动检查：

- train：营养不良12人、正常42人；
- test：营养不良3人、正常10人；
- 每人恰好有 `_01/_02` 两张；
- train/test 受试者没有交叉；
- 类别名固定为 `malnourished_hand` 和 `normal_hand`；
- 阳性类别固定为 `malnourished_hand`。

这些检查不通过时应直接终止训练。

## 任务二：增加手部专用入口

不继续复用当前包含人脸分组、合成数据准备的 `run_both_real_and_synth.sh`。

建议新增：

```text
04_classify_hand.sh
classify/run_hand_real.sh
```

入口只负责：

- 加载真实手部 train/test；
- 关闭合成数据；
- 关闭人脸 ROI 辅助头；
- 关闭 MixUp/CutMix；
- 指定输出目录和随机种子；
- 调用 CLIP 训练及测试。

建议输出到：

```text
/root/autodl-tmp/runs/hand_clip_baseline/
```

## 任务三：修正数据加载和预处理

### 动态平衡采样

每个 epoch 使用：

```text
营养不良：全部12人，共24张
正常：随机12人，共24张
总计：24人，共48张
```

要求：

- 按受试者抽样；
- 同一人的两张姿势同时入选；
- 第 `epoch` 轮使用 `seed + epoch`；
- 每轮把选中的正常受试者 ID 写入日志；
- 磁盘上仍保留全部42名正常训练者。

### Batch size

当前每轮只有48张图，不建议继续用默认 `batch_size=64`，否则一个 epoch 只有一个优化步骤。

首轮建议：

```text
batch_size=8
batch_size_eval=8
epochs=40
lr=1e-5
warmup_epochs=4
weight_decay=1e-4
```

这样每个 epoch 约6个训练 step。

### 图像增强

手部图片是宽图，当前中心裁剪可能截掉两侧手部，强颜色变化也可能破坏甲床和皮肤颜色。

首轮建议：

- pad 成正方形后 resize 到 `224×224`；
- 小幅旋转和平移；
- 可使用水平翻转；
- 不使用随机灰度；
- 不使用 Solarization；
- 不使用强 ColorJitter；
- 不使用 MixUp/CutMix；
- 测试阶段完全确定性处理。

## 任务四：修正 CLIP 类别提示

不能继续使用类似：

```text
a human photo of a malnourished_hand
```

建议固定两个自然语言提示：

```text
a clinical photograph of the hands of a person with malnutrition
a clinical photograph of the hands of a person with normal nutritional status
```

首轮先固定提示，不根据测试结果反复调整。

## 任务五：修正训练和测试泄漏

<!-- 当前代码每个 epoch 都使用测试目录评估并选择最佳 checkpoint，这不适用于只有 train/test 的实验。

本次采用固定训练方案：

1. 训练参数在运行前固定；
2. 只在 train 上训练40个 epoch；
3. 保存最终 checkpoint；
4. 训练全部结束后，test 只运行一次；
5. 不使用 test 选择最佳 epoch；
6. 不使用 test 调整分类阈值。

首轮分类阈值固定为 `0.5`。 -->
保留原代码，不删除，同时增加手部实验开关：
--select_best_on_test=True
True：保持原始逻辑，用当前 test 选择最佳 checkpoint；
False：训练固定40轮，保存最终 checkpoint，test 只评估一次。

## 任务六：设置三组基线实验

### 实验A：多数类基线

全部预测正常：

```text
Accuracy = 10 / 13 = 76.9%
Balanced Accuracy = 50%
ROC-AUC = 50%
Sensitivity = 0%
Specificity = 100%
```

因此训练结果即使 accuracy 达到77%，也不代表模型有效。

### 实验B：CLIP zero-shot

不训练模型，只使用自然语言提示直接预测测试集。

用途：

- 判断预训练 CLIP 是否已经具有一些手部营养状态语义；
- 给微调结果提供参照。

这需要增加真正的 zero-shot 评估模式，不能简单把两个 LoRA 开关都关闭后进入现有优化器。

### 实验C：CLIP LoRA 微调

首选设置：

```text
图像编码器 LoRA：开启
文本编码器 LoRA：关闭
动态12:12采样：开启
MixUp/CutMix：关闭
ROI辅助头：关闭
```

小数据下先只调整图像编码器，降低过拟合风险。

如果实验C有明显信号，再补一组：

```text
图像 LoRA + 文本 LoRA
```

作为对照，不作为首轮必要任务。

## 任务七：按受试者评估两种姿势

每张图片输出：

```text
subject_id
pose
true_label
predicted_label
malnourished_probability
```

其中：

```text
46_01.png → subject_id=46, pose=01
46_02.png → subject_id=46, pose=02
```

受试者级概率：

```python
subject_probability = mean(pose01_probability, pose02_probability)
```

需要分别报告：

- 图片级指标，共26张；
- 受试者级指标，共13人，作为主要结果；
- 姿势 `_01` 单独指标；
- 姿势 `_02` 单独指标；
- 两姿势预测是否一致。

## 任务八：需要输出的指标

必须报告：

- Accuracy；
- Balanced Accuracy；
- ROC-AUC；
- PR-AUC；
- sensitivity；
- specificity；
- F1；
- MCC；
- 混淆矩阵；
- 每个受试者两张图及融合后的概率。

结果文件建议为：

```text
hand_clip_baseline/
└── seed22/
    ├── config.json
    ├── train.log
    ├── sampling_history.json
    ├── final_checkpoint.pth
    ├── image_predictions.csv
    ├── subject_predictions.csv
    ├── metrics.json
    └── confusion_matrix.png
```

## 任务九：稳定性验证

第一次先运行 `seed=22`，确认流程正常。

如果结果优于随机水平，再保持 train/test 划分不变，运行三个训练随机种子：

```text
22
23
24
```

这里改变的是：

- 模型初始化；
- DataLoader 顺序；
- 每轮正常受试者抽样顺序。

测试集不变。最终报告三个种子的均值和波动，防止某一次随机抽样偶然得到较好结果。

## 初步判断标准

可以把下面条件作为“存在初步分类信号”的探索性标准：

- 受试者级 Balanced Accuracy 明显高于 `0.5`；
- ROC-AUC 高于随机水平；
- sensitivity 和 specificity 不是一个很高、另一个接近零；
- 两种姿势融合优于或不差于单姿势；
- 三个训练种子的结果方向基本一致；
- 结果不是单纯依靠全部预测正常获得的高 accuracy。

例如受试者级 Balanced Accuracy 达到约 `0.65`、AUC 达到约 `0.70`，且多个随机种子表现一致，可以认为“值得继续扩大数据验证”。这只是探索性参考，不是统计或临床判定标准。

## 预计需要修改的项目文件

主要工作范围：

- 新增 `04_classify_hand.sh`；
- 新增 `classify/run_hand_real.sh`；
- 修改 `classify/config.py`，支持手部 train/test 路径；
- 修改 `classify/data.py`，实现受试者动态平衡采样和手部预处理；
- 修改 `classify/models/clip.py`，增加手部提示和 zero-shot 支持；
- 修改 `classify/main.py`，取消测试集选 checkpoint，修正手部阳性类别及受试者级评估；
- 增加采样器、数据泄漏和指标计算测试。

实施顺序建议是：先完成数据/评估正确性，再跑 zero-shot，最后跑 LoRA。这样最终能够回答三个问题：CLIP 原始能力如何、微调是否有效、结果是否超过多数类假象。