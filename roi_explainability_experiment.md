# 基于临床 ROI 的人脸营养不良筛查模型可解释性实验

## 1. 实验目的

为验证基于人脸图像的老年营养不良二分类模型是否关注具有临床意义的面部区域，本研究进一步开展模型可解释性实验。该实验以已训练完成的 CLIP-LoRA 分类器为分析对象，在测试集图像上结合人脸关键点定位、临床感兴趣区域（Region of Interest, ROI）划分、类别特异性注意力归因以及 ROI 遮挡验证，分析模型在预测营养不良与正常状态时所依赖的面部证据来源。

本实验重点回答以下问题：

1. 模型的预测结果能否与主分类实验保持一致；
2. 模型在判断营养不良时是否重点关注眶周、颞部、颧颊、下颌缘等与营养状态相关的面部区域；
3. 不同 ROI 对模型输出的贡献是否具有可验证的方向性；
4. ROI 可解释性结果是否能够为模型决策提供临床可理解的证据。

## 2. 实验数据与模型

### 2.1 测试集

本实验使用与主分类实验一致的独立测试集，路径为：

```text
/root/autodl-tmp/test_data
```

测试集按照二分类标签组织：

```text
test_data/
├── malnourished_face/
└── normal_face/
```

其中，每个类别包含 27 名受试者，每名受试者对应 3 张不同视角的人脸图像：

```text
*_01.png = 正脸 front
*_02.png = 左 45 度脸 left_45
*_03.png = 右 45 度脸 right_45
```

因此，测试集共包含：

```text
营养不良组：27 人 × 3 视角 = 81 张
正常组：    27 人 × 3 视角 = 81 张
总计：      54 人 × 3 视角 = 162 张
```

### 2.2 分类模型

本实验分析的模型为主实验中表现较优的 NutriDiff 分类器，即基于 CLIP ViT-B/16 的 LoRA 微调模型。模型 checkpoint 路径为：

```text
/root/autodl-tmp/runs/ablation/classify_outputs/
clip_real_plus_synth_qc_pool0.7_nipc330_lr1e-5_nomix/
my_dataset/clipViT-B/16/n_img_per_cls_500/sd2.1/
shot20_seed0_template1_ddlr0.0001_ddep240_lbd0.8/
lr1e-05_wd0.0001_mixuag/best_checkpoint.pth
```

该模型以 `malnourished_face` 和 `normal_face` 作为二分类标签。其中，在医学评价指标计算中，将 `malnourished_face` 视为阳性类，即：

```text
malnourished_face = 1
normal_face       = 0
```

## 3. 实验方法

### 3.1 预测一致性验证

在进行可解释性分析之前，首先验证 `roi_attention_analysis.py` 中的模型加载、图像预处理和预测流程是否能够复现主分类脚本 `main.py` 输出的预测结果。

主分类实验的预测结果保存于：

```text
detailed_prediction_results.json
```

其中包含每张测试图像的 `malnourished_prob`。可解释性脚本对同一 checkpoint 和同一测试集进行推理后，将每张图像的 `malnourished_probability` 与主实验结果按 `image_path` 对齐比较。

为保证复现一致性，可解释性脚本中采用与 `main.py` 完全一致的 CLIP 测试预处理流程：

```text
Resize(224, bicubic)
CenterCrop(224)
ToTensor
Normalize(CLIP_NORM_MEAN, CLIP_NORM_STD)
```

同时，模型加载时需注意 LoRA 层的状态。由于 `loralib.MergedLinear` 在模型切换至 `eval()` 状态时会对 LoRA 权重进行合并，因此本实验采用与主实验结果一致的加载顺序：

```python
model = model.to(DEVICE).eval()
model.load_state_dict(checkpoint["model"], strict=False)
```

该顺序可避免在加载 checkpoint 后再次触发 LoRA 权重合并，从而保证预测概率与主实验一致。

### 3.2 临床 ROI 定义

本实验关注 4 类与营养不良面部表征相关的临床 ROI：

1. 颞部区域（temporal）
2. 眶周区域（orbital）
3. 颧颊区域（malar）
4. 下颌缘区域（jawline）

ROI 的构建基于 MediaPipe FaceMesh 人脸关键点。对于部分侧脸图像，若 MediaPipe 直接检测失败，则使用 InsightFace 人脸框进行裁剪后再次尝试关键点检测，以提高侧脸图像的 ROI 构建成功率。

各 ROI 与营养不良临床表征的对应关系如下：

| ROI | 临床含义 |
|---|---|
| temporal | 颞部凹陷、软组织减少 |
| orbital | 眶周凹陷、眼周软组织减少 |
| malar | 颧颊脂肪垫减少、面颊消瘦 |
| jawline | 下颌缘锐化、脂肪及肌肉减少 |

### 3.3 类别特异性注意力归因

本实验使用 CLIP 视觉 Transformer 中的注意力层进行类别特异性梯度加权 attention rollout。对于每张图像，分别计算目标类别为 `malnourished_face` 和 `normal_face` 时的注意力归因图。

具体而言，对于目标类别的 logit margin 进行反向传播，获得注意力权重及其梯度。根据梯度正负方向，构建三类 attribution map：

1. 正向归因（positive attribution）：促进目标类别判断的区域；
2. 负向归因（negative attribution）：抑制目标类别判断的区域；
3. 绝对归因（absolute attribution）：不区分方向的总贡献强度。

随后，将 attribution map resize 到图像空间，并计算每个 ROI 内的平均归因强度及相对于全图平均归因强度的 enrichment：

```text
ROI enrichment = mean(attribution within ROI) / mean(attribution over whole image)
```

当 enrichment 大于 1 时，说明该 ROI 相比全图平均水平获得更高关注。

### 3.4 ROI 遮挡验证

为进一步验证 ROI 是否真正影响模型输出，本实验对每个 ROI 进行遮挡实验。具体做法是：对原图中的某一 ROI 区域进行遮挡或替换，然后重新输入模型，计算遮挡前后 `malnourished` logit margin 的变化：

```text
delta_mal_margin = base_mal_margin - occluded_mal_margin
```

若 `delta_mal_margin > 0`，说明遮挡该 ROI 后模型对营养不良的支持下降，该 ROI 对营养不良判断具有正向贡献。

若 `delta_mal_margin < 0`，说明遮挡该 ROI 后模型对营养不良的支持反而增强，该 ROI 可能提供了支持正常类别或抑制营养不良判断的证据。

## 4. 实验结果

### 4.1 预测结果复现情况

可解释性脚本共成功生成 ROI 并完成分析的图像为 150 张。由于部分侧脸图像关键点检测失败，有 12 张测试图像未进入 ROI 分析。

ROI 分析成功图像分布如下：

```text
总成功图像数：150
营养不良图像：75
正常图像：    75
```

按视角统计：

```text
front:    54
left_45:  50
right_45: 46
```

缺失图像主要来自左右 45 度侧脸，说明侧脸图像仍然是关键点检测和 ROI 构建中的主要困难来源。

将 ROI 分析脚本输出的 `malnourished_probability` 与主实验 `detailed_prediction_results.json` 中的 `malnourished_prob` 按图像路径对齐比较，结果为：

```text
对齐图像数：150
最大绝对误差：0.002227
平均绝对误差：0.000312
```

该误差很小，不影响阈值判断和 subject-level 结果。基于 ROI 成功图像进行 subject-level mean fusion 后，模型预测结果为：

```text
TP = 27
TN = 24
FP = 3
FN = 0
Acc = 0.9444
```

该结果与主分类实验的 subject-level 结果一致，说明可解释性实验中的模型推理流程能够复现主实验预测结果。

### 4.2 错误样本分析

subject-level 下，模型未出现营养不良漏诊，即：

```text
FN = 0
```

共有 3 名正常受试者被误判为营养不良：

```text
FP subject: 199, 204, 90
```

这与主实验中的误判 subject 完全一致。该结果说明可解释性实验并未改变模型预测，而是在相同预测结果基础上进一步分析模型关注区域。

### 4.3 营养不良样本中的 ROI 归因结果

对于真实标签为 `malnourished_face` 且目标类别为 `malnourished_face` 的样本，正向 attribution enrichment 的平均排序如下：

| ROI | Positive attribution enrichment |
|---|---:|
| orbital | 1.569 |
| temporal | 1.533 |
| malar | 1.471 |
| jawline | 1.199 |

结果显示，在模型判断营养不良时，眶周、颞部和颧颊区域的正向归因强度均高于全图平均水平。其中，眶周区域的 enrichment 最高，提示模型在营养不良判别中较强依赖眼周软组织变化；颞部和颧颊区域也表现出较高关注，符合营养不良患者常见的颞部凹陷和面颊消瘦等临床表征。

下颌缘区域的 enrichment 虽然也高于 1，但相对低于其他三个区域，说明其在本模型中的贡献相对较弱。

### 4.4 正常样本中的 ROI 归因结果

对于真实标签为 `normal_face` 且目标类别为 `normal_face` 的样本，正向 attribution enrichment 的平均排序如下：

| ROI | Positive attribution enrichment |
|---|---:|
| orbital | 3.408 |
| temporal | 2.591 |
| malar | 2.024 |
| jawline | 1.762 |

可以观察到，模型在判断正常类别时同样高度关注眶周、颞部、颧颊和下颌缘区域，且 enrichment 数值整体高于营养不良类别中的正向归因。这说明这些面部 ROI 不仅可作为营养不良的阳性证据，也可作为正常营养状态的反向证据。

尤其是眶周和颞部区域，在正常样本中的正向归因显著增强，提示模型可能通过这些区域中软组织饱满度、面部凹陷程度较低等视觉特征来支持正常类别判断。

### 4.5 ROI 遮挡验证结果

#### 4.5.1 营养不良样本

对于真实标签为 `malnourished_face` 的样本，遮挡各 ROI 后 `malnourished` margin 的平均下降量如下：

| ROI | delta_mal_margin |
|---|---:|
| orbital | 1.317 |
| jawline | 0.890 |
| malar | 0.799 |
| temporal | 0.448 |

所有 ROI 的 `delta_mal_margin` 均为正值，说明遮挡这些区域后，模型对营养不良类别的支持下降。其中，眶周区域的下降幅度最大，表明眶周区域是模型判断营养不良的最关键局部证据。

该结果与 attribution enrichment 分析一致，进一步验证模型确实依赖眶周区域进行营养不良判断。

#### 4.5.2 正常样本

对于真实标签为 `normal_face` 的样本，遮挡各 ROI 后 `malnourished` margin 的平均变化如下：

| ROI | delta_mal_margin |
|---|---:|
| orbital | -0.692 |
| temporal | -0.936 |
| jawline | -1.119 |
| malar | -1.332 |

所有 ROI 的 `delta_mal_margin` 均为负值，说明遮挡这些区域后，模型对营养不良类别的支持反而增强。换言之，在正常样本中，这些 ROI 提供了抑制营养不良判断、支持正常类别判断的证据。

其中，颧颊区域和下颌缘区域的负向变化幅度最大，提示正常人脸中颧颊及下颌缘的视觉特征可能是模型识别正常状态的重要依据。

### 4.6 不同视角下的 ROI 作用

从视角分层结果看，眶周区域在三个视角下均表现出较强贡献。

对于营养不良样本：

```text
front:    orbital delta = 1.610
left_45:  orbital delta = 0.928
right_45: orbital delta = 1.416
```

正脸和右 45 度视角中，眶周区域的遮挡影响尤其明显；左 45 度视角中，眶周与下颌缘贡献接近。

对于正常样本，遮挡 ROI 后 `malnourished` margin 多数呈负向变化，说明这些 ROI 在不同视角下均提供了正常状态的反向证据。该结果表明，多视角图像可以为模型提供互补的局部面部信息。

## 5. 讨论

本实验结果表明，CLIP-LoRA 分类器在进行人脸营养不良筛查时，并非仅依赖背景或无关图像区域，而是对具有临床意义的面部 ROI 表现出较高关注。尤其是眶周、颞部和颧颊区域，在 attribution 与 occlusion 两类分析中均表现出较强贡献。

眶周区域在营养不良样本中的正向归因和遮挡影响均最明显，说明模型可能学习到了眼周凹陷、软组织减少等与营养不良相关的视觉表征。颞部和颧颊区域同样具有较高 attribution enrichment，与临床上常见的颞部凹陷、颧颊脂肪减少等表现一致。

对于正常样本，ROI 遮挡后模型更倾向于预测营养不良，说明正常人脸中的这些局部区域提供了支持正常类别的视觉证据。这一现象从反向角度说明模型对营养状态相关面部区域具有一定判别能力。

此外，模型在 subject-level 下未出现营养不良漏诊，说明其对阳性样本具有较高敏感性。3 个假阳性 subject 的存在提示模型在部分正常样本中仍可能将局部消瘦、光照、姿态或个体面部结构误判为营养不良特征，这也是后续模型校准和误差分析的重要方向。

## 6. 局限性

本实验仍存在以下局限：

1. ROI 分析未覆盖全部 162 张测试图像，其中 12 张图像因关键点检测失败未进入 ROI 统计；
2. 关键点检测失败主要集中在侧脸图像，说明当前 ROI 构建方法对姿态变化仍较敏感；
3. attention attribution 本身只能反映模型内部注意力与梯度响应，不等价于严格因果解释；
4. 遮挡实验虽然能验证 ROI 对输出的影响，但遮挡操作可能引入分布外图像模式；
5. 当前实验样本量较小，ROI 统计结论仍需在更大规模、多中心数据集上验证。

## 7. 小结

本研究基于人脸关键点和临床 ROI 构建了模型可解释性分析流程，并从类别特异性 attention attribution 和 ROI 遮挡验证两个角度分析了 CLIP-LoRA 营养不良筛查模型的决策依据。

实验结果显示：

1. 可解释性脚本的预测结果能够复现主分类实验；
2. 模型在判断营养不良时重点关注眶周、颞部、颧颊等临床相关区域；
3. 遮挡眶周区域会显著降低模型对营养不良类别的支持；
4. 正常样本中的颧颊、下颌缘和眶周区域对抑制营养不良判断具有重要作用；
5. ROI 解释结果与营养不良面部临床表现具有较好一致性。

因此，该可解释性实验为模型预测提供了临床可理解的局部证据，增强了基于人脸图像进行老年营养不良筛查的可信度和可解释性。
