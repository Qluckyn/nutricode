# pose02 合成扩增与 QC 正式 V2

本目录只服务 `fold_0`、分类种子 `22`、全流程 `pose02` 的实验。旧候选与正式 V2 输出隔离，不能混用。

## 固定设计

- 候选：每类 2400 张，共 4800 张；FoundHand 外观锚图后接 ControlNet I2I。
- 外观参考：仅可用训练受试者，恶病质 11 名、正常 40 名；两类均衡轮换。
- 多样性：外观父图与结构父图不同；I2I 强度 `0.15/0.22/0.30` 每类各 800 张。
- 训练：C0 仅真实；C1/C2 每 epoch 每类 12 真 + 36 合成，合成:真实=3:1；40 epoch。
- 评价：固定 `fold_0/test`，记录 balanced accuracy、MCC、macro-F1、ROC-AUC、PR-AUC、敏感度、特异度。

## 生成完成后的 CPU/GPU 顺序

```bash
# 1. 完整性和近重复预筛（CPU；不完整时可加 --allow-incomplete）
python hand_synth_expansion_qc/audit_generation_v2.py --config hand_synth_expansion_qc/generation_config_v2.yaml --compute-dhash

# 2. 为后续 MediaPipe/DINO 评分准备严格的训练集参考清单（CPU）
python hand_synth_expansion_qc/prepare_qc_feature_manifest_v2.py --config hand_synth_expansion_qc/generation_config_v2.yaml

# 3. 生图结束后独占GPU：结构复检、DINO 同类/异类边际和多样性评分
python hand_synth_expansion_qc/run_qc_feature_scoring_v2.py --config hand_synth_expansion_qc/generation_config_v2.yaml --input-manifest /root/autodl-tmp/runs/hand_synth_expansion_qc/fold_0_pose02/generation_v2/qc/feature_input_manifest.jsonl

# 4. 汇总QC硬门与三项分数
python hand_synth_expansion_qc/build_qc_manifest_v2.py --config hand_synth_expansion_qc/generation_config_v2.yaml --feature-scores /root/autodl-tmp/runs/hand_synth_expansion_qc/fold_0_pose02/generation_v2/qc/feature_scores.jsonl

# 5. 在类别×强度×外观父图相同配额内，导出 C1 随机池和 C2 QC优选池
python hand_synth_expansion_qc/select_matched_c1_c2_v2.py --config hand_synth_expansion_qc/classification_config_v2.yaml --qc-manifest /root/autodl-tmp/runs/hand_synth_expansion_qc/fold_0_pose02/generation_v2/qc/qc_manifest.jsonl

# 6. 构建分类软链接目录，再在GPU空闲后训练
python hand_synth_expansion_qc/build_classifier_data.py --config hand_synth_expansion_qc/classification_config_v2.yaml --condition c0
python hand_synth_expansion_qc/build_classifier_data.py --config hand_synth_expansion_qc/classification_config_v2.yaml --condition c1_raw --selection-manifest SELECTION.json
python hand_synth_expansion_qc/build_classifier_data.py --config hand_synth_expansion_qc/classification_config_v2.yaml --condition c2_qc --selection-manifest SELECTION.json
bash hand_synth_expansion_qc/run_classifier.sh c0
```

QC 总分为 `Q = 0.40 S_structure + 0.35 S_semantic_margin + 0.25 S_diversity`，但必须先通过双手/pose02/完整性硬门。C1 与 C2 仅在“随机选择或按 Q 选择”上不同。
