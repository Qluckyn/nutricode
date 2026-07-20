#!/usr/bin/env python3
"""C2-MD：以真实训练图 ROI 分布约束合成手部候选。

流程：v10 椭圆 ROI -> DINO 特征 -> 训练折 PCA(2维) -> 类别条件
Ledoit-Wolf 马氏距离。结构异常与近重复为硬门；ROI 距离使用留一法
真实距离的 P95 作为硬门。整个脚本不读取测试集。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from run_c2_roi_feature_scoring import detector, detect_points, embed, roi_image, rows


CLASSES = ("malnourished_hand", "normal_hand")


def valid_roi(path: str, landmarker):
    """读取图片并返回严格椭圆掩膜后的 ROI；关键点失败返回空。"""
    image = Image.open(path).convert("RGB")
    points = detect_points(image, landmarker)
    return roi_image(image, points) if points is not None else None


def mahalanobis_squared(vector: np.ndarray, model: LedoitWolf) -> float:
    """Ledoit-Wolf 模型提供精度矩阵，适用于小样本低维分布。"""
    diff = vector - model.location_
    return float(diff @ model.precision_ @ diff)


def fit_class_model(values: np.ndarray, label: str):
    """拟合类别模型，并以留一法真实距离计算 P95 门槛。"""
    if len(values) < 5:
        raise RuntimeError(f"{label} 有效真实 ROI 少于 5 张，无法稳定拟合 C2-MD：{len(values)}")
    model = LedoitWolf().fit(values)
    loo_distances = []
    for index in range(len(values)):
        loo_model = LedoitWolf().fit(np.delete(values, index, axis=0))
        loo_distances.append(mahalanobis_squared(values[index], loo_model))
    return model, np.asarray(loo_distances, dtype=np.float64)


def load_by_id(path: Path):
    return {row["candidate_id"]: row for row in rows(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--structure-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pca-components", type=int, default=2)
    parser.add_argument("--tau-quantile", type=float, default=95.0)
    parser.add_argument("--class-separability-margin", type=float)
    parser.add_argument("--class-separability-mode", choices=("hard", "soft", "off"), default="hard",
                        help="hard 为硬门；soft 仅作为同层候选排序项；off 不使用类别可分性")
    parser.add_argument("--soft-separability-weight", type=float, default=0.25,
                        help="soft 模式中异类距离差的排序权重")
    args = parser.parse_args()

    import yaml
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    entries = rows(args.feature_manifest)
    structure = load_by_id(args.structure_scores)
    audit = load_by_id(Path(config["generation_output_root"]) / "audit" / "candidate_cpu_audit.jsonl")
    if not entries:
        raise RuntimeError("C2-MD 输入候选为空")

    # 清单中的两个引用列表均来自当前 fold 的真实训练图，绝不读取测试集。
    first = entries[0]
    reference_paths = {
        first["nutrition_class"]: first["same_class_references"],
        "normal_hand" if first["nutrition_class"] == "malnourished_hand" else "malnourished_hand": first["other_class_references"],
    }
    if set(reference_paths) != set(CLASSES):
        raise RuntimeError("训练折真实参考类别不完整")

    landmarker = detector(Path(config["models"]["hand_landmarker"]))
    try:
        # 先从真实训练图提取 ROI，记录无效样本以便审计。
        real_rois, real_labels, real_counts = [], [], {}
        for label in CLASSES:
            valid = 0
            for path in reference_paths[label]:
                crop = valid_roi(path, landmarker)
                if crop is not None:
                    real_rois.append(crop)
                    real_labels.append(label)
                    valid += 1
            real_counts[label] = valid

        candidate_rois, candidate_indices = [], []
        for index, entry in enumerate(entries):
            crop = valid_roi(entry["image_path"], landmarker)
            if crop is not None:
                candidate_rois.append(crop)
                candidate_indices.append(index)
    finally:
        landmarker.close()

    if not real_rois:
        raise RuntimeError("没有可用真实 ROI，无法进行 C2-MD")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("C2-MD DINO 特征提取需要 CUDA")

    processor = AutoImageProcessor.from_pretrained(config["models"]["dinov2"], local_files_only=True)
    model = AutoModel.from_pretrained(config["models"]["dinov2"], local_files_only=True).to(args.device).eval()
    real_features = embed(real_rois, model, processor, args.device)
    candidate_features = embed(candidate_rois, model, processor, args.device)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # PCA 只由当前 fold 的真实训练图拟合；维度固定为 2 以适配少数类约 12 张样本。
    pca = PCA(n_components=args.pca_components, svd_solver="full", random_state=22)
    real_values = pca.fit_transform(real_features.numpy())
    class_values = {
        label: real_values[np.asarray([item == label for item in real_labels])]
        for label in CLASSES
    }
    class_models, thresholds, loo_summary = {}, {}, {}
    for label in CLASSES:
        class_models[label], loo = fit_class_model(class_values[label], label)
        thresholds[label] = float(np.percentile(loo, args.tau_quantile))
        loo_summary[label] = {
            "count": int(len(loo)), "median": float(np.median(loo)),
            "max": float(np.max(loo)), "tau": thresholds[label],
        }

    candidate_values = pca.transform(candidate_features.numpy())
    value_by_index = {index: value for index, value in zip(candidate_indices, candidate_values)}
    output = []
    for index, entry in enumerate(entries):
        base = audit[entry["candidate_id"]]
        structural = structure[entry["candidate_id"]]
        label = entry["nutrition_class"]
        other = "normal_hand" if label == "malnourished_hand" else "malnourished_hand"
        if index not in value_by_index:
            d2_own = d2_other = None
            md_pass = False
        else:
            value = value_by_index[index]
            d2_own = mahalanobis_squared(value, class_models[label])
            d2_other = mahalanobis_squared(value, class_models[other])
            md_pass = bool(d2_own <= thresholds[label])
        # 该差分大于等于零时，候选离其标注类别的真实分布不远于异类分布。
        label_consistent = None if d2_own is None else bool(d2_own <= d2_other)
        separability_pass = (
            True if args.class_separability_margin is None else
            bool(d2_own is not None and d2_other is not None and d2_own + args.class_separability_margin <= d2_other)
        )
        # 软约束不淘汰正常类等表型重叠候选，仅在相同分层配额内优先选择类别差异更清晰的图。
        separability_hard_pass = separability_pass if args.class_separability_mode == "hard" else True
        hard_pass = bool(structural.get("structure_pass", False)) and not bool(structural.get("near_duplicate", False)) and md_pass and separability_hard_pass
        if d2_own is None:
            qc_score = None
        elif args.class_separability_mode == "soft":
            qc_score = -float(d2_own) + args.soft_separability_weight * float(d2_other - d2_own)
        else:
            qc_score = -float(d2_own)
        output.append({
            **base,
            "roi_definition": "v10_thumbbase_fdi_strict_ellipse",
            "structure_pass": bool(structural.get("structure_pass", False)),
            "near_duplicate": bool(structural.get("near_duplicate", False)),
            "md_d2_own": d2_own,
            "md_d2_other": d2_other,
            "md_tau": thresholds[label],
            "md_pass": md_pass,
            "label_consistent_audit": label_consistent,
            "class_separability_margin": args.class_separability_margin,
            "class_separability_mode": args.class_separability_mode,
            "soft_separability_weight": args.soft_separability_weight,
            "class_separability_pass": separability_pass,
            "qc_status": "approved" if hard_pass else "rejected",
            # 选择器按 qc_score 降序取样；软模式同时奖励“远离异类”的候选。
            "qc_score": qc_score,
            "uses_test_data": False,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output), encoding="utf-8")
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    args.model_output.write_text(json.dumps({
        "roi_definition": "v10_thumbbase_fdi_strict_ellipse",
        "pca_components": args.pca_components,
        "tau_quantile": args.tau_quantile,
        "class_separability_margin": args.class_separability_margin,
        "class_separability_mode": args.class_separability_mode,
        "soft_separability_weight": args.soft_separability_weight,
        "real_valid_counts": real_counts,
        "loo_thresholds": loo_summary,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.astype(float).tolist(),
        "uses_test_data": False,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output), "approved": sum(row["qc_status"] == "approved" for row in output),
        "total": len(output), "real_valid": real_counts, "uses_test_data": False,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
