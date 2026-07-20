#!/usr/bin/env python3
"""汇总 fold_0 两条路线 C1 与类别可分性 C2-MD 的真实差异。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


ROUTES = ("datadream_i2i_no_cn", "foundhand_datadream_i2i_no_cn")
CONDITIONS = ("c1_raw", "c2_qc")


def jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def one(path: Path, name: str) -> Path:
    matches = list(path.rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"{path} 下 {name} 数量异常：{len(matches)}")
    return matches[0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(path: Path):
    values = json.loads(path.read_text(encoding="utf-8"))["image_level_metrics"]
    return {key: values[key] for key in ("acc", "balanced_accuracy", "mcc", "f1", "sen", "spe", "tp", "tn", "fp", "fn")}


def predictions(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {row["image_path"]: row for row in csv.DictReader(handle)}


def mean(values):
    return None if not values else float(np.mean(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs-dir", default="classifier_runs_c2_md_separable")
    args = parser.parse_args()
    report = {"stage": "C2-MD-separable-fold0-result-audit", "routes": {}, "uses_test_data_for_selection": False}

    for route in ROUTES:
        root = args.root / route
        selection = json.loads((root / "selection_c1_c2_md_separable" / "matched_c1_c2_selection.json").read_text(encoding="utf-8"))["conditions"]
        c1, c2 = selection["c1_raw"], selection["c2_qc"]
        c1_ids, c2_ids = {row["candidate_id"] for row in c1}, {row["candidate_id"] for row in c2}
        # 同时比较 ID 与字节级图像哈希，排除“编号不同但实际图相同”的情况。
        c1_hash, c2_hash = {sha256(Path(row["output_path"])) for row in c1}, {sha256(Path(row["output_path"])) for row in c2}
        run_root = root / args.runs_dir
        result = {"selection": {
            "per_condition": {name: len(selection[name]) for name in CONDITIONS},
            "id_overlap": len(c1_ids & c2_ids), "id_overlap_ratio": len(c1_ids & c2_ids) / len(c1_ids),
            "pixel_hash_overlap": len(c1_hash & c2_hash), "pixel_hash_overlap_ratio": len(c1_hash & c2_hash) / len(c1_hash),
            "c1_only": len(c1_ids - c2_ids), "c2_only": len(c2_ids - c1_ids),
            "c1_label_consistent": sum(bool(row.get("label_consistent_audit")) for row in c1),
            "c2_label_consistent": sum(bool(row.get("label_consistent_audit")) for row in c2),
            "c1_md_d2_mean": mean([row["md_d2_own"] for row in c1 if row.get("md_d2_own") is not None]),
            "c2_md_d2_mean": mean([row["md_d2_own"] for row in c2 if row.get("md_d2_own") is not None]),
            "strength_counts": {name: dict(sorted(Counter(str(row["denoising_strength"]) for row in selection[name]).items())) for name in CONDITIONS},
        }, "metrics": {}, "prediction_comparison": {}}
        pred = {}
        for condition in CONDITIONS:
            result["metrics"][condition] = metrics(one(run_root / condition, "metrics.json"))
            pred[condition] = predictions(one(run_root / condition, "image_predictions.csv"))
        common = sorted(set(pred["c1_raw"]) & set(pred["c2_qc"]))
        changed = [path for path in common if pred["c1_raw"][path]["predicted_label"] != pred["c2_qc"][path]["predicted_label"]]
        deltas = [
            float(pred["c2_qc"][path]["malnourished_probability"]) - float(pred["c1_raw"][path]["malnourished_probability"])
            for path in common
        ]
        result["prediction_comparison"] = {
            "test_images": len(common), "changed_hard_labels": len(changed),
            "max_abs_malnourished_probability_delta": float(np.max(np.abs(deltas))),
            "mean_abs_malnourished_probability_delta": float(np.mean(np.abs(deltas))),
            "changed_image_paths": changed,
        }
        report["routes"][route] = result
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
