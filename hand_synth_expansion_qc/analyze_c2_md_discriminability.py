#!/usr/bin/env python3
"""评估 C2-MD 加入类别可分性约束后的候选容量。

约束定义为 ``D_own^2 + m <= D_other^2``。其中 D_own / D_other 是
候选至本类 / 异类真实训练 ROI 分布的马氏距离；m=0 即“离本类不远于
异类”。本脚本不改变清单，只为确定 fold_0 是否能维持每类、每强度
30 张的 C2 选择规模。
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


CLASSES = ("malnourished_hand", "normal_hand")


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summary(values):
    """返回距离差分位数，便于预先冻结 margin，避免据测试集调参。"""
    if not values:
        return {"count": 0}
    vector = np.asarray(values, dtype=float)
    return {
        "count": int(vector.size), "mean": float(vector.mean()), "median": float(np.median(vector)),
        "p10": float(np.percentile(vector, 10)), "p25": float(np.percentile(vector, 25)),
        "p75": float(np.percentile(vector, 75)), "p90": float(np.percentile(vector, 90)),
        "min": float(vector.min()), "max": float(vector.max()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--margins", type=float, nargs="+", default=[0.0])
    parser.add_argument("--target-per-strength", type=int, default=30)
    args = parser.parse_args()

    report = {
        "stage": "C2-MD-discriminability-feasibility-fold0",
        "constraint": "structure_pass AND not_near_duplicate AND md_pass AND md_d2_own + margin <= md_d2_other",
        "target_per_class_per_strength": args.target_per_strength,
        "margins": args.margins,
        "routes": {},
        "uses_test_data": False,
    }
    for route_name in ("datadream_i2i_no_cn", "foundhand_datadream_i2i_no_cn"):
        manifest = args.root / route_name / "qc" / "qc_manifest_c2_md.jsonl"
        rows = load_jsonl(manifest)
        route = {"all_margin_distribution": {}, "feasibility": {}}
        for label in CLASSES:
            eligible = [
                row for row in rows
                if row["nutrition_class"] == label
                and row.get("structure_pass")
                and not row.get("near_duplicate")
                and row.get("md_pass")
                and row.get("md_d2_own") is not None
                and row.get("md_d2_other") is not None
            ]
            # 正值越大，表示该候选在 ROI 特征空间内越偏向其标注类别。
            route["all_margin_distribution"][label] = summary([
                row["md_d2_other"] - row["md_d2_own"] for row in eligible
            ])
        for margin in args.margins:
            by_class_strength = defaultdict(list)
            by_class_strength_parent = defaultdict(Counter)
            for row in rows:
                if not (row.get("structure_pass") and not row.get("near_duplicate") and row.get("md_pass")):
                    continue
                own, other = row.get("md_d2_own"), row.get("md_d2_other")
                if own is None or other is None or own + margin > other:
                    continue
                key = (row["nutrition_class"], str(row["denoising_strength"]))
                by_class_strength[key].append(row)
                by_class_strength_parent[key][str(row["appearance_parent_subject_id"])] += 1
            strata = {}
            feasible = True
            for label in CLASSES:
                for strength in ("0.15", "0.22", "0.3"):
                    key = (label, strength)
                    count = len(by_class_strength[key])
                    strata[f"{label}/strength_{strength}"] = {
                        "available": count,
                        "target": args.target_per_strength,
                        "parent_count": len(by_class_strength_parent[key]),
                        "max_per_parent": max(by_class_strength_parent[key].values(), default=0),
                        "feasible": count >= args.target_per_strength,
                    }
                    feasible &= count >= args.target_per_strength
            route["feasibility"][str(margin)] = {"all_strata_feasible": feasible, "strata": strata}
        report["routes"][route_name] = route
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
