#!/usr/bin/env python3
"""汇总同一候选池内 C1-raw 与 C2-MD 的选择差异。"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


CLASSES = ("malnourished_hand", "normal_hand")
CONDITIONS = ("c1_raw", "c2_qc")


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def quantiles(values):
    """空列表显式返回空，避免把缺失统计误写为零。"""
    if not values:
        return None
    values = np.asarray(values, dtype=float)
    return {
        "count": int(len(values)), "mean": float(values.mean()),
        "median": float(np.median(values)), "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)), "min": float(values.min()), "max": float(values.max()),
    }


def summarize(items):
    """输出距离、门控与分层分布；C1 和 C2 使用同一字段计算。"""
    result = {}
    for label in CLASSES:
        group = [item for item in items if item["nutrition_class"] == label]
        result[label] = {
            "count": len(group),
            "md_d2": quantiles([item["md_d2_own"] for item in group if item.get("md_d2_own") is not None]),
            "md_pass": sum(bool(item.get("md_pass")) for item in group),
            "structure_pass": sum(bool(item.get("structure_pass")) for item in group),
            "near_duplicate": sum(bool(item.get("near_duplicate")) for item in group),
            "label_consistent": sum(bool(item.get("label_consistent_audit")) for item in group),
            "by_strength": dict(sorted(Counter(str(item["denoising_strength"]) for item in group).items())),
            "parent_count": len({str(item["appearance_parent_subject_id"]) for item in group}),
            "max_per_parent": max(Counter(str(item["appearance_parent_subject_id"]) for item in group).values(), default=0),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"stage": "C2-MD-fold0-selection-audit", "routes": {}, "uses_test_data": False}

    for route_dir in sorted(path for path in args.root.iterdir() if path.is_dir() and path.name in {
        "datadream_i2i_no_cn", "foundhand_datadream_i2i_no_cn"
    }):
        qc = {item["candidate_id"]: item for item in load_jsonl(route_dir / "qc" / "qc_manifest_c2_md.jsonl")}
        selection = json.loads((route_dir / "selection_c1_c2_md" / "matched_c1_c2_selection.json").read_text(encoding="utf-8"))["conditions"]
        selected = {
            condition: [qc[item["candidate_id"]] for item in selection[condition]]
            for condition in CONDITIONS
        }
        c1_ids = {item["candidate_id"] for item in selected["c1_raw"]}
        c2_ids = {item["candidate_id"] for item in selected["c2_qc"]}
        c1_failures = Counter()
        for item in selected["c1_raw"]:
            if not item.get("structure_pass"):
                c1_failures["structure_failed"] += 1
            if item.get("near_duplicate"):
                c1_failures["near_duplicate"] += 1
            if not item.get("md_pass"):
                c1_failures["md_outlier"] += 1

        report["routes"][route_dir.name] = {
            "candidate_pool": len(qc),
            "c1_c2_overlap": len(c1_ids & c2_ids),
            "c1_only": len(c1_ids - c2_ids),
            "c2_only": len(c2_ids - c1_ids),
            "c1_gate_failures": dict(c1_failures),
            "conditions": {name: summarize(items) for name, items in selected.items()},
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
