#!/usr/bin/env python3
"""冻结正式 V2 的自动通过 pose02 条件；仅接受 fold_0 训练图。"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = Path(config["v2_conditions_root"])
    draft = root / "draft" / "condition_manifest_draft.jsonl"
    rows = [json.loads(line) for line in draft.read_text(encoding="utf-8").splitlines() if line.strip()]
    final = []
    for row in rows:
        valid = (row["source_split"] == "fold_0_train" and row["pose"] == "02" and
                 row["automatic_status"] == "candidate" and row["keypoints_path"] and row["mask_path"])
        row["manual_status"] = "accepted" if valid else "rejected"
        row["manual_reviewer"] = "user_confirmed_auto_candidate_policy_v2" if valid else "auto_rule"
        row["manual_notes"] = "用户确认自动候选结构质量；V2仅接收自动通过的pose02训练条件" if valid else ";".join(row["failure_reasons"])
        row["usable_for_generation"] = bool(valid)
        if valid:
            final.append(row)
    counts = Counter(row["compound_class"] for row in final)
    expected = dict(config["usable_appearance_counts"])
    if dict(counts) != expected:
        raise RuntimeError(f"V2外观条件数量不足或异常：实际{dict(counts)}，期望{expected}")
    final_dir = root / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    manifest = final_dir / "condition_manifest.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in final), encoding="utf-8")
    (final_dir / "condition_summary.json").write_text(json.dumps({"stage": config["stage"], "records": len(final), "by_class": counts, "pose": "02", "uses_test_data": False}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest), "counts": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
