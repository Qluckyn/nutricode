#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""是建管任复检与 DINO 表征评分的输入清单；只使用训练折。"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import yaml


def load_rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = Path(config["generation_output_root"])
    audit = load_rows(root / "audit" / "candidate_cpu_audit.jsonl")
    metadata_root = root / ("op_i2i" if (root / "op_i2i").is_dir() else "foundhand_i2i") / "metadata"

    references = {}
    for cls in ("malnourished_hand", "normal_hand"):
        references[cls] = [str(path) for path in sorted(
            (Path(config["data"]["root"]) / f"fold_{config['fold']}" / "train" / cls).glob("*_02.png"),
            key=lambda path: int(path.stem.split("_")[0]),
        )]

    out = []
    for row in audit:
        meta = json.loads((metadata_root / f"{row['candidate_id']}.json").read_text(encoding="utf-8"))
        is_foundhand = meta.get("mode") == "foundhand_i2i"
        # DataDream I2I 无 ControlNet 以真实外观父图妋态作参考;
        # FoundHand 路线则以目标键定的相方丹扊克键定为参考。
        reference_keypoints = meta["keypoints_path"] if is_foundhand else meta["appearance_keypoints_path"]
        other = "normal_hand" if row["nutrition_class"] == "malnourished_hand" else "malnourished_hand"
        out.append({
            "candidate_id": row["candidate_id"], "image_path": row["output_path"],
            "nutrition_class": row["nutrition_class"], "pose": "02",
            "target_keypoints_path": reference_keypoints, "roi_reference_keypoints_path": reference_keypoints,
            "generation_mode": meta.get("mode"), "cpu_dhash": row.get("cpu_dhash"),
            "same_class_references": references[row["nutrition_class"]], "other_class_references": references[other],
        })
    dest = root / "qc" / "feature_input_manifest.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("".join(json.dumps(row, ensure_ascii=False) + chr(10) for row in out), encoding="utf-8")
    print(json.dumps({"output": str(dest), "candidates": len(out), "uses_test_data": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
