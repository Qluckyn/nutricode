#!/usr/bin/env python3
"""构建正式 V2 生成所需的 pose02 训练受试者条件源清单。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


def sha256_file(path: Path) -> str:
    """计算原图哈希，保证条件源可追溯。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    fold_name = f"fold_{int(config['fold'])}"
    split = json.loads(Path(config["data"]["split_manifest"]).read_text(encoding="utf-8"))["folds"][fold_name]
    records = []
    for source_class, lora_class in (("malnourished_hand", "malnourished_hand_pose02"), ("normal_hand", "normal_hand_pose02")):
        train_ids = sorted(split["train"][source_class]["subject_ids"], key=int)
        test_ids = set(split["test"][source_class]["subject_ids"])
        for subject_id in train_ids:
            if subject_id in test_ids:
                raise RuntimeError(f"训练/测试受试者重叠：{source_class}/{subject_id}")
            source = Path(config["data"]["root"]) / fold_name / "train" / source_class / f"{subject_id}_02.png"
            if not source.is_file():
                raise FileNotFoundError(source)
            records.append({
                "source_path": str(source.resolve()), "sha256": sha256_file(source),
                "subject_id": subject_id, "nutrition_class": source_class,
                "pose": "02", "lora_class": lora_class,
            })
    output = Path(config["v2_conditions_root"]) / "source_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 2, "stage": config["stage"], "fold": int(config["fold"]),
               "is_hand_only": True, "uses_test_data": False, "records": records,
               "counts": {"malnourished_hand_pose02": 12, "normal_hand_pose02": 42}}
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"source_manifest": str(output), "records": len(records), "counts": payload["counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
