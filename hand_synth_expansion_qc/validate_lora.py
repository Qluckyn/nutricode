#!/usr/bin/env python3
"""校验本实验pose02 LoRA权重、训练参数和测试集隔离。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


def sha256_file(path: Path) -> str:
    """计算文件哈希，冻结生成所用权重版本。"""
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
    root = Path(config["output"]["root"])
    manifest_path = root / "lora_data" / "hand_lora_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    test_ids = set().union(*(set(ids) for ids in manifest["excluded_test_subject_ids"].values()))
    expected = ["malnourished_hand_pose02", "normal_hand_pose02"]
    records = []
    for class_index, class_name in enumerate(expected):
        run_dir = root / "lora_weights" / class_name / "hand_nutrition_pose02" / "shot12_seed22_tpl1_notextlora" / "lr0.0001_epoch40" / class_name
        weight_path = run_dir / "pytorch_lora_weights.safetensors"
        metadata_path = run_dir / "hand_training_metadata.json"
        if not weight_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"缺少LoRA产物：{run_dir}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        arguments = metadata["arguments"]
        training_images = [Path(path) for path in metadata["training_images"]]
        subjects = {path.stem.split("_")[0] for path in training_images}
        if (metadata["class_name"] != class_name or int(arguments["target_class_idx"]) != class_index
                or int(arguments["num_train_epochs"]) != int(config["lora"]["epochs"])
                or int(arguments["rank"]) != int(config["lora"]["rank"])
                or bool(arguments["train_text_encoder"]) or len(training_images) != 12
                or subjects & test_ids):
            raise RuntimeError(f"LoRA参数或数据隔离校验失败：{class_name}")
        records.append({
            "class_name": class_name,
            "weight_path": str(weight_path),
            "weight_sha256": sha256_file(weight_path),
            "metadata_path": str(metadata_path),
            "training_subject_ids": sorted(subjects, key=int),
            "test_overlap": [],
        })
    report = {
        "stage": config["stage"], "status": "passed", "fold": int(config["fold"]),
        "allowed_poses": config["data"]["required_poses"], "seed": 22,
        "manifest_sha256": sha256_file(manifest_path), "weights": records,
    }
    path = root / "lora_weights" / "lora_validation_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
