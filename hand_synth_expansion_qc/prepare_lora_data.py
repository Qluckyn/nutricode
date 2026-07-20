#!/usr/bin/env python3
"""从fold_0训练集构建双姿势、两类别LoRA训练数据，并记录可追溯清单。"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import yaml


def sha256_file(path: Path) -> str:
    """计算原始图哈希，保证LoRA输入可复现。"""
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
    output_root = Path(config["output"]["root"]) / "lora_data"
    output_root.mkdir(parents=True, exist_ok=True)
    classes = config["data"]["classes"]
    poses = config["data"]["required_poses"]
    lora_cfg = config["lora"]

    # 营养不良类训练折恰为12名，正常类从42名训练受试者中固定种子抽取12名，避免类别失衡。
    # pose02 LoRA沿用明确的“类别+姿势”类别名，保证训练文本与生成标签一一对应。
    lora_name = {"malnourished_hand": "malnourished_hand_pose02", "normal_hand": "normal_hand_pose02"}
    selected = {"malnourished_hand": sorted(split["train"]["malnourished_hand"]["subject_ids"], key=int)}
    normal_ids = sorted(split["train"]["normal_hand"]["subject_ids"], key=int)
    rng = random.Random(int(lora_cfg["normal_subject_selection_seed"]))
    selected["normal_hand"] = sorted(rng.sample(normal_ids, int(lora_cfg["subjects_per_class"])), key=int)
    if len(selected["malnourished_hand"]) != int(lora_cfg["subjects_per_class"]):
        raise ValueError("fold_0营养不良训练受试者数与LoRA设定不一致")

    excluded = {name: sorted(split["test"][name]["subject_ids"], key=int) for name in classes}
    records: list[dict] = []
    for source_class in ("malnourished_hand", "normal_hand"):
        lora_class = lora_name[source_class]
        target_dir = output_root / lora_class
        target_dir.mkdir(exist_ok=True)
        for subject_id in selected[source_class]:
            for pose in poses:
                source = Path(config["data"]["root"]) / fold_name / "train" / source_class / f"{subject_id}_{pose}.png"
                if not source.is_file():
                    raise FileNotFoundError(f"缺少训练图：{source}")
                target = target_dir / source.name
                # 使用软链接避免复制真实数据；若已有目标，必须仍指向同一训练原图。
                if target.exists() or target.is_symlink():
                    if not target.is_symlink() or target.resolve() != source.resolve():
                        raise FileExistsError(f"目标文件冲突：{target}")
                else:
                    target.symlink_to(source)
                records.append({
                    "lora_class": lora_class,
                    "source_class": source_class,
                    "subject_id": subject_id,
                    "pose": pose,
                    "source_path": str(source.resolve()),
                    "output_path": str(target.resolve()),
                    "sha256": sha256_file(source),
                })

    manifest = {
        "schema_version": 1,
        "is_hand_only": True,
        "fold": int(config["fold"]),
        "dataset": lora_cfg["dataset"],
        "selected_subject_ids": selected,
        "excluded_test_subject_ids": excluded,
        "class_counts": {name: sum(row["lora_class"] == name for row in records) for name in lora_name.values()},
        "records": records,
    }
    (output_root / "hand_lora_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_root": str(output_root), "class_counts": manifest["class_counts"], "selected_subject_ids": selected}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
