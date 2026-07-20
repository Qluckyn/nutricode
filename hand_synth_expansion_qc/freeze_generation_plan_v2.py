#!/usr/bin/env python3
"""冻结正式 V2 候选：外观—结构解耦，三档 I2I 强度均衡。"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict, Counter
from pathlib import Path

import yaml


def seed_for(base: int, compound: str, index: int) -> int:
    payload = f"{base}:foundhand_i2i_v2:{compound}:{index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in Path(config["condition_manifest"]).read_text(encoding="utf-8").splitlines() if line.strip()]
    groups = defaultdict(list)
    for row in rows:
        if row["source_split"] != f"fold_{int(config['fold'])}_train" or row["pose"] != "02" or not row["usable_for_generation"]:
            raise RuntimeError(f"发现无效V2条件：{row['condition_id']}")
        groups[row["compound_class"]].append(row)
    strengths = [float(value) for value in config["generation"]["denoising_strengths"]]
    total = int(config["candidate_count_per_compound"])
    if total % len(strengths):
        raise ValueError("候选数必须可被强度档数整除")
    plan = []
    for compound in config["compound_classes"]:
        appearances = sorted(groups[compound], key=lambda row: int(row["parent_subject_id"]))
        structures = sorted(groups[compound], key=lambda row: int(row["parent_subject_id"]))
        for index in range(total):
            appearance = appearances[index % len(appearances)]
            # 使用互素步长和偏移，使目标结构与外观父图系统性交叉而非同图复用。
            structure = structures[(index * 7 + 1) % len(structures)]
            if len(structures) > 1:
                offset = 1
                while structure["parent_subject_id"] == appearance["parent_subject_id"]:
                    structure = structures[(index * 7 + offset) % len(structures)]
                    offset += 1
            strength = strengths[(index // (total // len(strengths))) % len(strengths)]
            plan.append({
                "candidate_id": f"Q_FOUNDHAND_I2I_V2_{compound}_{index:04d}", "mode": "foundhand_i2i",
                "compound_class": compound, "nutrition_class": appearance["nutrition_class"], "pose": "02",
                "appearance_parent_subject_id": appearance["parent_subject_id"], "appearance_parent_path": appearance["padded_image_path"],
                "appearance_keypoints_path": appearance["keypoints_path"], "appearance_mask_path": appearance["mask_path"],
                "label_reference_subject_id": appearance["parent_subject_id"], "label_reference_path": appearance["source_path"],
                "structure_condition_id": structure["condition_id"], "structure_parent_subject_id": structure["parent_subject_id"],
                "keypoints_path": structure["keypoints_path"], "seed": seed_for(int(config["base_seed"]), compound, index),
                "denoising_strength": strength, "generation_variant": f"strength_{strength:.2f}",
            })
    output = Path(config["generation_plan"])
    output.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in plan)
    if output.exists() and output.read_text(encoding="utf-8") != text:
        raise RuntimeError(f"候选计划已存在且不一致：{output}")
    output.write_text(text, encoding="utf-8")
    summary = {"total": len(plan), "by_class": Counter(row["compound_class"] for row in plan), "by_strength": Counter(str(row["denoising_strength"]) for row in plan)}
    (output.parent / "candidate_plan_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: (dict(value) if isinstance(value, Counter) else value) for key, value in summary.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
