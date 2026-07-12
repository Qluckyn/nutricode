#!/usr/bin/env python3
"""校验手部 train/test 划分，防止受试者泄漏或姿势缺失。"""

import argparse
import json
import re
from pathlib import Path


CLASS_NAMES = ("malnourished_hand", "normal_hand")
SPLIT_NAMES = ("train", "test")
IMAGE_PATTERN = re.compile(
    r"^(?P<subject>\d+)_(?P<pose>01|02)\.(?:png|jpg|jpeg)$", re.IGNORECASE
)


def _scan_class_dir(class_dir: Path):
    """读取类别目录，并验证每名受试者恰好具有 01/02 两种姿势。"""
    if not class_dir.is_dir():
        raise FileNotFoundError(f"缺少类别目录：{class_dir}")

    subjects = {}
    for path in sorted(class_dir.iterdir()):
        if path.is_symlink() and not path.exists():
            raise RuntimeError(f"发现失效软链接：{path}")
        if not path.is_file():
            continue
        match = IMAGE_PATTERN.fullmatch(path.name)
        if not match:
            raise ValueError(f"文件名不符合 <受试者ID>_<01|02>.<扩展名>：{path}")
        subject_id = match.group("subject")
        pose = match.group("pose")
        poses = subjects.setdefault(subject_id, set())
        if pose in poses:
            raise ValueError(f"同一姿势重复：{class_dir.name}/{subject_id}_{pose}")
        poses.add(pose)

    if not subjects:
        raise RuntimeError(f"类别目录中没有图片：{class_dir}")
    incomplete = {
        subject_id: sorted(poses)
        for subject_id, poses in subjects.items()
        if poses != {"01", "02"}
    }
    if incomplete:
        raise ValueError(f"以下受试者没有完整的 01/02 姿势：{incomplete}")
    return subjects


def _validate_global_subject_separation(scanned):
    """跨类别汇总受试者ID，确保任何受试者都不会同时进入train和test。"""
    train_subjects = set().union(
        *(set(scanned["train"][name]) for name in CLASS_NAMES)
    )
    test_subjects = set().union(
        *(set(scanned["test"][name]) for name in CLASS_NAMES)
    )
    overlap = train_subjects & test_subjects
    if overlap:
        raise ValueError(
            f"存在全局train/test受试者泄漏：{sorted(overlap, key=int)}"
        )


def validate_split(data_root: Path, manifest_path: Path, expected_counts):
    """执行数量、类别、姿势、清单一致性和 train/test 泄漏检查。"""
    scanned = {
        split: {
            class_name: _scan_class_dir(data_root / split / class_name)
            for class_name in CLASS_NAMES
        }
        for split in SPLIT_NAMES
    }

    for split in SPLIT_NAMES:
        class_overlap = (
            set(scanned[split]["malnourished_hand"])
            & set(scanned[split]["normal_hand"])
        )
        if class_overlap:
            raise ValueError(
                f"{split} 中受试者同时出现在两个类别："
                f"{sorted(class_overlap, key=int)}"
            )

    # 必须跨类别检查；例如“营养不良train”和“正常test”的同号受试者也属于泄漏。
    _validate_global_subject_separation(scanned)

    actual_counts = {
        split: {
            class_name: len(scanned[split][class_name])
            for class_name in CLASS_NAMES
        }
        for split in SPLIT_NAMES
    }
    if actual_counts != expected_counts:
        raise ValueError(
            f"受试者数量不符合预期：actual={actual_counts}, expected={expected_counts}"
        )

    if not manifest_path.is_file():
        raise FileNotFoundError(f"缺少划分清单：{manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for split in SPLIT_NAMES:
        for class_name in CLASS_NAMES:
            manifest_ids = set(manifest["splits"][split][class_name]["subject_ids"])
            scanned_ids = set(scanned[split][class_name])
            if manifest_ids != scanned_ids:
                raise ValueError(f"split.json 与目录不一致：{split}/{class_name}")

    return {
        "status": "ok",
        "data_root": str(data_root),
        "manifest": str(manifest_path),
        "seed": manifest.get("seed"),
        "subjects": actual_counts,
        "images": {
            split: {
                class_name: actual_counts[split][class_name] * 2
                for class_name in CLASS_NAMES
            }
            for split in SPLIT_NAMES
        },
    }


def main():
    parser = argparse.ArgumentParser(description="校验手部数据集 train/test 划分")
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("/root/autodl-tmp/data_hand/split_seed22"),
        help="包含 train、test 和 split.json 的划分根目录",
    )
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    expected_counts = {
        "train": {"malnourished_hand": 12, "normal_hand": 42},
        "test": {"malnourished_hand": 3, "normal_hand": 10},
    }
    summary = validate_split(
        data_root, data_root / "split.json", expected_counts
    )
    print("[OK] 手部数据划分校验通过")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
