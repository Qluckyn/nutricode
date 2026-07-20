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


def _load_manifest_split(manifest_path: Path, fold: int | None):
    """读取扁平划分或五折清单中的一个折，并统一为 splits 结构。"""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"缺少划分清单：{manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if fold is None:
        if "splits" not in manifest:
            raise ValueError(
                "清单不包含扁平 splits 字段；五折数据请显式传入 --fold=<编号>"
            )
        return manifest

    fold_key = f"fold_{fold}"
    folds = manifest.get("folds")
    if not isinstance(folds, dict) or fold_key not in folds:
        available = sorted(folds) if isinstance(folds, dict) else []
        raise ValueError(f"清单中不存在 {fold_key}，可用折：{available}")
    return {
        "seed": manifest.get("seed"),
        "splits": folds[fold_key],
        "fold": fold,
    }


def _expected_counts(manifest):
    """从清单派生各类别受试者数，兼容每折正常组数量不同的五折划分。"""
    return {
        split: {
            class_name: int(manifest["splits"][split][class_name]["subject_count"])
            for class_name in CLASS_NAMES
        }
        for split in SPLIT_NAMES
    }


def validate_split(data_root: Path, manifest_path: Path, expected_counts, fold=None):
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

    manifest = _load_manifest_split(manifest_path, fold)
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
        "fold": manifest.get("fold"),
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
        help="扁平划分根目录；五折模式下传入包含 fold_<编号>/ 和 split.json 的根目录",
    )
    parser.add_argument(
        "--fold", type=int, default=None,
        help="可选五折编号；例如 --fold=0 会校验 <data_root>/fold_0",
    )
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    manifest_path = data_root / "split.json"
    if args.fold is not None:
        if args.fold < 0:
            parser.error("--fold 必须为非负整数")
        split_root = data_root / f"fold_{args.fold}"
        manifest = _load_manifest_split(manifest_path, args.fold)
        expected_counts = _expected_counts(manifest)
    else:
        split_root = data_root
        # 保持扁平数据集的既有固定数量约束。
        expected_counts = {
            "train": {"malnourished_hand": 12, "normal_hand": 42},
            "test": {"malnourished_hand": 3, "normal_hand": 10},
        }
    summary = validate_split(
        split_root, manifest_path, expected_counts, fold=args.fold
    )
    print("[OK] 手部数据划分校验通过")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
