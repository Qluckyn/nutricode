#!/usr/bin/env python3
"""计算“全部预测为训练集多数类”的手部分类基线。"""

import argparse
import json
import re
from pathlib import Path


CLASS_NAMES = ("malnourished_hand", "normal_hand")
IMAGE_PATTERN = re.compile(
    r"^(?P<subject>\d+)_(?P<pose>01|02)\.(?:png|jpg|jpeg)$", re.IGNORECASE
)


def collect_class(class_dir: Path):
    """读取类别目录，返回受试者到两张姿势图片的映射。"""
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
            raise ValueError(f"图片命名不符合 <受试者ID>_<01|02>：{path}")
        subject_id = match.group("subject")
        subjects.setdefault(subject_id, []).append(str(path))
    incomplete = {
        subject_id: len(paths)
        for subject_id, paths in subjects.items()
        if len(paths) != 2
    }
    if incomplete:
        raise ValueError(f"存在姿势不完整的受试者：{incomplete}")
    return subjects


def majority_metrics(positive_count: int, negative_count: int):
    """全部预测正常时的二分类指标，营养不良定义为阳性。"""
    total = positive_count + negative_count
    prevalence = positive_count / total
    return {
        "accuracy": negative_count / total,
        "balanced_accuracy": 0.5,
        "roc_auc": 0.5,
        # 常数分数模型的 PR-AUC 基线等于阳性率。
        "pr_auc": prevalence,
        "sensitivity": 0.0,
        "specificity": 1.0,
        "f1": 0.0,
        "mcc": 0.0,
        "tp": 0,
        "tn": negative_count,
        "fp": 0,
        "fn": positive_count,
    }


def run(data_root: Path, output_dir: Path):
    train = {
        class_name: collect_class(data_root / "train" / class_name)
        for class_name in CLASS_NAMES
    }
    test = {
        class_name: collect_class(data_root / "test" / class_name)
        for class_name in CLASS_NAMES
    }
    majority_class = max(CLASS_NAMES, key=lambda name: len(train[name]))
    if majority_class != "normal_hand":
        raise ValueError(f"预期训练集多数类为 normal_hand，实际为 {majority_class}")

    mal_subjects = len(test["malnourished_hand"])
    normal_subjects = len(test["normal_hand"])
    mal_images = sum(len(paths) for paths in test["malnourished_hand"].values())
    normal_images = sum(len(paths) for paths in test["normal_hand"].values())
    result = {
        "experiment_mode": "majority",
        "description": "所有测试样本均预测为训练集多数类 normal_hand",
        "positive_class": "malnourished_hand",
        "predicted_class": majority_class,
        "train_subject_counts": {
            class_name: len(train[class_name]) for class_name in CLASS_NAMES
        },
        "test_subject_counts": {
            class_name: len(test[class_name]) for class_name in CLASS_NAMES
        },
        "test_image_counts": {
            class_name: sum(len(paths) for paths in test[class_name].values())
            for class_name in CLASS_NAMES
        },
        "image_level_metrics": majority_metrics(mal_images, normal_images),
        "subject_level_metrics": majority_metrics(mal_subjects, normal_subjects),
        "interpretation": "该结果是类别不均衡下的最低参照，不代表模型具有分类能力。",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "majority_baseline_results.json"
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("[OK] 多数类基线计算完成")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[OK] 结果已保存：{output_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description="计算手部数据多数类基线")
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("/root/autodl-tmp/data_hand/split_seed22"),
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.data_root.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
