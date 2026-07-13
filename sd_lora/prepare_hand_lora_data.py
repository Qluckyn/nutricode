#!/usr/bin/env python3
"""审计手部数据划分，并为首折 LoRA 训练准备四类软链接目录。

该脚本是手部实验专用入口，不复用也不修改面部数据准备逻辑。输出目录中只
创建指向真实训练集的软链接，原始图片与原始 split.json 均不会被改写。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


SOURCE_CLASSES = ("malnourished_hand", "normal_hand")
SPLITS = ("train", "test")
POSES = ("01", "02")
OUTPUT_CLASS = {
    ("malnourished_hand", "01"): "malnourished_hand_pose01",
    ("malnourished_hand", "02"): "malnourished_hand_pose02",
    ("normal_hand", "01"): "normal_hand_pose01",
    ("normal_hand", "02"): "normal_hand_pose02",
}
IMAGE_PATTERN = re.compile(
    r"^(?P<subject>\d+)_(?P<pose>01|02)\.(?P<ext>png|jpg|jpeg)$",
    re.IGNORECASE,
)


def numeric_sort(values):
    """按受试者数字 ID 排序，同时保留 split.json 中的原始字符串形式。"""
    return sorted(values, key=lambda value: (int(value), value))


def sha256_file(path: Path) -> str:
    """计算文件摘要，用于追溯具体训练图片和原始划分清单。"""
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_image(path: Path) -> dict:
    """验证 PIL 可完整解码图片，并返回不会泄露图像内容的基础元数据。"""
    try:
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            mode = image.mode
            image_format = image.format
    except Exception as exc:  # Pillow 的解码异常类型会随图片格式而变化。
        raise ValueError(f"图片无法正常读取：{path}: {exc}") from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"图片尺寸无效：{path}: {width}x{height}")
    return {
        "width": width,
        "height": height,
        "mode": mode,
        "format": image_format,
        "sha256": sha256_file(path),
    }


def scan_class_dir(class_dir: Path) -> dict[str, dict[str, Path]]:
    """扫描单个状态目录，并强制每名受试者恰好具有 01、02 两张图。"""
    if not class_dir.is_dir():
        raise FileNotFoundError(f"缺少类别目录：{class_dir}")

    subjects: dict[str, dict[str, Path]] = {}
    unexpected = []
    for path in sorted(class_dir.iterdir()):
        if path.name.startswith("."):
            continue
        if path.is_symlink() and not path.exists():
            raise RuntimeError(f"发现失效软链接：{path}")
        if not path.is_file():
            unexpected.append(str(path))
            continue
        match = IMAGE_PATTERN.fullmatch(path.name)
        if not match:
            unexpected.append(str(path))
            continue
        subject_id = match.group("subject")
        pose = match.group("pose")
        pose_files = subjects.setdefault(subject_id, {})
        if pose in pose_files:
            raise ValueError(f"同一受试者姿势重复：{class_dir.name}/{subject_id}_{pose}")
        pose_files[pose] = path.resolve()

    if unexpected:
        raise ValueError(f"类别目录中存在非预期文件或目录：{unexpected}")
    if not subjects:
        raise RuntimeError(f"类别目录中没有有效图片：{class_dir}")

    incomplete = {
        subject_id: sorted(pose_files)
        for subject_id, pose_files in subjects.items()
        if set(pose_files) != set(POSES)
    }
    if incomplete:
        raise ValueError(f"以下受试者缺少完整 01/02 姿势：{incomplete}")
    return subjects


def load_and_validate_source(data_root: Path) -> tuple[dict, dict]:
    """校验目录、split.json、跨类别关系及 train/test 受试者隔离。"""
    split_path = data_root / "split.json"
    if not split_path.is_file():
        raise FileNotFoundError(f"缺少原始划分清单：{split_path}")
    split_manifest = json.loads(split_path.read_text(encoding="utf-8"))

    scanned = {
        split: {
            class_name: scan_class_dir(data_root / split / class_name)
            for class_name in SOURCE_CLASSES
        }
        for split in SPLITS
    }

    for split in SPLITS:
        mal_ids = set(scanned[split]["malnourished_hand"])
        normal_ids = set(scanned[split]["normal_hand"])
        overlap = mal_ids & normal_ids
        if overlap:
            raise ValueError(
                f"{split} 中受试者同时出现在两个状态：{numeric_sort(overlap)}"
            )
        for class_name in SOURCE_CLASSES:
            declared = set(
                split_manifest["splits"][split][class_name]["subject_ids"]
            )
            actual = set(scanned[split][class_name])
            if declared != actual:
                raise ValueError(
                    f"split.json 与目录不一致：{split}/{class_name}; "
                    f"manifest_only={numeric_sort(declared - actual)}, "
                    f"directory_only={numeric_sort(actual - declared)}"
                )

    # 必须跨状态汇总 ID 后检查，防止“营养不良训练者”进入“正常测试集”等泄漏。
    train_ids = set().union(
        *(set(scanned["train"][class_name]) for class_name in SOURCE_CLASSES)
    )
    test_ids = set().union(
        *(set(scanned["test"][class_name]) for class_name in SOURCE_CLASSES)
    )
    leakage = train_ids & test_ids
    if leakage:
        raise ValueError(f"发现 train/test 受试者泄漏：{numeric_sort(leakage)}")

    audit = {
        "status": "ok",
        "source_data_root": str(data_root),
        "source_split_manifest": str(split_path.resolve()),
        "source_split_sha256": sha256_file(split_path),
        "source_split_seed": split_manifest.get("seed"),
        "counts": {
            split: {
                class_name: {
                    "subjects": len(scanned[split][class_name]),
                    "images": len(scanned[split][class_name]) * len(POSES),
                }
                for class_name in SOURCE_CLASSES
            }
            for split in SPLITS
        },
        "train_test_subject_overlap": [],
    }
    return scanned, audit


def select_training_subjects(scanned: dict, selection_seed: int) -> dict[str, list[str]]:
    """保留全部营养不良训练者，并按受试者固定抽取等量正常训练者。"""
    malnourished = numeric_sort(scanned["train"]["malnourished_hand"])
    normal_pool = numeric_sort(scanned["train"]["normal_hand"])
    if len(normal_pool) < len(malnourished):
        raise ValueError(
            "正常训练受试者数量少于营养不良训练受试者，无法建立首轮均衡集"
        )
    rng = random.Random(selection_seed)
    selected_normal = numeric_sort(rng.sample(normal_pool, len(malnourished)))
    return {
        "malnourished_hand": malnourished,
        "normal_hand": selected_normal,
    }


def discover_existing_selection(output_root: Path, scanned: dict) -> dict[str, list[str]] | None:
    """若用户已准备四类目录，则采用并严格校验其受试者选择。

    用户先于脚本准备的数据不应被覆盖。这里只接受四类姿势成对、全部来自训练集、
    且内容与训练源图片完全一致的实体文件或软链接。
    """
    class_dirs = [output_root / output_class for output_class in OUTPUT_CLASS.values()]
    if not output_root.exists() or not any(path.exists() for path in class_dirs):
        return None

    discovered: dict[tuple[str, str], set[str]] = {}
    for (source_class, pose), output_class in OUTPUT_CLASS.items():
        class_dir = output_root / output_class
        if not class_dir.is_dir():
            raise FileNotFoundError(f"已存在的输出中缺少四类之一：{class_dir}")
        subject_ids = set()
        for path in sorted(class_dir.iterdir()):
            if path.name.startswith("."):
                continue
            if path.is_symlink() and not path.exists():
                raise RuntimeError(f"已准备数据中存在失效软链接：{path}")
            if not path.is_file():
                raise ValueError(f"已准备类别目录中存在非图片内容：{path}")
            match = IMAGE_PATTERN.fullmatch(path.name)
            if not match or match.group("pose") != pose:
                raise ValueError(f"图片姿势与类别目录不一致：{path}")
            subject_id = match.group("subject")
            if subject_id not in scanned["train"][source_class]:
                raise ValueError(f"图片受试者不属于对应真实训练集：{path}")
            source = scanned["train"][source_class][subject_id][pose]
            if sha256_file(path) != sha256_file(source):
                raise ValueError(f"已准备图片与真实训练源内容不一致：{path}")
            subject_ids.add(subject_id)
        if not subject_ids:
            raise RuntimeError(f"已准备类别目录为空：{class_dir}")
        discovered[(source_class, pose)] = subject_ids

    selected = {}
    for source_class in SOURCE_CLASSES:
        pose01_ids = discovered[(source_class, "01")]
        pose02_ids = discovered[(source_class, "02")]
        if pose01_ids != pose02_ids:
            raise ValueError(
                f"已准备数据的两个姿势受试者不一致：{source_class}; "
                f"pose01_only={numeric_sort(pose01_ids - pose02_ids)}, "
                f"pose02_only={numeric_sort(pose02_ids - pose01_ids)}"
            )
        selected[source_class] = numeric_sort(pose01_ids)

    if len(selected["malnourished_hand"]) != len(selected["normal_hand"]):
        raise ValueError(f"已准备数据不均衡：{ {k: len(v) for k, v in selected.items()} }")
    return selected


def ensure_output_is_safe(output_root: Path) -> None:
    """拒绝混入未知文件，避免脚本覆盖人工数据或其他实验结果。"""
    allowed = set(OUTPUT_CLASS.values()) | {"manifest.json", "audit_report.json"}
    if not output_root.exists():
        return
    unexpected = [path.name for path in output_root.iterdir() if path.name not in allowed]
    if unexpected:
        raise RuntimeError(
            f"输出目录含有非本脚本管理的内容，拒绝继续：{sorted(unexpected)}"
        )


def create_or_verify_training_file(source: Path, destination: Path) -> str:
    """幂等准备训练文件，并接受用户已准备的内容相同实体图片。"""
    if destination.is_symlink():
        if not destination.exists():
            raise RuntimeError(f"输出目录存在失效软链接：{destination}")
        if destination.resolve() != source.resolve():
            raise RuntimeError(
                f"软链接目标不一致：{destination} -> {destination.resolve()}, "
                f"expected={source.resolve()}"
            )
        return "symlink"
    if destination.exists():
        if not destination.is_file():
            raise RuntimeError(f"输出位置已存在且不是图片文件：{destination}")
        if sha256_file(destination) != sha256_file(source):
            raise RuntimeError(f"已存在图片与训练源内容不一致：{destination}")
        return "verified_regular_file"
    destination.symlink_to(source.resolve())
    return "symlink"


def synchronize_managed_symlinks(
    output_root: Path,
    scanned: dict,
    selected: dict[str, list[str]],
) -> list[str]:
    """按显式受试者集合移除过期软链接，但绝不删除已有实体图片。"""
    expected_paths = set()
    for source_class in SOURCE_CLASSES:
        for subject_id in selected[source_class]:
            for pose in POSES:
                source = scanned["train"][source_class][subject_id][pose]
                output_class = OUTPUT_CLASS[(source_class, pose)]
                expected_paths.add((output_root / output_class / source.name).absolute())

    stale_symlinks = []
    for output_class in OUTPUT_CLASS.values():
        class_dir = output_root / output_class
        if not class_dir.exists():
            continue
        for path in class_dir.iterdir():
            if path.name.startswith("."):
                raise RuntimeError(f"同步前请先移除类别目录中的隐藏缓存：{path}")
            if path.absolute() in expected_paths:
                continue
            if not path.is_symlink():
                raise RuntimeError(
                    f"同步选择时拒绝删除非软链接文件，请人工核对：{path}"
                )
            stale_symlinks.append(path)

    # 完整预检通过后才统一删除，避免中途报错造成只替换了一半的状态。
    removed = []
    for path in stale_symlinks:
        # 这里只清理本脚本管理目录中的旧受试者软链接，不影响真实训练源。
        if path.is_symlink():
            path.unlink()
            removed.append(str(path.absolute()))
    return removed


def prepare_output(
    scanned: dict,
    selected: dict[str, list[str]],
    output_root: Path,
    audit: dict,
    selection_seed: int,
    sync_selection: bool = False,
) -> dict:
    """建立四类目录并生成包含图片摘要的首折训练 manifest。"""
    ensure_output_is_safe(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for output_class in OUTPUT_CLASS.values():
        (output_root / output_class).mkdir(exist_ok=True)

    if sync_selection:
        audit["removed_stale_symlinks"] = synchronize_managed_symlinks(
            output_root=output_root,
            scanned=scanned,
            selected=selected,
        )

    records = []
    expected_destinations = set()
    for source_class in SOURCE_CLASSES:
        for subject_id in selected[source_class]:
            for pose in POSES:
                source = scanned["train"][source_class][subject_id][pose]
                output_class = OUTPUT_CLASS[(source_class, pose)]
                destination = output_root / output_class / source.name
                storage_type = create_or_verify_training_file(source, destination)
                expected_destinations.add(destination.absolute())
                records.append(
                    {
                        "subject_id": subject_id,
                        "nutrition_class": source_class,
                        "pose": pose,
                        "lora_class": output_class,
                        "source_path": str(source),
                        "output_path": str(destination.absolute()),
                        "output_relative_path": str(destination.relative_to(output_root)),
                        "storage_type": storage_type,
                        **inspect_image(source),
                    }
                )

    # 防止上一次运行遗留的额外软链接悄悄进入 LoRA 训练集。
    for output_class in OUTPUT_CLASS.values():
        for path in (output_root / output_class).iterdir():
            if path.absolute() not in expected_destinations:
                raise RuntimeError(f"LoRA 类别目录中存在非预期文件：{path}")

    manifest = {
        "schema_version": 1,
        "task": "hand_lora_first_fold_balanced_training_data",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "is_hand_only": True,
        "storage_policy": "验证并保留已有等价实体图片；缺失图片使用绝对软链接创建",
        "source_data_root": audit["source_data_root"],
        "source_split_manifest": audit["source_split_manifest"],
        "source_split_sha256": audit["source_split_sha256"],
        "source_split_seed": audit["source_split_seed"],
        "selection_seed": selection_seed,
        "selection_policy": audit["selection_policy"],
        "selected_subject_ids": selected,
        "excluded_test_subject_ids": {
            class_name: numeric_sort(scanned["test"][class_name])
            for class_name in SOURCE_CLASSES
        },
        "class_counts": {
            output_class: sum(
                record["lora_class"] == output_class for record in records
            )
            for output_class in OUTPUT_CLASS.values()
        },
        "records": records,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "audit_report.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="审计 split_seed22，并准备手部首折四类 LoRA 训练数据"
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("/root/autodl-tmp/data_hand/split_seed22"),
        help="包含 train、test、split.json 的真实手部划分目录",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("/root/autodl-tmp/data_hand/lora_train/fold_0"),
        help="手部 LoRA 四类软链接目录；必须与面部数据目录隔离",
    )
    parser.add_argument(
        "--selection_seed",
        type=int,
        default=22,
        help="从正常训练池抽取匹配受试者的固定随机种子",
    )
    parser.add_argument(
        "--normal_subject_ids",
        type=str,
        default=None,
        help=(
            "可选：逗号分隔的正常训练受试者 ID。用于复现用户已选集合；"
            "未指定且输出目录为空时才按 selection_seed 随机抽取"
        ),
    )
    parser.add_argument(
        "--sync_selection",
        action="store_true",
        help=(
            "按 normal_subject_ids 同步已有输出；只删除过期软链接，"
            "若遇到实体文件则拒绝操作"
        ),
    )
    parser.add_argument(
        "--audit_only",
        action="store_true",
        help="只审计并打印结果，不创建目录或软链接",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_root = args.output_root.absolute()
    scanned, audit = load_and_validate_source(data_root)
    # 显式名单优先，便于经过授权后安全替换受试者；原自动发现逻辑继续保留在后续分支。
    if args.normal_subject_ids:
        normal_ids = numeric_sort(
            {value.strip() for value in args.normal_subject_ids.split(",") if value.strip()}
        )
        malnourished_ids = numeric_sort(scanned["train"]["malnourished_hand"])
        unknown = set(normal_ids) - set(scanned["train"]["normal_hand"])
        if unknown:
            raise ValueError(f"指定的正常受试者不属于训练集：{numeric_sort(unknown)}")
        if len(normal_ids) != len(malnourished_ids):
            raise ValueError(
                f"指定正常受试者数量必须为 {len(malnourished_ids)}，实际为 {len(normal_ids)}"
            )
        selected = {
            "malnourished_hand": malnourished_ids,
            "normal_hand": normal_ids,
        }
        audit["selection_policy"] = (
            "使用全部营养不良训练受试者；正常训练受试者由命令行显式指定；"
            "同一受试者的 pose01 和 pose02 同时入选"
        )
    else:
        existing_selection = discover_existing_selection(output_root, scanned)
        if existing_selection is not None:
            selected = existing_selection
            audit["selection_policy"] = (
                "采用用户已准备的均衡受试者集合；脚本已验证四类姿势配对、训练集来源及文件摘要"
            )
        else:
            selected = select_training_subjects(scanned, args.selection_seed)
            audit["selection_policy"] = (
                "使用全部营养不良训练受试者；从正常训练池按固定随机种子抽取等量受试者；"
                "同一受试者的 pose01 和 pose02 必须同时入选"
            )

    if args.audit_only:
        result = {
            "audit": audit,
            "selection_seed": args.selection_seed,
            "selected_subject_ids": selected,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    manifest = prepare_output(
        scanned=scanned,
        selected=selected,
        output_root=output_root,
        audit=audit,
        selection_seed=args.selection_seed,
        sync_selection=args.sync_selection,
    )
    print("[OK] 手部数据审计及首折 LoRA 数据准备完成")
    print(f"[OK] 输出目录：{output_root}")
    print(json.dumps(manifest["class_counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
