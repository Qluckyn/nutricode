#!/usr/bin/env python3
"""阶段 D：手部合成图的非破坏性质量控制与盲审材料生成。"""

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import clip
import cv2
import mediapipe as mp
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps


CLASSES = (
    "malnourished_hand_pose01",
    "malnourished_hand_pose02",
    "normal_hand_pose01",
    "normal_hand_pose02",
)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_symlink(source, target):
    """只创建指向原图的链接，阶段 D 永不移动或删除阶段 C 原图。"""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() and target.resolve() == source.resolve():
        return
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"拒绝覆盖已有 QC 产物：{target}")
    target.symlink_to(source.resolve())


def dhash(image, hash_size=16):
    gray = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    values = np.asarray(gray)
    bits = values[:, 1:] > values[:, :-1]
    return "".join("1" if value else "0" for value in bits.ravel())


def hamming(left, right):
    return sum(a != b for a, b in zip(left, right))


def load_manifest(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("records", [])
    if not records:
        raise ValueError(f"真实训练 manifest 无记录：{path}")
    return data, records


def load_generation_records(input_root):
    records = []
    for class_name in CLASSES:
        class_dir = input_root / "train" / class_name
        metadata_path = class_dir / "metadata.jsonl"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"缺少生成元数据：{metadata_path}")
        with metadata_path.open(encoding="utf-8") as handle:
            class_records = [json.loads(line) for line in handle if line.strip()]
        images = sorted(path for path in class_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        if len(class_records) != len(images):
            raise ValueError(
                f"{class_name} 图片/元数据数量不一致：{len(images)} != {len(class_records)}"
            )
        for record in class_records:
            path = Path(record["output_path"])
            if record.get("class_name") != class_name or path.parent.resolve() != class_dir.resolve():
                raise ValueError(f"类别或路径追溯不一致：{record}")
            if not path.is_file():
                raise FileNotFoundError(path)
            record = dict(record)
            record["path"] = path
            records.append(record)
    return records


def create_landmarker(model_path):
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=3,
        min_hand_detection_confidence=0.35,
        min_hand_presence_confidence=0.35,
        min_tracking_confidence=0.35,
    )
    return HandLandmarker.create_from_options(options)


def hand_geometry(landmarks):
    """用指尖到掌根的相对距离估计伸展手指数；握拳时该距离通常更短。"""
    points = np.asarray([[point.x, point.y] for point in landmarks], dtype=np.float32)
    palm_scale = np.linalg.norm(points[9] - points[0]) + 1e-6
    extended = 0
    ratios = []
    for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
        tip_distance = np.linalg.norm(points[tip] - points[0]) / palm_scale
        pip_distance = np.linalg.norm(points[pip] - points[0]) / palm_scale
        ratios.append(float(tip_distance))
        extended += int(tip_distance > pip_distance * 1.18 and tip_distance > 1.35)
    return extended, ratios


def inspect_image(path, class_name, landmarker):
    reasons = []
    metrics = {}
    try:
        with Image.open(path) as opened:
            opened.verify()
        image = Image.open(path).convert("RGB")
    except Exception as exc:
        return None, ["file_unreadable"], {"read_error": str(exc)}

    array = np.asarray(image)
    metrics.update(
        width=image.width,
        height=image.height,
        pixel_std=float(array.std()),
        brightness_mean=float(array.mean()),
        brightness_std=float(cv2.cvtColor(array, cv2.COLOR_RGB2GRAY).std()),
        saturation_mean=float(cv2.cvtColor(array, cv2.COLOR_RGB2HSV)[:, :, 1].mean()),
        dhash=dhash(image),
    )
    if image.size != (512, 512):
        reasons.append("invalid_dimensions")
    if metrics["pixel_std"] < 8.0:
        reasons.append("blank_or_low_variance")
    if metrics["brightness_mean"] < 25.0 or metrics["brightness_mean"] > 245.0:
        reasons.append("extreme_exposure")

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(array))
    result = landmarker.detect(mp_image)
    hand_count = len(result.hand_landmarks)
    metrics["detected_hand_count"] = hand_count
    if hand_count != 2:
        reasons.append("hand_count_not_two")
    extended_counts = []
    for landmarks in result.hand_landmarks:
        extended, _ = hand_geometry(landmarks)
        extended_counts.append(extended)
    metrics["extended_fingers_per_hand"] = extended_counts
    total_extended = sum(extended_counts)
    metrics["extended_fingers_total"] = total_extended
    if hand_count == 2 and class_name.endswith("pose01") and total_extended > 3:
        reasons.append("pose01_not_fists")
    if hand_count == 2 and class_name.endswith("pose02") and total_extended < 6:
        reasons.append("pose02_not_extended")
    return image, reasons, metrics


def extract_clip_features(paths, model, preprocess, device, batch_size=32):
    features = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        tensors = torch.stack([preprocess(Image.open(path).convert("RGB")) for path in batch_paths])
        with torch.no_grad():
            output = model.encode_image(tensors.to(device)).float()
            output /= output.norm(dim=-1, keepdim=True)
        features.append(output.cpu().numpy())
    return np.concatenate(features, axis=0)


def robust_distribution_threshold(real_features):
    centroid = real_features.mean(axis=0)
    centroid /= np.linalg.norm(centroid)
    distances = 1.0 - real_features @ centroid
    median = float(np.median(distances))
    mad = float(np.median(np.abs(distances - median)))
    # 阈值只由当前折真实训练图确定；小样本下同时给 MAD 一个保守下限。
    threshold = max(float(np.quantile(distances, 0.99)), median + 4.0 * max(mad, 0.005))
    return centroid, distances, threshold


def make_grid(items, destination, title, columns=5, thumb_size=220):
    rows = max(1, (len(items) + columns - 1) // columns)
    canvas = Image.new("RGB", (columns * thumb_size, rows * (thumb_size + 28) + 36), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title, fill="black")
    for index, (label, path) in enumerate(items):
        row, column = divmod(index, columns)
        image = Image.open(path).convert("RGB")
        thumb = ImageOps.contain(image, (thumb_size - 8, thumb_size - 8))
        x = column * thumb_size + (thumb_size - thumb.width) // 2
        y = 36 + row * (thumb_size + 28) + (thumb_size - thumb.height) // 2
        canvas.paste(thumb, (x, y))
        draw.text((column * thumb_size + 6, 36 + row * (thumb_size + 28) + thumb_size + 3), label, fill="black")
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="手部合成图阶段 D 非破坏性 QC")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--real-manifest", type=Path, required=True)
    parser.add_argument("--hand-model", type=Path, required=True)
    parser.add_argument("--clip-model", default="ViT-B/16")
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--blind-sample-per-class", type=int, default=10)
    args = parser.parse_args()

    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"拒绝覆盖非空 QC 输出目录：{args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    generation_records = load_generation_records(args.input_root)
    manifest, real_records = load_manifest(args.real_manifest)
    real_by_class = defaultdict(list)
    for record in real_records:
        real_by_class[record["lora_class"]].append(Path(record["source_path"]))
    if set(real_by_class) != set(CLASSES):
        raise ValueError("真实训练 manifest 未覆盖四个手部复合类别")

    landmarker = create_landmarker(args.hand_model)
    inspected = []
    for record in generation_records:
        image, reasons, metrics = inspect_image(record["path"], record["class_name"], landmarker)
        inspected.append({**record, "image": image, "reasons": reasons, "metrics": metrics})
    landmarker.close()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load(args.clip_model, device=device, jit=False)
    clip_model.eval()
    all_real_paths = [path for class_name in CLASSES for path in real_by_class[class_name]]
    real_features_all = extract_clip_features(all_real_paths, clip_model, preprocess, device)
    real_feature_map = {str(path.resolve()): feature for path, feature in zip(all_real_paths, real_features_all)}
    synth_paths = [record["path"] for record in inspected]
    synth_features = extract_clip_features(synth_paths, clip_model, preprocess, device)

    thresholds = {}
    real_hashes = []
    for path in all_real_paths:
        real_hashes.append((path, dhash(Image.open(path).convert("RGB"))))
    previous_synth = []
    for index, record in enumerate(inspected):
        class_name = record["class_name"]
        class_real = np.stack([real_feature_map[str(path.resolve())] for path in real_by_class[class_name]])
        centroid, real_distances, threshold = robust_distribution_threshold(class_real)
        thresholds[class_name] = {
            "clip_centroid_distance_threshold": threshold,
            "real_distance_median": float(np.median(real_distances)),
            "real_distance_mad": float(np.median(np.abs(real_distances - np.median(real_distances)))),
        }
        feature = synth_features[index]
        distance = float(1.0 - feature @ centroid)
        record["metrics"]["clip_centroid_distance"] = distance
        if distance > threshold:
            record["reasons"].append("clip_distribution_outlier")

        nearest_real_index = int(np.argmax(real_features_all @ feature))
        nearest_real_similarity = float(real_features_all[nearest_real_index] @ feature)
        record["metrics"]["nearest_real_clip_similarity"] = nearest_real_similarity
        record["metrics"]["nearest_real_path"] = str(all_real_paths[nearest_real_index])
        hash_value = record["metrics"].get("dhash")
        real_hash_distance = min(hamming(hash_value, item_hash) for _, item_hash in real_hashes)
        record["metrics"]["nearest_real_dhash_distance"] = real_hash_distance
        if nearest_real_similarity >= 0.995 or real_hash_distance <= 4:
            record["reasons"].append("near_duplicate_real")

        if previous_synth:
            similarities = synth_features[:index] @ feature
            nearest_index = int(np.argmax(similarities))
            similarity = float(similarities[nearest_index])
            hash_distance = min(hamming(hash_value, item[1]) for item in previous_synth)
            record["metrics"]["nearest_previous_synth_clip_similarity"] = similarity
            record["metrics"]["nearest_previous_synth_dhash_distance"] = hash_distance
            if similarity >= 0.995 or hash_distance <= 4:
                record["reasons"].append("near_duplicate_synthetic")
        previous_synth.append((record["path"], hash_value))

    del clip_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    counts = defaultdict(Counter)
    metadata_path = args.output_root / "qc_metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as handle:
        for record in inspected:
            reasons = list(dict.fromkeys(record["reasons"]))
            status = "accepted" if not reasons else "rejected"
            class_name = record["class_name"]
            pose = record["pose"]
            filename = f"{pose}_seed{record['seed']}_cfg001_{record['output_index']:06d}.png"
            raw_link = args.output_root / "raw" / class_name / filename
            atomic_symlink(record["path"], raw_link)
            qc_link = None
            if status == "accepted":
                qc_link = args.output_root / "qc" / class_name / filename
                atomic_symlink(record["path"], qc_link)
            counts[class_name][status] += 1
            output = {
                "schema_version": 1,
                "source_generation_metadata": {key: value for key, value in record.items() if key not in {"path", "image", "reasons", "metrics"}},
                "source_path": str(record["path"].resolve()),
                "raw_link": str(raw_link.resolve(strict=False)),
                "qc_link": str(qc_link.resolve(strict=False)) if qc_link else None,
                "status": status,
                "rejection_reasons": reasons,
                "manual_review_status": "pending_blind_review",
                "metrics": record["metrics"],
                "sha256": sha256(record["path"]),
            }
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")

    shortcut_report = {}
    for nutrition in ("malnourished", "normal"):
        subset = [item for item in inspected if item["nutrition_status"] == nutrition]
        shortcut_report[nutrition] = {
            metric: {
                "mean": float(np.mean([item["metrics"][metric] for item in subset])),
                "std": float(np.std([item["metrics"][metric] for item in subset])),
            }
            for metric in ("brightness_mean", "brightness_std", "saturation_mean")
        }

    rng = random.Random(args.seed)
    blind_rows = []
    blind_items = []
    blind_key = {}
    for class_name in CLASSES:
        candidates = [item for item in inspected if item["class_name"] == class_name]
        sample = rng.sample(candidates, min(args.blind_sample_per_class, len(candidates)))
        for item in sample:
            blind_rows.append(item)
    rng.shuffle(blind_rows)
    blind_dir = args.output_root / "blind_review" / "images"
    for index, item in enumerate(blind_rows, 1):
        blind_id = f"B{index:03d}"
        target = blind_dir / f"{blind_id}.png"
        atomic_symlink(item["path"], target)
        blind_items.append((blind_id, item["path"]))
        blind_key[blind_id] = {
            "class_name": item["class_name"],
            "source_path": str(item["path"]),
            "automatic_status": "accepted" if not item["reasons"] else "rejected",
        }
    with (args.output_root / "blind_review" / "review_form.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["blind_id", "two_hands", "anatomy_ok", "pose_ok", "scene_ok", "accept", "failure_types", "notes"])
        for blind_id, _ in blind_items:
            writer.writerow([blind_id, "", "", "", "", "", "", ""])
    write_json(args.output_root / "blind_review" / "private_key.json", blind_key)
    make_grid(blind_items, args.output_root / "blind_review" / "blind_grid.jpg", "Stage D blind review")

    rejected_items = []
    accepted_items = []
    for item in inspected:
        label = f"{item['class_name']}:{item['output_index']}"
        (accepted_items if not item["reasons"] else rejected_items).append((label, item["path"]))
    make_grid(accepted_items[:40], args.output_root / "reports" / "accepted_examples.jpg", "Automatic accepted examples")
    make_grid(rejected_items[:40], args.output_root / "reports" / "rejected_examples.jpg", "Automatic rejected examples")

    report = {
        "schema_version": 1,
        "task": "hand_synthetic_quality_control",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_root": str(args.input_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "real_manifest": str(args.real_manifest.resolve()),
        "real_manifest_sha256": sha256(args.real_manifest),
        "real_split_manifest_sha256": manifest.get("source_split_sha256"),
        "test_data_used": False,
        "models": {
            "hand_landmarker": {"path": str(args.hand_model.resolve()), "sha256": sha256(args.hand_model), "mediapipe_version": mp.__version__},
            "distribution_and_duplicate_feature_extractor": {"name": args.clip_model, "package": "openai-clip", "weight_cache": "/root/.cache/clip/ViT-B-16.pt"},
        },
        "fixed_rules": {
            "dimensions": [512, 512],
            "pixel_std_min": 8.0,
            "brightness_range": [25.0, 245.0],
            "detected_hand_count": 2,
            "pose01_max_extended_non_thumb_fingers": 3,
            "pose02_min_extended_non_thumb_fingers": 6,
            "clip_distribution_threshold": "per-class real-train centroid: max(real q99, median + 4*max(MAD,0.005))",
            "near_duplicate_clip_similarity_min": 0.995,
            "near_duplicate_dhash_distance_max": 4,
        },
        "thresholds": thresholds,
        "counts": {class_name: dict(counts[class_name]) for class_name in CLASSES},
        "rejection_reason_counts": dict(Counter(reason for item in inspected for reason in set(item["reasons"]))),
        "class_shortcut_statistics": shortcut_report,
        "blind_review": {"sample_count": len(blind_rows), "status": "pending_human_review", "seed": args.seed},
    }
    write_json(args.output_root / "quality_report.json", report)
    write_json(args.output_root / "qc_config.json", vars(args) | {"input_root": str(args.input_root), "output_root": str(args.output_root), "real_manifest": str(args.real_manifest), "hand_model": str(args.hand_model)})
    print(json.dumps(report["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
