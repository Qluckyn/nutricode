#!/usr/bin/env python3
"""阶段 V2-B：为 fold 内真实训练图准备 FoundHand 手部条件。

仅处理手部数据；不读取测试集、面部数据或面部模型。原图只读，所有派生产物写入独立目录。
"""

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from PIL import Image, ImageDraw


HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def square_pad_with_transform(image, size):
    """保持原始比例白色补边，并返回从原图到补边图的仿射参数。"""
    image = image.convert("RGB")
    original_width, original_height = image.size
    scale = min(size / original_width, size / original_height)
    resized_width = max(1, round(original_width * scale))
    resized_height = max(1, round(original_height * scale))
    resized = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
    pad_left = (size - resized_width) // 2
    pad_top = (size - resized_height) // 2
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(resized, (pad_left, pad_top))
    transform = {
        "type": "scale_then_pad",
        "original_size": [original_width, original_height],
        "output_size": [size, size],
        "scale_x": resized_width / original_width,
        "scale_y": resized_height / original_height,
        "resized_size": [resized_width, resized_height],
        "pad_left": pad_left,
        "pad_top": pad_top,
        "matrix_3x3": [
            [resized_width / original_width, 0.0, float(pad_left)],
            [0.0, resized_height / original_height, float(pad_top)],
            [0.0, 0.0, 1.0],
        ],
    }
    return canvas, transform


class HandDetector:
    """复用单个 MediaPipe 检测器，避免每张图重复初始化。"""

    def __init__(self, model_path):
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.20,
            min_hand_presence_confidence=0.20,
            min_tracking_confidence=0.20,
        )
        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def close(self):
        self.detector.close()

    def detect(self, image):
        array = np.ascontiguousarray(np.asarray(image))
        result = self.detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=array))
        raw = []
        for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            points = np.asarray(
                [[point.x * image.width, point.y * image.height] for point in landmarks],
                dtype=np.float32,
            )
            category = handedness[0]
            raw.append({
                "points": points,
                "wrist_x": float(points[0, 0]),
                "mediapipe_label": category.category_name,
                "confidence": float(category.score),
            })

        # 本项目固定为俯拍双手：图像左侧对应被摄者右手，右侧对应左手。
        raw.sort(key=lambda item: item["wrist_x"])
        keypoints = None
        hand_meta = []
        if len(raw) == 2:
            keypoints = np.zeros((42, 2), dtype=np.float32)
            for rank, item in enumerate(raw):
                start = 0 if rank == 0 else 21
                keypoints[start:start + 21] = item["points"]
                hand_meta.append({
                    "slot": start,
                    "anatomical_side": "right" if start == 0 else "left",
                    "mediapipe_label": item["mediapipe_label"],
                    "confidence": item["confidence"],
                    "slot_assignment": "image_x_order_for_fixed_top_down_capture",
                    "bbox_xyxy": [
                        float(item["points"][:, 0].min()),
                        float(item["points"][:, 1].min()),
                        float(item["points"][:, 0].max()),
                        float(item["points"][:, 1].max()),
                    ],
                })
        return keypoints, hand_meta, len(raw), [
            {k: v for k, v in item.items() if k != "points"} for item in raw
        ]


def draw_preview(image, keypoints, mask, title, status):
    preview = image.copy()
    if mask is not None:
        # 只画轮廓，不遮挡皮肤和关键点。
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        array = cv2.cvtColor(np.asarray(preview), cv2.COLOR_RGB2BGR)
        cv2.drawContours(array, contours, -1, (0, 220, 0), 1)
        preview = Image.fromarray(cv2.cvtColor(array, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(preview)
    if keypoints is not None:
        for hand_index, start in enumerate((0, 21)):
            color = (230, 35, 35) if hand_index == 0 else (25, 90, 230)
            points = [(int(x), int(y)) for x, y in keypoints[start:start + 21]]
            for a, b in HAND_CONNECTIONS:
                draw.line((points[a], points[b]), fill=color, width=2)
            for x, y in points:
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill="white", outline=color)
    banner = Image.new("RGB", (preview.width, 28), "white")
    banner_draw = ImageDraw.Draw(banner)
    banner_draw.text((4, 3), f"{title} | {status}", fill="black")
    canvas = Image.new("RGB", (preview.width, preview.height + 28), "white")
    canvas.paste(banner, (0, 0))
    canvas.paste(preview, (0, 28))
    return canvas


def contact_sheet(items, output_path, columns=4):
    if not items:
        return
    cell_width, cell_height = items[0].size
    rows = (len(items) + columns - 1) // columns
    sheet = Image.new("RGB", (cell_width * columns, cell_height * rows), "white")
    for index, image in enumerate(items):
        sheet.paste(image, ((index % columns) * cell_width, (index // columns) * cell_height))
    sheet.save(output_path, quality=92)


def main():
    parser = argparse.ArgumentParser(description="V2-B 手部条件准备")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--hand-model", type=Path, required=True)
    parser.add_argument("--sam-weight", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--low-confidence-threshold", type=float, default=0.85)
    args = parser.parse_args()

    for path in (args.manifest, args.hand_model, args.sam_weight):
        if not path.exists():
            raise FileNotFoundError(path)
    source_manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not source_manifest.get("is_hand_only"):
        raise RuntimeError("拒绝处理非手部 manifest")

    output_dirs = {
        name: args.output_dir / name
        for name in ("images", "keypoints", "masks", "previews", "metadata", "review")
    }
    for path in output_dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    from segment_anything import SamPredictor, sam_model_registry

    detector = HandDetector(args.hand_model)
    sam = sam_model_registry["vit_h"](checkpoint=str(args.sam_weight)).to("cuda")
    sam_predictor = SamPredictor(sam)
    records = []
    preview_groups = {"pose01": [], "pose02": []}
    started = time.perf_counter()

    try:
        for index, source in enumerate(source_manifest["records"], start=1):
            condition_id = f"C{index:04d}"
            source_path = Path(source["source_path"])
            image, transform = square_pad_with_transform(Image.open(source_path), args.resolution)
            image_path = output_dirs["images"] / f"{condition_id}.png"
            image.save(image_path)

            keypoints, hands, detected_count, raw_detections = detector.detect(image)
            keypoint_path = None
            mask_path = None
            mask = None
            sam_score = None
            failure_reasons = []
            if detected_count != 2:
                failure_reasons.append(f"hand_count_not_two:{detected_count}")
            if keypoints is not None:
                keypoint_path = output_dirs["keypoints"] / f"{condition_id}.npy"
                np.save(keypoint_path, keypoints)
                sam_predictor.set_image(np.asarray(image))
                masks, scores, _ = sam_predictor.predict(
                    point_coords=np.asarray([keypoints[0], keypoints[21]], dtype=np.float32),
                    point_labels=np.ones(2, dtype=np.int32),
                    multimask_output=True,
                )
                best = int(np.argmax(scores))
                mask = masks[best].astype(np.uint8)
                sam_score = float(scores[best])
                mask_path = output_dirs["masks"] / f"{condition_id}.png"
                Image.fromarray(mask * 255).save(mask_path)
                area_ratio = float(mask.mean())
                if not 0.01 <= area_ratio <= 0.65:
                    failure_reasons.append(f"sam_mask_area_out_of_range:{area_ratio:.6f}")
            else:
                area_ratio = None

            confidence = min((hand["confidence"] for hand in hands), default=None)
            pose_name = f"pose{source['pose']}"
            if pose_name == "pose01":
                manual_requirement = "full_review_required"
            elif confidence is None or confidence < args.low_confidence_threshold:
                manual_requirement = "low_confidence_full_review_required"
            else:
                manual_requirement = "high_confidence_sample_review_required"

            automatic_status = "candidate" if not failure_reasons else "failed"
            preview = draw_preview(
                image, keypoints, mask, condition_id,
                f"{automatic_status};hands={detected_count};conf={confidence if confidence is not None else 'NA'}",
            )
            preview_path = output_dirs["previews"] / f"{condition_id}.jpg"
            preview.save(preview_path, quality=92)
            preview_groups[pose_name].append(preview)

            record = {
                "schema_version": 1,
                "condition_id": condition_id,
                "fold": 0,
                "is_hand_only": True,
                "uses_face_assets": False,
                "parent_subject_id": source["subject_id"],
                "nutrition_class": source["nutrition_class"],
                "pose": source["pose"],
                "compound_class": source["lora_class"],
                "source_split": "fold_0_train",
                "source_path": str(source_path.resolve()),
                "source_sha256": source["sha256"],
                "padded_image_path": str(image_path.resolve()),
                "padded_image_sha256": sha256(image_path),
                "coordinate_transform": transform,
                "detected_hand_count": detected_count,
                "raw_detections": raw_detections,
                "hands": hands,
                "keypoints_path": str(keypoint_path.resolve()) if keypoint_path else None,
                "keypoints_sha256": sha256(keypoint_path) if keypoint_path else None,
                "keypoints_shape": [42, 2] if keypoint_path else None,
                "mask_path": str(mask_path.resolve()) if mask_path else None,
                "mask_sha256": sha256(mask_path) if mask_path else None,
                "mask_shape": [args.resolution, args.resolution] if mask_path else None,
                "mask_area_ratio": area_ratio,
                "sam_score": sam_score,
                "min_handedness_confidence": confidence,
                "automatic_status": automatic_status,
                "failure_reasons": failure_reasons,
                "manual_requirement": manual_requirement,
                "manual_status": "pending" if automatic_status == "candidate" else "not_applicable_auto_failed",
                "manual_reviewer": None,
                "manual_notes": None,
                "preview_path": str(preview_path.resolve()),
                "models": {
                    "mediapipe_path": str(args.hand_model.resolve()),
                    "mediapipe_sha256": sha256(args.hand_model),
                    "sam_path": str(args.sam_weight.resolve()),
                    "sam_sha256": sha256(args.sam_weight),
                },
            }
            metadata_path = output_dirs["metadata"] / f"{condition_id}.json"
            write_json(metadata_path, record)
            record["metadata_path"] = str(metadata_path.resolve())
            records.append(record)
            print(f"[{index:02d}/{len(source_manifest['records'])}] {condition_id} {source['lora_class']} -> {automatic_status}")
    finally:
        detector.close()
        del sam_predictor, sam
        torch.cuda.empty_cache()

    manifest_path = args.output_dir / "condition_manifest_draft.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    contact_sheet(preview_groups["pose01"], output_dirs["review"] / "pose01_all_review.jpg")
    contact_sheet(preview_groups["pose02"], output_dirs["review"] / "pose02_all_review.jpg")
    review_template = {
        record["condition_id"]: {
            "decision": "reject" if record["automatic_status"] == "failed" else "pending",
            "reviewer": "auto_rule" if record["automatic_status"] == "failed" else None,
            "notes": ";".join(record["failure_reasons"]) if record["failure_reasons"] else None,
        }
        for record in records
    }
    write_json(output_dirs["review"] / "manual_review_template.json", review_template)

    counts = {}
    for record in records:
        key = record["compound_class"]
        counts.setdefault(key, {"total": 0, "candidate": 0, "failed": 0})
        counts[key]["total"] += 1
        counts[key][record["automatic_status"]] += 1
    summary = {
        "schema_version": 1,
        "task": "hand_v2b_condition_preparation_draft",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(args.manifest.resolve()),
        "source_manifest_sha256": sha256(args.manifest),
        "is_hand_only": True,
        "uses_face_assets": False,
        "uses_test_subjects": False,
        "records": len(records),
        "counts_by_class": counts,
        "elapsed_seconds": time.perf_counter() - started,
        "manual_review_complete": False,
        "manifest_path": str(manifest_path.resolve()),
    }
    write_json(args.output_dir / "condition_summary_draft.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
