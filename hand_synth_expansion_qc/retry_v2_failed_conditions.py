#!/usr/bin/env python3
"""对 MediaPipe 初检失败的 pose02 训练图在原始尺度方图上重试。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hand_synthesis_v2"))
from prepare_conditions import HandDetector, sha256, square_pad_with_transform  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-manifest", type=Path, required=True)
    parser.add_argument("--hand-model", type=Path, required=True)
    parser.add_argument("--sam-weight", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.draft_manifest.read_text(encoding="utf-8").splitlines() if line]
    from segment_anything import SamPredictor, sam_model_registry
    detector = HandDetector(args.hand_model)
    sam = sam_model_registry["vit_h"](checkpoint=str(args.sam_weight)).to("cuda")
    predictor = SamPredictor(sam)
    repaired = []
    try:
        for row in rows:
            if row["automatic_status"] != "failed" or row["failure_reasons"] != ["hand_count_not_two:1"]:
                continue
            original = Image.open(row["source_path"]).convert("RGB")
            original_size = max(original.size)
            large, _ = square_pad_with_transform(original, original_size)
            points, hands, count, raw = detector.detect(large)
            if count != 2 or points is None:
                print(f"[UNREPAIRED] {row['condition_id']} detected={count}")
                continue
            points = (points * (512.0 / original_size)).astype(np.float32)
            image = Image.open(row["padded_image_path"]).convert("RGB")
            predictor.set_image(np.asarray(image))
            masks, scores, _ = predictor.predict(
                point_coords=np.asarray([points[0], points[21]], dtype=np.float32),
                point_labels=np.ones(2, dtype=np.int32), multimask_output=True,
            )
            best = int(np.argmax(scores))
            mask = masks[best].astype(np.uint8)
            area = float(mask.mean())
            if not 0.01 <= area <= 0.65:
                print(f"[UNREPAIRED] {row['condition_id']} mask_area={area:.5f}")
                continue
            keypoint_path = Path(row["keypoints_path"] or (Path(row["padded_image_path"]).parents[1] / "keypoints" / f"{row['condition_id']}.npy"))
            mask_path = Path(row["mask_path"] or (Path(row["padded_image_path"]).parents[1] / "masks" / f"{row['condition_id']}.png"))
            keypoint_path.parent.mkdir(parents=True, exist_ok=True)
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(keypoint_path, points)
            Image.fromarray(mask * 255).save(mask_path)
            row.update({"detected_hand_count": 2, "raw_detections": raw, "hands": hands,
                        "keypoints_path": str(keypoint_path.resolve()), "keypoints_sha256": sha256(keypoint_path),
                        "keypoints_shape": [42, 2], "mask_path": str(mask_path.resolve()), "mask_sha256": sha256(mask_path),
                        "mask_shape": [512, 512], "mask_area_ratio": area, "sam_score": float(scores[best]),
                        "automatic_status": "candidate", "failure_reasons": [],
                        "condition_detection_attempt": "padded_original_resolution_retry"})
            Path(row["metadata_path"]).write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            repaired.append(row["condition_id"])
            print(f"[REPAIRED] {row['condition_id']}")
    finally:
        detector.close()
        del predictor, sam
        torch.cuda.empty_cache()
    args.draft_manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"repaired": repaired}, ensure_ascii=False))


if __name__ == "__main__":
    main()
