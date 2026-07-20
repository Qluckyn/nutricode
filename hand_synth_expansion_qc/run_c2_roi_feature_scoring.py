#!/usr/bin/env python3
"""按第一背侧骨间肌投影区进行 C2-QC 特征评分（仅训练折）。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mediapipe as mp
import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, AutoModel

from c2_roi_geometry import roi_specs_for_two_hands


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def detector(model_path):
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE, num_hands=2,
        min_hand_detection_confidence=.20, min_hand_presence_confidence=.20,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def detect_points(image, landmarker):
    result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(image)))
    if len(result.hand_landmarks) != 2:
        return None
    hands = [np.asarray([[p.x, p.y] for p in hand], dtype=np.float32) for hand in result.hand_landmarks]
    hands.sort(key=lambda hand: float(hand[0, 0]))
    points = np.concatenate(hands)
    if np.any(points < .01) or np.any(points > .99):
        return None
    return points


def roi_image(image, points):
    """提取左右手的严格椭圆 ROI，并以白底屏蔽椭圆外像素。"""
    specs = roi_specs_for_two_hands(points)
    if specs is None:
        return None
    crops = []
    for spec in specs:
        side = max(24, int(2.4 * spec.long_radius * max(image.size)))
        cx, cy = spec.center[0] * image.width, spec.center[1] * image.height
        crop = image.crop((int(cx-side/2), int(cy-side/2), int(cx+side/2), int(cy+side/2)))
        # 在局部坐标绘制未旋转椭圆，再按核验图相同的符号旋转为最终 ROI。
        mask = Image.new("L", crop.size, 0)
        draw = ImageDraw.Draw(mask)
        crop_width, crop_height = crop.size
        radius_x = spec.long_radius * image.width
        radius_y = spec.short_radius * image.height
        draw.ellipse((crop_width/2-radius_x, crop_height/2-radius_y, crop_width/2+radius_x, crop_height/2+radius_y), fill=255)
        mask = mask.rotate(spec.angle_deg, resample=Image.Resampling.BICUBIC)
        crop = Image.composite(crop, Image.new("RGB", crop.size, "white"), mask)
        crops.append(crop.resize((224, 224), Image.Resampling.LANCZOS))
    output = Image.new("RGB", (448, 224), "white")
    output.paste(crops[0], (0, 0)); output.paste(crops[1], (224, 0))
    return output


def procrustes(predicted, target):
    predicted = predicted - predicted.mean(0); target = target - target.mean(0)
    predicted /= max(float(np.linalg.norm(predicted)), 1e-8); target /= max(float(np.linalg.norm(target)), 1e-8)
    u, _, vt = np.linalg.svd(predicted.T @ target)
    return float(np.mean(np.linalg.norm(predicted @ (u @ vt) - target, axis=1)))


def embed(images, model, processor, device, batch=16):
    values = []
    for start in range(0, len(images), batch):
        inputs = processor(images=images[start:start+batch], return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = model(**inputs)
            value = output.pooler_output if output.pooler_output is not None else output.last_hidden_state[:, 0]
            values.append(torch.nn.functional.normalize(value.float(), dim=1).cpu())
    return torch.cat(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--landmarker", type=Path, required=True)
    parser.add_argument("--dinov2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    entries = rows(args.manifest)
    if not entries:
        raise RuntimeError("QC 输入清单为空")
    landmarker = detector(args.landmarker)
    try:
        # 所有真实原型来自当前 fold 的 train 文件清单，明确隔离测试集。
        refs, ref_labels = [], []
        for label in ("malnourished_hand", "normal_hand"):
            for path in entries[0]["same_class_references"] if label == entries[0]["nutrition_class"] else entries[0]["other_class_references"]:
                image = Image.open(path).convert("RGB"); points = detect_points(image, landmarker); crop = roi_image(image, points) if points is not None else None
                if crop is not None: refs.append(crop); ref_labels.append(label)
        candidate_crops, geometries = [], []
        for entry in entries:
            image = Image.open(entry["image_path"]).convert("RGB"); points = detect_points(image, landmarker)
            if points is None:
                candidate_crops.append(None); geometries.append((False, 0., None)); continue
            target = np.load(entry["target_keypoints_path"]).astype(np.float32) / image.width
            error = procrustes(points, target); score = float(np.exp(-error/.08))
            candidate_crops.append(roi_image(image, points)); geometries.append((score >= .78, score, error))
        valid = [i for i, crop in enumerate(candidate_crops) if crop is not None]
        model = AutoModel.from_pretrained(args.dinov2, local_files_only=True).to(args.device).eval()
        processor = AutoImageProcessor.from_pretrained(args.dinov2, local_files_only=True)
        ref_features = embed(refs, model, processor, args.device)
        prototypes = {label: ref_features[[i for i,x in enumerate(ref_labels) if x == label]].mean(0) for label in set(ref_labels)}
        features = embed([candidate_crops[i] for i in valid], model, processor, args.device)
        feature_map = dict(zip(valid, features))
        output=[]
        for index, entry in enumerate(entries):
            passed, structure, error = geometries[index]
            if index not in feature_map:
                output.append({"candidate_id":entry["candidate_id"],"structure_pass":False,"structure_score":0.,"structure_error":None,"semantic_margin":0.,"diversity_score":0.,"near_duplicate":False,"uses_test_data":False}); continue
            feature=feature_map[index]; own=entry["nutrition_class"]; other="normal_hand" if own=="malnourished_hand" else "malnourished_hand"
            margin=float(torch.clamp((torch.dot(feature, prototypes[own])-torch.dot(feature, prototypes[other])+1)/2,0,1))
            same=[feature_map[i] for i in valid if i != index and entries[i]["nutrition_class"] == own]
            nearest=max((float(torch.dot(feature, value)) for value in same), default=0.)
            diversity=float(np.clip((1-nearest)/.15,0,1))
            output.append({"candidate_id":entry["candidate_id"],"structure_pass":passed,"structure_score":structure,"structure_error":error,"semantic_margin":margin,"diversity_score":diversity,"nearest_same_class_similarity":nearest,"near_duplicate":nearest>=.995,"uses_test_data":False,"feature_scope":"first_dorsal_interosseous_roi_train_only"})
    finally:
        landmarker.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row,ensure_ascii=False)+"\n" for row in output),encoding="utf-8")
    print({"output":str(args.output),"candidates":len(output),"uses_test_data":False})


if __name__ == "__main__":
    main()
