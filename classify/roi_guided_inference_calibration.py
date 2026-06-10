# ROI-guided image 推理
import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets.folder import default_loader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "classify"))
sys.path.insert(0, str(PROJECT_ROOT / "passing"))

from models.clip import CLIP as NutriCLIP
from data import get_transforms
from qc_filter import (
    ClinicalFeatureExtractor,
    INSIGHTFACE_DET_SIZE,
    LEFT_JAW_LINE,
    LEFT_MALAR,
    LEFT_ORBITAL,
    LEFT_TEMPORAL,
    MODEL_PATH,
    RIGHT_JAW_LINE,
    RIGHT_MALAR,
    RIGHT_ORBITAL,
    RIGHT_TEMPORAL,
)
from util_data import SUBSET_NAMES


DEFAULT_CHECKPOINT = (
    "/root/autodl-tmp/runs/ablation/classify_outputs/"
    "clip_real_plus_synth_qc_pool0.7_nipc330_lr1e-5_nomix/"
    "my_dataset/clipViT-B/16/n_img_per_cls_500/sd2.1/"
    "shot20_seed0_template1_ddlr0.0001_ddep240_lbd0.8/"
    "lr1e-05_wd0.0001_mixuag/best_checkpoint.pth"
)
DEFAULT_TEST_DIR = "/root/autodl-tmp/test_data"
DEFAULT_OUTPUT_DIR = (
    "/root/autodl-tmp/runs/ablation/classify_outputs/"
    "clip_real_plus_synth_qc_pool0.7_nipc330_lr1e-5_nomix/"
    "my_dataset/clipViT-B/16/n_img_per_cls_500/sd2.1/"
    "shot20_seed0_template1_ddlr0.0001_ddep240_lbd0.8/"
    "lr1e-05_wd0.0001_mixuag"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = SUBSET_NAMES["my_dataset"]
POSITIVE_CLASS = "malnourished_face"
NEGATIVE_CLASS = "normal_face"
POSITIVE_IDX = CLASS_NAMES.index(POSITIVE_CLASS)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIEW_ORDER = ("front", "left", "right")
METHODS = ("global_only", "roi_only", "global_roi_fusion")


def parse_subject_id(path):
    name = os.path.basename(path)
    match = re.match(r"^(\d+)_", name)
    return match.group(1) if match else ""


def parse_view(path):
    name = os.path.basename(path)
    match = re.match(r"^\d+_(01|02|03)(?:$|[_\-.])", name)
    if match:
        return {"01": "front", "02": "left", "03": "right"}[match.group(1)]
    path_lower = path.lower()
    if "front" in path_lower:
        return "front"
    if "left" in path_lower:
        return "left"
    if "right" in path_lower:
        return "right"
    return ""


def get_transform():
    _, test_transform = get_transforms("clip")
    return test_transform


def load_model(checkpoint_path, clip_download_dir=None):
    model = NutriCLIP(
        dataset="my_dataset",
        is_lora_image=True,
        is_lora_text=True,
        clip_download_dir=clip_download_dir,
        clip_version="ViT-B/16",
    )
    # loralib.MergedLinear merges LoRA weights when switching to eval().
    # Match the already-eval model state used by main.py detailed prediction export:
    # move/eval first, then load the checkpoint weights. Loading first then eval()
    # would merge LoRA into qkv weights and changes global probabilities.
    model = model.to(DEVICE).eval()
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    incompatible = model.load_state_dict(ckpt["model"], strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print("[WARN] checkpoint and model structure are not an exact match.")
        if incompatible.missing_keys:
            print(f"[WARN] missing_keys: {incompatible.missing_keys[:10]}")
        if incompatible.unexpected_keys:
            print(f"[WARN] unexpected_keys: {incompatible.unexpected_keys[:10]}")
    return model


class DynamicROIBuilder:
    def __init__(self, use_insightface_fallback=False):
        self.extractors = {}
        self.use_insightface_fallback = bool(use_insightface_fallback)

    def _extractor_for_view(self, view_name):
        if view_name == "front":
            key = ("front", None)
            kwargs = dict(
                use_insightface_fallback=self.use_insightface_fallback,
                insightface_det_sizes=(INSIGHTFACE_DET_SIZE,),
                insightface_crop_expands=(1.30, 1.60, 1.90),
                insightface_det_thresh=0.35,
                insightface_primary=False,
                view_hint="front",
                side_hint=None,
            )
        elif view_name == "left":
            key = ("three_quarter", "left")
            kwargs = dict(
                use_insightface_fallback=self.use_insightface_fallback,
                insightface_det_sizes=(INSIGHTFACE_DET_SIZE,),
                insightface_crop_expands=(1.45, 1.70, 2.00),
                insightface_det_thresh=0.30,
                insightface_primary=True,
                mp_min_face_detection_confidence=0.25,
                mp_min_face_presence_confidence=0.25,
                mp_min_tracking_confidence=0.25,
                view_hint="three_quarter",
                side_hint="left",
            )
        elif view_name == "right":
            key = ("three_quarter", "right")
            kwargs = dict(
                use_insightface_fallback=self.use_insightface_fallback,
                insightface_det_sizes=(INSIGHTFACE_DET_SIZE,),
                insightface_crop_expands=(1.45, 1.70, 2.00),
                insightface_det_thresh=0.30,
                insightface_primary=True,
                mp_min_face_detection_confidence=0.25,
                mp_min_face_presence_confidence=0.25,
                mp_min_tracking_confidence=0.25,
                view_hint="three_quarter",
                side_hint="right",
            )
        else:
            key = ("front", "unknown")
            kwargs = dict(
                use_insightface_fallback=self.use_insightface_fallback,
                view_hint="front",
                side_hint=None,
            )

        if key not in self.extractors:
            self.extractors[key] = ClinicalFeatureExtractor(MODEL_PATH, **kwargs)
        return self.extractors[key]

    def build_mask(self, image_rgb, view_name):
        extractor = self._extractor_for_view(view_name)
        landmarks = extractor.get_landmarks(image_rgb)
        if landmarks is None:
            return None

        masks = [
            extractor.get_mask_from_points(image_rgb.shape, landmarks[LEFT_TEMPORAL], mode="hull"),
            extractor.get_mask_from_points(image_rgb.shape, landmarks[RIGHT_TEMPORAL], mode="hull"),
            extractor.get_mask_from_points(image_rgb.shape, landmarks[LEFT_ORBITAL], mode="hull"),
            extractor.get_mask_from_points(image_rgb.shape, landmarks[RIGHT_ORBITAL], mode="hull"),
            extractor.get_mask_from_points(image_rgb.shape, landmarks[LEFT_MALAR], mode="hull"),
            extractor.get_mask_from_points(image_rgb.shape, landmarks[RIGHT_MALAR], mode="hull"),
            extractor.get_mask_from_points(image_rgb.shape, landmarks[LEFT_JAW_LINE], mode="line", thickness=15),
            extractor.get_mask_from_points(image_rgb.shape, landmarks[RIGHT_JAW_LINE], mode="line", thickness=15),
        ]
        mask = np.maximum.reduce(masks).astype(np.uint8)
        return mask


def make_roi_guided_image(image_rgb, roi_mask, mode="dim", dim_factor=0.5, blur_kernel=31):
    if roi_mask is None or int(np.sum(roi_mask)) == 0:
        return image_rgb.copy(), "landmark_failed"

    mask = roi_mask > 0
    guided = image_rgb.copy()
    if mode == "blur":
        kernel = int(blur_kernel)
        if kernel % 2 == 0:
            kernel += 1
        blurred = cv2.GaussianBlur(image_rgb, (kernel, kernel), 0)
        guided[~mask] = blurred[~mask]
        roi_quality = "ok_blur"
    else:
        guided[~mask] = np.clip(guided[~mask].astype(np.float32) * float(dim_factor), 0, 255).astype(np.uint8)
        roi_quality = "ok_dim"
    return guided, roi_quality


@torch.no_grad()
def predict_positive_prob_from_pil(model, transform, image_pil):
    tensor = transform(image_pil).unsqueeze(0).to(DEVICE)
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)
    return float(probs[0, POSITIVE_IDX].detach().cpu().item())


@torch.no_grad()
def predict_positive_prob_from_rgb(model, transform, image_rgb):
    return predict_positive_prob_from_pil(model, transform, Image.fromarray(image_rgb))


def iter_test_images(test_dir):
    test_dir = Path(test_dir)
    for class_name in CLASS_NAMES:
        class_dir = test_dir / class_name
        if not class_dir.is_dir():
            print(f"[WARN] missing class dir: {class_dir}")
            continue
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() in IMG_EXTS:
                yield class_name, path


def binary_auc_score(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(len(y_score), dtype=float)
    sorted_scores = y_score[order]
    i = 0
    while i < len(y_score):
        j = i + 1
        while j < len(y_score) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    sum_ranks_pos = float(ranks[pos].sum())
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def binary_metrics(y_true, y_score, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    precision = tp / max(1, tp + fp)
    sensitivity = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    f1 = (2 * precision * sensitivity) / max(1e-12, precision + sensitivity)
    mcc_num = tp * tn - fp * fn
    mcc_den = math.sqrt(max(1e-12, (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))

    return {
        "n": int(len(y_true)),
        "acc": float(acc),
        "auc": binary_auc_score(y_true, y_score),
        "f1": float(f1),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "mcc": float(mcc_num / mcc_den),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def compute_image_metrics(records, threshold=0.5):
    rows = []
    for method, key in (
        ("global_only", "p_global"),
        ("roi_only", "p_roi"),
        ("global_roi_fusion", "p_fusion"),
    ):
        y_true = [r["positive_true_label"] for r in records]
        y_score = [r[key] for r in records]
        rows.append({"level": "image", "method": method, **binary_metrics(y_true, y_score, threshold)})
    return rows


def build_subject_results(records, threshold=0.5):
    grouped = defaultdict(list)
    for record in records:
        grouped[record["subject_id"]].append(record)

    subject_results = []
    for subject_id in sorted(grouped):
        items = grouped[subject_id]
        labels = {int(item["positive_true_label"]) for item in items}
        if len(labels) != 1:
            raise ValueError(f"Inconsistent labels for subject {subject_id}")

        row = {
            "subject_id": subject_id,
            "positive_true_label": labels.pop(),
            "views_present": sorted(item["view"] for item in items),
        }
        for key in ("p_global", "p_roi", "p_fusion"):
            row[key] = float(np.mean([item[key] for item in items]))

        for method, key in (
            ("global_only", "p_global"),
            ("roi_only", "p_roi"),
            ("global_roi_fusion", "p_fusion"),
        ):
            row[f"{method}_pred_label"] = int(row[key] >= threshold)
        subject_results.append(row)
    return subject_results


def compute_subject_metrics(subject_results, threshold=0.5):
    rows = []
    for method, key in (
        ("global_only", "p_global"),
        ("roi_only", "p_roi"),
        ("global_roi_fusion", "p_fusion"),
    ):
        y_true = [r["positive_true_label"] for r in subject_results]
        y_score = [r[key] for r in subject_results]
        rows.append({"level": "subject", "method": method, **binary_metrics(y_true, y_score, threshold)})
    return rows


def collect_error_cases(image_records, subject_results, threshold=0.5):
    rows = []
    for record in image_records:
        label = int(record["positive_true_label"])
        method_scores = {
            "global_only": record["p_global"],
            "roi_only": record["p_roi"],
            "global_roi_fusion": record["p_fusion"],
        }
        wrong_methods = [
            method for method, score in method_scores.items()
            if int(score >= threshold) != label
        ]
        if wrong_methods or not str(record["roi_quality"]).startswith("ok"):
            rows.append({
                "level": "image",
                "image_path": record["image_path"],
                "subject_id": record["subject_id"],
                "view": record["view"],
                "true_class_name": record["true_class_name"],
                "positive_true_label": label,
                "wrong_methods": ";".join(wrong_methods),
                "roi_quality": record["roi_quality"],
                "p_global": record["p_global"],
                "p_roi": record["p_roi"],
                "p_fusion": record["p_fusion"],
            })

    for record in subject_results:
        label = int(record["positive_true_label"])
        method_scores = {
            "global_only": record["p_global"],
            "roi_only": record["p_roi"],
            "global_roi_fusion": record["p_fusion"],
        }
        wrong_methods = [
            method for method, score in method_scores.items()
            if int(score >= threshold) != label
        ]
        if wrong_methods:
            rows.append({
                "level": "subject",
                "image_path": "",
                "subject_id": record["subject_id"],
                "view": ",".join(record["views_present"]),
                "true_class_name": POSITIVE_CLASS if label == 1 else NEGATIVE_CLASS,
                "positive_true_label": label,
                "wrong_methods": ";".join(wrong_methods),
                "roi_quality": "",
                "p_global": record["p_global"],
                "p_roi": record["p_roi"],
                "p_fusion": record["p_fusion"],
            })
    return rows


def compare_global_with_main_json(image_records, main_json_path):
    if not main_json_path or not os.path.exists(main_json_path):
        return {
            "available": False,
            "path": main_json_path,
            "n_compared": 0,
            "max_abs_diff": None,
        }

    with open(main_json_path, "r") as f:
        main_json = json.load(f)
    main_by_path = {
        row.get("image_path"): row
        for row in main_json.get("all_samples", [])
        if row.get("image_path")
    }

    diffs = []
    missing = []
    for row in image_records:
        image_path = row["image_path"]
        ref = main_by_path.get(image_path)
        if ref is None:
            missing.append(image_path)
            continue
        diffs.append(abs(float(row["p_global"]) - float(ref["malnourished_prob"])))

    return {
        "available": True,
        "path": main_json_path,
        "n_compared": len(diffs),
        "max_abs_diff": max(diffs) if diffs else None,
        "mean_abs_diff": float(np.mean(diffs)) if diffs else None,
        "missing_in_main_json": missing,
    }


def write_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_csv(rows, path, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def main():
    parser = argparse.ArgumentParser(description="ROI-guided inference calibration for a trained CLIP-LoRA classifier.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--test-dir", default=DEFAULT_TEST_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--clip-download-dir", default=os.environ.get("CLIP_DOWNLOAD_DIR", None))
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--main-json", default=None)
    parser.add_argument("--background-mode", default="dim", choices=["dim", "blur"])
    parser.add_argument("--dim-factor", type=float, default=0.5)
    parser.add_argument("--blur-kernel", type=int, default=31)
    parser.add_argument("--use-insightface-fallback", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    main_json_path = args.main_json or str(output_dir / "detailed_prediction_results.json")

    print(f"[INFO] loading model: {args.checkpoint}")
    model = load_model(args.checkpoint, clip_download_dir=args.clip_download_dir)
    transform = get_transform()
    roi_builder = DynamicROIBuilder(use_insightface_fallback=args.use_insightface_fallback)

    image_records = []
    warnings = []
    for true_class_name, image_path in iter_test_images(args.test_dir):
        image_pil = default_loader(str(image_path)).convert("RGB")
        image_rgb = np.asarray(image_pil)
        subject_id = parse_subject_id(str(image_path))
        view = parse_view(str(image_path))
        positive_true_label = 1 if true_class_name == POSITIVE_CLASS else 0

        p_global = predict_positive_prob_from_pil(model, transform, image_pil)
        roi_mask = roi_builder.build_mask(image_rgb, view)
        roi_image_rgb, roi_quality = make_roi_guided_image(
            image_rgb,
            roi_mask,
            mode=args.background_mode,
            dim_factor=args.dim_factor,
            blur_kernel=args.blur_kernel,
        )
        if not str(roi_quality).startswith("ok"):
            roi_image_rgb = image_rgb
            warning = f"landmark failed, fallback to original image: {image_path}"
            warnings.append(warning)
            print(f"[WARN] {warning}")

        p_roi = predict_positive_prob_from_rgb(model, transform, roi_image_rgb)
        p_fusion = float(args.alpha * p_global + (1.0 - args.alpha) * p_roi)

        image_records.append(
            {
                "image_path": str(image_path),
                "subject_id": subject_id,
                "view": view,
                "true_class_name": true_class_name,
                "positive_true_label": positive_true_label,
                "p_global": float(p_global),
                "p_roi": float(p_roi),
                "p_fusion": p_fusion,
                "roi_quality": roi_quality,
            }
        )

    subject_results = build_subject_results(image_records, threshold=args.threshold)
    metric_rows = compute_image_metrics(image_records, threshold=args.threshold)
    metric_rows.extend(compute_subject_metrics(subject_results, threshold=args.threshold))
    error_cases = collect_error_cases(image_records, subject_results, threshold=args.threshold)
    global_reference_check = compare_global_with_main_json(image_records, main_json_path)

    prediction_results = {
        "checkpoint": args.checkpoint,
        "test_dir": args.test_dir,
        "alpha": args.alpha,
        "threshold": args.threshold,
        "positive_class_name": POSITIVE_CLASS,
        "negative_class_name": NEGATIVE_CLASS,
        "warnings": warnings,
        "global_reference_check": global_reference_check,
        "image_results": image_records,
        "metrics": metric_rows,
    }

    write_json(prediction_results, output_dir / "roi_guided_prediction_results.json")
    write_json(
        {
            "checkpoint": args.checkpoint,
            "test_dir": args.test_dir,
            "alpha": args.alpha,
            "threshold": args.threshold,
            "subject_results": subject_results,
        },
        output_dir / "roi_guided_subject_results.json",
    )
    write_csv(
        metric_rows,
        output_dir / "roi_guided_metrics.csv",
        ["level", "method", "n", "acc", "auc", "f1", "sensitivity", "specificity", "mcc", "tp", "tn", "fp", "fn"],
    )
    write_csv(
        error_cases,
        output_dir / "roi_guided_error_cases.csv",
        [
            "level",
            "image_path",
            "subject_id",
            "view",
            "true_class_name",
            "positive_true_label",
            "wrong_methods",
            "roi_quality",
            "p_global",
            "p_roi",
            "p_fusion",
        ],
    )

    n_landmark_failed = sum(1 for r in image_records if not str(r["roi_quality"]).startswith("ok"))
    print(f"[INFO] images: {len(image_records)}")
    print(f"[INFO] subjects: {len(subject_results)}")
    print(f"[INFO] landmark fallback images: {n_landmark_failed}")
    print(
        "[INFO] global vs main.py detailed JSON: "
        f"available={global_reference_check['available']}, "
        f"n_compared={global_reference_check['n_compared']}, "
        f"max_abs_diff={global_reference_check['max_abs_diff']}"
    )
    print(f"[INFO] saved: {output_dir / 'roi_guided_prediction_results.json'}")
    print(f"[INFO] saved: {output_dir / 'roi_guided_metrics.csv'}")
    print(f"[INFO] saved: {output_dir / 'roi_guided_subject_results.json'}")
    print(f"[INFO] saved: {output_dir / 'roi_guided_error_cases.csv'}")


if __name__ == "__main__":
    main()
