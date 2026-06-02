import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.stats import ttest_ind
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "classify"))
sys.path.insert(0, str(PROJECT_ROOT / "passing"))

from models.clip import CLIP as NutriCLIP
from util_data import SUBSET_NAMES
from qc_filter import (
    ClinicalFeatureExtractor,
    LEFT_TEMPORAL,
    RIGHT_TEMPORAL,
    LEFT_ORBITAL,
    RIGHT_ORBITAL,
    LEFT_MALAR,
    RIGHT_MALAR,
    LEFT_JAW_LINE,
    RIGHT_JAW_LINE,
    MODEL_PATH,
    INSIGHTFACE_DET_SIZE,
)

DEFAULT_CHECKPOINT_PATH = (
    "/root/autodl-tmp/runs/ablation/classify_outputs/"
    "clip_real_plus_synth_qc_pool0.7_nipc330_lr1e-5_nomix/"
    "my_dataset/clipViT-B/16/n_img_per_cls_500/sd2.1/"
    "shot20_seed0_template1_ddlr0.0001_ddep240_lbd0.8/"
    "lr1e-05_wd0.0001_mixuag/best_checkpoint.pth"
)


DEFAULT_TEST_DIR = "/root/autodl-tmp/test_data"
DEFAULT_OUTPUT_DIR = "/root/autodl-tmp/runs/vis/roi_attention_analysis"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = SUBSET_NAMES["my_dataset"]
POSITIVE_CLASS = "malnourished_face"
NEGATIVE_CLASS = "normal_face"
POSITIVE_CLASS_IDX = CLASS_NAMES.index(POSITIVE_CLASS)
NEGATIVE_CLASS_IDX = CLASS_NAMES.index(NEGATIVE_CLASS)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ROI_NAMES = ["temporal", "orbital", "malar", "jawline"]
ATTR_KINDS = ["pos", "neg", "abs"]

ROI_COLORS = {
    "temporal": (255, 80, 80),
    "orbital": (80, 255, 255),
    "malar": (80, 180, 255),
    "jawline": (100, 255, 100),
}


def get_transform():
    return transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])


def get_tensor_transform_only():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])


def load_model(checkpoint_path, clip_download_dir=None):
    model = NutriCLIP(
        dataset="my_dataset",
        is_lora_image=True,
        is_lora_text=True,
        clip_download_dir=clip_download_dir,
        clip_version="ViT-B/16",
    )
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    incompatible = model.load_state_dict(ckpt["model"], strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print("[WARN] checkpoint与模型结构不完全一致。")
        if incompatible.missing_keys:
            print(f"[WARN] missing_keys: {incompatible.missing_keys[:10]}")
        if incompatible.unexpected_keys:
            print(f"[WARN] unexpected_keys: {incompatible.unexpected_keys[:10]}")
    return model.to(DEVICE).eval()


def preprocess_for_model_and_roi(img_pil, transform):
    geom = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
    ])
    img_for_roi = geom(img_pil)
    tensor = transform(img_pil).unsqueeze(0)
    return tensor, np.array(img_for_roi)


def parse_subject_and_view(img_path):
    stem = Path(img_path).stem
    match = re.match(r"^(.+?)_(\d+)$", stem)
    if not match:
        return "", "unknown"
    subject_id, view_code = match.groups()
    view_name = {"01": "front", "02": "left_45", "03": "right_45"}.get(view_code, view_code)
    return subject_id, view_name


def normalize_for_vis(attn_map):
    arr = np.asarray(attn_map, dtype=np.float32)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)


def safe_float(value):
    value = float(value)
    if np.isfinite(value):
        return value
    return None


class ClassSpecificGradientRollout:
    """Class-specific gradient-weighted rollout for CLIP visual ViT attention."""

    def __init__(self, model):
        self.model = model
        self.attn_modules = list(self.model.clip.visual.transformer.resblocks)
        missing = [i for i, block in enumerate(self.attn_modules) if not hasattr(block.attn, "capture_attention")]
        if missing:
            raise RuntimeError(
                "当前可视化需要 LoRA attention 的 capture_attention 支持；"
                f"以下block不支持: {missing[:5]}"
            )

    def _set_capture(self, enabled):
        for block in self.attn_modules:
            block.attn.capture_attention = enabled
            block.attn.captured_attn = None

    @staticmethod
    def _normalize_rows(mat):
        return mat / mat.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    def _rollout_nonnegative(self, cams):
        if not cams:
            raise RuntimeError("没有捕获到可用于解释的attention梯度。")
        n_tokens = cams[0].shape[-1]
        result = torch.eye(n_tokens)
        for cam in cams:
            cam = torch.clamp(cam, min=0)
            cam = cam + torch.eye(n_tokens)
            cam = self._normalize_rows(cam)
            result = torch.matmul(cam, result)
        cls_attention = result[0, 1:]
        grid_size = int(np.sqrt(cls_attention.numel()))
        if grid_size * grid_size != cls_attention.numel():
            raise RuntimeError(f"patch token数量不是平方数: {cls_attention.numel()}")
        return cls_attention.reshape(grid_size, grid_size).numpy()

    def __call__(self, image_tensor, target_class="malnourished_face"):
        self.model.zero_grad(set_to_none=True)
        image_tensor = image_tensor.detach().clone().to(DEVICE).requires_grad_(True)

        self._set_capture(True)
        try:
            with torch.enable_grad():
                logits = self.model(image_tensor)
                probs = torch.softmax(logits, dim=1)

                if target_class == "predicted":
                    target_idx = int(logits.argmax(dim=1).item())
                else:
                    target_idx = CLASS_NAMES.index(target_class)

                if logits.shape[1] == 2:
                    other_idx = 1 - target_idx
                    score = logits[:, target_idx] - logits[:, other_idx]
                else:
                    score = logits[:, target_idx]

                score.sum().backward()

            pos_cams = []
            neg_cams = []
            abs_cams = []
            for block in self.attn_modules:
                attn = block.attn.captured_attn
                if attn is None or attn.grad is None:
                    continue
                grad = attn.grad.detach()
                attn_value = attn.detach()
                pos_cams.append((attn_value * torch.relu(grad)).mean(dim=1)[0].cpu())
                neg_cams.append((attn_value * torch.relu(-grad)).mean(dim=1)[0].cpu())
                abs_cams.append((attn_value * torch.abs(grad)).mean(dim=1)[0].cpu())

            maps = {
                "pos": self._rollout_nonnegative(pos_cams),
                "neg": self._rollout_nonnegative(neg_cams),
                "abs": self._rollout_nonnegative(abs_cams),
            }
            maps["signed"] = maps["pos"] - maps["neg"]

            return {
                "maps": maps,
                "attn_map": maps["pos"],
                "logits": logits.detach().cpu().squeeze(0).numpy(),
                "probs": probs.detach().cpu().squeeze(0).numpy(),
                "target_idx": target_idx,
                "target_class": CLASS_NAMES[target_idx],
            }
        finally:
            self._set_capture(False)
            self.model.zero_grad(set_to_none=True)


class DynamicROIAttentionAnalyzer:
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
        elif view_name == "left_45":
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
        elif view_name == "right_45":
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
            kwargs = dict(use_insightface_fallback=self.use_insightface_fallback, view_hint="front", side_hint=None)

        if key not in self.extractors:
            self.extractors[key] = ClinicalFeatureExtractor(MODEL_PATH, **kwargs)
        return self.extractors[key]

    def generate_roi_masks(self, image_np, view_name="front"):
        extractor = self._extractor_for_view(view_name)
        landmarks = extractor.get_landmarks(image_np)
        if landmarks is None:
            return None

        l_temp = extractor.get_mask_from_points(image_np.shape, landmarks[LEFT_TEMPORAL], mode="hull")
        r_temp = extractor.get_mask_from_points(image_np.shape, landmarks[RIGHT_TEMPORAL], mode="hull")
        l_orb = extractor.get_mask_from_points(image_np.shape, landmarks[LEFT_ORBITAL], mode="hull")
        r_orb = extractor.get_mask_from_points(image_np.shape, landmarks[RIGHT_ORBITAL], mode="hull")
        l_malar = extractor.get_mask_from_points(image_np.shape, landmarks[LEFT_MALAR], mode="hull")
        r_malar = extractor.get_mask_from_points(image_np.shape, landmarks[RIGHT_MALAR], mode="hull")
        l_jaw = extractor.get_mask_from_points(image_np.shape, landmarks[LEFT_JAW_LINE], mode="line", thickness=15)
        r_jaw = extractor.get_mask_from_points(image_np.shape, landmarks[RIGHT_JAW_LINE], mode="line", thickness=15)

        return {
            "temporal": np.maximum(l_temp, r_temp),
            "orbital": np.maximum(l_orb, r_orb),
            "malar": np.maximum(l_malar, r_malar),
            "jawline": np.maximum(l_jaw, r_jaw),
        }


def compute_roi_map_scores(attn_map, roi_masks):
    scores = {}
    attn_map = np.asarray(attn_map, dtype=np.float32)
    for roi_name, mask in roi_masks.items():
        attn_resized = cv2.resize(attn_map, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
        mask_bool = mask > 0
        if np.sum(mask_bool) == 0:
            scores[f"roi_{roi_name}_mean"] = 0.0
            scores[f"roi_{roi_name}_enrichment"] = 0.0
            continue
        roi_attention = attn_resized[mask_bool]
        global_attention = float(np.mean(attn_resized))
        scores[f"roi_{roi_name}_mean"] = float(np.mean(roi_attention))
        scores[f"roi_{roi_name}_enrichment"] = float(np.mean(roi_attention) / (global_attention + 1e-8))
    return scores


def compute_all_attribution_scores(maps, roi_masks):
    scores = {}
    for kind in ("pos", "neg", "abs"):
        kind_scores = compute_roi_map_scores(maps[kind], roi_masks)
        for key, value in kind_scores.items():
            scores[f"attr_{kind}_{key}"] = value

    signed_scores = compute_roi_map_scores(maps["signed"], roi_masks)
    for roi_name in ROI_NAMES:
        scores[f"attr_signed_roi_{roi_name}_mean"] = signed_scores.get(f"roi_{roi_name}_mean", 0.0)
        pos = scores.get(f"attr_pos_roi_{roi_name}_enrichment", 0.0)
        neg = scores.get(f"attr_neg_roi_{roi_name}_enrichment", 0.0)
        scores[f"attr_signed_roi_{roi_name}_balance"] = float(pos - neg)
    return scores


@torch.no_grad()
def predict_tensor(model, tensor):
    tensor = tensor.to(DEVICE)
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)
    logits_np = logits.detach().cpu().squeeze(0).numpy()
    probs_np = probs.detach().cpu().squeeze(0).numpy()
    return logits_np, probs_np


def margin_values(logits, target_idx=None):
    mal_margin = float(logits[POSITIVE_CLASS_IDX] - logits[NEGATIVE_CLASS_IDX])
    if target_idx is None:
        return mal_margin, None
    other_idx = 1 - target_idx if len(logits) == 2 else None
    if other_idx is None:
        target_margin = float(logits[target_idx])
    else:
        target_margin = float(logits[target_idx] - logits[other_idx])
    return mal_margin, target_margin


def make_occluded_image(img_np, mask, fill="mean"):
    occluded = img_np.copy()
    mask_bool = mask > 0
    if not np.any(mask_bool):
        return occluded
    if fill == "gray":
        fill_value = np.array([127, 127, 127], dtype=np.float32)
    elif fill == "black":
        fill_value = np.array([0, 0, 0], dtype=np.float32)
    else:
        bg = img_np[~mask_bool]
        fill_value = bg.mean(axis=0) if bg.size else img_np.reshape(-1, 3).mean(axis=0)
    occluded[mask_bool] = fill_value.astype(np.uint8)
    return occluded


def run_roi_occlusion(model, img_np, roi_masks, target_idx, fill="mean"):
    tensor_only = get_tensor_transform_only()
    base_tensor = tensor_only(Image.fromarray(img_np)).unsqueeze(0)
    base_logits, base_probs = predict_tensor(model, base_tensor)
    base_mal_margin, base_target_margin = margin_values(base_logits, target_idx)

    result = {
        "base_mal_margin": base_mal_margin,
        "base_target_margin": base_target_margin,
    }
    for roi_name, mask in roi_masks.items():
        occluded_np = make_occluded_image(img_np, mask, fill=fill)
        occluded_tensor = tensor_only(Image.fromarray(occluded_np)).unsqueeze(0)
        occ_logits, occ_probs = predict_tensor(model, occluded_tensor)
        occ_mal_margin, occ_target_margin = margin_values(occ_logits, target_idx)
        result[f"occ_{roi_name}_mal_margin"] = occ_mal_margin
        result[f"occ_{roi_name}_target_margin"] = occ_target_margin
        result[f"occ_{roi_name}_delta_mal_margin"] = float(base_mal_margin - occ_mal_margin)
        result[f"occ_{roi_name}_delta_target_margin"] = float(base_target_margin - occ_target_margin)
        result[f"occ_{roi_name}_mal_prob"] = float(occ_probs[POSITIVE_CLASS_IDX])
    return result


def draw_visualization(img_np, roi_masks, maps, title, save_path=None, show=False):
    heat = normalize_for_vis(maps["pos"])
    heat_resized = cv2.resize(heat, (224, 224), interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap((heat_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.5 * img_np + 0.5 * heatmap).astype(np.uint8)

    neg_heat = normalize_for_vis(maps["neg"])
    neg_resized = cv2.resize(neg_heat, (224, 224), interpolation=cv2.INTER_LINEAR)
    neg_heatmap = cv2.applyColorMap((neg_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    neg_heatmap = cv2.cvtColor(neg_heatmap, cv2.COLOR_BGR2RGB)
    neg_overlay = (0.5 * img_np + 0.5 * neg_heatmap).astype(np.uint8)

    roi_vis = img_np.copy()
    for roi_name, mask in roi_masks.items():
        color = ROI_COLORS[roi_name]
        roi_vis[mask > 0] = (0.7 * roi_vis[mask > 0] + 0.3 * np.array(color)).astype(np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(title)
    axes[0].imshow(img_np)
    axes[0].set_title("Input")
    axes[1].imshow(overlay)
    axes[1].set_title("Positive attribution")
    axes[2].imshow(neg_overlay)
    axes[2].set_title("Negative attribution")
    axes[3].imshow(roi_vis)
    axes[3].set_title("Clinical ROI")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def analyze_image_target(
    img_path,
    true_class,
    model,
    explainer,
    transform,
    roi_analyzer,
    target_class,
    output_dir=None,
    run_occlusion=False,
    occlusion_fill="mean",
    save_image=True,
    show=False,
):
    img_path = Path(img_path)
    subject_id, view_name = parse_subject_and_view(img_path)
    img_pil = Image.open(img_path).convert("RGB")
    tensor, img_np = preprocess_for_model_and_roi(img_pil, transform)

    explanation = explainer(tensor, target_class=target_class)
    maps = explanation["maps"]
    probs = explanation["probs"]
    logits = explanation["logits"]
    target_idx = explanation["target_idx"]
    pred_idx = int(np.argmax(logits))
    mal_margin, target_margin = margin_values(logits, target_idx)

    roi_masks = roi_analyzer.generate_roi_masks(img_np, view_name=view_name)
    if roi_masks is None:
        print(f"[WARN] ROI生成失败: {img_path}")
        return None

    attr_scores = compute_all_attribution_scores(maps, roi_masks)
    occ_scores = {}
    if run_occlusion:
        occ_scores = run_roi_occlusion(model, img_np, roi_masks, target_idx, fill=occlusion_fill)

    if output_dir is not None and save_image:
        save_path = (
            Path(output_dir)
            / "images"
            / target_class
            / (true_class or "unknown")
            / f"{img_path.stem}_roi_validation.png"
        )
        title = (
            f"{img_path.name} | true={true_class} | view={view_name} | "
            f"target={CLASS_NAMES[target_idx]} | pred={CLASS_NAMES[pred_idx]} | "
            f"mal_prob={float(probs[POSITIVE_CLASS_IDX]) * 100:.1f}%"
        )
        draw_visualization(img_np, roi_masks, maps, title, save_path=save_path, show=show)

    record = {
        "true_class": true_class,
        "image_path": str(img_path),
        "image_name": img_path.name,
        "subject_id": subject_id,
        "view": view_name,
        "target_class": CLASS_NAMES[target_idx],
        "predicted_class": CLASS_NAMES[pred_idx],
        "malnourished_probability": float(probs[POSITIVE_CLASS_IDX]),
        "normal_probability": float(probs[NEGATIVE_CLASS_IDX]),
        "mal_margin": mal_margin,
        "target_margin": target_margin,
        "is_correct": bool(true_class == CLASS_NAMES[pred_idx]) if true_class else None,
    }
    record.update(attr_scores)
    record.update(occ_scores)
    return record


def iter_test_images(test_dir, max_per_class=None):
    test_dir = Path(test_dir)
    for cls_name in CLASS_NAMES:
        cls_dir = test_dir / cls_name
        if not cls_dir.is_dir():
            print(f"[WARN] 类别目录不存在，跳过: {cls_dir}")
            continue
        imgs = sorted([p for p in cls_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
        if max_per_class is not None:
            imgs = imgs[:max_per_class]
        for img_path in imgs:
            yield cls_name, img_path


def resolve_targets(args):
    targets = getattr(args, "targets", None)
    if targets is None:
        return [args.target_class]
    if targets == "both":
        return [POSITIVE_CLASS, NEGATIVE_CLASS]
    if targets == "predicted":
        return ["predicted"]
    return [targets]


def flatten_records_for_csv(records):
    keys = set()
    for row in records:
        keys.update(row.keys())
    preferred = [
        "true_class",
        "image_name",
        "subject_id",
        "view",
        "target_class",
        "predicted_class",
        "is_correct",
        "malnourished_probability",
        "normal_probability",
        "mal_margin",
        "target_margin",
    ]
    roi_attr = []
    for kind in ATTR_KINDS:
        for metric in ("mean", "enrichment"):
            for roi in ROI_NAMES:
                roi_attr.append(f"attr_{kind}_roi_{roi}_{metric}")
    roi_attr.extend([f"attr_signed_roi_{roi}_mean" for roi in ROI_NAMES])
    roi_attr.extend([f"attr_signed_roi_{roi}_balance" for roi in ROI_NAMES])
    occ = []
    for roi in ROI_NAMES:
        occ.extend([
            f"occ_{roi}_delta_mal_margin",
            f"occ_{roi}_delta_target_margin",
            f"occ_{roi}_mal_margin",
            f"occ_{roi}_target_margin",
            f"occ_{roi}_mal_prob",
        ])
    ordered = [k for k in preferred + roi_attr + occ if k in keys]
    extras = sorted(k for k in keys if k not in ordered and k != "image_path")
    return ordered + extras + (["image_path"] if "image_path" in keys else [])


def save_records(records, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "roi_attention_records.json"
    csv_path = output_dir / "roi_attention_records.csv"
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    fieldnames = flatten_records_for_csv(records)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow({k: row.get(k) for k in fieldnames})
    print(f"[INFO] saved: {json_path}")
    print(f"[INFO] saved: {csv_path}")


def aggregate_records(records, group_keys, value_keys):
    groups = defaultdict(list)
    for row in records:
        key = tuple(row.get(k, "") for k in group_keys)
        groups[key].append(row)

    out = []
    for key, rows in sorted(groups.items()):
        item = {group_keys[i]: key[i] for i in range(len(group_keys))}
        item["n"] = len(rows)
        for value_key in value_keys:
            vals = [r.get(value_key) for r in rows]
            vals = [float(v) for v in vals if v is not None and np.isfinite(float(v))]
            item[f"{value_key}_mean"] = float(np.mean(vals)) if vals else None
            item[f"{value_key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
        out.append(item)
    return out


def save_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] saved: {path}")


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[INFO] saved: {path}")


def add_subject_key(row):
    out = dict(row)
    out["subject_key"] = f"{row.get('true_class', '')}:{row.get('subject_id', '')}:{row.get('target_class', '')}"
    return out


def save_group_summaries(records, output_dir, run_occlusion=False):
    output_dir = Path(output_dir)
    attr_values = []
    for kind in ATTR_KINDS:
        for roi in ROI_NAMES:
            attr_values.append(f"attr_{kind}_roi_{roi}_enrichment")
    attr_values.extend([f"attr_signed_roi_{roi}_mean" for roi in ROI_NAMES])
    attr_values.extend([f"attr_signed_roi_{roi}_balance" for roi in ROI_NAMES])

    occ_values = []
    if run_occlusion:
        for roi in ROI_NAMES:
            occ_values.extend([f"occ_{roi}_delta_mal_margin", f"occ_{roi}_delta_target_margin"])

    attr_summary = aggregate_records(records, ["true_class", "target_class"], attr_values)
    view_summary = aggregate_records(records, ["true_class", "target_class", "view"], attr_values + occ_values)

    subject_input = [add_subject_key(row) for row in records if row.get("subject_id")]
    subject_level = aggregate_records(
        subject_input,
        ["true_class", "target_class", "subject_id"],
        attr_values + occ_values + ["malnourished_probability", "mal_margin"],
    )
    subject_summary = aggregate_records(subject_level, ["true_class", "target_class"], [k for k in subject_level[0] if k.endswith("_mean")] if subject_level else [])

    save_csv(attr_summary, output_dir / "roi_attribution_group_summary.csv")
    save_json(attr_summary, output_dir / "roi_attribution_group_summary.json")
    save_csv(view_summary, output_dir / "roi_viewwise_summary.csv")
    save_csv(subject_level, output_dir / "roi_subject_level_records.csv")
    save_csv(subject_summary, output_dir / "roi_subject_summary.csv")

    if run_occlusion:
        occ_summary = aggregate_records(records, ["true_class", "target_class"], occ_values)
        save_csv(occ_summary, output_dir / "roi_occlusion_group_summary.csv")
        save_json(occ_summary, output_dir / "roi_occlusion_group_summary.json")

    save_summary_plots(records, output_dir, run_occlusion=run_occlusion)


def mean_for(records, true_class, target_class, key):
    vals = [r.get(key) for r in records if r.get("true_class") == true_class and r.get("target_class") == target_class]
    vals = [float(v) for v in vals if v is not None and np.isfinite(float(v))]
    return float(np.mean(vals)) if vals else np.nan


def save_summary_plots(records, output_dir, run_occlusion=False):
    output_dir = Path(output_dir)
    pairs = [(t, target) for t in CLASS_NAMES for target in CLASS_NAMES]
    labels = [f"true={t}\ntarget={target}" for t, target in pairs]
    x = np.arange(len(ROI_NAMES))

    for kind in ("pos", "neg", "abs"):
        fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4), sharey=True)
        if len(pairs) == 1:
            axes = [axes]
        for ax, (true_class, target_class), label in zip(axes, pairs, labels):
            vals = [mean_for(records, true_class, target_class, f"attr_{kind}_roi_{roi}_enrichment") for roi in ROI_NAMES]
            ax.bar(x, vals, alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(ROI_NAMES, rotation=25, ha="right")
            ax.set_title(label)
            ax.set_ylabel(f"{kind} enrichment")
        fig.tight_layout()
        fig_path = output_dir / f"attribution_{kind}_by_roi.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] saved: {fig_path}")

    if run_occlusion:
        fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4), sharey=True)
        if len(pairs) == 1:
            axes = [axes]
        for ax, (true_class, target_class), label in zip(axes, pairs, labels):
            vals = [mean_for(records, true_class, target_class, f"occ_{roi}_delta_mal_margin") for roi in ROI_NAMES]
            ax.bar(x, vals, alpha=0.8)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(ROI_NAMES, rotation=25, ha="right")
            ax.set_title(label)
            ax.set_ylabel("delta mal margin")
        fig.tight_layout()
        fig_path = output_dir / "occlusion_delta_mal_margin_by_roi.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] saved: {fig_path}")


def run_single(args, model, explainer, transform, roi_analyzer):
    targets = resolve_targets(args)
    records = []
    true_class = Path(args.image).parent.name if Path(args.image).parent.name in CLASS_NAMES else ""
    for target_class in targets:
        record = analyze_image_target(
            img_path=args.image,
            true_class=true_class,
            model=model,
            explainer=explainer,
            transform=transform,
            roi_analyzer=roi_analyzer,
            target_class=target_class,
            output_dir=args.output_dir,
            run_occlusion=args.run_occlusion,
            occlusion_fill=args.occlusion_fill,
            save_image=True,
            show=args.show,
        )
        if record is not None:
            records.append(record)
            print(json.dumps(record, indent=2, ensure_ascii=False))
    if records:
        save_records(records, args.output_dir)


def run_batch(args, model, explainer, transform, roi_analyzer):
    records = []
    targets = resolve_targets(args)
    for true_class, img_path in iter_test_images(args.test_dir, max_per_class=args.max_per_class):
        for target_class in targets:
            record = analyze_image_target(
                img_path=img_path,
                true_class=true_class,
                model=model,
                explainer=explainer,
                transform=transform,
                roi_analyzer=roi_analyzer,
                target_class=target_class,
                output_dir=args.output_dir,
                run_occlusion=args.run_occlusion,
                occlusion_fill=args.occlusion_fill,
                save_image=not args.no_save_images,
                show=False,
            )
            if record is not None:
                records.append(record)
    save_records(records, args.output_dir)
    save_group_summaries(records, args.output_dir, run_occlusion=args.run_occlusion)
    print(f"[INFO] completed {len(records)} target-image records")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Class-specific CLIP ROI attribution and occlusion validation for NutriCode."
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--clip-download-dir", default=os.environ.get("CLIP_DOWNLOAD_DIR", None))
    parser.add_argument("--target-class", default=POSITIVE_CLASS, choices=[*CLASS_NAMES, "predicted"])
    parser.add_argument("--targets", default=None, choices=["both", *CLASS_NAMES, "predicted"])
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mode", default="single", choices=["single", "batch"])
    parser.add_argument("--image", default="/root/autodl-tmp/test_data/malnourished_face/35_01.png")
    parser.add_argument("--test-dir", default=DEFAULT_TEST_DIR)
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--run-occlusion", action="store_true")
    parser.add_argument("--occlusion-fill", default="mean", choices=["mean", "gray", "black"])
    parser.add_argument("--no-save-images", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--use-insightface-fallback",
        action="store_true",
        help="Enable InsightFace bbox fallback for hard side-view cases. May download InsightFace weights if missing.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    import matplotlib
    from matplotlib import font_manager

    font_names = {font.name for font in font_manager.fontManager.ttflist}
    if "WenQuanYi Micro Hei" in font_names:
        matplotlib.rcParams["font.family"] = "WenQuanYi Micro Hei"
    args = parse_args()
    print("[INFO] loading model...")
    model = load_model(args.checkpoint, clip_download_dir=args.clip_download_dir)
    explainer = ClassSpecificGradientRollout(model)
    transform = get_transform()
    roi_analyzer = DynamicROIAttentionAnalyzer(use_insightface_fallback=args.use_insightface_fallback)

    if args.mode == "single":
        run_single(args, model, explainer, transform, roi_analyzer)
    else:
        run_batch(args, model, explainer, transform, roi_analyzer)
