# 可解释性分析的三个验证
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

from data import get_transforms
from models.clip import CLIP as NutriCLIP
from util_data import SUBSET_NAMES
from qc_filter import (
    ClinicalFeatureExtractor,
    FACE_OVAL,
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

# 真实+合成过滤（NutriDiff）
DEFAULT_CHECKPOINT_PATH = (
    "/root/autodl-tmp/runs/ablation/classify_outputs/"
    "clip_real_plus_synth_qc_pool0.7_nipc330_lr1e-5_nomix/"
    "my_dataset/clipViT-B/16/n_img_per_cls_500/sd2.1/"
    "shot20_seed0_template1_ddlr0.0001_ddep240_lbd0.8/"
    "lr1e-05_wd0.0001_mixuag/best_checkpoint.pth"
)
# 只用真实数据（Baseline）
# DEFAULT_CHECKPOINT_PATH = (
#     "/root/autodl-tmp/runs/ablation/classify_outputs/clip_real_plus_synth_raw_pool0.7_nipc330_lr1e-5_nomix/my_dataset/clipViT-B/16/n_img_per_cls_500/baseline_shot20_seed0/lr1e-05_wd0.0001_mixuag/best_checkpoint.pth"
# )
# 真实+合成不过滤（DataDream）
# DEFAULT_CHECKPOINT_PATH = (
#     "/root/autodl-tmp/runs/ablation/classify_outputs/clip_real_plus_synth_raw_pool0.7_nipc330_lr1e-5_nomix/my_dataset/clipViT-B/16/n_img_per_cls_500/sd2.1/shot20_seed0_template1_ddlr0.0001_ddep240_lbd0.8/lr1e-05_wd0.0001_mixuag/best_checkpoint.pth"
# )

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
    _, test_transform = get_transforms("clip")
    return test_transform


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
    # loralib.MergedLinear merges LoRA weights when switching to eval().
    # Match the checkpoint state used by main.py detailed prediction export:
    # eval first, then load weights. Loading first and then eval() would merge
    # LoRA again and change the global probabilities.
    model = model.to(DEVICE).eval()
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    incompatible = model.load_state_dict(ckpt["model"], strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print("[WARN] checkpoint与模型结构不完全一致。")
        if incompatible.missing_keys:
            print(f"[WARN] missing_keys: {incompatible.missing_keys[:10]}")
        if incompatible.unexpected_keys:
            print(f"[WARN] unexpected_keys: {incompatible.unexpected_keys[:10]}")
    return model


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

    @staticmethod
    def _bbox_to_expanded_mask(shape, bbox_xyxy, expand=1.10):
        h, w = shape[:2]
        x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        bw = max(0.0, x2 - x1) * float(expand)
        bh = max(0.0, y2 - y1) * float(expand)

        x0 = max(0, int(np.floor(cx - bw / 2.0)))
        y0 = max(0, int(np.floor(cy - bh / 2.0)))
        x3 = min(w, int(np.ceil(cx + bw / 2.0)))
        y3 = min(h, int(np.ceil(cy + bh / 2.0)))

        mask = np.zeros(shape[:2], dtype=np.uint8)
        if x3 > x0 and y3 > y0:
            mask[y0:y3, x0:x3] = 1
        return mask

    def generate_face_mask(self, image_np, view_name="front"):
        extractor = self._extractor_for_view(view_name)
        landmarks = extractor.get_landmarks(image_np)
        if landmarks is not None:
            face_mask = extractor.get_mask_from_points(
                image_np.shape,
                landmarks[FACE_OVAL],
                mode="hull",
            )
            if np.any(face_mask):
                return face_mask, "facemesh"

        if getattr(extractor, "_face_analyzer", None) is None:
            extractor._init_insightface()
        if getattr(extractor, "_face_analyzer", None) is None:
            return None, "failed"

        try:
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            faces = extractor._insightface_get_faces(image_bgr)
            bbox = extractor._select_largest_bbox(faces) if faces else None
        except Exception:
            bbox = None

        if bbox is None:
            return None, "failed"
        face_mask = self._bbox_to_expanded_mask(image_np.shape, bbox, expand=1.10)
        if not np.any(face_mask):
            return None, "failed"
        return face_mask, "insightface_bbox"


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


def compute_face_normalized_roi_map_scores(attn_map, roi_masks, face_mask, eps=1e-8):
    scores = {}
    if face_mask is None or not np.any(face_mask):
        for roi_name in roi_masks:
            scores[f"roi_{roi_name}_face_mean"] = None
            scores[f"roi_{roi_name}_face_enrichment"] = None
        return scores

    attn_map = np.asarray(attn_map, dtype=np.float32)
    # Use the detected face as the baseline, so background attribution cannot
    # dilute or inflate ROI enrichment. Keep the image-space masks to preserve
    # thin clinical ROIs such as the jawline.
    attn_resized = cv2.resize(
        attn_map,
        (face_mask.shape[1], face_mask.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    face_bool = face_mask > 0
    if not np.any(face_bool):
        for roi_name in roi_masks:
            scores[f"roi_{roi_name}_face_mean"] = None
            scores[f"roi_{roi_name}_face_enrichment"] = None
        return scores

    face_attention = float(np.mean(attn_resized[face_bool]))
    for roi_name, mask in roi_masks.items():
        roi_face_bool = (mask > 0) & face_bool
        if not np.any(roi_face_bool):
            scores[f"roi_{roi_name}_face_mean"] = None
            scores[f"roi_{roi_name}_face_enrichment"] = None
            continue
        roi_face_attention = float(np.mean(attn_resized[roi_face_bool]))
        scores[f"roi_{roi_name}_face_mean"] = roi_face_attention
        scores[f"roi_{roi_name}_face_enrichment"] = float(roi_face_attention / (face_attention + eps))
    return scores


def compute_all_attribution_scores(maps, roi_masks, face_mask=None):
    scores = {}
    for kind in ("pos", "neg", "abs"):
        kind_scores = compute_roi_map_scores(maps[kind], roi_masks)
        for key, value in kind_scores.items():
            scores[f"attr_{kind}_{key}"] = value
        face_kind_scores = compute_face_normalized_roi_map_scores(maps[kind], roi_masks, face_mask)
        for key, value in face_kind_scores.items():
            scores[f"attr_{kind}_{key}"] = value

    signed_scores = compute_roi_map_scores(maps["signed"], roi_masks)
    signed_face_scores = compute_face_normalized_roi_map_scores(maps["signed"], roi_masks, face_mask)
    for roi_name in ROI_NAMES:
        scores[f"attr_signed_roi_{roi_name}_mean"] = signed_scores.get(f"roi_{roi_name}_mean", 0.0)
        scores[f"attr_signed_roi_{roi_name}_face_mean"] = signed_face_scores.get(
            f"roi_{roi_name}_face_mean"
        )
        pos = scores.get(f"attr_pos_roi_{roi_name}_enrichment", 0.0)
        neg = scores.get(f"attr_neg_roi_{roi_name}_enrichment", 0.0)
        scores[f"attr_signed_roi_{roi_name}_balance"] = float(pos - neg)
        face_pos = scores.get(f"attr_pos_roi_{roi_name}_face_enrichment")
        face_neg = scores.get(f"attr_neg_roi_{roi_name}_face_enrichment")
        scores[f"attr_signed_roi_{roi_name}_face_balance"] = (
            None if face_pos is None or face_neg is None else float(face_pos - face_neg)
        )
    return scores


def compute_face_background_attribution_scores(maps, face_mask, face_mask_source, eps=1e-8):
    scores = {"face_mask_source": face_mask_source}
    if face_mask is None or not np.any(face_mask):
        scores["face_mask_area_ratio"] = None
        for kind in ("abs", "pos", "neg"):
            scores[f"face_attr_sum_{kind}"] = None
            scores[f"background_attr_sum_{kind}"] = None
            scores[f"face_attr_ratio_{kind}"] = None
            scores[f"background_attr_ratio_{kind}"] = None
        scores["face_attr_density_abs"] = None
        scores["background_attr_density_abs"] = None
        scores["face_background_enrichment_abs"] = None
        return scores

    for kind in ("abs", "pos", "neg"):
        attn = np.asarray(maps[kind], dtype=np.float32)
        mask_resized = cv2.resize(
            face_mask.astype(np.uint8),
            (attn.shape[1], attn.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ) > 0
        if kind == "abs":
            scores["face_mask_area_ratio"] = float(np.mean(mask_resized))

        face_pixels = int(np.sum(mask_resized))
        background_pixels = int(np.sum(~mask_resized))
        face_sum = float(np.sum(attn[mask_resized])) if face_pixels > 0 else 0.0
        background_sum = float(np.sum(attn[~mask_resized])) if background_pixels > 0 else 0.0
        face_ratio = face_sum / (face_sum + background_sum + eps)
        scores[f"face_attr_sum_{kind}"] = face_sum
        scores[f"background_attr_sum_{kind}"] = background_sum
        scores[f"face_attr_ratio_{kind}"] = float(face_ratio)
        scores[f"background_attr_ratio_{kind}"] = float(1.0 - face_ratio)

        if kind == "abs":
            face_density = face_sum / (face_pixels + eps)
            background_density = background_sum / (background_pixels + eps)
            scores["face_attr_density_abs"] = float(face_density)
            scores["background_attr_density_abs"] = float(background_density)
            scores["face_background_enrichment_abs"] = float(face_density / (background_density + eps))
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


def draw_visualization(img_np, roi_masks, maps, title, face_mask=None, face_mask_source=None, save_path=None, show=False):
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

    face_vis = img_np.copy()
    face_title = "Face mask"
    if face_mask is not None and np.any(face_mask):
        # This panel visualizes the denominator used by face-normalized ROI enrichment.
        face_bool = face_mask > 0
        face_color = np.array([255, 220, 40], dtype=np.uint8)
        face_vis[face_bool] = (0.65 * face_vis[face_bool] + 0.35 * face_color).astype(np.uint8)
        face_title = "Face mask ({})".format(face_mask_source or "unknown")
    else:
        face_title = "Face mask unavailable"

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle(title)
    axes[0].imshow(img_np)
    axes[0].set_title("Input")
    axes[1].imshow(overlay)
    axes[1].set_title("Positive attribution")
    axes[2].imshow(neg_overlay)
    axes[2].set_title("Negative attribution")
    axes[3].imshow(roi_vis)
    axes[3].set_title("Clinical ROI")
    axes[4].imshow(face_vis)
    axes[4].set_title(face_title)
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

    face_mask, face_mask_source = roi_analyzer.generate_face_mask(img_np, view_name=view_name)
    attr_scores = compute_all_attribution_scores(maps, roi_masks, face_mask=face_mask)
    face_background_scores = compute_face_background_attribution_scores(
        maps,
        face_mask,
        face_mask_source,
    )
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
        draw_visualization(
            img_np,
            roi_masks,
            maps,
            title,
            face_mask=face_mask,
            face_mask_source=face_mask_source,
            save_path=save_path,
            show=show,
        )

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
    record.update(face_background_scores)
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
        for metric in ("mean", "enrichment", "face_mean", "face_enrichment"):
            for roi in ROI_NAMES:
                roi_attr.append(f"attr_{kind}_roi_{roi}_{metric}")
    roi_attr.extend([f"attr_signed_roi_{roi}_mean" for roi in ROI_NAMES])
    roi_attr.extend([f"attr_signed_roi_{roi}_balance" for roi in ROI_NAMES])
    roi_attr.extend([f"attr_signed_roi_{roi}_face_mean" for roi in ROI_NAMES])
    roi_attr.extend([f"attr_signed_roi_{roi}_face_balance" for roi in ROI_NAMES])
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
            attr_values.append(f"attr_{kind}_roi_{roi}_face_enrichment")
    attr_values.extend([f"attr_signed_roi_{roi}_mean" for roi in ROI_NAMES])
    attr_values.extend([f"attr_signed_roi_{roi}_balance" for roi in ROI_NAMES])
    attr_values.extend([f"attr_signed_roi_{roi}_face_mean" for roi in ROI_NAMES])
    attr_values.extend([f"attr_signed_roi_{roi}_face_balance" for roi in ROI_NAMES])

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


FACE_BACKGROUND_VALUE_KEYS = [
    "face_mask_area_ratio",
    "face_attr_sum_abs",
    "background_attr_sum_abs",
    "face_attr_ratio_abs",
    "background_attr_ratio_abs",
    "face_attr_density_abs",
    "background_attr_density_abs",
    "face_background_enrichment_abs",
    "face_attr_sum_pos",
    "background_attr_sum_pos",
    "face_attr_ratio_pos",
    "background_attr_ratio_pos",
    "face_attr_sum_neg",
    "background_attr_sum_neg",
    "face_attr_ratio_neg",
    "background_attr_ratio_neg",
]


def _valid_face_background_records(records):
    return [
        row for row in records
        if row.get("face_attr_ratio_abs") is not None
        and np.isfinite(float(row.get("face_attr_ratio_abs")))
    ]


def _aggregate_face_background(records, group_by=None):
    group_by = group_by or []
    groups = defaultdict(list)
    if not group_by:
        groups[("all",)].extend(records)
    else:
        for row in records:
            groups[tuple(row.get(k, "") for k in group_by)].append(row)

    rows = []
    for key, items in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        out = {
            "group_by": "+".join(group_by) if group_by else "all",
            "group_value": "all" if not group_by else "|".join(str(x) for x in key),
            "n": len(items),
        }
        if group_by:
            for i, group_key in enumerate(group_by):
                out[group_key] = key[i]
        for value_key in FACE_BACKGROUND_VALUE_KEYS:
            vals = []
            for item in items:
                value = item.get(value_key)
                if value is None:
                    continue
                value = float(value)
                if np.isfinite(value):
                    vals.append(value)
            out[f"{value_key}_mean"] = float(np.mean(vals)) if vals else None
            out[f"{value_key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
        rows.append(out)
    return rows


def _write_face_background_markdown(records, summary_rows, path):
    path = Path(path)
    overall = next((row for row in summary_rows if row.get("group_by") == "all"), {})
    top_background = sorted(
        records,
        key=lambda row: float(row.get("background_attr_ratio_abs") or -1.0),
        reverse=True,
    )[:10]

    def fmt(value, digits=4):
        if value is None:
            return "NA"
        try:
            value = float(value)
        except Exception:
            return "NA"
        if not np.isfinite(value):
            return "NA"
        return f"{value:.{digits}f}"

    lines = [
        "# Face/Background Attribution Ratio Analysis",
        "",
        "本分析用于验证 CLIP-LoRA 的 attribution 是否主要集中在人脸区域，而不是背景区域。face mask 优先由 MediaPipe FaceMesh 的 face oval convex hull 生成；FaceMesh 失败时，使用 InsightFace 最大人脸 bbox，并向外扩展 10%。",
        "",
        "## Overall",
        "",
        f"- 有效 target-image records: {len(records)}",
        f"- face_attr_ratio_abs: {fmt(overall.get('face_attr_ratio_abs_mean'))} ± {fmt(overall.get('face_attr_ratio_abs_std'))}",
        f"- background_attr_ratio_abs: {fmt(overall.get('background_attr_ratio_abs_mean'))} ± {fmt(overall.get('background_attr_ratio_abs_std'))}",
        f"- face_mask_area_ratio: {fmt(overall.get('face_mask_area_ratio_mean'))} ± {fmt(overall.get('face_mask_area_ratio_std'))}",
        f"- face_attr_density_abs: {fmt(overall.get('face_attr_density_abs_mean'))} ± {fmt(overall.get('face_attr_density_abs_std'))}",
        f"- background_attr_density_abs: {fmt(overall.get('background_attr_density_abs_mean'))} ± {fmt(overall.get('background_attr_density_abs_std'))}",
        f"- face_background_enrichment_abs: {fmt(overall.get('face_background_enrichment_abs_mean'))} ± {fmt(overall.get('face_background_enrichment_abs_std'))}",
        "",
        "## Top 10 Background Attribution Cases",
        "",
        "| rank | image | true_class | target_class | predicted_class | view | background_abs | face_abs | mask_source |",
        "|---:|---|---|---|---|---|---:|---:|---|",
    ]
    for rank, row in enumerate(top_background, start=1):
        lines.append(
            "| {rank} | {image} | {true_class} | {target_class} | {predicted_class} | {view} | {bg} | {face} | {source} |".format(
                rank=rank,
                image=row.get("image_path", ""),
                true_class=row.get("true_class", ""),
                target_class=row.get("target_class", ""),
                predicted_class=row.get("predicted_class", ""),
                view=row.get("view", ""),
                bg=fmt(row.get("background_attr_ratio_abs")),
                face=fmt(row.get("face_attr_ratio_abs")),
                source=row.get("face_mask_source", ""),
            )
        )
    lines.extend([
        "",
        "## Outputs",
        "",
        "- `face_background_attribution_records.csv`: 每张图、每个 target class 的 face/background attribution sum 与 ratio。",
        "- `face_background_attribution_summary.csv`: 按 all、true_class、view、is_correct、predicted_class 分组的 mean/std。",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] saved: {path}")


def save_face_background_outputs(records, output_dir):
    output_dir = Path(output_dir)
    fb_records = _valid_face_background_records(records)
    if not fb_records:
        print("[WARN] no valid face/background attribution records to save")
        return

    record_fields = [
        "true_class",
        "image_name",
        "subject_id",
        "view",
        "target_class",
        "predicted_class",
        "is_correct",
        "malnourished_probability",
        "normal_probability",
        "face_mask_source",
        *FACE_BACKGROUND_VALUE_KEYS,
        "image_path",
    ]
    save_csv(
        [{k: row.get(k) for k in record_fields} for row in fb_records],
        output_dir / "face_background_attribution_records.csv",
    )

    summary_rows = []
    summary_rows.extend(_aggregate_face_background(fb_records, []))
    for group_by in (["true_class"], ["view"], ["is_correct"], ["predicted_class"]):
        summary_rows.extend(_aggregate_face_background(fb_records, group_by))
    save_csv(summary_rows, output_dir / "face_background_attribution_summary.csv")
    _write_face_background_markdown(
        fb_records,
        summary_rows,
        output_dir / "face_background_attribution_analysis.md",
    )


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
        for metric_suffix, label_suffix, file_suffix in (
            ("enrichment", "enrichment", "by_roi"),
            ("face_enrichment", "face-normalized enrichment", "face_enrichment_by_roi"),
        ):
            fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4), sharey=True)
            if len(pairs) == 1:
                axes = [axes]
            for ax, (true_class, target_class), label in zip(axes, pairs, labels):
                vals = [
                    mean_for(records, true_class, target_class, f"attr_{kind}_roi_{roi}_{metric_suffix}")
                    for roi in ROI_NAMES
                ]
                ax.bar(x, vals, alpha=0.8)
                ax.set_xticks(x)
                ax.set_xticklabels(ROI_NAMES, rotation=25, ha="right")
                ax.set_title(label)
                ax.set_ylabel(f"{kind} {label_suffix}")
            fig.tight_layout()
            fig_path = output_dir / f"attribution_{kind}_{file_suffix}.png"
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
        save_face_background_outputs(records, args.output_dir)


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
    save_face_background_outputs(records, args.output_dir)
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
