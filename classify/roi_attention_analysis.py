import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
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
POSITIVE_CLASS_IDX = CLASS_NAMES.index(POSITIVE_CLASS)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

ROI_COLORS = {
    "temporal": (255, 80, 80),
    "orbital": (80, 255, 255),
    "malar": (80, 180, 255),
    "jawline": (100, 255, 100),
}
ROI_NAMES = ["temporal", "orbital", "malar", "jawline"]


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


class ClassSpecificGradientRollout:
    """Gradient-weighted attention rollout for the CLIP visual ViT."""

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

            cams = []
            for block in self.attn_modules:
                attn = block.attn.captured_attn
                if attn is None or attn.grad is None:
                    continue
                grad = torch.relu(attn.grad.detach())
                weighted = (attn.detach() * grad).mean(dim=1)[0]
                if float(weighted.sum()) <= 0:
                    weighted = attn.detach().mean(dim=1)[0]
                cams.append(weighted.cpu())

            if not cams:
                raise RuntimeError("没有捕获到可用于解释的attention梯度。")

            n_tokens = cams[0].shape[-1]
            result = torch.eye(n_tokens)
            for cam in cams:
                cam = cam + torch.eye(n_tokens)
                cam = self._normalize_rows(cam)
                result = torch.matmul(cam, result)

            cls_attention = result[0, 1:]
            grid_size = int(np.sqrt(cls_attention.numel()))
            if grid_size * grid_size != cls_attention.numel():
                raise RuntimeError(f"patch token数量不是平方数: {cls_attention.numel()}")

            attn_map = cls_attention.reshape(grid_size, grid_size).numpy()
            return {
                "attn_map": attn_map,
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


def compute_dynamic_roi_attention_scores(attn_map, roi_masks):
    scores = {}
    for roi_name, mask in roi_masks.items():
        attn_resized = cv2.resize(attn_map, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
        mask_bool = mask > 0
        if np.sum(mask_bool) == 0:
            scores[roi_name] = 0.0
            continue
        roi_attention = attn_resized[mask_bool]
        global_attention = float(np.mean(attn_resized))
        scores[roi_name] = float(np.mean(roi_attention) / (global_attention + 1e-8))
    return scores


def normalize_for_vis(attn_map):
    return (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)


def visualize_single(img_path, model, explainer, transform, roi_analyzer, target_class, save_path=None, show=False):
    img_path = Path(img_path)
    subject_id, view_name = parse_subject_and_view(img_path)
    img_pil = Image.open(img_path).convert("RGB")
    tensor, img_np = preprocess_for_model_and_roi(img_pil, transform)

    explanation = explainer(tensor, target_class=target_class)
    raw_attn_map = explanation["attn_map"]
    probs = explanation["probs"]
    logits = explanation["logits"]
    target_idx = explanation["target_idx"]
    pred_idx = int(np.argmax(logits))
    mal_prob = float(probs[POSITIVE_CLASS_IDX])

    roi_masks = roi_analyzer.generate_roi_masks(img_np, view_name=view_name)
    if roi_masks is None:
        print(f"[WARN] ROI生成失败: {img_path}")
        return None

    roi_scores = compute_dynamic_roi_attention_scores(raw_attn_map, roi_masks)
    attn_resized = cv2.resize(normalize_for_vis(raw_attn_map), (224, 224), interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap((attn_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.5 * img_np + 0.5 * heatmap).astype(np.uint8)

    roi_vis = img_np.copy()
    for roi_name, mask in roi_masks.items():
        color = ROI_COLORS[roi_name]
        roi_vis[mask > 0] = (0.7 * roi_vis[mask > 0] + 0.3 * np.array(color)).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"{img_path.name} | view={view_name} | target={CLASS_NAMES[target_idx]} | "
        f"pred={CLASS_NAMES[pred_idx]} | mal_prob={mal_prob * 100:.1f}%"
    )
    axes[0].imshow(img_np)
    axes[0].set_title("Input")
    axes[0].axis("off")
    axes[1].imshow(overlay)
    axes[1].set_title("Class-specific gradient rollout")
    axes[1].axis("off")
    axes[2].imshow(roi_vis)
    axes[2].set_title("Clinical ROI")
    axes[2].axis("off")
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)

    return {
        "image_path": str(img_path),
        "image_name": img_path.name,
        "subject_id": subject_id,
        "view": view_name,
        "target_class": CLASS_NAMES[target_idx],
        "predicted_class": CLASS_NAMES[pred_idx],
        "malnourished_probability": mal_prob,
        "normal_probability": float(probs[CLASS_NAMES.index("normal_face")]),
        "roi_scores": roi_scores,
    }


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


def save_records(records, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "roi_attention_records.json"
    csv_path = output_dir / "roi_attention_records.csv"
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    fieldnames = [
        "true_class", "image_name", "subject_id", "view", "target_class",
        "predicted_class", "malnourished_probability", "normal_probability",
        *[f"roi_{name}_enrichment" for name in ROI_NAMES],
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            flat = {k: row.get(k) for k in fieldnames if not k.startswith("roi_")}
            for roi_name in ROI_NAMES:
                flat[f"roi_{roi_name}_enrichment"] = row["roi_scores"].get(roi_name)
            writer.writerow(flat)
    print(f"[INFO] saved: {json_path}")
    print(f"[INFO] saved: {csv_path}")


def save_group_summary(records, output_dir):
    output_dir = Path(output_dir)
    grouped = {cls: {roi: [] for roi in ROI_NAMES} for cls in CLASS_NAMES}
    for row in records:
        true_class = row["true_class"]
        for roi in ROI_NAMES:
            grouped[true_class][roi].append(row["roi_scores"][roi])

    summary = []
    for roi in ROI_NAMES:
        mal_vals = grouped["malnourished_face"][roi]
        nor_vals = grouped["normal_face"][roi]
        p_value = float("nan")
        if len(mal_vals) >= 2 and len(nor_vals) >= 2:
            _, p_value = ttest_ind(mal_vals, nor_vals, equal_var=False)
        summary.append({
            "roi": roi,
            "malnourished_mean": float(np.mean(mal_vals)) if mal_vals else float("nan"),
            "normal_mean": float(np.mean(nor_vals)) if nor_vals else float("nan"),
            "p_value": float(p_value) if np.isfinite(p_value) else None,
            "n_malnourished": len(mal_vals),
            "n_normal": len(nor_vals),
        })

    summary_path = output_dir / "roi_attention_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    x = np.arange(len(ROI_NAMES))
    width = 0.35
    mal_means = [r["malnourished_mean"] for r in summary]
    nor_means = [r["normal_mean"] for r in summary]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, mal_means, width, label="Malnourished", alpha=0.8)
    ax.bar(x + width / 2, nor_means, width, label="Normal", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(ROI_NAMES)
    ax.set_ylabel("Class-specific attention enrichment")
    ax.set_title("ROI Attention Comparison")
    ax.legend()
    for i, row in enumerate(summary):
        y_base = max([v for v in (mal_means[i], nor_means[i]) if np.isfinite(v)], default=0.0)
        p_text = f"p={row['p_value']:.4f}" if row["p_value"] is not None else "p=NA"
        ax.text(i, y_base + 0.01, p_text, ha="center")
    plt.tight_layout()
    fig_path = output_dir / "roi_attention_comparison.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] saved: {summary_path}")
    print(f"[INFO] saved: {fig_path}")


def run_single(args, model, explainer, transform, roi_analyzer):
    save_path = args.save_path
    if save_path is None:
        save_path = Path(args.output_dir) / f"single_{Path(args.image).stem}.png"
    record = visualize_single(
        img_path=args.image,
        model=model,
        explainer=explainer,
        transform=transform,
        roi_analyzer=roi_analyzer,
        target_class=args.target_class,
        save_path=save_path,
        show=args.show,
    )
    if record is not None:
        print(json.dumps(record, indent=2, ensure_ascii=False))


def run_batch(args, model, explainer, transform, roi_analyzer):
    records = []
    output_dir = Path(args.output_dir)
    for true_class, img_path in iter_test_images(args.test_dir, max_per_class=args.max_per_class):
        save_path = output_dir / "images" / true_class / f"{img_path.stem}_roi_attention.png"
        record = visualize_single(
            img_path=img_path,
            model=model,
            explainer=explainer,
            transform=transform,
            roi_analyzer=roi_analyzer,
            target_class=args.target_class,
            save_path=save_path,
            show=False,
        )
        if record is None:
            continue
        record["true_class"] = true_class
        records.append(record)
    save_records(records, output_dir)
    save_group_summary(records, output_dir)
    print(f"[INFO] completed {len(records)} images")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Class-specific CLIP ROI attention visualization for NutriCode."
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--clip-download-dir", default=os.environ.get("CLIP_DOWNLOAD_DIR", None))
    parser.add_argument("--target-class", default=POSITIVE_CLASS, choices=[*CLASS_NAMES, "predicted"])
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mode", default="single", choices=["single", "batch"])
    parser.add_argument("--image", default="/root/autodl-tmp/test_data/malnourished_face/35_01.png")
    parser.add_argument("--save-path", default=None)
    parser.add_argument("--test-dir", default=DEFAULT_TEST_DIR)
    parser.add_argument("--max-per-class", type=int, default=None)
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
