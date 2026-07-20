#!/usr/bin/env python3
"""V2-A：FoundHand 双姿势离线冒烟验收。

本脚本不修改 FoundHand 原仓库。营养标签仅由真实外观父图继承；FoundHand 只负责
根据目标关键点改变手部姿势，不使用提示词注入营养语义。
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps


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


def square_pad(image, size):
    """保持比例并白色补边，避免直接拉伸改变手部几何。"""
    image = ImageOps.contain(image.convert("RGB"), (size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    offset = ((size - image.width) // 2, (size - image.height) // 2)
    canvas.paste(image, offset)
    return canvas


def extract_keypoints(image, model_path):
    """按 FoundHand 约定返回 42 点：右手在前、左手在后。"""
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.20,
        min_hand_presence_confidence=0.20,
        min_tracking_confidence=0.20,
    )
    array = np.ascontiguousarray(np.asarray(image))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=array)
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
        result = detector.detect(mp_image)

    if len(result.hand_landmarks) != 2:
        raise RuntimeError(f"FoundHand 双手条件要求检测到 2 只手，实际为 {len(result.hand_landmarks)}")

    keypoints = np.zeros((42, 2), dtype=np.float32)
    handedness = []
    occupied = set()
    for landmarks, categories in zip(result.hand_landmarks, result.handedness):
        category = categories[0]
        label = category.category_name
        # 与 FoundHand 官方 Notebook 一致：MediaPipe 对镜像自拍的 Left 实际映射到右手槽位。
        start = 0 if label == "Left" else 21
        if start in occupied:
            raise RuntimeError(f"MediaPipe 双手 handedness 重复，无法安全构造 42 点条件：{label}")
        occupied.add(start)
        handedness.append({"mediapipe_label": label, "score": float(category.score), "slot": start})
        for index, landmark in enumerate(landmarks):
            keypoints[start + index] = [landmark.x * image.width, landmark.y * image.height]

    if occupied != {0, 21}:
        raise RuntimeError(f"左右手槽位不完整：{sorted(occupied)}")
    return keypoints, handedness


def draw_keypoints(image, keypoints):
    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    for hand_index, start in enumerate((0, 21)):
        color = (230, 40, 40) if hand_index == 0 else (30, 110, 230)
        points = [(int(x), int(y)) for x, y in keypoints[start:start + 21]]
        for a, b in HAND_CONNECTIONS:
            draw.line((points[a], points[b]), fill=color, width=2)
        for x, y in points:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(255, 255, 255), outline=color)
    return preview


def build_sam_mask(image, keypoints, sam_checkpoint):
    """使用官方 Demo 相同的双腕正点提示生成外观参考手部掩码。"""
    from segment_anything import SamPredictor, sam_model_registry

    sam = sam_model_registry["vit_h"](checkpoint=str(sam_checkpoint)).to("cuda")
    predictor = SamPredictor(sam)
    predictor.set_image(np.asarray(image))
    input_points = np.asarray([keypoints[0], keypoints[21]], dtype=np.float32)
    masks, scores, _ = predictor.predict(
        point_coords=input_points,
        point_labels=np.ones(2, dtype=np.int32),
        multimask_output=True,
    )
    best = int(np.argmax(scores))
    mask = masks[best].astype(np.uint8)
    score = float(scores[best])
    del predictor, sam
    torch.cuda.empty_cache()
    return mask, score


def keypoint_heatmap(points, size, variance=1.0):
    height, width = size
    xs, ys = np.meshgrid(np.arange(width), np.arange(height))
    grid = np.stack((xs, ys), axis=-1)
    diff = grid[None, ...] - points[:, None, None, :]
    return np.exp(-np.sum(diff ** 2, axis=-1) / (2 * variance)) / (2 * np.pi * variance)


def make_heatmaps(keypoints, image_size, latent_size, device, dtype):
    scaled = keypoints.copy()
    scaled[:, 0] *= latent_size[1] / image_size[1]
    scaled[:, 1] *= latent_size[0] / image_size[0]
    valid = (
        (keypoints[:, 0] > 0) & (keypoints[:, 0] < image_size[1])
        & (keypoints[:, 1] > 0) & (keypoints[:, 1] < image_size[0])
    )
    heatmaps = keypoint_heatmap(scaled, latent_size) * valid[:, None, None]
    return torch.as_tensor(heatmaps[None], device=device, dtype=dtype)


def image_tensor(image, device, dtype):
    array = np.asarray(image).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)


def load_foundhand_models(foundhand_root, model_path, vae_path, device, dtype):
    """使用 mmap + meta 参数加载大权重，降低 11.6GB checkpoint 的内存复制峰值。"""
    sys.path.insert(0, str(foundhand_root))
    from models import vit, vqvae
    from diffusion import create_diffusion

    diffusion = create_diffusion("250")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False, mmap=True)
    state = checkpoint["ema_state_dict"]
    with torch.device("meta"):
        model = vit.DiT_XL_2(
            input_size=32,
            latent_dim=4,
            in_channels=47,
            learn_sigma=True,
        )
    incompatible = model.load_state_dict(state, strict=False, assign=True)
    if incompatible.missing_keys:
        raise RuntimeError(f"FoundHand 主模型缺少权重：{incompatible.missing_keys}")
    model = model.eval().requires_grad_(False).to(device=device, dtype=dtype)
    del state, checkpoint

    vae_checkpoint = torch.load(vae_path, map_location="cpu", weights_only=False)
    autoencoder = vqvae.create_model(3, 3, 4).eval().requires_grad_(False)
    vae_incompatible = autoencoder.load_state_dict(vae_checkpoint["state_dict"], strict=False)
    if vae_incompatible.missing_keys:
        raise RuntimeError(f"VAE 缺少权重：{vae_incompatible.missing_keys}")
    autoencoder = autoencoder.to(device=device, dtype=dtype)
    del vae_checkpoint
    return diffusion, model, autoencoder, incompatible.unexpected_keys, vae_incompatible.unexpected_keys


@torch.inference_mode()
def generate_one(diffusion, model, autoencoder, reference, ref_keypoints, hand_mask,
                 target_keypoints, seed, cfg_scale, device, dtype):
    latent_scale = 0.18215
    latent_size = (32, 32)
    image_size = (256, 256)
    ref_image = image_tensor(reference, device, dtype)
    ref_heatmaps = make_heatmaps(ref_keypoints, image_size, latent_size, device, dtype)
    resized_mask = cv2.resize(hand_mask, latent_size[::-1], interpolation=cv2.INTER_NEAREST)
    mask_tensor = torch.as_tensor(resized_mask[None, None], device=device, dtype=dtype)
    ref_latent = latent_scale * autoencoder.encode(ref_image).sample()
    ref_cond = torch.cat([ref_latent, ref_heatmaps, mask_tensor], dim=1)

    target_heatmaps = make_heatmaps(target_keypoints, image_size, latent_size, device, dtype)
    target_cond = torch.cat([target_heatmaps, torch.zeros_like(mask_tensor)], dim=1)

    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn((1, 4, 32, 32), generator=generator, device=device, dtype=dtype)
    noise = torch.cat([noise, noise], dim=0)
    nvs = torch.zeros(1, dtype=torch.int64, device=device)
    model_kwargs = {
        "target_cond": torch.cat([target_cond, torch.zeros_like(target_cond)], dim=0),
        "ref_cond": torch.cat([ref_cond, torch.zeros_like(ref_cond)], dim=0),
        "nvs": torch.cat([nvs, 2 * torch.ones_like(nvs)], dim=0),
        "cfg_scale": cfg_scale,
    }
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg,
        noise.shape,
        noise,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        progress=True,
        device=device,
    ).chunk(2)[0]
    decoded = autoencoder.decode(samples / latent_scale).clamp(-1, 1)
    output = ((decoded[0].permute(1, 2, 0).float().cpu().numpy() + 1) * 127.5).round().astype(np.uint8)
    return Image.fromarray(output, mode="RGB")


def main():
    parser = argparse.ArgumentParser(description="FoundHand V2-A 双姿势冒烟验收")
    parser.add_argument("--foundhand-root", type=Path, required=True)
    parser.add_argument("--model-weight", type=Path, required=True)
    parser.add_argument("--vae-weight", type=Path, required=True)
    parser.add_argument("--sam-weight", type=Path, required=True)
    parser.add_argument("--hand-model", type=Path, required=True)
    parser.add_argument("--reference-image", type=Path, required=True)
    parser.add_argument("--pose01-image", type=Path, required=True)
    parser.add_argument("--pose02-image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=22011)
    parser.add_argument("--cfg-scale", type=float, default=2.5)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    required = (
        args.foundhand_root, args.model_weight, args.vae_weight, args.sam_weight,
        args.hand_model, args.reference_image, args.pose01_image, args.pose02_image,
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    inputs = {
        "reference": square_pad(Image.open(args.reference_image), 256),
        "pose01": square_pad(Image.open(args.pose01_image), 256),
        "pose02": square_pad(Image.open(args.pose02_image), 256),
    }
    keypoints = {}
    handedness = {}
    for name, image in inputs.items():
        keypoints[name], handedness[name] = extract_keypoints(image, args.hand_model)
        image.save(args.output_dir / f"{name}_padded.png")
        draw_keypoints(image, keypoints[name]).save(args.output_dir / f"{name}_keypoints.png")

    # 关键点预检先落盘；即使后续大模型失败，也保留可诊断证据。
    preflight = {
        "pose01_detected_hands": len(handedness["pose01"]),
        "pose02_detected_hands": len(handedness["pose02"]),
        "handedness": handedness,
    }
    (args.output_dir / "preflight.json").write_text(
        json.dumps(preflight, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.preflight_only:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    mask, sam_score = build_sam_mask(inputs["reference"], keypoints["reference"], args.sam_weight)
    Image.fromarray(mask * 255).save(args.output_dir / "reference_sam_mask.png")
    sam_seconds = time.perf_counter() - started

    dtype = torch.float16
    load_started = time.perf_counter()
    diffusion, model, autoencoder, model_extra, vae_extra = load_foundhand_models(
        args.foundhand_root, args.model_weight, args.vae_weight, "cuda", dtype
    )
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started

    outputs = {}
    inference_seconds = {}
    for index, pose in enumerate(("pose01", "pose02")):
        torch.cuda.synchronize()
        inference_started = time.perf_counter()
        output = generate_one(
            diffusion, model, autoencoder,
            inputs["reference"], keypoints["reference"], mask, keypoints[pose],
            args.seed + index, args.cfg_scale, "cuda", dtype,
        )
        torch.cuda.synchronize()
        inference_seconds[pose] = time.perf_counter() - inference_started
        output_path = args.output_dir / f"generated_{pose}.png"
        output.save(output_path)
        outputs[pose] = {
            "path": str(output_path.resolve()),
            "sha256": sha256(output_path),
            "seed": args.seed + index,
        }

    report = {
        "schema_version": 1,
        "task": "hand_v2a_foundhand_dual_pose_smoke",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "is_hand_only": True,
        "uses_face_assets": False,
        "semantic_control": "无提示词；营养标签仅继承自真实外观父图",
        "foundhand_root": str(args.foundhand_root.resolve()),
        "foundhand_code_git_commit": None,
        "foundhand_code_note": "下载目录不含 .git，无法读取 commit",
        "weights": {
            "model": {"path": str(args.model_weight.resolve()), "sha256": sha256(args.model_weight)},
            "vae": {"path": str(args.vae_weight.resolve()), "sha256": sha256(args.vae_weight)},
            "sam": {"path": str(args.sam_weight.resolve()), "sha256": sha256(args.sam_weight)},
            "mediapipe": {"path": str(args.hand_model.resolve()), "sha256": sha256(args.hand_model)},
        },
        "inputs": {
            "reference": {"path": str(args.reference_image.resolve()), "sha256": sha256(args.reference_image)},
            "pose01": {"path": str(args.pose01_image.resolve()), "sha256": sha256(args.pose01_image)},
            "pose02": {"path": str(args.pose02_image.resolve()), "sha256": sha256(args.pose02_image)},
        },
        "parent_subject_id": "15",
        "parent_split": "fold_0_train",
        "nutrition_label": "malnourished_hand",
        "resolution": 256,
        "sampling_steps": 250,
        "cfg_scale": args.cfg_scale,
        "dtype": "float16",
        "torch_version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "sam_mask_score": sam_score,
        "sam_seconds": sam_seconds,
        "model_load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "unexpected_model_keys": model_extra,
        "unexpected_vae_keys": vae_extra,
        "handedness": handedness,
        "outputs": outputs,
        "manual_review_required": {
            "pose01": True,
            "pose02": True,
            "reason": "V2-A 仅验证可运行性与初步双姿势能力，不能由同一 MediaPipe 模型自证结构正确",
        },
    }
    report_path = args.output_dir / "smoke_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    # 避免宿主环境中非法 OMP_NUM_THREADS 值触发额外告警。
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
