#!/usr/bin/env python3
"""DataDream + OpenPose/FoundHand 三种快速手部生成入口。"""

import argparse
import gc
import hashlib
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from diffusers import (
    StableDiffusionImg2ImgPipeline,
)
from PIL import Image, ImageDraw


HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)


class HandDiffOpts:
    """兼容 FoundHand 旧 checkpoint 中序列化的 __main__.HandDiffOpts。"""

    pass


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path):
    return [
        json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def derived_seed(base_seed, mode, compound, index):
    """由实验主键确定 seed，避免补生成时改变已有候选。"""
    payload = f"{base_seed}:{mode}:{compound}:{index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def render_keypoints(path, resolution):
    """把 FoundHand 的左右手 42 点渲染成现有 OpenPose ControlNet 条件图。"""
    points = np.load(path).astype(np.float32)
    if points.shape != (42, 2):
        raise ValueError(f"关键点形状必须为(42,2)，实际为{points.shape}: {path}")
    scale = resolution / 256.0
    points = points * scale
    image = Image.new("RGB", (resolution, resolution), "black")
    draw = ImageDraw.Draw(image)
    colors = ((255, 80, 80), (80, 180, 255))
    for hand_index, offset in enumerate((0, 21)):
        hand = points[offset:offset + 21]
        for start, end in HAND_CONNECTIONS:
            draw.line(
                (tuple(hand[start]), tuple(hand[end])),
                fill=colors[hand_index], width=max(3, resolution // 128),
            )
        radius = max(3, resolution // 128)
        for x, y in hand:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="white")
    return image


def make_plan(config, mode, count):
    records = load_jsonl(config["condition_manifest"])
    parents, usable = defaultdict(list), defaultdict(list)
    for record in records:
        if record["source_split"] != "fold_0_train":
            raise ValueError(f"发现非训练折条件：{record['condition_id']}")
        parents[record["compound_class"]].append(record)
        if record["usable_for_generation"]:
            usable[record["compound_class"]].append(record)
    plan = []
    for compound in config["compound_classes"]:
        group_parents = sorted(parents[compound], key=lambda item: item["condition_id"])
        group_usable = sorted(usable[compound], key=lambda item: item["condition_id"])
        if not group_parents or not group_usable:
            raise RuntimeError(f"{compound} 缺少父图或可用姿势条件")
        for index in range(count):
            # T2I/I2I 使用更广的标签参考父图；FoundHand 外观父图必须同时具有关键点和掩码。
            label_parent = group_parents[index % len(group_parents)]
            structure = group_usable[index % len(group_usable)]
            appearance = structure if mode == "foundhand_i2i" else label_parent
            candidate_id = f"Q_{mode.upper()}_{compound}_{index:03d}"
            plan.append({
                "candidate_id": candidate_id,
                "mode": mode,
                "compound_class": compound,
                "nutrition_class": label_parent["nutrition_class"],
                "pose": label_parent["pose"],
                "appearance_parent_subject_id": appearance["parent_subject_id"],
                "appearance_parent_path": appearance["padded_image_path"],
                "appearance_keypoints_path": appearance.get("keypoints_path"),
                "appearance_mask_path": appearance.get("mask_path"),
                "label_reference_subject_id": label_parent["parent_subject_id"],
                "label_reference_path": label_parent["padded_image_path"],
                "structure_condition_id": structure["condition_id"],
                "structure_parent_subject_id": structure["parent_subject_id"],
                "keypoints_path": structure["keypoints_path"],
                "seed": derived_seed(config["base_seed"], mode, compound, index),
            })
    return plan


def load_controlnet(config):
    """显式按 SD2.1 线性投影配置转换，避免单文件被误判为 SD1.5。"""
    from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
        download_controlnet_from_original_ckpt,
    )
    return download_controlnet_from_original_ckpt(
        checkpoint_path=str(config["models"]["controlnet_weight"]),
        original_config_file=str(config["models"]["controlnet_config"]),
        image_size=int(config["generation"]["resolution"]),
        from_safetensors=True,
        device="cpu",
        use_linear_projection=True,
        cross_attention_dim=1024,
    ).to(dtype=torch.float16)


def load_pipeline(config, mode):
    """纯 I2I 消融：不加载 OpenPose-ControlNet。"""
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        config["models"]["base_model"], torch_dtype=torch.float16,
        safety_checker=None, local_files_only=True,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=False)
    return pipe


def load_foundhand(config):
    # FoundHand 的兼容加载器位于旧生成模块；新增 C2 候选仍复用同一权重。
    script_dir = Path(__file__).resolve().parent.parent / "hand_synthesis_v2"
    sys.path.insert(0, str(script_dir))
    import smoke_foundhand as foundhand
    diffusion, model, autoencoder, model_extra, vae_extra = foundhand.load_foundhand_models(
        Path(config["models"]["foundhand_root"]),
        Path(config["models"]["foundhand_weight"]),
        Path(config["models"]["foundhand_vae"]),
        "cuda", torch.float32,
    )
    return foundhand, diffusion, model, autoencoder, model_extra, vae_extra


def lora_weight_for(config, compound_class):
    """优先使用实验显式登记的权重路径；保留旧配置的目录拼接兼容性。"""
    explicit_paths = config["models"].get("lora_paths", {})
    if compound_class in explicit_paths:
        return Path(explicit_paths[compound_class])
    return (
        Path(config["models"]["lora_root"]) / config["models"]["lora_mid"]
        / compound_class / "pytorch_lora_weights.safetensors"
    )


def generate_foundhand_anchor(bundle, item, config, output_path):
    foundhand, diffusion, model, autoencoder, _, _ = bundle
    # FoundHand 的DiT/VAE固定输入256×256；本实验的条件预览可为512，
    # 因此锚定前同步缩放参考图及两套关键点，最终再由ControlNet放大到目标分辨率。
    reference = Image.open(item["appearance_parent_path"]).convert("RGB")
    source_width = reference.width
    foundhand_size = 256
    reference = reference.resize((foundhand_size, foundhand_size), Image.Resampling.LANCZOS)
    point_scale = foundhand_size / float(source_width)
    reference_keypoints = np.load(item["appearance_keypoints_path"]).astype(np.float32) * point_scale
    target_keypoints = np.load(item["keypoints_path"]).astype(np.float32) * point_scale
    hand_mask = (
        np.asarray(Image.open(item["appearance_mask_path"]).convert("L")) > 127
    ).astype(np.uint8)
    anchor = foundhand.generate_one(
        diffusion, model, autoencoder, reference, reference_keypoints, hand_mask,
        target_keypoints, int(item["seed"]),
        float(config["generation"]["foundhand_cfg"]), "cuda", torch.float32,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    anchor.save(output_path)
    return anchor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("op_t2i", "op_i2i", "foundhand_i2i"), required=True)
    parser.add_argument("--limit-per-compound", type=int)
    parser.add_argument("--limit-total", type=int)
    # 基础 SD2.1 对照禁用类别 LoRA，其他生成参数保持一致。
    parser.add_argument("--disable-lora", action="store_true")
    # 可传入已冻结的候选计划，以确保外观、结构和强度分层可复现。
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--compound", choices=tuple(config_name for config_name in ("malnourished_hand_pose01", "malnourished_hand_pose02", "normal_hand_pose01", "normal_hand_pose02")))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    generation = config["generation"]
    count = args.limit_per_compound or int(config["candidate_count_per_compound"])

    required = [
        config["condition_manifest"], config["models"]["base_model"],
    ]
    if args.mode == "foundhand_i2i":
        required.extend((
            config["models"]["foundhand_weight"], config["models"]["foundhand_vae"],
        ))
    for path in required:
        if not Path(path).exists():
            raise FileNotFoundError(path)

    plan = load_jsonl(args.plan) if args.plan else make_plan(config, args.mode, count)
    if args.plan and not plan:
        raise ValueError(f"冻结候选计划为空：{args.plan}")
    # 补生成时只处理未达到接收门槛的复合类别，避免重复扩量。
    if args.compound is not None:
        plan = [item for item in plan if item["compound_class"] == args.compound]
    if args.limit_total is not None:
        plan = plan[:args.limit_total]
    root = Path(config["output_root"])
    # 新实验可指定独立输出根目录，避免与已归档的候选图混合。
    mode_root = Path(config.get("generation_output_root", root / "generation")) / args.mode
    metadata_root = mode_root / "metadata"
    condition_root = mode_root / "conditions"
    anchor_root = mode_root / "foundhand_anchors"
    for path in (metadata_root, condition_root):
        path.mkdir(parents=True, exist_ok=True)

    pipe = load_pipeline(config, args.mode)
    # 已有锚图直接复用；扩充候选缺少锚图时按冻结权重生成新锚图。
    foundhand_bundle = load_foundhand(config) if args.mode == "foundhand_i2i" else None
    current_compound = None
    completed = 0
    torch.cuda.reset_peak_memory_stats()
    for item in plan:
        output_dir = mode_root / item["compound_class"]
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{item['candidate_id']}.png"
        metadata_path = metadata_root / f"{item['candidate_id']}.json"
        if output_path.is_file() and metadata_path.is_file():
            print(f"[SKIP] {item['candidate_id']} 已存在")
            continue
        if output_path.exists() or metadata_path.exists():
            raise RuntimeError(f"候选产物不完整，拒绝覆盖：{item['candidate_id']}")

        if not args.disable_lora and current_compound != item["compound_class"]:
            if current_compound is not None:
                pipe.unload_lora_weights()
            lora_path = lora_weight_for(config, item["compound_class"])
            if not lora_path.is_file():
                raise FileNotFoundError(lora_path)
            pipe.load_lora_weights(str(lora_path.parent), weight_name=lora_path.name)
            current_compound = item["compound_class"]

        resolution = int(generation["resolution"])
        condition = render_keypoints(item["keypoints_path"], resolution)
        condition_path = condition_root / f"{item['candidate_id']}.png"
        condition.save(condition_path)
        init_path = None
        if args.mode == "op_i2i":
            init_image = Image.open(item["appearance_parent_path"]).convert("RGB").resize(
                (resolution, resolution), Image.Resampling.LANCZOS
            )
            init_path = item["appearance_parent_path"]
        elif args.mode == "foundhand_i2i":
            anchor_path = anchor_root / item["compound_class"] / f"{item['candidate_id']}.png"
            if anchor_path.is_file():
                anchor = Image.open(anchor_path).convert("RGB")
            else:
                anchor = generate_foundhand_anchor(foundhand_bundle, item, config, anchor_path)
            init_image = anchor.resize((resolution, resolution), Image.Resampling.LANCZOS)
            init_path = str(anchor_path)

        generator = torch.Generator(device="cuda").manual_seed(int(item["seed"]))
        kwargs = {
            "prompt": config["prompts"][item["compound_class"]],
            "negative_prompt": config["prompts"][f"negative_pose{item['pose']}"],
            "image": init_image,
            "num_inference_steps": int(generation["inference_steps"]),
            "guidance_scale": float(generation["guidance_scale"]),
            "generator": generator,
        }
        if args.mode == "op_t2i":
            raise ValueError("本消融仅支持 I2I")
        kwargs.update(strength=float(item.get("denoising_strength", generation["denoising_strength"])))
        torch.cuda.synchronize()
        started = time.perf_counter()
        output = pipe(**kwargs).images[0]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        output.save(output_path)
        metadata = {
            **item,
            "schema_version": 1,
            "stage": config["stage"],
            "status": "generated",
            "is_hand_only": True,
            "uses_test_data": False,
            "output_path": str(output_path),
            "output_sha256": sha256_file(output_path),
            "condition_path": str(condition_path),
            "condition_sha256": sha256_file(condition_path),
            "init_image_path": init_path,
            "lora_path": None if args.disable_lora else str(lora_weight_for(config, item["compound_class"])),
            "base_model": config["models"]["base_model"],
            "controlnet_weight": None,
            "resolution": resolution,
            "inference_steps": int(generation["inference_steps"]),
            "guidance_scale": float(generation["guidance_scale"]),
            "controlnet_scale": float(generation["controlnet_scale"]),
            "denoising_strength": None if args.mode == "op_t2i" else float(item.get("denoising_strength", generation["denoising_strength"])),
            "inference_seconds": elapsed,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        completed += 1
        print(f"[{completed}/{len(plan)}] {item['candidate_id']} 完成，{elapsed:.2f}s")

    if current_compound is not None:
        pipe.unload_lora_weights()
    del pipe, foundhand_bundle
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps({
        "status": "complete", "mode": args.mode, "requested": len(plan),
        "newly_completed": completed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
