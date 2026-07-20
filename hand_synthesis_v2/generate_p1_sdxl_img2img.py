#!/usr/bin/env python3
"""执行 V2-D P1：SDXL 低强度 img2img。"""

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
import torch
import yaml
from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    method = config["methods"]["P1"]
    if not method["enabled"]:
        raise RuntimeError("配置已禁用 P1")
    root = Path(config["output_root"])
    plans = [item for item in load_jsonl(root / "candidate_plan.jsonl") if item["method"] == "P1"]
    if args.limit is not None:
        plans = plans[:args.limit]

    model_root = Path(method["model_root"])
    required = (
        model_root / "model_index.json",
        model_root / "unet/diffusion_pytorch_model.fp16.safetensors",
        model_root / "vae/diffusion_pytorch_model.fp16.safetensors",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"SDXL 组件未下载完整：{path}")

    # 数 GB 权重只在批次开始时哈希一次，逐图元数据复用同一份可信记录。
    component_paths = sorted(model_root.glob("**/*.safetensors"))
    component_hashes = {str(path.relative_to(model_root)): sha256_file(path) for path in component_paths}
    model_info = {
        "model_repo": method["model_repo"], "model_revision": method["model_revision"],
        "model_index_sha256": sha256_file(model_root / "model_index.json"),
        "license_sha256": sha256_file(model_root / "LICENSE.md"),
        "components": component_hashes,
    }
    (root / "p1_sdxl_component_hashes.json").write_text(
        json.dumps(model_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        model_root, torch_dtype=torch.float16, variant="fp16",
        use_safetensors=True, local_files_only=True,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    torch.cuda.reset_peak_memory_stats()

    for index, plan in enumerate(plans, 1):
        output_dir = root / "p1_sdxl_img2img" / plan["compound_class"]
        metadata_dir = root / "p1_sdxl_img2img" / "metadata"
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{plan['candidate_id']}.png"
        metadata_path = metadata_dir / f"{plan['candidate_id']}.json"
        if output_path.is_file() and metadata_path.is_file():
            print(f"[SKIP] {plan['candidate_id']} 已完成")
            continue
        if output_path.exists() or metadata_path.exists():
            raise RuntimeError(f"候选存在不完整产物，拒绝覆盖：{plan['candidate_id']}")

        parent = Path(plan["appearance_parent_path"])
        if sha256_file(parent) != plan["appearance_parent_sha256"]:
            raise RuntimeError(f"外观父图哈希不符：{parent}")
        image = Image.open(parent).convert("RGB").resize(
            (int(method["resolution"]), int(method["resolution"])), Image.Resampling.LANCZOS)
        generator = torch.Generator(device="cuda").manual_seed(int(plan["seed"]))
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = pipe(
            prompt=plan["prompt"], negative_prompt=plan["negative_prompt"], image=image,
            strength=float(plan["denoising_strength"]),
            num_inference_steps=int(method["num_inference_steps"]),
            guidance_scale=float(method["guidance_scale"]), generator=generator,
        ).images[0]
        torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - started
        result.save(output_path)

        metadata = {
            **plan, "schema_version": 2, "execution_status": "generated",
            "is_hand_only": True, "uses_face_assets": False,
            "label_inheritance": "仅继承真实外观父图，不由 prompt 或模型确认",
            "model": model_info,
            "vae": {
                "path": str(model_root / "vae/diffusion_pytorch_model.fp16.safetensors"),
                "sha256": component_hashes["vae/diffusion_pytorch_model.fp16.safetensors"],
            },
            "lora": None, "controlnet": None, "ip_adapter": None, "handrefiner": None,
            "refined": False, "repair_mask_path": None, "pre_refine_path": None,
            "resolution": list(result.size),
            "num_inference_steps": int(method["num_inference_steps"]),
            "guidance_scale": float(method["guidance_scale"]),
            "output_path": str(output_path), "output_sha256": sha256_file(output_path),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "inference_seconds": inference_seconds,
            "software": {
                "python": platform.python_version(), "torch": torch.__version__,
                "diffusers": __import__("diffusers").__version__,
                "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                "dtype": method["dtype"],
            },
        }
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[{index}/{len(plans)}] P1 完成 {plan['candidate_id']}，{inference_seconds:.2f}s")

    print(json.dumps({
        "status": "completed", "method": "P1", "requested": len(plans),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
