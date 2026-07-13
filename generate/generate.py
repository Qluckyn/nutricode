
import json
import os
import pickle
import random
import string
from datetime import datetime, timezone
from pathlib import Path
from os.path import join as ospj
from typing import Tuple

import fire
import numpy as np
import torch
import torchvision as tv
import yaml
from safetensors import safe_open
from tqdm import tqdm
from transformers import T5Tokenizer, T5EncoderModel

from util import (
    batch_iteration, 
    make_dirs, 
    set_seed,
    SUBSET_NAMES,
    TEMPLATES_SMALL,
    HAND_GENERATION_PROMPTS,
    HAND_POSE_NEGATIVE_PROMPTS,
)


HAND_DATASET_NAME = "hand_nutrition"


def validate_hand_prompt_lengths(pipe, classname):
    """拒绝超过 CLIP 上限的手部 prompt，避免关键姿势或场景描述被静默截断。"""
    pose = "pose01" if classname.endswith("pose01") else "pose02"
    prompt_map = {
        "positive": HAND_GENERATION_PROMPTS[classname],
        "negative": HAND_POSE_NEGATIVE_PROMPTS[pose],
    }
    max_length = pipe.tokenizer.model_max_length
    for prompt_type, prompt in prompt_map.items():
        token_count = len(pipe.tokenizer(prompt, truncation=False).input_ids)
        if token_count > max_length:
            raise ValueError(
                f"{classname} {prompt_type} prompt 超过 CLIP 上限："
                f"{token_count}>{max_length}"
            )

def get_pipe(model_type, model_dir, device, is_tqdm):
    # CUDA_VISIBLE_DEVICES issue
    # https://discuss.pytorch.org/t/cuda-visible-device-is-of-no-use/10018
    from diffusers import DiffusionPipeline, StableDiffusionPipeline, StableDiffusionXLPipeline

    if model_type in ("sdxl", "sdxl-base", "sdxl-base-1.0"):
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
    elif model_type not in ("sdxl-turbo",):
        pipe = StableDiffusionPipeline.from_pretrained(
            model_dir,
            revision="fp16",
            torch_dtype=torch.float16,
        )
    else:
        pipe = DiffusionPipeline.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
    pipe = pipe.to(device)

    pipe.set_progress_bar_config(disable=not is_tqdm)

    return pipe


def get_prompt_embeds(pipe, prompts, device):
    text_inputs = pipe.tokenizer(
        prompts,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids

    if (
        hasattr(pipe.text_encoder.config, "use_attention_mask")
        and pipe.text_encoder.config.use_attention_mask
    ):
        attention_mask = text_inputs.attention_mask.to(device)
    else:
        attention_mask = None

    prompt_embeds = pipe.text_encoder(
        text_input_ids.to(device),
        attention_mask=attention_mask,
    )
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds


def update_pipe(
    pipe,
    n_shot,
    n_template,
    dataset,
    datadream_dir,
    datadream_lr,
    datadream_epoch,
    datadream_train_text_encoder,
    fewshot_seed,
    classname,
):

    if datadream_dir is None:
        raise ValueError("`datadream_dir` should be defined.")

    print("Update pipe with DataDream.")
    mid = f"shot{n_shot}_{fewshot_seed}_tpl{n_template}"
    if not datadream_train_text_encoder:
        mid += "_notextlora"
    # 原代码：
    # fpath = ospj(
    #     datadream_dir, dataset, mid,
    #     f"lr{datadream_lr}_epoch{datadream_epoch}", classname,
    # )
    # 保留原目录规则，并在加载前显式校验，避免四类手部权重错配后静默生成。
    fpath = ospj(
        datadream_dir,
        dataset,
        mid,
        f"lr{datadream_lr}_epoch{datadream_epoch}",
        classname,
    )
    weight_path = ospj(fpath, "pytorch_lora_weights.safetensors")
    if not os.path.isfile(weight_path):
        raise FileNotFoundError(f"Missing LoRA weight: {weight_path}")
    pipe.load_lora_weights(fpath, weight_name="pytorch_lora_weights.safetensors")
    # 仅附加追溯信息，不改变 Diffusers pipeline 的推理行为。
    pipe._datadream_lora_path = weight_path
    if dataset == HAND_DATASET_NAME:
        validate_hand_prompt_lengths(pipe, classname)

    return pipe


def get_dataset_name_for_template(dataset):
    dataset_name = {
        "imagenet": "",
        "imagenet_100": "",
        "pets": "pet ",
        "fgvc_aircraft": "aircraft ",
        "cars": "car ",
        "eurosat": "satellite ",
        "dtd": "texture ",
        "flowers102": "flower ",
        "food101": "food ",
        "sun397": "scene ",
        "caltech101": "",
        "my_dataset": "human ",
        "my_dataset_binary": "human ",
        "hand_nutrition": "",
    }[dataset]
    return dataset_name


@torch.no_grad()
def get_text_embeds_for_weight(pipe, device, dataset, prompt2="both"):
    dataset_name = get_dataset_name_for_template(dataset)
    embeds_original = []
    embeds_soft = []
    for template in TEMPLATES_SMALL:
        prompts_original = [
            template.format(dataset_name, clsname) for clsname in SUBSET_NAMES[dataset]
        ]

        if prompt2 == "both":
            prompts_soft = [
                template.format(dataset_name, f"<{clsname}>, {clsname}")
                for clsname in SUBSET_NAMES[dataset]
            ]
        elif prompt2 == "short":
            prompts_soft = [
                template.format(dataset_name, f"<{clsname}>")
                for clsname in SUBSET_NAMES[dataset]
            ]
        else:
            raise ValueError('`prompt2` should be either "both" or "short".')

        _embeds_original = get_prompt_embeds(pipe, prompts_original, device)
        _embeds_soft = get_prompt_embeds(pipe, prompts_soft, device)

        embeds_original.append(_embeds_original)
        embeds_soft.append(_embeds_soft)

    embeds_original = torch.stack(
        embeds_original
    )  # size = [n_temp, n_cls, n_seq, f_dim]
    embeds_soft = torch.stack(embeds_soft)

    return embeds_original, embeds_soft


class GenerateImage:
    def __init__(
        self,
        pipe,
        device,
        mode,
        guidance_scale,
        num_inference_steps,
        n_img_per_class,
        save_dir,
        count_start,
        bs,
        n_shot,
        n_template,
        dataset,
        seed=42,
        sd_version=None,
        model_dir=None,
    ):
        self.pipe = pipe
        self.device = device
        self.mode = mode
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.n_img_per_class = n_img_per_class
        self.save_dir = save_dir
        self.count_start = count_start
        self.bs = bs
        self.n_shot = n_shot
        self.n_template = n_template
        self.dataset = dataset
        self.seed = int(seed)
        self.sd_version = sd_version
        self.model_dir = model_dir
        self.hand_prompts = HAND_GENERATION_PROMPTS
        self.dataset_name = get_dataset_name_for_template(dataset)

        self.resize_fn = tv.transforms.Resize(
            224, interpolation=tv.transforms.InterpolationMode.BICUBIC
        )

        self.run = self.name_template_method

    def update_pipe(self, pipe):
        # for datadream
        self.pipe = pipe

    def save_data(
        self,
        outputs,
        save_dir,
        count,
        prompts=None,
        seeds=None,
        classname=None,
        negative_prompt=None,
    ):
        images = outputs.images
        metadata_path = Path(save_dir) / "metadata.jsonl"
        for batch_index, image in enumerate(images):
            fpath = ospj(save_dir, f"{count}.png")
            if self.dataset == HAND_DATASET_NAME and os.path.exists(fpath):
                raise FileExistsError(
                    f"手部生成拒绝覆盖已有图片，请调整 count_start 或输出目录：{fpath}"
                )
            image = image.resize((512, 512))
            image.save(fpath)

            if self.dataset == HAND_DATASET_NAME:
                pose = "pose01" if classname.endswith("pose01") else "pose02"
                nutrition_status = (
                    "malnourished" if classname.startswith("malnourished") else "normal"
                )
                record = {
                    "schema_version": 1,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "output_path": str(Path(fpath).resolve()),
                    "output_index": count,
                    "dataset": self.dataset,
                    "class_name": classname,
                    "nutrition_status": nutrition_status,
                    "pose": pose,
                    "prompt": prompts[batch_index],
                    "negative_prompt": negative_prompt,
                    "seed": int(seeds[batch_index]),
                    "lora_weight_path": getattr(
                        self.pipe, "_datadream_lora_path", None
                    ),
                    "sd_version": self.sd_version,
                    "model_dir": self.model_dir,
                    "guidance_scale": self.guidance_scale,
                    "num_inference_steps": self.num_inference_steps,
                    "lora_scale": 1 if self.mode == "datadream" else 0,
                    "width": 512,
                    "height": 512,
                }
                with metadata_path.open("a", encoding="utf-8") as file_obj:
                    file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
        return count

    def run_pipe(self, prompts, seeds=None, negative_prompt=None):
        if isinstance(prompts, list):
            prompt_embeds = None
        elif isinstance(prompts, torch.Tensor):
            prompt_embeds = prompts
            prompts = None

        lora_scale = 1 if self.mode == "datadream" else 0

        pipe_kwargs = dict(
            prompt=prompts,
            prompt_embeds=prompt_embeds,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            cross_attention_kwargs={"scale": lora_scale},
        )
        if self.dataset == HAND_DATASET_NAME:
            pipe_kwargs["negative_prompt"] = [negative_prompt] * len(prompts)
            pipe_kwargs["generator"] = [
                torch.Generator(device=self.device).manual_seed(int(seed))
                for seed in seeds
            ]
        outputs = self.pipe(**pipe_kwargs)
        return outputs

    def set_save_dir(self, classname, prompts):
        save_dir = ospj(self.save_dir, "train", classname)

        make_dirs(save_dir)
        if isinstance(prompts, list):
            with open(ospj(save_dir, "prompts.json"), "w") as f:
                json.dump(prompts, f, indent=4, ensure_ascii=False)

        return save_dir

    def decorator_batch_prompts(prompt_fn):
        def wrapper(self, classname):
            # prompts for input to SD
            prompts = prompt_fn(self, classname)

            # make directory
            save_dir = self.set_save_dir(classname, prompts)

            count = self.count_start
            prompts = prompts[self.count_start :]

            for prompts_batch in batch_iteration(prompts, self.bs):
                if self.dataset == HAND_DATASET_NAME:
                    seeds_batch = list(
                        range(self.seed + count, self.seed + count + len(prompts_batch))
                    )
                    pose = "pose01" if classname.endswith("pose01") else "pose02"
                    negative_prompt = HAND_POSE_NEGATIVE_PROMPTS[pose]
                else:
                    # 原面部流程继续依赖类级 set_seed，不额外注入 generator 或 negative prompt。
                    seeds_batch = None
                    negative_prompt = None
                # generate images
                outputs = self.run_pipe(
                    prompts_batch,
                    seeds=seeds_batch,
                    negative_prompt=negative_prompt,
                )

                # save
                count = self.save_data(
                    outputs,
                    save_dir,
                    count,
                    prompts=prompts_batch,
                    seeds=seeds_batch,
                    classname=classname,
                    negative_prompt=negative_prompt,
                )

        return wrapper

    @decorator_batch_prompts
    def name_template_method(self, classname):
        if self.dataset == HAND_DATASET_NAME:
            # 手部只使用阶段 C 定义的固定临床 prompt，不套用面部/通用模板。
            return [self.hand_prompts[classname]] * self.n_img_per_class

        # 原始面部/通用模板生成逻辑完整保留。
        templates = TEMPLATES_SMALL[: self.n_template]
        n_repeat = self.n_img_per_class // len(templates) + 1
        prompts = [
            template.format(self.dataset_name, classname)
            for _ in range(n_repeat)
            for template in templates
        ]
        prompts = prompts[: self.n_img_per_class]
        return prompts



def set_local(dataset):
    yaml_file = "local.yaml"
    with open(yaml_file, "r") as f:
        args_local = yaml.safe_load(f)
    return args_local


def main(
    seed=42,
    sd_version="sd2.1",
    mode="datadream",  # zeroshot, datadream
    guidance_scale=2.0,
    num_inference_steps=50,
    n_img_per_class=100,
    count_start=0,
    n_set_split=5,
    split_idx=0,
    bs=10,
    # few-shot
    n_shot=0,
    n_template=0,
    dataset="imagenet",
    fewshot_seed="seed0",  # best or seed{number}.
    datadream_lr: float = 1e-4,
    datadream_epoch: int = 200,
    datadream_train_text_encoder: bool = True,
    is_tqdm: bool = True,
    is_dataset_wise_model: bool = False,
    save_dir: str = None,
    datadream_dir: str = None,
    model_dir: str = None,
):
    if isinstance(n_set_split, str):
        n_set_split = int(n_set_split)
    if isinstance(split_idx, str):
        split_idx = int(split_idx)
    if isinstance(n_shot, str):
        n_shot = int(n_shot)

    assert mode in ("zeroshot", "datadream"), "Wrong `mode` argument."
    assert mode == "datadream" and n_shot >= 1, \
           "`n_shot` should be integer when `mode` is datadream."
    if mode == "zeroshot":
        n_shot = 0

    # set local arguments
    args_local = set_local(dataset)
    model_dir = model_dir or args_local["model_dir"][sd_version]
    datadream_dir = datadream_dir or (args_local["datadream_dir"] if mode == "datadream" else None)

    # save directory
    mid_dir = f"gs{guidance_scale}_nis{num_inference_steps}"
    if mode == "zeroshot":
        mid2_dir = f"shot{n_shot}_template{n_template}"
    elif mode == "datadream":
        mid2_dir = f"shot{n_shot}_{fewshot_seed}_template{n_template}"
        mid2_dir += f"_lr{datadream_lr}_ep{datadream_epoch}"
        if not datadream_train_text_encoder:
            mid2_dir += "_notextlora"
        if is_dataset_wise_model:
            mid2_dir += "_dswise"
    mid_dir = ospj(mid_dir, mid2_dir)
    save_dir_base = save_dir or args_local["save_dir"]
    save_dir = ospj(
        save_dir_base,
        dataset,
        sd_version,
        mid_dir,
    )
    print(save_dir)

    if dataset == HAND_DATASET_NAME:
        if mode != "datadream" or is_dataset_wise_model:
            raise ValueError("手部阶段 C 仅支持逐类 DataDream LoRA 生成")
        if datadream_train_text_encoder:
            raise ValueError(
                "阶段 B 权重为 notextlora，手部生成必须设置 "
                "datadream_train_text_encoder=False"
            )
        selected_prompts = HAND_GENERATION_PROMPTS
        if set(SUBSET_NAMES[dataset]) != set(selected_prompts):
            raise ValueError("手部四类与生成 prompt 映射不一致")
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        run_config = {
            "schema_version": 1,
            "task": "hand_lora_image_generation",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": dataset,
            "classes": SUBSET_NAMES[dataset],
            "seed_base": int(seed),
            "sd_version": sd_version,
            "model_dir": model_dir,
            "datadream_dir": datadream_dir,
            "n_shot": int(n_shot),
            "fewshot_seed": fewshot_seed,
            "n_template": int(n_template),
            "datadream_lr": float(datadream_lr),
            "datadream_epoch": int(datadream_epoch),
            "guidance_scale": float(guidance_scale),
            "num_inference_steps": int(num_inference_steps),
            "n_img_per_class": int(n_img_per_class),
            "count_start": int(count_start),
            "batch_size": int(bs),
            "split_idx": int(split_idx),
            "n_set_split": int(n_set_split),
            "positive_prompts": selected_prompts,
            "negative_prompts": HAND_POSE_NEGATIVE_PROMPTS,
        }
        (Path(save_dir) / "generation_config.json").write_text(
            json.dumps(run_config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # load SD pipeline
    pipe = get_pipe(sd_version, model_dir, device, is_tqdm)
    if mode == "datadream":
        if is_dataset_wise_model:
            classname = "dataset-wise"
            pipe = update_pipe(
                pipe,
                n_shot,
                n_template,
                dataset,
                datadream_dir,
                datadream_lr,
                datadream_epoch,
                datadream_train_text_encoder,
                fewshot_seed,
                classname,
            )
        else:
            # update pipe in every class
            pass

    # load instance
    generate_image = GenerateImage(
        pipe=pipe,
        device=device,
        mode=mode,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        n_img_per_class=n_img_per_class,
        save_dir=save_dir,
        count_start=count_start,
        bs=bs,
        n_shot=n_shot,
        n_template=n_template,
        dataset=dataset,
        seed=seed,
        sd_version=sd_version,
        model_dir=model_dir,
    )

    iters = SUBSET_NAMES[dataset]

    # parallel computing
    step = len(iters) // n_set_split
    start_idx = split_idx * step
    end_idx = (split_idx + 1) * step if (split_idx + 1) != n_set_split else len(iters)
    print(
        f"SPLIT!! Out of {len(SUBSET_NAMES[dataset])} pairs, we generate from idx {start_idx} to {end_idx}."
    )
    iters_partial = iters[start_idx:end_idx]

    # generate & save synthetic images
    for classname in tqdm(iters_partial, total=len(iters_partial)):
        if mode == "datadream":
            if is_dataset_wise_model:
                # update pipe just in the beginning
                pass
            else:
                # update pipe
                pipe = get_pipe(sd_version, model_dir, device, is_tqdm)
                pipe = update_pipe(
                    pipe,
                    n_shot,
                    n_template,
                    dataset,
                    datadream_dir,
                    datadream_lr,
                    datadream_epoch,
                    datadream_train_text_encoder,
                    fewshot_seed,
                    classname,
                )
                generate_image.update_pipe(pipe)

        # run
        set_seed(seed)
        generate_image.run(classname)


if __name__ == "__main__":
    fire.Fire(main)
