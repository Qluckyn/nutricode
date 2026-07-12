
import sys
import logging
import random
import atexit
import getpass
import shutil
import time
import os
import yaml
import json
import argparse
from os.path import join as ospj

from util_data import SUBSET_NAMES

_MODEL_TYPE = ("resnet50", "clip")


class Logger(object):
    """Log stdout messages."""

    def __init__(self, outfile):
        self.terminal = sys.stdout
        self.log = open(outfile, "a")
        # 原始逻辑：sys.stdout = self.log，只写内部log.log，导致外层tee的train.log为空。
        # 使用Logger自身同时写终端和文件，保留原日志并让实验入口捕获完整控制台输出。
        sys.stdout = self

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def str2bool(v):
    if v == "":
        return None
    elif v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2none(v):
    if v is None:
        return v
    elif v.lower() in ('none', 'null'):
        return None
    else:
        return v

def int2none(v):
    if v is None or v == "":
        return v
    elif v.lower() in ('none', 'null'):
        return None
    else:
        return int(v)

def float2none(v):
    if v is None or v == "":
        return v
    elif v.lower() in ('none', 'null'):
        return None
    else:
        return float(v)

def list_int2none(vs):
    return_vs = []
    for v in vs:
        if v is None:
            pass
        elif v.lower() in ('none', 'null'):
            v = None
        else:
            v = int(v)
        return_vs.append(v)
    return return_vs


def set_local(args):
    yaml_file = "local.yaml"
    with open(yaml_file, "r") as f:
        args_local = yaml.safe_load(f)

    # 原始逻辑：真实训练集路径只能从 local.yaml 中读取。
    # args.real_train_data_dir = args_local["real_train_data_dir"][args.dataset]
    # 手部实验需要独立传入划分路径，因此优先使用命令行覆盖值。
    args.real_train_data_dir = (
        args.real_train_data_dir_override
        or args_local["real_train_data_dir"][args.dataset]
    )
    # 原始逻辑：无论是否启用 few-shot，都会强制读取对应配置。
    # args.real_train_fewshot_data_dir = ospj(
    #     args_local["real_train_fewshot_data_dir"][args.dataset],
    #     args.fewshot_seed
    # )
    # 手部真实数据基线不使用 few-shot；配置缺失时回退到真实训练目录。
    fewshot_dirs = args_local.get("real_train_fewshot_data_dir") or {}
    fewshot_root = fewshot_dirs.get(args.dataset)
    if fewshot_root:
        args.real_train_fewshot_data_dir = ospj(fewshot_root, args.fewshot_seed)
    else:
        args.real_train_fewshot_data_dir = args.real_train_data_dir
    # 原始逻辑：真实测试集路径只能从 local.yaml 中读取。
    # args.real_test_data_dir = args_local["real_test_data_dir"][args.dataset]
    # 测试集也允许覆盖，以免改写现有人脸实验的 local.yaml。
    args.real_test_data_dir = (
        args.real_test_data_dir_override
        or args_local["real_test_data_dir"][args.dataset]
    )
    # 原始逻辑会强制索引以下可选项，local.yaml 注释这些字段时会触发 KeyError。
    # args.synth_train_data_dir = args_local["synth_train_data_dir"]
    # args.metadata_dir = args_local["metadata_dir"]
    # args.clip_download_dir = args_local["clip_download_dir"]
    # args.wandb_key = args_local["wandb_key"]
    # 真实手部实验不依赖合成目录和 metadata，因此允许安全缺省。
    args.synth_train_data_dir = args_local.get("synth_train_data_dir", "")
    args.metadata_dir = args_local.get("metadata_dir", "metadata")
    args.clip_download_dir = args_local.get("clip_download_dir")
    args.wandb_key = args_local.get("wandb_key")


def set_output_dir(args):
    n_img_per_cls = "full" if args.n_img_per_cls is None else args.n_img_per_cls
    mid2 = f"n_img_per_cls_{n_img_per_cls}"
    if args.is_synth_train:
        mid3 = args.sd_version
    else:
        mid3 = "baseline"
        if args.is_pooled_fewshot:
            mid3 += f"_shot{args.n_shot}_{args.fewshot_seed}"
    if args.is_synth_train:
        if args.n_shot == 0: # zeroshot
            mid3 = ospj(mid3, f"shot{args.n_shot}_template{args.n_template}")
        else: # datadream
            mid3 = ospj(mid3, f"shot{args.n_shot}_{args.fewshot_seed}_template{args.n_template}")
            mid3 += f"_ddlr{args.datadream_lr}"
            mid3 += f"_ddep{args.datadream_epoch}"
            if not args.datadream_train_text_encoder:
                mid3 += "_notextlora"
            if args.is_dataset_wise:
                mid3 += "_dswise"
        if args.is_pooled_fewshot:
            mid3 += f"_lbd{args.lambda_1}"
    mixaug = "_mixuag" if args.is_mix_aug else ""
    mid4 = f"lr{args.lr}_wd{args.wd}{mixaug}"

    model_type = args.model_type
    if model_type == 'clip':
        model_type += args.clip_version

    args.output_dir = ospj(args.output_dir, args.dataset, model_type, mid2, mid3, mid4)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)


def set_synth_train_data_dir(args):
    if args.is_synth_train: 
        if args.n_shot == 0:
            mid_dir = f"shot{args.n_shot}_template{args.n_template}"
        else:
            mid_dir = f"shot{args.n_shot}_{args.fewshot_seed}_template{args.n_template}"
            mid_dir += f"_lr{args.datadream_lr}"
            mid_dir += f"_ep{args.datadream_epoch}"
            if not args.datadream_train_text_encoder:
                mid_dir += "_notextlora"
            if args.is_dataset_wise:
                mid_dir += "_dswise"
        args.synth_train_data_dir = ospj(
            args.synth_train_data_dir,
            args.dataset,
            args.sd_version, 
            f"gs{args.guidance_scale}_nis{args.num_inference_steps}",
            mid_dir, 
            "train",
        ) 


def set_log(output_dir):
    log_file_name = ospj(output_dir, 'log.log')
    Logger(log_file_name)


def set_wandb_group(args):
    pooled = f"pool_lbd{args.lambda_1}" if args.is_pooled_fewshot else ""
    mixaug = "_mixaug" if args.is_mix_aug else ""
    synth_setting = ""
    if args.is_synth_train: 
        if args.n_shot == 0:
            synth_setting = "zeroshot"
        else:
            synth_setting = "datadream"
            synth_setting += f"_ddlr{args.datadream_lr}"
            synth_setting += f"_ddep{args.datadream_epoch}"
            if not args.datadream_train_text_encoder:
                synth_setting += "_notl"
            if args.is_dataset_wise:
                synth_setting += "_dswise"
    model_type = args.model_type
    if model_type == 'clip':
        model_type += args.clip_version
    args.wandb_group = f"{args.dataset[:4]}_{model_type}_{pooled}_shot{args.n_shot}_{args.fewshot_seed}_{synth_setting}_gs{args.guidance_scale}_nipc{args.n_img_per_cls}_lr{args.lr}_wd{args.wd}{mixaug}"


def set_follow_up_configs(args):
    set_output_dir(args)
    # Allow overriding synth train dir (useful for one-off experiments / tau subsets)
    # If set, we use it as-is and skip the auto path construction.
    override = getattr(args, "synth_train_data_dir_override", None)
    if override:
        args.synth_train_data_dir = override
    else:
        set_synth_train_data_dir(args)
    set_log(args.output_dir)
    if args.wandb_group is None:
        set_wandb_group(args)

def get_args():
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument('--model_type', type=str2none, default=None,
                        choices=_MODEL_TYPE)
    parser.add_argument('--clip_version', type=str, default='ViT-B/16')

    # CLIP setting
    parser.add_argument("--is_lora_image", type=str2bool, default=True)
    parser.add_argument("--is_lora_text", type=str2bool, default=True)


    # Data
    parser.add_argument('--dataset', type=str, default='imagenet')
    parser.add_argument(
        "--real_train_data_dir_override",
        type=str2none,
        default=None,
        help="覆盖 local.yaml 中的真实训练集目录，供手部实验入口使用。",
    )
    parser.add_argument(
        "--real_test_data_dir_override",
        type=str2none,
        default=None,
        help="覆盖 local.yaml 中的真实测试集目录，供手部实验入口使用。",
    )
    parser.add_argument("--n_img_per_cls", type=int2none, default=100)
    parser.add_argument("--is_mix_aug", type=str2bool, default=False,
                        help="use mixup and cutmix")
    parser.add_argument(
        "--hand_pose", type=str, default="all", choices=("all", "01", "02"),
        help="手部姿势过滤：all保留_01和_02，01或02仅保留对应文件。",
    )
    parser.add_argument(
        "--use_hand_transforms", type=str2bool, default=False,
        help="是否启用保留完整双手的方形补边和轻量几何增强。",
    )
    parser.add_argument(
        "--is_hand_subject_balanced", type=str2bool, default=False,
        help="是否按受试者执行手部12:12动态平衡采样。",
    )
    parser.add_argument(
        "--hand_subjects_per_class", type=int, default=12,
        help="手部动态采样时每个类别每轮使用的受试者数。",
    )
    parser.add_argument(
        "--sampling_history_path", type=str2none, default=None,
        help="可选：保存每轮正常受试者抽样记录的JSON路径。",
    )
    # 手部探索实验可选择保留原始的“使用 test 选择最佳 checkpoint”方式。
    parser.add_argument(
        "--select_best_on_test", type=str2bool, default=True,
        help="True时每轮在test评估并选最佳模型；False时固定训练轮数后仅测试一次。",
    )
    parser.add_argument("--is_pooled_fewshot", type=str2bool, default=False)
    parser.add_argument("--lambda_1", type=float2none, default=0,
                        help="weight for loss from real/synth data")
    parser.add_argument("--fewshot_seed", type=str2none, default="seed0",
                        help="best or seed{number}.")
    parser.add_argument("--is_dataset_wise", type=str2bool, default=False)
    parser.add_argument("--datadream_lr", type=float2none, default=1e-4)
    parser.add_argument("--datadream_epoch", type=int2none, default=200)
    parser.add_argument("--datadream_train_text_encoder", type=str2bool, default=True)

    # ROI 辅助监督相关
    parser.add_argument("--use_roi_aux_head", type=str2bool, default=False,
                        help="是否启用 ROI 描述符辅助回归头")
    parser.add_argument("--alpha_roi", type=float, default=0.1,
                        help="ROI 辅助损失权重")
    parser.add_argument("--roi_descriptor_cache_path", type=str,
                        default="/root/autodl-tmp/runs/roi_descriptor_cache.json",
                        help="描述符缓存文件路径")
    parser.add_argument("--mediapipe_model_path", type=str,
                        default="/root/autodl-tmp/face_landmarker.task",
                        help="MediaPipe face_landmarker.task 文件路径")

    # stable diffusion
    parser.add_argument("--is_synth_train", type=str2bool, default=False)
    parser.add_argument("--sd_version", type=str2none, default=None)
    parser.add_argument(
        "--synth_train_data_dir_override",
        type=str2none,
        default=None,
        help=(
            "Optional: override the final synth train directory path (expects class subfolders, e.g. "
            "{malnourished_face,normal_face}). If set, skips the auto construction under local.yaml synth root."
        ),
    )

    # Evaluation-only (useful for external_test)
    parser.add_argument("--eval_only", type=str2bool, default=False,
                        help="If true, skip training and only run evaluation/prediction analysis")
    parser.add_argument("--eval_ckpt", type=str2none, default=None,
                        help="Path to checkpoint (.pth) to load for eval_only; default is <output_dir>/best_checkpoint.pth")
    parser.add_argument("--guidance_scale", type=float2none, default=2.0)
    parser.add_argument("--num_inference_steps", type=int2none, default=50)
    # for few-shot
    parser.add_argument("--n_shot", type=int2none, default=16)
    parser.add_argument("--n_template", type=int2none, default=1)


    # Training/Optimization parameters
    parser.add_argument(
        "--use_fp16",
        type=str2bool,
        default=True,
        help="Whether or not to use mixed precision for training.",
    )
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size per GPU. Total batch size is proportional to the number of GPUs.",
    )
    parser.add_argument(
        "--batch_size_eval",
        default=8,
        type=int,
    )
    parser.add_argument(
        "--epochs",
        default=100,
        type=int,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--wd",
        type=float2none,
        default=1e-4,
        help="Weight decay for the SGD optimizer.",
    )
    parser.add_argument(
        "--lr",
        default=0.1,
        type=float2none,
        help="Maximum learning rate at the end of linear warmup.",
    )
    parser.add_argument(
        "--warmup_epochs",
        default=25,
        type=int,
        help="Number of training epochs for the learning-rate-warm-up phase.",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        help="Minimum learning rate at the end of training.",
    )

    parser.add_argument(
        "--output_dir",
        default="./output",
        type=str,
        help="Path to the output folder to save logs and checkpoints.",
    )
    parser.add_argument(
        "--saveckpt_freq",
        default=100,
        type=int,
        help="Frequency of intermediate checkpointing.",
    )
    parser.add_argument(
        "--seed",
        default=22,
        type=int,
        help="Random seed",
    )
    parser.add_argument(
        "--num_workers",
        default=12,
        type=int,
        help="Number of data loading workers per GPU.",
    )
    parser.add_argument(
        "--dist_url",
        default="env://",
        type=str,
        help="Url used to set up distributed training.",
    )
    parser.add_argument(
        "--local_rank",
        default=0,
        type=int,
        help="Please ignore this argument; No need to set it manually.",
    )
    # wandb args
    parser.add_argument('--log', type=str, default='tensorboard', help='How to log')
    parser.add_argument('--wandb_entity', type=str, default='regaz', help='Wandb entity')
    parser.add_argument('--wandb_project', type=str, default='datadream', help='Wandb project name')
    parser.add_argument('--wandb_group', type=str2none, default=None, help='Name of the group for wandb runs')
    parser.add_argument('--wandb_key', default='<your_api_key_here>', type=str, help='API key for W&B.')

    args = parser.parse_args()

    set_local(args)
    set_follow_up_configs(args)

    return args


