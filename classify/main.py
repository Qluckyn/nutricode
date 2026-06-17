# 修改了analyze_predictions() 的输出
import os
import sys
import json
import time
import math
import random
import datetime
import traceback
import numpy as np
from pathlib import Path
from os.path import join as ospj
import wandb
import re

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2

from utils import (
    fix_random_seeds,
    cosine_scheduler,
    MetricLogger,
)

from config import get_args
from data import get_data_loader, get_synth_train_data_loader
from models.clip import CLIP
from models.resnet50 import ResNet50
from util_data import SUBSET_NAMES


def _infer_roi_descriptor_dirs(args):
    # 归一化统计只从真实训练目录计算；raw/qc 合成目录用于增量补充缓存。
    default_raw = (
        "/root/autodl-tmp/datadream_outputs/generated_images/my_dataset/"
        "sd2.1/gs3.5_nis50/shot20_seed0_template1_lr0.0001_ep240/train"
    )
    synth_raw_dir = os.environ.get("SYNTH_RAW_DIR") or default_raw
    synth_qc_dir = os.environ.get("SYNTH_QC_DIR")
    if not synth_qc_dir:
        if synth_raw_dir.endswith("/train"):
            synth_qc_dir = synth_raw_dir[:-len("/train")] + "/filtered_train"
        else:
            synth_qc_dir = os.path.join(os.path.dirname(synth_raw_dir), "filtered_train")
    return [args.real_train_data_dir, synth_raw_dir, synth_qc_dir]


#加载真实数据
def load_data_loader(args, descriptor_cache=None):
    train_loader, test_loader = get_data_loader(
        real_train_data_dir=args.real_train_data_dir,
        real_test_data_dir=args.real_test_data_dir,
        metadata_dir=args.metadata_dir,
        dataset=args.dataset, 
        bs=args.batch_size,
        eval_bs=args.batch_size_eval,
        n_img_per_cls=args.n_img_per_cls,
        is_synth_train=args.is_synth_train,
        n_shot=args.n_shot,
        real_train_fewshot_data_dir=args.real_train_fewshot_data_dir,
        is_pooled_fewshot=args.is_pooled_fewshot,
        model_type=args.model_type,
        descriptor_cache=descriptor_cache,
    )
    return train_loader, test_loader


#加载合成数据
def load_synth_train_data_loader(args, descriptor_cache=None):
    synth_train_loader = get_synth_train_data_loader(
        synth_train_data_dir=args.synth_train_data_dir,
        bs=args.batch_size,
        n_img_per_cls=args.n_img_per_cls,
        dataset=args.dataset,
        n_shot=args.n_shot,
        real_train_fewshot_data_dir=args.real_train_fewshot_data_dir,
        is_pooled_fewshot=args.is_pooled_fewshot,
        model_type=args.model_type,
        descriptor_cache=descriptor_cache,
    )
    return synth_train_loader

#训练+评估
def main(args):
    args.n_classes = len(SUBSET_NAMES[args.dataset])

    print(f"[INFO] model_type={args.model_type}")
    if args.model_type == "clip":
        print(f"[INFO] clip_version={args.clip_version}")

    os.makedirs(args.output_dir, exist_ok=True)

    fix_random_seeds(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    cudnn.benchmark = True

    # ==================================================
    # ROI descriptor cache
    # ==================================================
    descriptor_cache = None
    if getattr(args, "use_roi_aux_head", False) and not getattr(args, "eval_only", False):
        from roi_descriptor import ROIDescriptorCache

        descriptor_cache = ROIDescriptorCache(
            cache_path=args.roi_descriptor_cache_path,
            model_path=args.mediapipe_model_path,
        )
        descriptor_cache.build(_infer_roi_descriptor_dirs(args))

    # ==================================================
    # Data loader
    # ==================================================
    train_loader, val_loader = load_data_loader(args, descriptor_cache=descriptor_cache)
    if (not getattr(args, "eval_only", False)) and args.is_synth_train:
        train_loader = load_synth_train_data_loader(args, descriptor_cache=descriptor_cache)

        
    # ==================================================
    # Model and optimizer
    # ==================================================
    if args.model_type == "clip":
        # TODO
        model = CLIP(
            dataset=args.dataset,
            is_lora_image=args.is_lora_image,
            is_lora_text=args.is_lora_text,
            clip_download_dir=args.clip_download_dir,
            clip_version=args.clip_version,
            use_roi_aux_head=getattr(args, "use_roi_aux_head", False),
        )
        params_groups = model.learnable_params()
    elif args.model_type == "resnet50": 
        model = ResNet50(n_classes=args.n_classes)
        params_groups = model.parameters()

    model = model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()

    # ==================================================
    # Eval-only shortcut (external_test)
    # ==================================================
    if getattr(args, "eval_only", False):
        print("=> Eval-only mode: skipping training")
        _ = analyze_predictions(args, model, val_loader)
        return

    # CutMix and MixUp augmentation
    if args.is_mix_aug:
        cutmix = v2.CutMix(num_classes=args.n_classes)
        mixup = v2.MixUp(num_classes=args.n_classes)
        cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
    else:
        cutmix_or_mixup = None

    scheduler = None
    optimizer = torch.optim.AdamW(
        params_groups, lr=args.lr, weight_decay=args.wd,
    )
    args.lr_schedule = cosine_scheduler(
        args.lr,
        args.min_lr,
        args.epochs,
        len(train_loader),
        warmup_epochs=args.warmup_epochs,
        start_warmup_value=args.min_lr,
    )

    fp16_scaler = None
    if args.use_fp16:
        # mixed precision training
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ==================================================
    # Loading previous checkpoint & initializing tensorboard
    # ==================================================

    if args.log == 'wandb':
        assert wandb is not None, "Wandb not installed, please install it or run without wandb"
        _ = os.system('wandb login {}'.format(args.wandb_key))
        os.environ['WANDB_API_KEY'] = args.wandb_key
        wandb.init(
            project=args.wandb_project, 
            group=args.wandb_group, 
            name=args.wandb_group,
            settings=wandb.Settings(start_method='fork'),
            config=vars(args)
        )
        args.wandb_url = wandb.run.get_url()
    elif args.log == 'tensorboard':
        from torch.utils.tensorboard import SummaryWriter
        tb_dir = os.path.join(args.output_dir, "tb-{}".format(args.local_rank))
        Path(tb_dir).mkdir(parents=True, exist_ok=True)
        tb_writer = SummaryWriter(tb_dir, flush_secs=30)

    # ==================================================
    # Training
    # ==================================================
    print("=> Training starts ...")
    start_time = time.time()

    best_stats = {}
    best_top1 = 0.

    for epoch in range(0, args.epochs):
        train_stats, best_stats, best_top1 = train_one_epoch(
            model, criterion, train_loader, optimizer, scheduler, epoch, fp16_scaler, cutmix_or_mixup, args,
            val_loader, best_stats, best_top1, 
        )

#         if args.dataset in ("imagenet", "sun397"):
#             # evaluate ten times in each epoch
#             # here we only save train stats
#             if args.log == 'wandb':
#                 train_stats.update({"epoch": epoch})
#                 wandb.log(train_stats)
#         else:
        # ============ evaluate model ... ============
        test_stats = eval(
            model, criterion, val_loader, epoch, fp16_scaler, args)

        # ============ saving logs and model checkpoint ... ============
        if test_stats["test/top1"] > best_top1:
            best_top1 = test_stats["test/top1"]
            best_stats = test_stats
            save_model(args, model, optimizer, epoch, fp16_scaler, "best_checkpoint.pth")

        if epoch + 1 == args.epochs:
            test_stats['test/best_top1'] = best_stats["test/top1"]
            test_stats['test/best_loss'] = best_stats["test/loss"]

        if args.log == 'wandb':
            train_stats.update({"epoch": epoch})
            wandb.log(train_stats)
            wandb.log(test_stats)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))
    # 在训练结束后添加预测分析
    print("\n=> 加载最佳模型进行预测分析...")
    prediction_stats = analyze_predictions(args, model, val_loader)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))

    # 详细结果已在analyze_predictions函数中保存
    # 这里可以选择保存一个简明版本
    summary_results = {
        "overall_accuracy": float(prediction_stats["overall_accuracy"]),
        "class_accuracies": prediction_stats["class_accuracies"],
    }


def train_one_epoch(
    model, criterion, data_loader, optimizer, scheduler, epoch, fp16_scaler, cutmix_or_mixup, args,
    val_loader, best_stats, best_top1,
):
    metric_logger = MetricLogger(delimiter="  ")
    header = "Epoch: [{}/{}]".format(epoch, args.epochs)

    model.train()

    for it, batch in enumerate(
        metric_logger.log_every(data_loader, 100, header)
    ):
        descriptor = None
        has_descriptor = False
        is_real = None
        if args.is_synth_train and args.is_pooled_fewshot:
            if len(batch) == 4:
                image, label, is_real, descriptor = batch
                has_descriptor = True
            else:
                image, label, is_real = batch
        else:
            if len(batch) == 3:
                image, label, descriptor = batch
                has_descriptor = True
            else:
                image, label = batch
        descriptor_matches_image = has_descriptor

        label_origin = label
        label_origin = label_origin.cuda(non_blocking=True)

        # apply CutMix and MixUp augmentation
        if args.is_mix_aug:
            p = random.random()
            if p >= 0.2:
                pass
            else:
                if args.is_synth_train and args.is_pooled_fewshot:
                    new_image = torch.zeros_like(image)
                    new_label = torch.stack([torch.zeros_like(label)] * args.n_classes, dim=1).mul(1.0)

                    real_mask = (is_real == 1)
                    synth_mask = (is_real == 0)

                    image_real, label_real = image[real_mask], label[real_mask]
                    image_synth, label_synth = image[synth_mask], label[synth_mask]

                    if image_real.numel() > 0:
                        image_real, label_real = cutmix_or_mixup(image_real, label_real)
                    if image_synth.numel() > 0:
                        image_synth, label_synth = cutmix_or_mixup(image_synth, label_synth)

                    if image_real.numel() > 0:
                        new_image[real_mask] = image_real
                        new_label[real_mask] = label_real
                    if image_synth.numel() > 0:
                        new_image[synth_mask] = image_synth
                        new_label[synth_mask] = label_synth

                    image = new_image
                    label = new_label
                    descriptor_matches_image = False

                else:
                    image, label = cutmix_or_mixup(image, label)
                    descriptor_matches_image = False
            

        it = len(data_loader) * epoch + it  # global training iteration

        image = image.squeeze(1).cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)

        # update weight decay and learning rate according to their schedule
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = args.lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = args.wd

        # forward pass
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            roi_pred = None
            if getattr(args, "use_roi_aux_head", False):
                logit, roi_pred = model(image)
            else:
                logit = model(image)

            if args.is_synth_train and args.is_pooled_fewshot:
                real_mask = (is_real == 1)
                synth_mask = (is_real == 0)

                has_real = bool(real_mask.any().item())
                has_synth = bool(synth_mask.any().item())

                if has_real and has_synth:
                    loss_real = criterion(logit[real_mask], label[real_mask])
                    loss_synth = criterion(logit[synth_mask], label[synth_mask])
                    loss = args.lambda_1 * loss_real + (1 - args.lambda_1) * loss_synth
                elif has_real:
                    loss = criterion(logit[real_mask], label[real_mask])
                elif has_synth:
                    loss = criterion(logit[synth_mask], label[synth_mask])
                else:
                    loss = criterion(logit, label)
            else:
                loss = criterion(logit, label)

            if (
                getattr(args, "use_roi_aux_head", False)
                and descriptor_matches_image
                and descriptor is not None
                and roi_pred is not None
            ):
                descriptor = descriptor.cuda(non_blocking=True)
                valid_mask = ~torch.isnan(descriptor).any(dim=1)
                if valid_mask.sum() > 0:
                    loss_roi = F.mse_loss(roi_pred[valid_mask], descriptor[valid_mask])
                    loss = loss + args.alpha_roi * loss_roi
                    metric_logger.update(loss_roi=loss_roi.item())

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        # parameter update
        optimizer.zero_grad()
        if fp16_scaler is None:
            loss.backward()
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # logging
        with torch.no_grad():
            #acc1, acc5 = get_accuracy(logit.detach(), label_origin, topk=(1, 5))
            acc1, = get_accuracy(logit.detach(), label_origin, topk=(1,)) 
            metric_logger.update(top1=acc1.item())
            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

        if scheduler is not None:
            scheduler.step()

#         # eval in the middle if there's too much iterations
#         if args.dataset in ("imagenet", "sun397") and it % (len(data_loader) // 10) == 0:
#             test_stats = eval(
#                 model, criterion, val_loader, epoch, fp16_scaler, args)
#             if test_stats["test/top1"] > best_top1:
#                 best_top1 = test_stats["test/top1"]
#                 best_stats = test_stats
#                 save_model(args, model, optimizer, epoch, fp16_scaler, "best_checkpoint.pth")
#             if epoch + 1 == args.epochs:
#                 test_stats['test/best_top1'] = best_stats["test/top1"]
#                 test_stats['test/best_loss'] = best_stats["test/loss"]
#             if args.log == 'wandb':
#                 wandb.log(test_stats)
#             model.train()

#         if it % len(data_loader) == 5:
#             break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged train stats:", metric_logger)

    return {"train/{}".format(k): meter.global_avg for k, meter in metric_logger.meters.items()}, best_stats, best_top1


@torch.no_grad()
def eval(model, criterion, data_loader, epoch, fp16_scaler, args):
    metric_logger = MetricLogger(delimiter="  ")
    header = "Epoch: [{}/{}]".format(epoch, args.epochs)

    is_last = epoch + 1 == args.epochs
    if is_last:
        targets = []
        outputs = []

    model.eval()

    for it, (image, label) in enumerate(
        metric_logger.log_every(data_loader, 100, header)
    ):

        image = image.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            output = model(image)
            if isinstance(output, tuple):
                output = output[0]
            loss = criterion(output, label)

        #acc1, acc5 = get_accuracy(output, label, topk=(1, 5))
        acc1, = get_accuracy(output, label, topk=(1,))

        # record logs
        metric_logger.update(loss=loss.item())
        metric_logger.update(top1=acc1.item())
        #metric_logger.update(top5=acc5.item())

        if is_last:
            targets.append(label)
            outputs.append(output)

    metric_logger.synchronize_between_processes()
    print("Averaged test stats:", metric_logger)

    stat_dict = {"test/{}".format(k): meter.global_avg for k, meter in metric_logger.meters.items()}

    if is_last:
        targets = torch.cat(targets)
        outputs = torch.cat(outputs)

        # calculate per class accuracy
        acc_per_class = [
            get_accuracy(outputs[targets == cls_idx], targets[targets == cls_idx], topk=(1,))[0].item() 
            for cls_idx in range(args.n_classes)
        ]
        for cls_idx, acc in enumerate(acc_per_class):
            print("{} [{}]: {}".format(SUBSET_NAMES[args.dataset][cls_idx], cls_idx, str(acc)))
            stat_dict[SUBSET_NAMES[args.dataset][cls_idx] + '_cls-acc'] = acc

    return stat_dict


def get_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    
    num_classes = output.size(1)
    maxk = min(maxk, num_classes)
    
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        if k <= num_classes:
            res.append(correct[:k].reshape(-1).float().sum(0) * 100.0 / batch_size)
        else:
            # 如果k大于类别数，返回top-1的准确率
            res.append(correct[:1].reshape(-1).float().sum(0) * 100.0 / batch_size)
    
    return res


def _parse_subject_id_from_path(path: str) -> str:
    if not path:
        return ""
    name = os.path.basename(path)
    m = re.match(r"^(\d+)_", name)
    return m.group(1) if m else ""


def _parse_view_from_path(path: str) -> str:
    if not path:
        return ""

    name = os.path.basename(path)
    m = re.match(r"^\d+_(01|02|03)(?:$|[_\-.])", name)
    if m:
        return {
            "01": "front",
            "02": "left",
            "03": "right",
        }[m.group(1)]

    path_lower = path.lower()
    if "front" in path_lower:
        return "front"
    if "left" in path_lower:
        return "left"
    if "right" in path_lower:
        return "right"
    return ""


def _binary_auc_score(y_true, y_score):
    # Rank-based AUC (Mann–Whitney). Returns NaN if only one class present.
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(y_score) + 1)
    sum_ranks_pos = ranks[pos].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _binary_metrics(y_true, y_score, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    spe = tn / max(1, (tn + fp))
    f1 = (2 * prec * rec) / max(1e-12, (prec + rec))
    mcc_num = (tp * tn - fp * fn)
    mcc_den = math.sqrt(max(1e-12, (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = mcc_num / mcc_den

    auc = _binary_auc_score(y_true, y_score)
    return {
        "acc": float(acc),
        "auc": float(auc),
        "f1": float(f1),
        "sen": float(rec),
        "spe": float(spe),
        "mcc": float(mcc),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }

@torch.no_grad()
def analyze_predictions(args, model, data_loader):
    """在训练结束后加载最佳模型权重并分析预测结果"""
    print("\n=== 预测结果分析 ===")
    
    # 加载最佳模型权重
    checkpoint_path = getattr(args, "eval_ckpt", None) or os.path.join(args.output_dir, "best_checkpoint.pth")
    if os.path.exists(checkpoint_path):
        print(f"加载最佳模型权重: {checkpoint_path}")
        try:
            from torch.serialization import add_safe_globals
            
            add_safe_globals([np._core.multiarray.scalar])
            
            checkpoint = torch.load(checkpoint_path, map_location="cuda")
        except Exception as e:
            checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
            
        # 加载模型权重
        model.load_state_dict(checkpoint["model"], strict=False)
    else:
        print(f"未找到最佳模型权重: {checkpoint_path}，使用当前模型")
    
    model.eval()
    
    # 收集所有预测和真实标签
    all_preds = []
    all_targets = []
    all_probs = [] 
    
    for batch_idx, (image, label) in enumerate(data_loader):
        image = image.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)
        
        # 计算输出（CLIP/ResNet 统一走 logits 输出）
        output = model(image)
        if isinstance(output, tuple):
            output = output[0]
        
        # 获取预测概率和类别
        probs = torch.nn.functional.softmax(output, dim=1)
        _, preds = output.max(1)
        
        all_preds.append(preds.cpu())
        all_targets.append(label.cpu())
        all_probs.append(probs.cpu())
    
    # 合并批次结果
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    all_probs = torch.cat(all_probs).numpy()
    
    # 创建混淆矩阵
    confusion_matrix = torch.zeros(args.n_classes, args.n_classes, dtype=torch.int)
    for t, p in zip(all_targets, all_preds):
        confusion_matrix[t, p] += 1
    
    class_names = SUBSET_NAMES[args.dataset]
    
    # 计算每个类别的准确率
    class_accuracies = []
    print("\n每个类别的准确率:")
    for i, cls_name in enumerate(class_names):
        correct = confusion_matrix[i, i].item()
        total = confusion_matrix[i].sum().item()
        accuracy = 100.0 * correct / total if total > 0 else 0
        class_accuracies.append({"class": cls_name, "accuracy": accuracy, "correct": int(correct), "total": int(total)})
        print(f"{cls_name} [{i}]: {accuracy:.2f}% ({correct}/{total})")
    
    # 计算总体准确率
    overall_accuracy = 100.0 * confusion_matrix.diag().sum().item() / confusion_matrix.sum().item()
    print(f"\n总体准确率: {overall_accuracy:.2f}%")

    # 计算 image-level 二分类指标（malnourished 作为正类）
    if "malnourished_face" in class_names:
        pos_idx = class_names.index("malnourished_face")
    else:
        pos_idx = 1 if args.n_classes > 1 else 0
    normal_idx = class_names.index("normal_face") if "normal_face" in class_names else None
    y_true_img = (all_targets == pos_idx).astype(int)
    y_score_img = all_probs[:, pos_idx]
    image_level_metrics = _binary_metrics(y_true_img, y_score_img)

    # subject-level 聚合（若能拿到样本路径）
    subject_level_metrics = None
    subject_results = []
    sample_paths = None
    dset = data_loader.dataset
    try:
        if hasattr(dset, "samples"):
            sample_paths = [s[0] for s in dset.samples]
        elif hasattr(dset, "dataset") and hasattr(dset, "indices") and hasattr(dset.dataset, "samples"):
            sample_paths = [dset.dataset.samples[i][0] for i in dset.indices]
    except Exception:
        sample_paths = None

    if sample_paths is not None and len(sample_paths) == len(all_preds):
        subj_scores = {}
        subj_labels = {}
        for i, path in enumerate(sample_paths):
            sid = _parse_subject_id_from_path(path)
            if not sid:
                continue
            subj_scores.setdefault(sid, []).append(float(y_score_img[i]))
            subj_labels.setdefault(sid, []).append(int(y_true_img[i]))

        y_true_sub = []
        y_score_sub = []
        for sid, scores in subj_scores.items():
            label_list = subj_labels.get(sid, [])
            if not label_list:
                continue
            subj_label = int(round(float(np.mean(label_list))))
            subj_score = float(np.mean(scores))
            y_true_sub.append(subj_label)
            y_score_sub.append(subj_score)
            subject_results.append({
                "subject_id": sid,
                "true_label": subj_label,
                "pred_score": subj_score,
            })

        if y_true_sub:
            subject_level_metrics = _binary_metrics(y_true_sub, y_score_sub)
    
    # 创建所有样本的详细结果
    all_sample_results = []
    sample_paths_aligned = sample_paths is not None and len(sample_paths) == len(all_preds)
    parsed_subject_count = 0
    parsed_view_count = 0
    view_counts = {"front": 0, "left": 0, "right": 0, "unknown": 0}

    for i, (pred, target) in enumerate(zip(all_preds, all_targets)):
        is_correct = (pred == target)
        confidence = all_probs[i][pred] * 100
        image_path = sample_paths[i] if sample_paths_aligned else ""
        subject_id = _parse_subject_id_from_path(image_path)
        view = _parse_view_from_path(image_path)
        true_class_name = class_names[target]
        predicted_class_name = class_names[pred]
        normal_prob = float(all_probs[i][normal_idx]) if normal_idx is not None else None

        if subject_id:
            parsed_subject_count += 1
        if view:
            parsed_view_count += 1
            view_counts[view] = view_counts.get(view, 0) + 1
        else:
            view_counts["unknown"] += 1

        sample_result = {
            "sample_id": i,
            "true_label": int(target),
            "true_class_name": true_class_name,
            "predicted_label": int(pred),
            "predicted_class_name": predicted_class_name,
            "is_correct": bool(is_correct),
            "confidence": float(confidence),
            "image_path": image_path,
            "subject_id": subject_id,
            "view": view,
            "positive_class_name": "malnourished_face",
            "positive_true_label": int(true_class_name == "malnourished_face"),
            "positive_pred_label": int(predicted_class_name == "malnourished_face"),
            "malnourished_prob": float(all_probs[i][pos_idx]),
            "normal_prob": normal_prob,
            "all_class_probabilities": {class_names[j]: float(all_probs[i][j] * 100) for j in range(args.n_classes)}
        }
        all_sample_results.append(sample_result)
    
    # 输出一些高置信度但错误的预测
    print("\n高置信度错误预测:")
    high_conf_errors = []
    
    for i, (pred, target) in enumerate(zip(all_preds, all_targets)):
        if pred != target:
            confidence = all_probs[i][pred] * 100
            high_conf_errors.append((i, target, pred, confidence))
    
    # 按置信度排序
    high_conf_errors.sort(key=lambda x: x[3], reverse=True)
    
    # 输出前5个
    print(f"{'样本序号':<10}{'真实标签':<15}{'预测标签':<15}{'置信度':<10}")
    for i, target, pred, conf in high_conf_errors[:5]:
        print(f"{i:<10}{class_names[target]:<15}{class_names[pred]:<15}{conf:.2f}%")

    print("\n样本路径与视角解析自检:")
    print(f"sample_paths 可用: {sample_paths is not None}")
    print(f"sample_paths 数量等于预测数量: {sample_paths_aligned} ({len(sample_paths) if sample_paths is not None else 0}/{len(all_preds)})")
    print(f"成功解析 subject_id 的样本数: {parsed_subject_count}/{len(all_preds)}")
    print(f"成功解析 view 的样本数: {parsed_view_count}/{len(all_preds)}")
    print(
        "视角数量: "
        f"front={view_counts.get('front', 0)}, "
        f"left={view_counts.get('left', 0)}, "
        f"right={view_counts.get('right', 0)}, "
        f"unknown={view_counts.get('unknown', 0)}"
    )
    
    # 将所有结果保存到JSON文件
    detailed_results = {
        "overall_accuracy": float(overall_accuracy),
        "confusion_matrix": confusion_matrix.tolist(),
        "class_accuracies": class_accuracies,
        "image_level_metrics": image_level_metrics,
        "subject_level_metrics": subject_level_metrics,
        "subject_results": subject_results,
        "all_samples": all_sample_results,
    }
    
    # 保存详细结果
    detailed_json_path = os.path.join(args.output_dir, "detailed_prediction_results.json")
    with open(detailed_json_path, "w") as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"\n所有预测结果已保存至: {detailed_json_path}")
    
    return {
        "confusion_matrix": confusion_matrix,
        "overall_accuracy": overall_accuracy,
        "predictions": all_preds,
        "targets": all_targets,
        "probabilities": all_probs,
        "all_sample_results": all_sample_results,  # 添加所有样本的详细结果
        "class_accuracies": class_accuracies       # 添加每个类的准确率统计
    }

def save_model(args, model, optimizer, epoch, fp16_scaler, file_name):
    state_dict = model.state_dict()
    save_dict = {
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "epoch": epoch + 1,
        "args": args,
    }
    if fp16_scaler is not None:
        save_dict["fp16_scaler"] = fp16_scaler.state_dict()
    torch.save(save_dict, os.path.join(args.output_dir, file_name))


if __name__ == "__main__":
    try:
        args = get_args()
        main(args)
    except Exception as e:
        print(traceback.format_exc())
