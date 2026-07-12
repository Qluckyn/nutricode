"""手部营养分类的图片级、姿势级和受试者级评估。"""

import csv
import json
import os
import re
from pathlib import Path

import numpy as np
import torch

from util_data import SUBSET_NAMES


HAND_DATASET = "hand_nutrition"
POSITIVE_CLASS = "malnourished_hand"
NEGATIVE_CLASS = "normal_hand"
POSE_PATTERN = re.compile(r"^(?P<subject>\d+)_(?P<pose>01|02)(?:$|[_\-.])")


def parse_subject_and_pose(path):
    """从 46_01.png 形式的文件名解析受试者ID和手部姿势。"""
    if not path:
        return "", ""
    match = POSE_PATTERN.match(os.path.basename(path))
    if not match:
        return "", ""
    return match.group("subject"), f"pose_{match.group('pose')}"


def _sample_paths_from_loader(data_loader):
    dataset = data_loader.dataset
    if hasattr(dataset, "samples"):
        return [sample[0] for sample in dataset.samples]
    if (
        hasattr(dataset, "dataset")
        and hasattr(dataset, "indices")
        and hasattr(dataset.dataset, "samples")
    ):
        return [dataset.dataset.samples[index][0] for index in dataset.indices]
    return None


def _load_requested_checkpoint(args, model, device):
    """加载显式指定的权重；纯zero-shot禁止自动读取输出目录旧checkpoint。"""
    checkpoint_path = getattr(args, "eval_ckpt", None)
    if checkpoint_path is None and getattr(args, "eval_only", False):
        print("Zero-shot未显式指定eval_ckpt，使用当前预训练CLIP，不加载旧权重")
        return None
    if checkpoint_path is None:
        checkpoint_path = os.path.join(args.output_dir, "best_checkpoint.pth")
    if not os.path.exists(checkpoint_path):
        print(f"未找到模型权重: {checkpoint_path}，使用当前模型")
        return checkpoint_path

    print(f"加载模型权重: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
    model.load_state_dict(checkpoint["model"], strict=False)
    return checkpoint_path


def _collect_predictions(model, data_loader, device):
    all_preds = []
    all_targets = []
    all_probs = []
    model.eval()
    with torch.no_grad():
        for image, label in data_loader:
            image = image.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            output = model(image)
            if isinstance(output, tuple):
                output = output[0]
            probs = torch.softmax(output, dim=1)
            preds = output.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(label.cpu())
            all_probs.append(probs.cpu())
    return (
        torch.cat(all_preds).numpy(),
        torch.cat(all_targets).numpy(),
        torch.cat(all_probs).numpy(),
    )


def _class_accuracy_records(confusion_matrix, class_names):
    records = []
    for index, class_name in enumerate(class_names):
        correct = int(confusion_matrix[index, index].item())
        total = int(confusion_matrix[index].sum().item())
        accuracy = 100.0 * correct / total if total else 0.0
        records.append(
            {
                "class": class_name,
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
            }
        )
        print(f"{class_name} [{index}]: {accuracy:.2f}% ({correct}/{total})")
    return records


def analyze_hand_predictions(args, model, data_loader, binary_metrics_fn):
    """执行手部专用评估；营养不良始终作为二分类阳性。"""
    print("\n=== 手部预测结果分析 ===")
    class_names = SUBSET_NAMES[args.dataset]
    required = {POSITIVE_CLASS, NEGATIVE_CLASS}
    if set(class_names) != required:
        raise ValueError(f"手部类别必须为 {sorted(required)}，实际为 {class_names}")
    pos_idx = class_names.index(POSITIVE_CLASS)
    normal_idx = class_names.index(NEGATIVE_CLASS)

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = _load_requested_checkpoint(args, model, device)
    all_preds, all_targets, all_probs = _collect_predictions(
        model, data_loader, device
    )

    sample_paths = _sample_paths_from_loader(data_loader)
    if sample_paths is None or len(sample_paths) != len(all_preds):
        raise RuntimeError(
            "手部评估必须取得与预测一一对应的样本路径，无法执行受试者聚合"
        )

    confusion_matrix = torch.zeros(
        args.n_classes, args.n_classes, dtype=torch.int64
    )
    for target, pred in zip(all_targets, all_preds):
        confusion_matrix[int(target), int(pred)] += 1
    class_accuracies = _class_accuracy_records(confusion_matrix, class_names)
    overall_accuracy = (
        100.0
        * confusion_matrix.diag().sum().item()
        / confusion_matrix.sum().item()
    )
    print(f"总体图片准确率: {overall_accuracy:.2f}%")

    y_true_img = (all_targets == pos_idx).astype(int)
    y_score_img = all_probs[:, pos_idx]
    image_level_metrics = binary_metrics_fn(y_true_img, y_score_img)

    parsed = []
    for index, path in enumerate(sample_paths):
        subject_id, pose = parse_subject_and_pose(path)
        if not subject_id or not pose:
            raise ValueError(f"无法解析手部受试者或姿势：{path}")
        parsed.append((subject_id, pose))

    # 姿势级指标只计算本次实验实际选择的姿势。
    hand_pose = getattr(args, "hand_pose", "all")
    required_poses = {
        "all": ("pose_01", "pose_02"),
        "01": ("pose_01",),
        "02": ("pose_02",),
    }[hand_pose]
    pose_level_metrics = {}
    pose_counts = {}
    for pose in required_poses:
        indices = [i for i, (_, current_pose) in enumerate(parsed) if current_pose == pose]
        if not indices:
            raise ValueError(f"测试集中缺少 {pose}")
        pose_level_metrics[pose] = binary_metrics_fn(
            y_true_img[indices], y_score_img[indices]
        )
        pose_counts[pose] = len(indices)

    # 使用“真实二分类标签 + 受试者ID”作为内部键，避免跨类别同号ID误合并。
    subject_buckets = {}
    for index, (subject_id, pose) in enumerate(parsed):
        binary_label = int(y_true_img[index])
        key = (binary_label, subject_id)
        bucket = subject_buckets.setdefault(
            key,
            {"label": binary_label, "subject_id": subject_id, "pose_scores": {}},
        )
        if pose in bucket["pose_scores"]:
            raise ValueError(f"受试者 {subject_id} 的 {pose} 重复")
        bucket["pose_scores"][pose] = float(y_score_img[index])

    subject_results = []
    y_true_subject = []
    y_score_subject = []
    agreement_count = 0
    complete_pair_count = 0
    for _, bucket in sorted(subject_buckets.items(), key=lambda item: int(item[0][1])):
        pose_scores = bucket["pose_scores"]
        missing = set(required_poses) - set(pose_scores)
        if missing:
            raise ValueError(
                f"受试者 {bucket['subject_id']} 缺少姿势：{sorted(missing)}"
            )
        subject_score = float(np.mean(list(pose_scores.values())))
        predicted_binary = int(subject_score >= 0.5)
        predicted_class_index = pos_idx if predicted_binary else normal_idx
        pose_predictions = {
            pose: int(score >= 0.5) for pose, score in pose_scores.items()
        }
        # 单姿势实验不存在跨姿势一致性，使用None明确表示“不适用”。
        poses_agree = (
            len(set(pose_predictions.values())) == 1
            if len(required_poses) > 1 else None
        )
        if len(required_poses) > 1:
            complete_pair_count += 1
            agreement_count += int(poses_agree)
        y_true_subject.append(bucket["label"])
        y_score_subject.append(subject_score)

        # true_label/predicted_label统一表示CLIP类别索引；
        # 医学阳性编码单独保存为positive_*_label，避免图片级与受试者级语义相反。
        true_class_index = pos_idx if bucket["label"] == 1 else normal_idx
        subject_results.append(
            {
                "subject_id": bucket["subject_id"],
                "true_label": true_class_index,
                "true_class_name": class_names[true_class_index],
                "predicted_label": predicted_class_index,
                "predicted_class_index": predicted_class_index,
                "predicted_class_name": class_names[predicted_class_index],
                "positive_true_label": bucket["label"],
                "positive_pred_label": predicted_binary,
                "malnourished_probability": subject_score,
                "pose_probabilities": pose_scores,
                "pose_predictions": pose_predictions,
                "poses_agree": poses_agree,
            }
        )

    subject_level_metrics = binary_metrics_fn(y_true_subject, y_score_subject)
    pose_agreement = {
        "applicable": len(required_poses) > 1,
        "complete_subject_pairs": complete_pair_count,
        "agreement_count": agreement_count,
        "disagreement_count": complete_pair_count - agreement_count,
        "agreement_rate": (
            agreement_count / complete_pair_count if complete_pair_count else None
        ),
    }

    all_sample_results = []
    for index, (pred, target) in enumerate(zip(all_preds, all_targets)):
        subject_id, pose = parsed[index]
        true_class_name = class_names[int(target)]
        predicted_class_name = class_names[int(pred)]
        all_sample_results.append(
            {
                "sample_id": index,
                "image_path": sample_paths[index],
                "subject_id": subject_id,
                "pose": pose,
                "true_label": int(target),
                "true_class_name": true_class_name,
                "predicted_label": int(pred),
                "predicted_class_name": predicted_class_name,
                "is_correct": bool(pred == target),
                "confidence": float(all_probs[index, int(pred)] * 100.0),
                "positive_class_name": POSITIVE_CLASS,
                "positive_true_label": int(int(target) == pos_idx),
                "positive_pred_label": int(int(pred) == pos_idx),
                "malnourished_probability": float(all_probs[index, pos_idx]),
                "normal_probability": float(all_probs[index, normal_idx]),
                "all_class_probabilities": {
                    class_names[j]: float(all_probs[index, j] * 100.0)
                    for j in range(args.n_classes)
                },
            }
        )

    detailed_results = {
        "dataset": HAND_DATASET,
        "checkpoint_path": checkpoint_path,
        "positive_class_name": POSITIVE_CLASS,
        "hand_pose": hand_pose,
        "classification_threshold": 0.5,
        "overall_accuracy": float(overall_accuracy),
        "confusion_matrix": confusion_matrix.tolist(),
        "class_accuracies": class_accuracies,
        "image_level_metrics": image_level_metrics,
        "pose_level_metrics": pose_level_metrics,
        "pose_counts": pose_counts,
        "subject_level_metrics": subject_level_metrics,
        "pose_agreement": pose_agreement,
        "subject_results": subject_results,
        "all_samples": all_sample_results,
    }
    output_path = Path(args.output_dir) / "detailed_prediction_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(detailed_results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    from hand_report_outputs import save_hand_report_outputs

    report_paths = save_hand_report_outputs(
        args,
        checkpoint_path,
        confusion_matrix,
        class_names,
        overall_accuracy,
        class_accuracies,
        image_level_metrics,
        pose_level_metrics,
        subject_level_metrics,
        pose_agreement,
        all_sample_results,
        subject_results,
    )
    print(f"任务八结果文件: {report_paths}")
    print(f"姿势数量: {pose_counts}")
    print(f"姿势预测一致性: {pose_agreement}")
    print(f"受试者级结果数: {len(subject_results)}")
    print(f"所有手部预测结果已保存至: {output_path}")

    return {
        "confusion_matrix": confusion_matrix,
        "overall_accuracy": overall_accuracy,
        "predictions": all_preds,
        "targets": all_targets,
        "probabilities": all_probs,
        "all_sample_results": all_sample_results,
        "class_accuracies": class_accuracies,
        "image_level_metrics": image_level_metrics,
        "pose_level_metrics": pose_level_metrics,
        "subject_level_metrics": subject_level_metrics,
        "pose_agreement": pose_agreement,
        "subject_results": subject_results,
    }
