"""保存手部分类任务八要求的配置、指标、CSV和混淆矩阵图。"""

import csv
import json
from pathlib import Path


def save_hand_report_outputs(
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
):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_payload = {
        "dataset": "hand_nutrition",
        "positive_class_name": "malnourished_hand",
        "hand_pose": getattr(args, "hand_pose", "all"),
        "classification_threshold": 0.5,
        "overall_accuracy_percent": float(overall_accuracy),
        "confusion_matrix": confusion_matrix.tolist(),
        "class_accuracies": class_accuracies,
        "image_level_metrics": image_level_metrics,
        "pose_level_metrics": pose_level_metrics,
        "subject_level_metrics": subject_level_metrics,
        "pose_agreement": pose_agreement,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # 保存最终解析后的运行参数，便于复现实验。
    config_payload = dict(vars(args))
    config_payload["resolved_checkpoint_path"] = checkpoint_path
    config_path = output_dir / "config.json"
    config_path.write_text(
        json.dumps(config_payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    image_csv_path = output_dir / "image_predictions.csv"
    image_fields = [
        "sample_id",
        "image_path",
        "subject_id",
        "pose",
        "true_label",
        "true_class_name",
        "predicted_label",
        "predicted_class_name",
        "is_correct",
        "confidence",
        "positive_true_label",
        "positive_pred_label",
        "malnourished_probability",
        "normal_probability",
    ]
    with image_csv_path.open("w", newline="", encoding="utf-8-sig") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=image_fields)
        writer.writeheader()
        for record in all_sample_results:
            writer.writerow({field: record.get(field) for field in image_fields})

    subject_csv_path = output_dir / "subject_predictions.csv"
    subject_fields = [
        "subject_id",
        "true_label",
        "true_class_name",
        "predicted_label",
        "predicted_class_name",
        "positive_true_label",
        "positive_pred_label",
        "malnourished_probability",
        "pose_01_probability",
        "pose_02_probability",
        "pose_01_prediction",
        "pose_02_prediction",
        "poses_agree",
    ]
    with subject_csv_path.open("w", newline="", encoding="utf-8-sig") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=subject_fields)
        writer.writeheader()
        for record in subject_results:
            writer.writerow(
                {
                    "subject_id": record["subject_id"],
                    "true_label": record["true_label"],
                    "true_class_name": record["true_class_name"],
                    "predicted_label": record["predicted_label"],
                    "predicted_class_name": record["predicted_class_name"],
                    "positive_true_label": record["positive_true_label"],
                    "positive_pred_label": record["positive_pred_label"],
                    "malnourished_probability": record["malnourished_probability"],
                    "pose_01_probability": record["pose_probabilities"].get("pose_01"),
                    "pose_02_probability": record["pose_probabilities"].get("pose_02"),
                    "pose_01_prediction": record["pose_predictions"].get("pose_01"),
                    "pose_02_prediction": record["pose_predictions"].get("pose_02"),
                    "poses_agree": record["poses_agree"],
                }
            )

    # 服务器环境使用Agg后端，不依赖图形界面。
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(5, 4))
    matrix_array = confusion_matrix.numpy()
    image_artist = axis.imshow(matrix_array, cmap="Blues")
    figure.colorbar(image_artist, ax=axis)
    axis.set_xticks(range(len(class_names)), labels=class_names, rotation=20)
    axis.set_yticks(range(len(class_names)), labels=class_names)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title("Hand nutrition confusion matrix")
    for row in range(len(class_names)):
        for column in range(len(class_names)):
            axis.text(
                column,
                row,
                int(matrix_array[row, column]),
                ha="center",
                va="center",
            )
    figure.tight_layout()
    confusion_matrix_path = output_dir / "confusion_matrix.png"
    figure.savefig(confusion_matrix_path, dpi=160)
    plt.close(figure)

    return {
        "metrics": str(metrics_path),
        "config": str(config_path),
        "image_predictions": str(image_csv_path),
        "subject_predictions": str(subject_csv_path),
        "confusion_matrix": str(confusion_matrix_path),
    }
