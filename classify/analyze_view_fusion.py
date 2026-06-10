# 三视角融合策略
import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict

import numpy as np


DEFAULT_INPUT_JSON = (
    "/root/autodl-tmp/runs/ablation/classify_outputs/"
    "clip_real_plus_synth_qc_pool0.7_nipc330_lr1e-5_nomix/"
    "my_dataset/clipViT-B/16/n_img_per_cls_500/sd2.1/"
    "shot20_seed0_template1_ddlr0.0001_ddep240_lbd0.8/"
    "lr1e-05_wd0.0001_mixuag/detailed_prediction_results.json"
)

VIEW_ORDER = ("front", "left", "right")
STRATEGIES = (
    "front_only",
    "left_only",
    "right_only",
    "mean",
    "max",
    "min",
    "confidence_weighted",
)


def binary_auc_score(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(len(y_score), dtype=float)
    sorted_scores = y_score[order]
    i = 0
    while i < len(y_score):
        j = i + 1
        while j < len(y_score) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j

    sum_ranks_pos = float(ranks[pos].sum())
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def binary_metrics(y_true, y_score, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    precision = tp / max(1, tp + fp)
    sensitivity = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    f1 = (2 * precision * sensitivity) / max(1e-12, precision + sensitivity)
    mcc_num = tp * tn - fp * fn
    mcc_den = math.sqrt(max(1e-12, (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))

    return {
        "n_subjects": int(len(y_true)),
        "acc": float(acc),
        "auc": binary_auc_score(y_true, y_score),
        "f1": float(f1),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "mcc": float(mcc_num / mcc_den),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def load_subjects(samples):
    subjects = {}
    duplicate_views = defaultdict(list)
    skipped = []

    for idx, sample in enumerate(samples):
        subject_id = str(sample.get("subject_id") or "")
        view = str(sample.get("view") or "")
        if not subject_id or view not in VIEW_ORDER:
            skipped.append({"sample_index": idx, "subject_id": subject_id, "view": view})
            continue

        subject = subjects.setdefault(
            subject_id,
            {
                "subject_id": subject_id,
                "positive_true_label": int(sample["positive_true_label"]),
                "views": {},
            },
        )
        label = int(sample["positive_true_label"])
        if label != subject["positive_true_label"]:
            raise ValueError(f"Inconsistent positive_true_label for subject {subject_id}")
        if view in subject["views"]:
            duplicate_views[subject_id].append(view)

        subject["views"][view] = {
            "malnourished_prob": float(sample["malnourished_prob"]),
            "image_path": sample.get("image_path", ""),
        }

    return subjects, duplicate_views, skipped


def fuse_score(strategy, view_probs):
    if strategy.endswith("_only"):
        view = strategy.replace("_only", "")
        return view_probs.get(view)

    probs = [view_probs[v] for v in VIEW_ORDER if v in view_probs]
    if not probs:
        return None

    arr = np.asarray(probs, dtype=float)
    if strategy == "mean":
        return float(np.mean(arr))
    if strategy == "max":
        return float(np.max(arr))
    if strategy == "min":
        return float(np.min(arr))
    if strategy == "confidence_weighted":
        weights = np.abs(arr - 0.5)
        denom = float(np.sum(weights))
        if denom > 1e-8:
            return float(np.sum(weights * arr) / denom)
        return float(np.mean(arr))
    raise ValueError(f"Unsupported fusion strategy: {strategy}")


def build_subject_results(subjects, threshold=0.5):
    by_strategy = {strategy: [] for strategy in STRATEGIES}

    for subject_id in sorted(subjects):
        subject = subjects[subject_id]
        view_probs = {
            view: item["malnourished_prob"]
            for view, item in subject["views"].items()
        }
        label = int(subject["positive_true_label"])

        for strategy in STRATEGIES:
            score = fuse_score(strategy, view_probs)
            if score is None:
                continue
            by_strategy[strategy].append(
                {
                    "subject_id": subject_id,
                    "strategy": strategy,
                    "positive_true_label": label,
                    "pred_score": float(score),
                    "pred_label": int(score >= threshold),
                    "views_present": sorted(subject["views"].keys()),
                    "view_probs": {view: float(view_probs[view]) for view in sorted(view_probs)},
                }
            )

    return by_strategy


def compute_strategy_metrics(subject_results, threshold=0.5):
    rows = []
    for strategy in STRATEGIES:
        records = subject_results[strategy]
        y_true = [r["positive_true_label"] for r in records]
        y_score = [r["pred_score"] for r in records]
        metrics = binary_metrics(y_true, y_score, threshold=threshold) if records else {
            "n_subjects": 0,
            "acc": float("nan"),
            "auc": float("nan"),
            "f1": float("nan"),
            "sensitivity": float("nan"),
            "specificity": float("nan"),
            "mcc": float("nan"),
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
        }
        rows.append({"strategy": strategy, **metrics})
    return rows


def compare_mean_with_input_subject_results(mean_results, input_subject_results):
    input_scores = {
        str(row.get("subject_id")): float(row.get("pred_score"))
        for row in input_subject_results
        if row.get("subject_id") is not None and row.get("pred_score") is not None
    }
    diffs = []
    missing_in_input = []
    for row in mean_results:
        sid = row["subject_id"]
        if sid not in input_scores:
            missing_in_input.append(sid)
            continue
        diffs.append(abs(float(row["pred_score"]) - input_scores[sid]))
    max_abs_diff = max(diffs) if diffs else None
    return {
        "max_abs_diff": max_abs_diff,
        "n_compared": len(diffs),
        "missing_in_input": missing_in_input,
        "extra_in_input": sorted(set(input_scores) - {r["subject_id"] for r in mean_results}),
    }


def write_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_metrics_csv(rows, path):
    fieldnames = ["strategy", "n_subjects", "acc", "auc", "f1", "sensitivity", "specificity", "mcc", "tp", "tn", "fp", "fn"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_float(value):
    if value is None:
        return ""
    if isinstance(value, float) and not np.isfinite(value):
        return "nan"
    return f"{float(value):.6f}"


def write_metrics_md(rows, path):
    headers = ["strategy", "n_subjects", "acc", "auc", "f1", "sensitivity", "specificity", "mcc", "tp", "tn", "fp", "fn"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = []
        for key in headers:
            value = row[key]
            if isinstance(value, float):
                values.append(format_float(value))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Subject-level three-view fusion analysis from detailed_prediction_results.json.")
    parser.add_argument("--input-json", default=DEFAULT_INPUT_JSON)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    input_json_path = os.path.abspath(args.input_json)
    output_dir = os.path.abspath(args.output_dir or os.path.dirname(input_json_path))
    os.makedirs(output_dir, exist_ok=True)

    with open(input_json_path, "r") as f:
        input_json = json.load(f)

    samples = input_json.get("all_samples", [])
    subjects, duplicate_views, skipped = load_subjects(samples)

    view_counts = Counter(str(s.get("view") or "unknown") for s in samples)
    subject_view_count_distribution = Counter(len(subject["views"]) for subject in subjects.values())
    subject_results = build_subject_results(subjects, threshold=args.threshold)
    metric_rows = compute_strategy_metrics(subject_results, threshold=args.threshold)
    mean_comparison = compare_mean_with_input_subject_results(
        subject_results["mean"],
        input_json.get("subject_results", []),
    )

    print(f"sample 总数: {len(samples)}")
    print(f"subject 总数: {len(subjects)}")
    print(
        "front/left/right 数量: "
        f"front={view_counts.get('front', 0)}, "
        f"left={view_counts.get('left', 0)}, "
        f"right={view_counts.get('right', 0)}"
    )
    print(f"每个 subject 的视角数量分布: {dict(sorted(subject_view_count_distribution.items()))}")
    print(f"mean fusion vs input_json['subject_results'] pred_score max_abs_diff: {mean_comparison['max_abs_diff']}")
    if skipped:
        print(f"跳过无法用于三视角分组的 sample 数: {len(skipped)}")
    if duplicate_views:
        print(f"检测到重复 subject-view 项的 subject 数: {len(duplicate_views)}")

    subject_output = {
        "input_json": input_json_path,
        "threshold": args.threshold,
        "strategies": subject_results,
    }
    summary = {
        "input_json": input_json_path,
        "threshold": args.threshold,
        "n_samples": len(samples),
        "n_subjects": len(subjects),
        "view_counts": {
            "front": int(view_counts.get("front", 0)),
            "left": int(view_counts.get("left", 0)),
            "right": int(view_counts.get("right", 0)),
            "unknown": int(view_counts.get("unknown", 0)),
        },
        "subject_view_count_distribution": {str(k): int(v) for k, v in sorted(subject_view_count_distribution.items())},
        "mean_subject_results_comparison": mean_comparison,
        "skipped_samples": skipped,
        "duplicate_views": {sid: views for sid, views in duplicate_views.items()},
        "metrics": metric_rows,
    }

    subject_results_path = os.path.join(output_dir, "view_fusion_subject_results.json")
    metrics_csv_path = os.path.join(output_dir, "view_fusion_metrics.csv")
    summary_json_path = os.path.join(output_dir, "view_fusion_summary.json")
    metrics_md_path = os.path.join(output_dir, "view_fusion_metrics.md")

    write_json(subject_output, subject_results_path)
    write_metrics_csv(metric_rows, metrics_csv_path)
    write_json(summary, summary_json_path)
    write_metrics_md(metric_rows, metrics_md_path)

    print(f"saved: {subject_results_path}")
    print(f"saved: {metrics_csv_path}")
    print(f"saved: {summary_json_path}")
    print(f"saved: {metrics_md_path}")


if __name__ == "__main__":
    main()
