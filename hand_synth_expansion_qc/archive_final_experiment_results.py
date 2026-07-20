#!/usr/bin/env python3
"""归档手部营养状态论文主实验的逐折指标与关键配置。

不复制模型权重、训练日志和图像，仅将可用于论文复核的 metrics、config 与
详细预测结果统一保存到项目目录，便于后续清理外部历史运行目录。
"""

from __future__ import annotations

import csv
import glob
import json
import shutil
import statistics
from pathlib import Path


RUNS_ROOT = Path("/root/autodl-tmp/runs")
QC_ROOT = RUNS_ROOT / "hand_synth_expansion_qc"
ARCHIVE_ROOT = Path(__file__).resolve().parent / "final_experiment_archive"
METRIC_KEYS = (
    "acc",
    "balanced_accuracy",
    "mcc",
    "f1",
    "sensitivity",
    "specificity",
)


def one_match(pattern: str) -> Path:
    """每个方法每折必须且只能使用一个指标文件，避免混入历史运行。"""
    matches = [Path(path) for path in glob.glob(pattern, recursive=True)]
    if len(matches) != 1:
        raise RuntimeError(f"指标文件数量异常（期望 1，实际 {len(matches)}）：{pattern}\n{matches}")
    return matches[0]


def c0_40(fold: int) -> Path:
    if fold == 0:
        return one_match(str(RUNS_ROOT / "hand_pose01_pose02_fold0/fold0_c0_prompt/**/metrics.json"))
    return one_match(str(RUNS_ROOT / f"hand_pose01_pose02_5fold_v4pilot/fold{fold}_c0/**/metrics.json"))


def c0_sweep(epoch: int, fold: int) -> Path:
    return one_match(str(QC_ROOT / f"c0_epoch_sweep_5fold_seed22/fold_{fold}/c0_real_{epoch}ep/seed22/**/metrics.json"))


def c0_step(fold: int) -> Path:
    """无 A-ROI 的 160 epoch 真实数据步数匹配对照。"""
    return one_match(
        str(QC_ROOT / f"c0_stepmatched_clean_no_roi_5fold_seed22/fold_{fold}/seed22/**/metrics.json")
    )


def c1_matched(fold: int, method: str) -> Path:
    return one_match(str(QC_ROOT / f"c1_matched_5fold_seed22/fold_{fold}/{method}/classifier_runs_c1_matched_textlora/c1_raw/seed22/**/metrics.json"))


def c2_route(fold: int, route: str, condition: str) -> Path:
    if fold == 0:
        return one_match(str(QC_ROOT / f"c2_qc_scaleup/fold_0/{route}/classifier_runs_c2_md_soft_textlora/{condition}/seed22/**/metrics.json"))
    return one_match(str(QC_ROOT / f"c2_md_5fold_seed22/fold_{fold}/{route}/classifier_runs_c2_md_separable_textlora/{condition}/seed22/**/metrics.json"))


EXPERIMENTS = {
    "C0-real-epoch": c0_40,
    "C0-real-60ep": lambda fold: c0_sweep(60, fold),
    "C0-real-80ep": lambda fold: c0_sweep(80, fold),
    "C0-real-120ep": lambda fold: c0_sweep(120, fold),
    "C0-real-step": c0_step,
    "SD2.1-I2I-noCN-C1-matched": lambda fold: c1_matched(fold, "sd21_i2i_no_cn"),
    "SDXL-I2I-noCN-C1-matched": lambda fold: c1_matched(fold, "sdxl_i2i_no_cn"),
    "DataDream-I2I-noCN-C1-matched": lambda fold: c2_route(fold, "datadream_i2i_no_cn", "c1_raw"),
    "DataDream-I2I-noCN-C2-soft": lambda fold: c2_route(fold, "datadream_i2i_no_cn", "c2_qc"),
    "FoundHand-DataDream-I2I-noCN-C1-matched": lambda fold: c2_route(fold, "foundhand_datadream_i2i_no_cn", "c1_raw"),
    "FoundHand-DataDream-I2I-noCN-C2-soft": lambda fold: c2_route(fold, "foundhand_datadream_i2i_no_cn", "c2_qc"),
}


def read_subject_metrics(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)["subject_level_metrics"]


def write_readme(summary_rows: list[dict[str, float]]) -> None:
    lines = [
        "# 手部营养状态论文主实验归档",
        "",
        "本目录仅归档论文主实验的可复核证据：每折 `metrics.json`、配置 `config.json`、详细预测结果、来源清单和汇总 CSV。",
        "不含模型权重、训练日志或生成图像，可在清理历史运行目录后继续用于论文制表与结果核查。",
        "",
        "## 实验口径",
        "",
        "- 五折受试者级评估，`seed=22`，仅使用 pose02；",
        "- C0-real-epoch/60ep/80ep/120ep/step：仅真实图训练；",
        "- C1-matched：每类从 300 张候选合成图按统一分层规则选 90 张；",
        "- C2-soft：在相同入选数量下，采用结构、去重与类别可分性软约束筛选；",
        "- 均值和标准差为 5 折未加权均值及样本标准差（`ddof=1`）。",
        "",
        "## 五折汇总",
        "",
        "| 路线 | Acc(%) | BA(%) | MCC | F1(%) | 敏感度(%) | 特异度(%) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['experiment']} | "
            f"{row['acc_mean'] * 100:.2f} ± {row['acc_std'] * 100:.2f} | "
            f"{row['balanced_accuracy_mean'] * 100:.2f} ± {row['balanced_accuracy_std'] * 100:.2f} | "
            f"{row['mcc_mean']:.4f} ± {row['mcc_std']:.4f} | "
            f"{row['f1_mean'] * 100:.2f} ± {row['f1_std'] * 100:.2f} | "
            f"{row['sensitivity_mean'] * 100:.2f} ± {row['sensitivity_std'] * 100:.2f} | "
            f"{row['specificity_mean'] * 100:.2f} ± {row['specificity_std'] * 100:.2f} |"
        )
    (ARCHIVE_ROOT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    raw_rows: list[dict[str, object]] = []
    manifest: dict[str, dict[str, str]] = {}

    for experiment, resolver in EXPERIMENTS.items():
        manifest[experiment] = {}
        for fold in range(5):
            source = resolver(fold)
            destination = ARCHIVE_ROOT / "per_fold" / experiment / f"fold_{fold}"
            destination.mkdir(parents=True, exist_ok=True)
            # 归档结果、配置及详细预测，保留源路径供追溯。
            for filename in ("metrics.json", "config.json", "detailed_prediction_results.json"):
                candidate = source.parent / filename
                if candidate.exists():
                    shutil.copy2(candidate, destination / filename)
            manifest[experiment][f"fold_{fold}"] = str(source)
            metrics = read_subject_metrics(source)
            raw_rows.append({"experiment": experiment, "fold": fold, **{key: metrics[key] for key in METRIC_KEYS}})

    with (ARCHIVE_ROOT / "fivefold_metrics_by_method.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("experiment", "fold", *METRIC_KEYS))
        writer.writeheader()
        writer.writerows(raw_rows)

    summary_rows: list[dict[str, float]] = []
    for experiment in EXPERIMENTS:
        rows = [row for row in raw_rows if row["experiment"] == experiment]
        summary: dict[str, float] = {"experiment": experiment}  # type: ignore[dict-item]
        for key in METRIC_KEYS:
            values = [float(row[key]) for row in rows]
            summary[f"{key}_mean"] = statistics.mean(values)
            summary[f"{key}_std"] = statistics.stdev(values)
        summary_rows.append(summary)

    fieldnames = ["experiment"] + [f"{key}_{suffix}" for key in METRIC_KEYS for suffix in ("mean", "std")]
    with (ARCHIVE_ROOT / "fivefold_mean_std_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    (ARCHIVE_ROOT / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_readme(summary_rows)
    print(f"已归档 {len(raw_rows)} 个逐折结果至：{ARCHIVE_ROOT}")


if __name__ == "__main__":
    main()
