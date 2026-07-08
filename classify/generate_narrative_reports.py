"""Batch-generate structured clinical narrative reports from cached ROI descriptors and attention records."""

import argparse
import csv
import json
import random
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from classify.narrative_report import (  # noqa: E402
    REQUIRED_TARGET_CLASS,
    ROI_NAMES,
    aggregate_subject_descriptors,
    aggregate_subject_views,
    build_thresholds,
    generate_subject_report,
    iter_image_paths,
)


DEFAULT_DESCRIPTOR_CACHE = "/root/autodl-tmp/runs/roi_descriptor_cache_with_test.json"
DEFAULT_ATTENTION_RECORDS = "/root/autodl-tmp/runs/vis/roi_validation_full/roi_attention_records.json"
DEFAULT_REAL_TRAIN_DIR = "/root/autodl-tmp/runs/cv/fold_4/my_dataset_binary/seed0"
DEFAULT_OUTPUT_DIR = "/root/autodl-tmp/runs/narrative_reports"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor_cache", default=DEFAULT_DESCRIPTOR_CACHE)
    parser.add_argument("--attention_records", default=DEFAULT_ATTENTION_RECORDS)
    parser.add_argument("--real_train_dirs", nargs="+", default=[DEFAULT_REAL_TRAIN_DIR])
    parser.add_argument("--test_image_dir", default="/root/autodl-tmp/test_data")
    parser.add_argument("--target_class", default=REQUIRED_TARGET_CLASS)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--attended_threshold", type=float, default=1.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Allow non-default target_class without an interactive confirmation.",
    )
    return parser.parse_args()


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _report_to_dict(report):
    out = dict(report)
    out["roi_findings"] = [asdict(item) if is_dataclass(item) else item for item in report["roi_findings"]]
    return out


def _subject_keys(records, target_class):
    keys = set()
    for row in records:
        if row.get("target_class") != target_class:
            continue
        subject_id = row.get("subject_id")
        true_class = row.get("true_class")
        if subject_id and true_class:
            keys.add((str(true_class), str(subject_id)))
    return sorted(keys, key=lambda item: (item[0], int(item[1]) if item[1].isdigit() else item[1]))


def _median(values):
    values = sorted(float(v) for v in values)
    n = len(values)
    if n == 0:
        raise ValueError("cannot compute median of empty values")
    mid = n // 2
    if n % 2:
        return values[mid]
    return 0.5 * (values[mid - 1] + values[mid])


def _subject_prediction(records, subject_id, target_class):
    rows = [
        row for row in records
        if str(row.get("subject_id")) == str(subject_id)
        and row.get("target_class") == target_class
    ]
    if not rows:
        raise ValueError(f"no target attention records for subject_id={subject_id}, target_class={target_class}")
    mal_prob = _median(row["malnourished_probability"] for row in rows)
    predicted_class = "malnourished_face" if mal_prob >= 0.5 else "normal_face"
    return predicted_class, mal_prob


def _write_json(reports, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)


def _write_csv(reports, path):
    fieldnames = [
        "subject_id",
        "true_class",
        "predicted_class",
        "malnourished_probability",
        "views_used",
        "n_non_normal_findings",
        "attended_rois",
        "abnormal_rois",
        "overlap_rois",
        "attention_only_rois",
        "abnormal_only_rois",
        "attention_narrative",
        "abnormal_narrative",
        "narrative",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for report in reports:
            writer.writerow({
                "subject_id": report["subject_id"],
                "true_class": report.get("true_class", ""),
                "predicted_class": report["predicted_class"],
                "malnourished_probability": report["malnourished_probability"],
                "views_used": json.dumps(report["views_used"], ensure_ascii=False),
                "n_non_normal_findings": sum(1 for item in report["roi_findings"] if item["severity_level"] != "normal"),
                "attended_rois": json.dumps(report.get("attended_rois", []), ensure_ascii=False),
                "abnormal_rois": json.dumps(report.get("abnormal_rois", []), ensure_ascii=False),
                "overlap_rois": json.dumps(report.get("overlap_rois", []), ensure_ascii=False),
                "attention_only_rois": json.dumps(report.get("attention_only_rois", []), ensure_ascii=False),
                "abnormal_only_rois": json.dumps(report.get("abnormal_only_rois", []), ensure_ascii=False),
                "attention_narrative": report.get("attention_narrative", ""),
                "abnormal_narrative": report.get("abnormal_narrative", ""),
                "narrative": report["narrative"],
            })


def _write_readable(reports, path):
    lines = []
    for report in reports:
        lines.append(f"subject_id: {report['subject_id']} ({report.get('true_class', '')})")
        lines.append(f"predicted_class: {report['predicted_class']}")
        lines.append(f"malnourished_probability: {report['malnourished_probability']:.6f}")
        lines.append(f"views_used: {report['views_used']}")
        lines.append(report["narrative"])
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def generate_reports(args):
    if args.target_class != REQUIRED_TARGET_CLASS and not args.yes:
        raise SystemExit(
            f"--target_class should remain {REQUIRED_TARGET_CLASS!r}; changing it reverses the narrative attribution basis. "
            "Pass --yes only if you intentionally want to continue."
        )

    random.seed(args.seed)
    descriptor_cache = _load_json(args.descriptor_cache)
    attention_records = _load_json(args.attention_records)
    real_train_paths = iter_image_paths(args.real_train_dirs)
    print(f"[INFO] real_train_image_paths={len(real_train_paths)}")
    if not (108 <= len(real_train_paths) <= 118):
        raise RuntimeError(f"real train image path count should be around 113; got {len(real_train_paths)}")

    thresholds = {
        roi: build_thresholds(args.descriptor_cache, roi, real_train_paths)
        for roi in ROI_NAMES
    }
    print(f"[INFO] thresholds={thresholds}")

    subject_keys = _subject_keys(attention_records, args.target_class)
    print(f"[INFO] subjects_from_attention={len(subject_keys)}")
    reports = []
    for true_class, subject_id in subject_keys:
        descriptor_values = aggregate_subject_descriptors(descriptor_cache, subject_id, image_dir=args.test_image_dir)
        attention_scores = aggregate_subject_views(attention_records, subject_id)
        predicted_class, mal_prob = _subject_prediction(attention_records, subject_id, args.target_class)
        report = generate_subject_report(
            subject_id,
            descriptor_values,
            attention_scores,
            thresholds,
            predicted_class=predicted_class,
            malnourished_probability=mal_prob,
            attended_threshold=args.attended_threshold,
        )
        report_dict = _report_to_dict(report)
        report_dict["true_class"] = true_class
        reports.append(report_dict)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(reports, output_dir / "narrative_reports.json")
    _write_csv(reports, output_dir / "narrative_reports.csv")
    _write_readable(reports, output_dir / "narrative_reports_readable.txt")
    print(f"[INFO] saved: {output_dir / 'narrative_reports.json'}")
    print(f"[INFO] saved: {output_dir / 'narrative_reports.csv'}")
    print(f"[INFO] saved: {output_dir / 'narrative_reports_readable.txt'}")
    return reports


def main():
    generate_reports(parse_args())


if __name__ == "__main__":
    main()
