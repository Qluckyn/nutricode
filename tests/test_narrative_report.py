import json
import random
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path

from classify.narrative_report import (
    ROI_NAMES,
    aggregate_subject_descriptors,
    aggregate_subject_views,
    collect_subject_view_attention,
    collect_subject_view_descriptors,
    build_thresholds,
    classify_severity,
    generate_roi_sentence,
    generate_subject_report,
    iter_image_paths,
    TEMPLATE_BANK,
)


def _write_cache(tmp_path, n=113, descriptor_value_fn=None):
    descriptor_value_fn = descriptor_value_fn or (lambda i: i / (n - 1))
    paths = [str((tmp_path / f"{i:03d}.png").resolve()) for i in range(n)]
    descriptors = {
        path: [
            descriptor_value_fn(i),
            0.5,
            0.5,
            0.5,
        ]
        for i, path in enumerate(paths)
    }
    raw_descriptors = {
        path: [
            100.0 + i,
            200.0 + i,
            300.0 + i,
            400.0 + i,
        ]
        for i, path in enumerate(paths)
    }
    cache_path = tmp_path / "cache.json"
    cache_path.write_text(
        json.dumps(
            {
                "normalize_stats": {"n_real_valid": n},
                "descriptors": descriptors,
                "raw_descriptors": raw_descriptors,
            }
        ),
        encoding="utf-8",
    )
    return cache_path, paths


class NarrativeReportPhase1Test(unittest.TestCase):
    def test_classify_severity_low_is_concerning_boundaries(self):
        thresholds = (0.20, 0.40)

        self.assertEqual(classify_severity(0.199, True, thresholds), "severe")
        self.assertEqual(classify_severity(0.200, True, thresholds), "severe")
        self.assertEqual(classify_severity(0.201, True, thresholds), "mild")
        self.assertEqual(classify_severity(0.399, True, thresholds), "mild")
        self.assertEqual(classify_severity(0.400, True, thresholds), "mild")
        self.assertEqual(classify_severity(0.401, True, thresholds), "normal")

    def test_classify_severity_high_is_concerning_boundaries(self):
        thresholds = (0.80, 0.60)

        self.assertEqual(classify_severity(0.801, False, thresholds), "severe")
        self.assertEqual(classify_severity(0.800, False, thresholds), "severe")
        self.assertEqual(classify_severity(0.799, False, thresholds), "mild")
        self.assertEqual(classify_severity(0.600, False, thresholds), "mild")
        self.assertEqual(classify_severity(0.599, False, thresholds), "normal")

    def test_classify_severity_rejects_raw_descriptor_space(self):
        with self.assertRaisesRegex(ValueError, "normalized value"):
            classify_severity(44.7, False, (0.80, 0.60))

    def test_build_thresholds_uses_normalized_descriptors_not_raw(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_path, paths = _write_cache(Path(tmp))

            low_q, mid_q = build_thresholds(str(cache_path), "temporal", paths)

        self.assertAlmostEqual(low_q, 0.10)
        self.assertAlmostEqual(mid_q, 0.35)

    def test_build_thresholds_rejects_values_outside_normalized_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_path, paths = _write_cache(
                Path(tmp),
                descriptor_value_fn=lambda i: 44.7 if i == 0 else 0.5,
            )

            with self.assertRaisesRegex(ValueError, "normalized descriptors"):
                build_thresholds(str(cache_path), "temporal", paths)

    def test_generate_roi_sentence_normal_without_attended_is_empty(self):
        self.assertEqual(generate_roi_sentence("temporal", "normal", False, 0.2), "")

    def test_generate_roi_sentence_anomalous_attend_uses_template_bank(self):
        sentence = generate_roi_sentence("temporal", "normal", True, 0.2)
        self.assertIn(sentence, TEMPLATE_BANK["temporal"]["anomalous_attend"])

    def test_generate_roi_sentence_mild_and_severe_use_matching_template_bank(self):
        for roi, templates in TEMPLATE_BANK.items():
            for severity in ("mild", "severe"):
                sentence = generate_roi_sentence(roi, severity, False, -0.2)
                self.assertIn(sentence, templates[severity])

    def test_generate_roi_sentence_randomly_switches_between_synonym_templates(self):
        random.seed(0)

        sentences = {
            generate_roi_sentence("temporal", "mild", False, 0.0)
            for _ in range(20)
        }

        self.assertEqual(sentences, set(TEMPLATE_BANK["temporal"]["mild"]))

    def test_aggregate_subject_views_filters_required_target_class(self):
        records = []
        for target_class, balance, enrichment in (
            ("malnourished_face", 0.25, 1.5),
            ("normal_face", -0.99, 9.9),
        ):
            row = {
                "subject_id": "001",
                "view": "front",
                "target_class": target_class,
            }
            for roi in ROI_NAMES:
                row[f"attr_pos_roi_{roi}_face_enrichment"] = enrichment
                row[f"attr_signed_roi_{roi}_face_balance"] = balance
            records.append(row)

        result = aggregate_subject_views(records, "001")

        self.assertEqual(result["views_used"], ["front"])
        for roi in ROI_NAMES:
            self.assertEqual(result[roi]["balance"], 0.25)
            self.assertEqual(result[roi]["enrichment"], 1.5)

    def test_aggregate_subject_views_rejects_full_image_attention_only(self):
        records = [{
            "subject_id": "001",
            "view": "front",
            "target_class": "malnourished_face",
            **{f"attr_pos_roi_{roi}_enrichment": 1.5 for roi in ROI_NAMES},
            **{f"attr_signed_roi_{roi}_balance": 0.25 for roi in ROI_NAMES},
        }]

        with self.assertRaisesRegex(ValueError, "face-normalized attention scores"):
            aggregate_subject_views(records, "001")

    def test_subject_171_missing_view_regression_real_data(self):
        descriptor_path = "/root/autodl-tmp/runs/roi_descriptor_cache_with_test.json"
        attention_path = "/root/autodl-tmp/runs/vis/roi_validation_full_face/roi_attention_records.json"
        train_dir = "/root/autodl-tmp/runs/cv/fold_4/my_dataset_binary/seed0"
        for path in (descriptor_path, attention_path, train_dir):
            if not Path(path).exists():
                self.skipTest(f"real stage3 fixture not available: {path}")

        descriptor_cache = json.loads(Path(descriptor_path).read_text(encoding="utf-8"))
        attention_records = json.loads(Path(attention_path).read_text(encoding="utf-8"))
        descriptor_values = aggregate_subject_descriptors(descriptor_cache, "171")
        attention_scores = aggregate_subject_views(attention_records, "171")
        thresholds = {
            roi: build_thresholds(descriptor_path, roi, iter_image_paths(train_dir))
            for roi in ROI_NAMES
        }
        target_records = [
            row for row in attention_records
            if row.get("subject_id") == "171" and row.get("target_class") == "malnourished_face"
        ]
        report = generate_subject_report(
            "171",
            descriptor_values,
            attention_scores,
            thresholds,
            predicted_class=target_records[0]["predicted_class"],
            malnourished_probability=target_records[0]["malnourished_probability"],
        )

        self.assertEqual(descriptor_values["views_used"], ["front", "left_45"])
        self.assertEqual(attention_scores["views_used"], ["front", "left_45"])
        self.assertEqual(report["views_used"], ["front", "left_45"])
        self.assertTrue(report["narrative"])
        self.assertEqual(len(report["roi_findings"]), 4)

    def test_generate_subject_report_confidence_matches_predicted_class(self):
        descriptor_values = {
            "temporal": 0.5,
            "orbital": 0.5,
            "malar": 0.4,
            "jawline": 0.4,
            "views_used": ["front"],
        }
        attention_scores = {
            "views_used": ["front"],
            **{roi: {"enrichment": 0.5, "balance": -0.1} for roi in ROI_NAMES},
        }
        thresholds = {
            "temporal": (0.2, 0.4),
            "orbital": (0.2, 0.4),
            "malar": (0.8, 0.6),
            "jawline": (0.8, 0.6),
        }

        report = generate_subject_report(
            "normal-confidence",
            descriptor_values,
            attention_scores,
            thresholds,
            predicted_class="normal_face",
            malnourished_probability=0.2,
        )

        self.assertIn("预测为正常（置信度80.0%）", report["narrative"])

    def test_structured_narrative_splits_attention_and_abnormal_regions(self):
        descriptor_values = {
            "temporal": 0.5,
            "orbital": 0.3,
            "malar": 0.4,
            "jawline": 0.4,
            "views_used": ["front"],
        }
        attention_scores = {
            "views_used": ["front"],
            "temporal": {"enrichment": 1.3, "balance": 0.2},
            "orbital": {"enrichment": 0.8, "balance": 0.1},
            "malar": {"enrichment": 0.7, "balance": -0.1},
            "jawline": {"enrichment": 0.6, "balance": -0.1},
        }
        thresholds = {
            "temporal": (0.2, 0.4),
            "orbital": (0.2, 0.4),
            "malar": (0.8, 0.6),
            "jawline": (0.8, 0.6),
        }

        report = generate_subject_report(
            "split", descriptor_values, attention_scores, thresholds, "normal_face", 0.2
        )

        self.assertEqual(report["attended_rois"], ["temporal"])
        self.assertEqual(report["abnormal_rois"], ["orbital"])
        self.assertEqual(report["attention_only_rois"], ["temporal"])
        self.assertEqual(report["abnormal_only_rois"], ["orbital"])
        self.assertEqual(report["overlap_rois"], [])
        self.assertIn("模型关注区域：对颞部区域关注度较高。", report["narrative"])
        self.assertIn("ROI异常区域：", report["narrative"])
        self.assertNotIn("颞部区域给予较高关注，但该区域测量值处于正常范围", report["narrative"])

    def test_structured_narrative_overlap_rois(self):
        descriptor_values = {
            "temporal": 0.3,
            "orbital": 0.5,
            "malar": 0.4,
            "jawline": 0.4,
            "views_used": ["front"],
        }
        attention_scores = {
            "views_used": ["front"],
            "temporal": {"enrichment": 1.4, "balance": 0.2},
            "orbital": {"enrichment": 0.8, "balance": -0.1},
            "malar": {"enrichment": 0.7, "balance": -0.1},
            "jawline": {"enrichment": 0.6, "balance": -0.1},
        }
        thresholds = {
            "temporal": (0.2, 0.4),
            "orbital": (0.2, 0.4),
            "malar": (0.8, 0.6),
            "jawline": (0.8, 0.6),
        }

        report = generate_subject_report(
            "overlap", descriptor_values, attention_scores, thresholds, "malnourished_face", 0.8
        )

        self.assertEqual(report["attended_rois"], ["temporal"])
        self.assertEqual(report["abnormal_rois"], ["temporal"])
        self.assertEqual(report["overlap_rois"], ["temporal"])
        self.assertEqual(report["attention_only_rois"], [])
        self.assertEqual(report["abnormal_only_rois"], [])

    def test_structured_narrative_no_attended_no_abnormal_fallback_text(self):
        descriptor_values = {
            "temporal": 0.5,
            "orbital": 0.5,
            "malar": 0.4,
            "jawline": 0.4,
            "views_used": ["front"],
        }
        attention_scores = {
            "views_used": ["front"],
            **{roi: {"enrichment": 0.5, "balance": -0.1} for roi in ROI_NAMES},
        }
        thresholds = {
            "temporal": (0.2, 0.4),
            "orbital": (0.2, 0.4),
            "malar": (0.8, 0.6),
            "jawline": (0.8, 0.6),
        }

        report = generate_subject_report(
            "fallback", descriptor_values, attention_scores, thresholds, "normal_face", 0.2
        )

        self.assertEqual(report["attended_rois"], [])
        self.assertEqual(report["abnormal_rois"], [])
        self.assertIn("模型关注区域：未见关注度显著高于阈值的预设ROI区域。", report["narrative"])
        self.assertIn("ROI异常区域：各ROI描述符均处于正常范围。", report["narrative"])

    def test_viewwise_narrative_describes_each_view_and_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_dir = Path(tmp) / "test_data"
            cls_dir = image_dir / "malnourished_face"
            cls_dir.mkdir(parents=True)
            descriptor_cache = {
                "descriptors": {
                    str((cls_dir / "001_01.png").resolve()): [0.5, 0.3, 0.9, 0.4],
                    str((cls_dir / "001_02.png").resolve()): [0.5, 0.3, 0.4, 0.4],
                    str((cls_dir / "001_03.png").resolve()): None,
                }
            }

            records = []
            for view, temporal, malar in (("front", 1.2, 1.2), ("left_45", 0.9, 0.9)):
                row = {"subject_id": "001", "view": view, "target_class": "malnourished_face"}
                for roi in ROI_NAMES:
                    row[f"attr_pos_roi_{roi}_face_enrichment"] = 0.8
                    row[f"attr_signed_roi_{roi}_face_balance"] = 0.1
                row["attr_pos_roi_temporal_face_enrichment"] = temporal
                row["attr_pos_roi_malar_face_enrichment"] = malar
                records.append(row)

            descriptor_values = {"temporal": 0.5, "orbital": 0.3, "malar": 0.9, "jawline": 0.4, "views_used": ["front", "left_45"]}
            attention_scores = {"views_used": ["front", "left_45"], **{roi: {"enrichment": 1.0, "balance": 0.1} for roi in ROI_NAMES}}
            thresholds = {"temporal": (0.2, 0.4), "orbital": (0.2, 0.4), "malar": (0.8, 0.6), "jawline": (0.8, 0.6)}

            descriptor_view_scores = collect_subject_view_descriptors(descriptor_cache, "001", image_dir=str(image_dir))
            attention_view_scores = collect_subject_view_attention(records, "001")
            report = generate_subject_report(
                "001",
                descriptor_values,
                attention_scores,
                thresholds,
                predicted_class="malnourished_face",
                malnourished_probability=0.8,
                attended_threshold=1.1,
                descriptor_view_scores=descriptor_view_scores,
                attention_view_scores=attention_view_scores,
            )

        self.assertIn("模型关注区域：正面：对颞部区域关注度较高，对颧颊区域关注度较高", report["narrative"])
        self.assertIn("左45度：未见关注度显著高于阈值的预设ROI区域", report["narrative"])
        self.assertIn("右45度：检测失败", report["narrative"])
        self.assertIn("ROI异常区域：正面：", report["narrative"])
        self.assertIn("左45度：", report["narrative"])
        self.assertIn("右45度：检测失败", report["narrative"])
        self.assertEqual(report["view_findings"]["right_45"]["descriptor_status"], "failed")
        self.assertEqual(report["view_findings"]["right_45"]["attention_status"], "missing")
        self.assertIn("subject_level_attention_narrative", report)

    def test_real_report_structured_sections_use_allowed_sentences(self):
        descriptor_path = "/root/autodl-tmp/runs/roi_descriptor_cache_with_test.json"
        attention_path = "/root/autodl-tmp/runs/vis/roi_validation_full_face/roi_attention_records.json"
        train_dir = "/root/autodl-tmp/runs/cv/fold_4/my_dataset_binary/seed0"
        for path in (descriptor_path, attention_path, train_dir):
            if not Path(path).exists():
                self.skipTest(f"real regression fixture not available: {path}")

        descriptor_cache = json.loads(Path(descriptor_path).read_text(encoding="utf-8"))
        attention_records = json.loads(Path(attention_path).read_text(encoding="utf-8"))
        real_train_paths = iter_image_paths(train_dir)
        thresholds = {roi: build_thresholds(descriptor_path, roi, real_train_paths) for roi in ROI_NAMES}
        allowed_abnormal_sentences = {
            sentence
            for roi_templates in TEMPLATE_BANK.values()
            for severity in ("mild", "severe")
            for sentence in roi_templates[severity]
        }
        allowed_attention_sentences = {
            f"对{name}区域关注度较高"
            for name in ("颞部", "眶周", "颧颊", "下颌缘")
        }
        attention_fallback = "未见关注度显著高于阈值的预设ROI区域"
        abnormal_fallback = "各ROI描述符均处于正常范围"
        subjects = sorted({
            (row["true_class"], row["subject_id"])
            for row in attention_records
            if row.get("target_class") == "malnourished_face"
        })

        for _true_class, subject_id in subjects:
            descriptor_values = aggregate_subject_descriptors(descriptor_cache, subject_id)
            attention_scores = aggregate_subject_views(attention_records, subject_id)
            target_rows = [
                row for row in attention_records
                if row.get("subject_id") == subject_id and row.get("target_class") == "malnourished_face"
            ]
            mal_prob = sorted(float(row["malnourished_probability"]) for row in target_rows)[len(target_rows) // 2]
            predicted_class = "malnourished_face" if mal_prob >= 0.5 else "normal_face"
            report = generate_subject_report(
                subject_id, descriptor_values, attention_scores, thresholds, predicted_class, mal_prob
            )
            attention_body = report["attention_narrative"].removeprefix("模型关注区域：").rstrip("。")
            abnormal_body = report["abnormal_narrative"].removeprefix("ROI异常区域：").rstrip("。")
            for sentence in [item for item in attention_body.split("；") if item]:
                self.assertIn(sentence, allowed_attention_sentences | {attention_fallback})
            for sentence in [item for item in abnormal_body.split("；") if item]:
                self.assertIn(sentence, allowed_abnormal_sentences | {abnormal_fallback})

    def test_real_group_non_normal_finding_count_distribution(self):
        descriptor_path = "/root/autodl-tmp/runs/roi_descriptor_cache_with_test.json"
        attention_path = "/root/autodl-tmp/runs/vis/roi_validation_full_face/roi_attention_records.json"
        train_dir = "/root/autodl-tmp/runs/cv/fold_4/my_dataset_binary/seed0"
        for path in (descriptor_path, attention_path, train_dir):
            if not Path(path).exists():
                self.skipTest(f"real regression fixture not available: {path}")

        descriptor_cache = json.loads(Path(descriptor_path).read_text(encoding="utf-8"))
        attention_records = json.loads(Path(attention_path).read_text(encoding="utf-8"))
        real_train_paths = iter_image_paths(train_dir)
        thresholds = {roi: build_thresholds(descriptor_path, roi, real_train_paths) for roi in ROI_NAMES}
        subjects = sorted({
            (row["true_class"], row["subject_id"])
            for row in attention_records
            if row.get("target_class") == "malnourished_face"
        })
        counts_by_group = defaultdict(list)
        for true_class, subject_id in subjects:
            descriptor_values = aggregate_subject_descriptors(descriptor_cache, subject_id)
            attention_scores = aggregate_subject_views(attention_records, subject_id)
            target_rows = [
                row for row in attention_records
                if row.get("subject_id") == subject_id and row.get("target_class") == "malnourished_face"
            ]
            mal_prob = sorted(float(row["malnourished_probability"]) for row in target_rows)[len(target_rows) // 2]
            predicted_class = "malnourished_face" if mal_prob >= 0.5 else "normal_face"
            report = generate_subject_report(
                subject_id, descriptor_values, attention_scores, thresholds, predicted_class, mal_prob
            )
            n_non_normal = sum(1 for finding in report["roi_findings"] if finding.severity_level != "normal")
            counts_by_group[true_class].append(n_non_normal)

        self.assertEqual(len(counts_by_group["malnourished_face"]), 27)
        self.assertEqual(len(counts_by_group["normal_face"]), 27)
        mal_dist = Counter(counts_by_group["malnourished_face"])
        normal_dist = Counter(counts_by_group["normal_face"])
        mal_mean = sum(counts_by_group["malnourished_face"]) / len(counts_by_group["malnourished_face"])
        normal_mean = sum(counts_by_group["normal_face"]) / len(counts_by_group["normal_face"])

        self.assertEqual(mal_dist, Counter({0: 10, 1: 9, 2: 5, 3: 3}))
        self.assertEqual(normal_dist, Counter({0: 8, 1: 10, 2: 7, 3: 2}))
        self.assertAlmostEqual(mal_mean, 1.037037037037037)
        self.assertAlmostEqual(normal_mean, 1.1111111111111112)

    def test_build_thresholds_rejects_unexpected_real_train_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_path, paths = _write_cache(Path(tmp), n=20)

            with self.assertRaisesRegex(ValueError, r"113 \+/- 5"):
                build_thresholds(str(cache_path), "temporal", paths)


if __name__ == "__main__":
    unittest.main()
