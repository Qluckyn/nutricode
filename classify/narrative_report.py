import os
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


ROI_NAMES = ["temporal", "orbital", "malar", "jawline"]
ROI_CN_NAMES = {
    "temporal": "颞部",
    "orbital": "眶周",
    "malar": "颧颊",
    "jawline": "下颌缘",
}
ROI_TO_DESCRIPTOR_INDEX = {
    "temporal": 0,
    "orbital": 1,
    "malar": 2,
    "jawline": 3,
}
ROI_DIRECTION = {
    "temporal": True,
    "orbital": True,
    "malar": False,
    "jawline": False,
}
REQUIRED_TARGET_CLASS = "malnourished_face"
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIEW_ORDER = ["front", "left_45", "right_45"]
VIEW_CODE_TO_NAME = {"01": "front", "02": "left_45", "03": "right_45"}
PREDICTED_CLASS_CN = {"malnourished_face": "营养不良", "normal_face": "正常"}

TEMPLATE_BANK = {
    "temporal": {
        "severe": [
            "颞部相对亮度明显偏低，提示颞肌萎缩较为显著",
            "颞部区域亮度显著低于全脸均值，符合中重度颞肌萎缩表现",
        ],
        "mild": [
            "颞部相对亮度轻度偏低，提示可能存在颞肌轻度萎缩",
            "颞部区域亮度略低于正常范围，建议关注颞肌状态",
        ],
        "anomalous_attend": [
            "模型对颞部区域给予较高关注，但该区域测量值处于正常范围，建议结合其他证据综合判断",
        ],
    },
    "orbital": {
        "severe": [
            "眶周相对亮度明显偏低，提示眶周脂肪垫萎缩、凹陷较为明显",
            "眶周区域亮度显著低于全脸均值，符合中重度眶周凹陷表现",
        ],
        "mild": [
            "眶周相对亮度轻度偏低，提示可能存在轻度眶周凹陷",
            "眶周区域亮度略低于正常范围，建议关注眶周脂肪状态",
        ],
        "anomalous_attend": [
            "模型对眶周区域给予较高关注，但该区域测量值处于正常范围，建议结合其他证据综合判断",
        ],
    },
    "malar": {
        "severe": [
            "颧颊纹理方差显著升高，提示颧颊皮下脂肪明显流失、皮肤纹理起伏增大",
            "颧颊区域纹理复杂度明显高于正常范围，符合中重度脂肪流失表现",
        ],
        "mild": [
            "颧颊纹理方差轻度升高，提示可能存在轻度皮下脂肪流失",
            "颧颊区域纹理略高于正常范围，建议关注该区域软组织状态",
        ],
        "anomalous_attend": [
            "模型对颧颊区域给予较高关注，但该区域测量值处于正常范围，建议结合其他证据综合判断",
        ],
    },
    "jawline": {
        "severe": [
            "下颌轮廓梯度显著升高，提示皮下脂肪流失后骨性下颌轮廓明显锐化",
            "下颌缘区域轮廓锐利度明显高于正常范围，符合中重度脂肪流失表现",
        ],
        "mild": [
            "下颌轮廓梯度轻度升高，提示可能存在轻度皮下脂肪流失",
            "下颌缘区域轮廓略偏锐利，建议关注该区域软组织状态",
        ],
        "anomalous_attend": [
            "模型对下颌缘区域给予较高关注，但该区域测量值处于正常范围，建议结合其他证据综合判断",
        ],
    },
}


@dataclass
class ROIFinding:
    roi: str
    descriptor_value: float
    severity_level: str
    attention_enrichment: float
    attention_attended: bool
    attention_balance: float
    sentence: str


def classify_severity(
    descriptor_value: float,
    direction_low_is_concerning: bool,
    thresholds: tuple,
) -> str:
    value = float(descriptor_value)
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"descriptor_value must be a finite normalized value in [0, 1], got {descriptor_value!r}")

    low_or_high_q, mid_q = (float(thresholds[0]), float(thresholds[1]))
    if direction_low_is_concerning:
        if value <= low_or_high_q:
            return "severe"
        if value <= mid_q:
            return "mild"
        return "normal"

    if value >= low_or_high_q:
        return "severe"
    if value >= mid_q:
        return "mild"
    return "normal"


def iter_image_paths(image_dirs: list[str] | tuple[str, ...] | str) -> list[str]:
    if isinstance(image_dirs, (str, Path)):
        image_dirs = [str(image_dirs)]

    paths = []
    for image_dir in image_dirs:
        root = Path(image_dir)
        if not root.exists():
            raise FileNotFoundError(f"real train image dir does not exist: {root}")
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMG_EXTENSIONS:
                paths.append(os.path.abspath(path))
    return sorted(paths)


def _load_descriptor_cache(descriptor_cache_path: str) -> dict:
    with open(descriptor_cache_path, "r", encoding="utf-8") as f:
        cache = json.load(f)
    if not isinstance(cache, dict) or "descriptors" not in cache:
        raise ValueError(f"descriptor cache missing 'descriptors': {descriptor_cache_path}")
    return cache


def build_thresholds(
    descriptor_cache_path: str,
    roi: str,
    real_train_image_paths: list,
) -> tuple:
    if roi not in ROI_NAMES:
        raise ValueError(f"unknown ROI: {roi!r}")

    cache = _load_descriptor_cache(descriptor_cache_path)
    descriptors = cache["descriptors"]
    idx = ROI_TO_DESCRIPTOR_INDEX[roi]

    values = []
    missing_or_failed = []
    for image_path in real_train_image_paths:
        key = os.path.abspath(image_path)
        desc = descriptors.get(key)
        if desc is None:
            missing_or_failed.append(key)
            continue
        value = float(desc[idx])
        if not np.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError(
                "build_thresholds must read normalized descriptors in [0, 1]; "
                f"got {value!r} for {key} roi={roi}"
            )
        values.append(value)

    if not (108 <= len(values) <= 118):
        preview = "\n".join(str(p) for p in real_train_image_paths[:20])
        raise ValueError(
            "real train valid descriptor count must be within 113 +/- 5; "
            f"got valid={len(values)}, input_paths={len(real_train_image_paths)}, "
            f"missing_or_failed={len(missing_or_failed)}. First paths:\n{preview}"
        )

    arr = np.asarray(values, dtype=np.float64)
    if ROI_DIRECTION[roi]:
        return (float(np.percentile(arr, 10)), float(np.percentile(arr, 35)))
    return (float(np.percentile(arr, 90)), float(np.percentile(arr, 65)))

def generate_roi_sentence(roi: str, severity: str, attended: bool, balance: float) -> str:
    if roi not in TEMPLATE_BANK:
        raise ValueError(f"unknown ROI: {roi!r}")
    if severity not in {"normal", "mild", "severe"}:
        raise ValueError(f"unknown severity: {severity!r}")

    if severity == "normal":
        if attended and float(balance) > 0.0:
            return random.choice(TEMPLATE_BANK[roi]["anomalous_attend"])
        return ""

    return random.choice(TEMPLATE_BANK[roi][severity])


def _parse_subject_id_from_path(path: str) -> str:
    name = os.path.basename(path or "")
    match = re.match(r"^(\d+)_", name)
    return match.group(1) if match else ""


def _parse_view_from_path(path: str) -> str:
    name = os.path.basename(path or "")
    match = re.match(r"^\d+_(01|02|03)(?:$|[_\-.])", name)
    if match:
        return VIEW_CODE_TO_NAME[match.group(1)]
    return "unknown"


def _ordered_views(views: set[str]) -> list[str]:
    return [view for view in VIEW_ORDER if view in views]


def aggregate_subject_descriptors(
    descriptor_cache: dict,
    subject_id: str,
    image_dir: str = "/root/autodl-tmp/test_data",
) -> dict:
    descriptors = descriptor_cache.get("descriptors", descriptor_cache)
    image_dir_abs = os.path.abspath(image_dir)
    values_by_roi = {roi: [] for roi in ROI_NAMES}
    views_used = set()

    for path, desc in descriptors.items():
        path_abs = os.path.abspath(path)
        if not path_abs.startswith(image_dir_abs + os.sep):
            continue
        if _parse_subject_id_from_path(path_abs) != str(subject_id):
            continue
        if desc is None:
            continue

        view = _parse_view_from_path(path_abs)
        if view in VIEW_ORDER:
            views_used.add(view)
        for roi in ROI_NAMES:
            value = float(desc[ROI_TO_DESCRIPTOR_INDEX[roi]])
            if not np.isfinite(value) or value < 0.0 or value > 1.0:
                raise ValueError(f"descriptor for subject={subject_id} roi={roi} is not normalized: {value!r}")
            values_by_roi[roi].append(value)

    ordered_views = _ordered_views(views_used)
    if not ordered_views:
        raise ValueError(f"no valid descriptor views found for subject_id={subject_id}")
    for roi, values in values_by_roi.items():
        if not values:
            raise ValueError(f"no valid descriptor values for subject_id={subject_id} roi={roi}")

    result = {roi: float(np.median(values_by_roi[roi])) for roi in ROI_NAMES}
    result["views_used"] = ordered_views
    return result


def aggregate_subject_views(records: list, subject_id: str) -> dict:
    filtered = [
        row for row in records
        if str(row.get("subject_id")) == str(subject_id)
        and row.get("target_class") == REQUIRED_TARGET_CLASS
    ]
    if not filtered:
        raise ValueError(
            f"no attention records found for subject_id={subject_id} "
            f"with target_class={REQUIRED_TARGET_CLASS}"
        )

    views_used = {row.get("view") for row in filtered if row.get("view") in VIEW_ORDER}
    result = {"views_used": _ordered_views(views_used)}
    for roi in ROI_NAMES:
        enrichments = []
        balances = []
        enrich_key = f"attr_pos_roi_{roi}_enrichment"
        balance_key = f"attr_signed_roi_{roi}_balance"
        for row in filtered:
            enrichment = row.get(enrich_key)
            balance = row.get(balance_key)
            if enrichment is not None and np.isfinite(float(enrichment)):
                enrichments.append(float(enrichment))
            if balance is not None and np.isfinite(float(balance)):
                balances.append(float(balance))
        if not enrichments or not balances:
            raise ValueError(f"missing attention scores for subject_id={subject_id} roi={roi}")
        result[roi] = {
            "enrichment": float(np.median(enrichments)),
            "balance": float(np.median(balances)),
        }
    return result


def _prediction_confidence(predicted_class: str, malnourished_probability: float) -> float:
    mal_prob = float(malnourished_probability)
    return 1.0 - mal_prob if predicted_class == "normal_face" else mal_prob


def _attention_sentence(roi: str) -> str:
    return f"对{ROI_CN_NAMES[roi]}区域关注度较高"


def _build_structured_sections(findings: list[ROIFinding]) -> dict:
    attended_findings = [finding for finding in findings if finding.attention_attended]
    abnormal_findings = [finding for finding in findings if finding.severity_level in {"mild", "severe"}]

    attended_rois = [finding.roi for finding in attended_findings]
    abnormal_rois = [finding.roi for finding in abnormal_findings]
    overlap_rois = [roi for roi in attended_rois if roi in set(abnormal_rois)]
    attention_only_rois = [roi for roi in attended_rois if roi not in set(abnormal_rois)]
    abnormal_only_rois = [roi for roi in abnormal_rois if roi not in set(attended_rois)]

    if attended_findings:
        attention_narrative = "模型关注区域：" + "；".join(_attention_sentence(finding.roi) for finding in attended_findings) + "。"
    else:
        attention_narrative = "模型关注区域：未见关注度显著高于阈值的预设ROI区域。"

    abnormal_sentences = [finding.sentence for finding in abnormal_findings if finding.sentence]
    if abnormal_sentences:
        abnormal_narrative = "ROI异常区域：" + "；".join(abnormal_sentences) + "。"
    else:
        abnormal_narrative = "ROI异常区域：各ROI描述符均处于正常范围。"

    return {
        "attended_rois": attended_rois,
        "abnormal_rois": abnormal_rois,
        "overlap_rois": overlap_rois,
        "attention_only_rois": attention_only_rois,
        "abnormal_only_rois": abnormal_only_rois,
        "attention_narrative": attention_narrative,
        "abnormal_narrative": abnormal_narrative,
        "narrative_sentence_count": len(attended_findings) + (len(abnormal_sentences) if abnormal_sentences else 1),
    }


def generate_subject_report(
    subject_id: str,
    descriptor_values: dict,
    attention_scores: dict,
    thresholds: dict,
    predicted_class: str,
    malnourished_probability: float,
    attended_threshold: float = 1.15,
) -> dict:
    descriptor_views = descriptor_values.get("views_used", [])
    attention_views = attention_scores.get("views_used", [])
    views_used = descriptor_views or attention_views

    findings = []
    for roi in ROI_NAMES:
        descriptor_value = float(descriptor_values[roi])
        attention_enrichment = float(attention_scores[roi]["enrichment"])
        attention_balance = float(attention_scores[roi]["balance"])
        severity = classify_severity(descriptor_value, ROI_DIRECTION[roi], thresholds[roi])
        attended = attention_enrichment > float(attended_threshold)
        sentence = "" if severity == "normal" else generate_roi_sentence(roi, severity, attended, attention_balance)
        findings.append(
            ROIFinding(
                roi=roi,
                descriptor_value=descriptor_value,
                severity_level=severity,
                attention_enrichment=attention_enrichment,
                attention_attended=attended,
                attention_balance=attention_balance,
                sentence=sentence,
            )
        )

    findings.sort(key=lambda item: item.attention_enrichment, reverse=True)
    sections = _build_structured_sections(findings)
    predicted_class_cn = PREDICTED_CLASS_CN.get(predicted_class, predicted_class)
    confidence = _prediction_confidence(predicted_class, malnourished_probability)
    narrative = (
        f"该受试者预测为{predicted_class_cn}（置信度{confidence:.1%}）。"
        f"{sections['attention_narrative']}"
        f"{sections['abnormal_narrative']}"
    )
    return {
        "subject_id": str(subject_id),
        "predicted_class": predicted_class,
        "malnourished_probability": float(malnourished_probability),
        "views_used": list(views_used),
        "roi_findings": findings,
        **sections,
        "narrative": narrative,
    }

