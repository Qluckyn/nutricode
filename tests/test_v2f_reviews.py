"""V2-F 人工盲审汇总规则的回归测试。"""

from hand_synthesis_v2.finalize_v2f_reviews import (
    accepted,
    agreement_metric,
    map_candidate_fields,
    validate_row,
)


def passing_review():
    return {
        "candidate_hand_count": "two",
        "candidate_anatomy": "pass",
        "candidate_pose": "fist",
        "phenotype_preserved": "yes",
        "candidate_artifact": "no",
        "near_duplicate_concern": "no",
        "pair_accept": "yes",
    }


def test_acceptance_rejects_near_duplicate_concern():
    """人工认为疑似近重复时，即使其他字段通过也不得接收。"""
    review = passing_review()
    assert accepted(review, {"pose": "01"})
    review["near_duplicate_concern"] = "yes"
    assert not accepted(review, {"pose": "01"})


def test_candidate_fields_follow_private_side_key():
    """公共左右侧必须通过私钥映射回候选图，不能假设固定在左侧。"""
    row = {
        "left_hand_count": "not_two", "right_hand_count": "two",
        "left_anatomy": "fail", "right_anatomy": "pass",
        "left_pose": "other", "right_pose": "extended",
        "phenotype_preserved": "yes",
        "left_artifact": "yes", "right_artifact": "no",
        "near_duplicate_concern": "no", "pair_accept": "yes",
        "failure_types": "", "notes": "",
    }
    mapped = map_candidate_fields(row, {"candidate_side": "right"})
    assert mapped["candidate_hand_count"] == "two"
    assert mapped["candidate_anatomy"] == "pass"
    assert mapped["candidate_pose"] == "extended"
    assert mapped["candidate_artifact"] == "no"


def test_review_validation_requires_near_duplicate_answer():
    config = {
        "allowed_answers": {
            "hand_count": ["two", "not_two", "uncertain"],
            "anatomy": ["pass", "fail", "uncertain"],
            "pose": ["fist", "extended", "other", "uncertain"],
            "phenotype_preserved": ["yes", "no", "uncertain"],
            "artifact": ["yes", "no", "uncertain"],
            "near_duplicate_concern": ["yes", "no", "uncertain"],
            "pair_accept": ["yes", "no", "uncertain"],
        }
    }
    row = {
        "left_hand_count": "two", "right_hand_count": "two",
        "left_anatomy": "pass", "right_anatomy": "pass",
        "left_pose": "fist", "right_pose": "fist",
        "phenotype_preserved": "yes",
        "left_artifact": "no", "right_artifact": "no",
        "near_duplicate_concern": "", "pair_accept": "yes",
    }
    assert "near_duplicate_concern=''" in validate_row(row, config)


def test_defined_kappa_meets_preregistered_threshold():
    metric = agreement_metric(["yes", "no"], ["yes", "no"], minimum=0.60)
    assert metric["cohen_kappa"] == 1.0
    assert metric["meets_minimum"] is True
