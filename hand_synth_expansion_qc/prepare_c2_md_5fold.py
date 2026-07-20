#!/usr/bin/env python3
"""为 fold_1--fold_4 建立可复现的 C2-MD 候选扩增工作区。

保留既有每类 90 张无 ControlNet 候选，并以相同父图轮换、三档强度与
确定性种子补足至每类 300 张。旧图通过软链接复用，不覆盖原始 C1 实验。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


ROUTES = ("datadream_i2i_no_cn", "foundhand_datadream_i2i_no_cn")
STRENGTHS = (0.15, 0.22, 0.30)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stable_seed(base_seed: int, mode: str, compound: str, index: int) -> int:
    """使用候选主键派生种子，补生成不会改变已存在候选。"""
    payload = f"{base_seed}:{mode}:{compound}:{index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def strength_for_index(index: int) -> float:
    """前 90 张每档 30 张；补充的 210 张每档再增加 70 张。"""
    return STRENGTHS[index // 30] if index < 90 else STRENGTHS[(index - 90) // 70]


def link_existing(old_root: Path, new_root: Path, plan: list[dict], mode: str) -> None:
    """软链接已有图和元数据，使生成器仅计算新增的 210 张/类。"""
    for item in plan:
        if int(item["candidate_id"].rsplit("_", 1)[-1]) >= 90:
            continue
        name = f'{item["candidate_id"]}.png'
        old_image = old_root / mode / item["compound_class"] / name
        old_meta = old_root / mode / "metadata" / f'{item["candidate_id"]}.json'
        new_image = new_root / mode / item["compound_class"] / name
        new_meta = new_root / mode / "metadata" / old_meta.name
        if not old_image.is_file() or not old_meta.is_file():
            raise FileNotFoundError(f"缺少既有候选：{old_image} 或 {old_meta}")
        for source, target in ((old_image, new_image), (old_meta, new_meta)):
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() or target.is_symlink():
                if not target.is_symlink() or target.resolve() != source.resolve():
                    raise FileExistsError(f"拒绝覆盖非预期文件：{target}")
            else:
                target.symlink_to(source)


def make_extended_plan(base_rows: list[dict], base_seed: int) -> list[dict]:
    """从冻结的 90 张计划扩展到 300 张，保持类别、父图和强度分层。"""
    by_compound: dict[str, list[dict]] = {}
    for row in base_rows:
        by_compound.setdefault(row["compound_class"], []).append(row)
    result: list[dict] = []
    for compound, rows in sorted(by_compound.items()):
        rows = sorted(rows, key=lambda row: row["candidate_id"])
        if len(rows) != 90:
            raise ValueError(f"{compound} 既有候选应为 90 张，实际为 {len(rows)}")
        mode = rows[0]["mode"]
        for index, row in enumerate(rows):
            item = dict(row)
            item["denoising_strength"] = strength_for_index(index)
            item["generation_variant"] = f"strength_{item['denoising_strength']:.2f}"
            result.append(item)
        for index in range(90, 300):
            item = dict(rows[index % len(rows)])
            item["candidate_id"] = f'Q_{mode.upper()}_{compound}_{index:04d}'
            item["seed"] = stable_seed(base_seed, mode, compound, index)
            item["denoising_strength"] = strength_for_index(index)
            item["generation_variant"] = f"strength_{item['denoising_strength']:.2f}"
            result.append(item)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=(1, 2, 3, 4))
    args = parser.parse_args()

    for fold in args.folds:
        for route in ROUTES:
            old_root = args.source_root / f"fold_{fold}" / route
            new_root = args.output_root / f"fold_{fold}" / route
            source_config = old_root / "config.yaml"
            base_plan_path = old_root / "candidate_plan.jsonl"
            config = yaml.safe_load(source_config.read_text(encoding="utf-8"))
            base_rows = read_jsonl(base_plan_path)
            plan = make_extended_plan(base_rows, int(config["base_seed"]))

            # 新目录独立保存 C2-MD 结果；只通过软链接读取旧 C1 候选。
            config.update({
                "stage": f"C2-MD-separable-fold{fold}-{route}",
                "candidate_count_per_compound": 300,
                "generation_plan": str(new_root / "candidate_plan_c2_md.jsonl"),
                "output_root": str(new_root),
                "generation_output_root": str(new_root),
                "selection_dir": "selection_c1_c2_md_separable",
                "classifier_data_dir": "classifier_data_c2_md_separable",
            })
            config["classification"]["synthetic_pool_per_class"] = 90
            config["classification"]["synthetic_targets_by_strength"] = {"0.15": 30, "0.22": 30, "0.3": 30}
            new_root.mkdir(parents=True, exist_ok=True)
            (new_root / "candidate_plan_c2_md.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in plan), encoding="utf-8"
            )
            (new_root / "config_c2_md_separable.yaml").write_text(
                yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
            )
            link_existing(old_root, new_root, plan, base_rows[0]["mode"])
            print(json.dumps({"fold": fold, "route": route, "output": str(new_root), "candidates": len(plan)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
