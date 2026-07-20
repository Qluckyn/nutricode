"""Pose02 V3正式实验专用的真实/合成固定曝光采样器。"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict

from torch.utils.data import Sampler


HAND_PATTERN = re.compile(r"^(?P<subject>\d+)_(?P<pose>01|02)(?:$|[_\-.])")
SYNTH_MARKERS = ("__pose02_v3_synth__", "__pose02_synth__")


class HandPose02V3MixedSampler(Sampler):
    """按冻结预算抽取pose02真实图和合成图，并执行合成池无放回循环。"""

    def __init__(
        self,
        dataset,
        real_per_class=10,
        synth_per_class=2,
        synth_pool_per_class=49,
        max_synth_per_parent_per_epoch=1,
        seed=22,
        history_path=None,
    ):
        self.dataset = dataset
        self.real_per_class = int(real_per_class)
        self.synth_per_class = int(synth_per_class)
        self.synth_pool_per_class = int(synth_pool_per_class)
        self.max_synth_per_parent_per_epoch = int(max_synth_per_parent_per_epoch)
        self.seed = int(seed)
        self.history_path = history_path
        self.epoch = 0
        self.history = {}
        if min(self.real_per_class, self.synth_per_class, self.synth_pool_per_class, self.max_synth_per_parent_per_epoch) <= 0:
            raise ValueError("Pose02混合采样的真实、合成曝光量及合成池大小必须为正数")
        required_classes = {"malnourished_hand", "normal_hand"}
        if set(dataset.class_to_idx) != required_classes:
            raise ValueError("V3混合采样要求两个固定手部类别")

        self.real_indices = defaultdict(lambda: defaultdict(list))
        self.synth_indices = defaultdict(list)
        self.synth_parent = {}
        for index, (path, label) in enumerate(dataset.samples):
            filename = os.path.basename(path)
            match = HAND_PATTERN.match(filename)
            if not match or match.group("pose") != "02":
                raise ValueError(f"V3采样器只能读取可解析的pose02文件：{path}")
            subject_id = match.group("subject")
            if any(marker in filename for marker in SYNTH_MARKERS):
                self.synth_indices[label].append(index)
                self.synth_parent[index] = subject_id
            else:
                self.real_indices[label][subject_id].append(index)

        for label in dataset.class_to_idx.values():
            bad_real = {
                subject: len(indices)
                for subject, indices in self.real_indices[label].items()
                if len(indices) != 1
            }
            if bad_real:
                raise ValueError(f"每名真实受试者应恰有一张pose02图：label={label}, {bad_real}")
            if len(self.real_indices[label]) < self.real_per_class:
                raise ValueError(f"真实受试者不足：label={label}")
            if len(self.synth_indices[label]) != self.synth_pool_per_class:
                raise ValueError(
                    f"Pose02每折每类合成池数量不符：label={label}, "
                    f"expected={self.synth_pool_per_class}, "
                    f"actual={len(self.synth_indices[label])}"
                )

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return 2 * (self.real_per_class + self.synth_per_class)

    def _stable_seed(self, label: int, cycle: int) -> int:
        if (self.real_per_class, self.synth_per_class, self.synth_pool_per_class) == (10, 2, 49):
            # 保留V3.5既有随机域，确保未来复核时采样序列不变。
            text = f"pose02-v3.5:{self.seed}:{label}:{cycle}"
        else:
            # 新协议把冻结曝光量和池大小写入随机域，避免意外共享抽样序列。
            text = (
                f"pose02-mixed:{self.seed}:{self.real_per_class}:"
                f"{self.synth_per_class}:{self.synth_pool_per_class}:{label}:{cycle}"
            )
        return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")

    def _cycle(self, label: int, cycle: int, previous_parent: str | None, prefix_parents: list[str]) -> list[int]:
        """生成每张图恰出现一次的循环，并避免相邻及跨循环边界父图重复。"""
        base = sorted(self.synth_indices[label], key=lambda index: self.dataset.samples[index][0])
        rng = random.Random(self._stable_seed(label, cycle))
        # 确定性重排，避免相邻合成图来自同一父受试者。
        for _ in range(10000):
            ordered = base.copy()
            rng.shuffle(ordered)
            parents = [self.synth_parent[index] for index in ordered]
            if previous_parent is not None and parents[0] == previous_parent:
                continue
            if any(left == right for left, right in zip(parents, parents[1:])):
                continue
            # 合成池大小不一定能被每轮曝光量整除；跨循环的同一epoch也必须父图唯一。
            needed = self.synth_per_class - len(prefix_parents)
            cross_epoch = prefix_parents + parents[:needed]
            if len(set(cross_epoch)) != len(cross_epoch):
                continue
            return ordered
        raise RuntimeError("10000次确定性重排后仍无法满足父图相邻约束")

    def _synthetic_stream(self, label: int, count: int) -> list[int]:
        """按冻结池大小轮转取样；小池耗尽后才确定性复用候选。"""
        if self.max_synth_per_parent_per_epoch > 1:
            # 大比例扩增时单轮合成数可大于父受试者数，改为受控轮转。
            # 正式1500张池在40轮内不会耗尽；小规模预实验的90张池耗尽后，
            # 才以确定性顺序重新装填，从而允许跨epoch复用而不改变单轮父图上限。
            source_queues = defaultdict(list)
            for index in sorted(self.synth_indices[label], key=lambda value: self.dataset.samples[value][0]):
                source_queues[self.synth_parent[index]].append(index)

            def refill(cycle: int) -> dict[int, list[int]]:
                queues = {parent: values.copy() for parent, values in source_queues.items()}
                for parent, values in queues.items():
                    random.Random(self._stable_seed(label, int(parent) + 1000000 * cycle)).shuffle(values)
                return queues

            queues = refill(0)
            stream, cycle = [], 0
            while len(stream) < count:
                rng = random.Random(self._stable_seed(label, 100000 + cycle))
                parents = sorted(parent for parent, values in queues.items() if values)
                if not parents:
                    cycle += 1
                    queues = refill(cycle)
                    continue
                rng.shuffle(parents)
                selected, used = [], defaultdict(int)
                while len(selected) < self.synth_per_class:
                    eligible = []
                    for parent in parents:
                        if used[parent] >= self.max_synth_per_parent_per_epoch:
                            continue
                        if any(index not in selected for index in queues[parent]):
                            eligible.append(parent)
                    if not eligible:
                        # 当前轮已用尽未重复的候选，重置池后继续补足本轮；
                        # 同一epoch仍通过 selected 排除重复图片。
                        cycle += 1
                        refreshed = refill(cycle)
                        for parent, values in refreshed.items():
                            queues[parent].extend(values)
                        continue
                    parent = eligible[len(selected) % len(eligible)]
                    candidate = next(index for index in reversed(queues[parent]) if index not in selected)
                    queues[parent].remove(candidate)
                    selected.append(candidate)
                    used[parent] += 1
                stream.extend(selected)
                cycle += 1
            return stream[:count]
        stream = []
        cycle = 0
        previous_parent = None
        while len(stream) < count:
            remainder = len(stream) % self.synth_per_class
            prefix = [self.synth_parent[index] for index in stream[-remainder:]] if remainder else []
            current = self._cycle(label, cycle, previous_parent, prefix)
            stream.extend(current)
            previous_parent = self.synth_parent[current[-1]]
            cycle += 1
        return stream[:count]


    def _small_pool_epoch_sample(self, label: int) -> list[int]:
        """小于累计曝光量的候选池：每个epoch确定性重采样，允许跨epoch复用。"""
        queues = defaultdict(list)
        for index in sorted(self.synth_indices[label], key=lambda value: self.dataset.samples[value][0]):
            queues[self.synth_parent[index]].append(index)
        rng = random.Random(self._stable_seed(label, 200000 + self.epoch))
        for values in queues.values():
            rng.shuffle(values)
        selected, used = [], defaultdict(int)
        while len(selected) < self.synth_per_class:
            eligible = [parent for parent, values in sorted(queues.items()) if values and used[parent] < self.max_synth_per_parent_per_epoch]
            if not eligible:
                raise RuntimeError("小规模合成池无法满足单父图曝光上限")
            parent = eligible[rng.randrange(len(eligible))]
            selected.append(queues[parent].pop())
            used[parent] += 1
        return selected

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        selected_real, selected_synth = [], []
        labels = sorted(self.dataset.class_to_idx.values())
        for label in labels:
            subjects = sorted(self.real_indices[label], key=int)
            chosen_subjects = sorted(rng.sample(subjects, self.real_per_class), key=int)
            selected_real.extend(self.real_indices[label][subject][0] for subject in chosen_subjects)

            # 90张等小池不足以覆盖40轮无放回曝光时，改为每轮确定性重采样；
            # 正式1500张池仍保持原有跨轮无放回轮转。
            if len(self.synth_indices[label]) < (self.epoch + 1) * self.synth_per_class:
                chosen_synth = self._small_pool_epoch_sample(label)
            else:
                stream = self._synthetic_stream(label, (self.epoch + 1) * self.synth_per_class)
                start = self.epoch * self.synth_per_class
                chosen_synth = stream[start:start + self.synth_per_class]
            parents = [self.synth_parent[index] for index in chosen_synth]
            if max(Counter(parents).values()) > self.max_synth_per_parent_per_epoch:
                raise RuntimeError("同一epoch同类合成后代超过单父图曝光上限")
            selected_synth.extend(chosen_synth)

        indices = selected_real + selected_synth
        rng.shuffle(indices)
        record = {
            "epoch": self.epoch,
            "seed": self.seed + self.epoch,
            "real_count": len(selected_real),
            "synthetic_count": len(selected_synth),
            "real_count_per_class": self.real_per_class,
            "synthetic_count_per_class": self.synth_per_class,
            "selected_real_paths": [self.dataset.samples[index][0] for index in selected_real],
            "selected_synthetic_paths": [self.dataset.samples[index][0] for index in selected_synth],
            "selected_synthetic_parent_ids": [self.synth_parent[index] for index in selected_synth],
        }
        self.history[str(self.epoch)] = record
        print(
            f"[Pose02-V3混合采样] epoch={self.epoch}, "
            f"real={len(selected_real)}, synthetic={len(selected_synth)}"
        )
        if self.history_path:
            path = os.path.abspath(self.history_path)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(self.history, handle, ensure_ascii=False, indent=2)
        return iter(indices)
