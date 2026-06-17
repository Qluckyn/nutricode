import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

# classify/ 与 passing/ 同级，描述符提取逻辑复用现有 qc_filter.py。
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "passing"))
from qc_filter import ClinicalFeatureExtractor, DataFilter  # noqa: E402


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class ROIDescriptorCache:
    """
    预计算并缓存所有训练图像的 4 维描述符。
    缓存文件为 JSON，key = 图像绝对路径，value = [s1, s2, s3, s4]。
    """

    def __init__(self, cache_path: str, model_path: str):
        # cache_path: 缓存 JSON 文件路径
        # model_path: MediaPipe face_landmarker.task 模型路径
        self.cache_path = cache_path
        self.model_path = model_path
        self.cache: Dict[str, Optional[List[float]]] = {}
        self.raw_cache: Dict[str, Optional[List[float]]] = {}
        self.normalize_stats = None
        self._extractor_cache = {}
        self._load()

    def _load(self):
        """加载已有缓存，兼容旧版直接以路径为顶层 key 的 JSON。"""
        if not os.path.exists(self.cache_path):
            return
        with open(self.cache_path, "r") as f:
            data = json.load(f)

        if isinstance(data, dict) and "descriptors" in data:
            self.cache = data.get("descriptors", {})
            self.raw_cache = data.get("raw_descriptors", {})
            self.normalize_stats = data.get("normalize_stats")
        elif isinstance(data, dict):
            self.cache = data
            self.raw_cache = {}
            self.normalize_stats = None

    def _save(self):
        """把归一化描述符、原始描述符和统计量一起写入磁盘。"""
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        payload = {
            "normalize_stats": self.normalize_stats,
            "descriptors": self.cache,
            "raw_descriptors": self.raw_cache,
        }
        tmp_path = self.cache_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, self.cache_path)

    @staticmethod
    def _image_paths(image_dir: str) -> List[str]:
        """递归收集目录下的图像路径，并使用绝对路径作为缓存 key。"""
        paths = []
        for root, _, files in os.walk(image_dir):
            for name in files:
                if name.lower().endswith(IMG_EXTENSIONS):
                    paths.append(os.path.abspath(os.path.join(root, name)))
        return sorted(paths)

    @staticmethod
    def _view_hint_from_path(img_path: str) -> str:
        n = img_path.lower()
        if "profile" in n:
            return "profile"
        if "three-quarter" in n or "three_quarter" in n or "threequarter" in n:
            return "three_quarter"
        return "front"

    @staticmethod
    def _side_hint_from_path(img_path: str) -> Optional[str]:
        n = img_path.lower()
        if "left_" in n or "_left" in n:
            return "left"
        if "right_" in n or "_right" in n:
            return "right"
        return None

    def _get_extractor(self, img_path: str) -> ClinicalFeatureExtractor:
        """按视角复用 extractor，避免每张图重复初始化 MediaPipe。"""
        view_hint = self._view_hint_from_path(img_path)
        side_hint = self._side_hint_from_path(img_path)
        key = (view_hint, side_hint)
        if key in self._extractor_cache:
            return self._extractor_cache[key]

        if view_hint == "profile":
            extractor = ClinicalFeatureExtractor(
                self.model_path,
                use_insightface_fallback=True,
                insightface_crop_expands=(1.60, 1.90, 2.20),
                insightface_det_thresh=0.25,
                insightface_primary=True,
                view_hint=view_hint,
                side_hint=side_hint,
            )
        elif view_hint == "three_quarter":
            extractor = ClinicalFeatureExtractor(
                self.model_path,
                use_insightface_fallback=True,
                insightface_crop_expands=(1.45, 1.70, 2.00),
                insightface_det_thresh=0.30,
                insightface_primary=True,
                view_hint=view_hint,
                side_hint=side_hint,
            )
        else:
            extractor = ClinicalFeatureExtractor(
                self.model_path,
                use_insightface_fallback=True,
                insightface_crop_expands=(1.30, 1.60, 1.90),
                insightface_det_thresh=0.35,
                insightface_primary=False,
                view_hint=view_hint,
                side_hint=side_hint,
            )

        self._extractor_cache[key] = extractor
        return extractor

    def _extract_raw_descriptor(self, img_path: str) -> Optional[List[float]]:
        """提取单张图像的原始 4 维临床描述符。"""
        extractor = self._get_extractor(img_path)
        metrics = extractor.calculate_metrics(img_path)
        if metrics is None:
            return None
        desc = DataFilter._metrics_to_vector(metrics)
        if not np.isfinite(desc).all():
            return None
        return desc.astype(float).tolist()

    @staticmethod
    def _normalize(raw_desc: List[float], stats: dict) -> List[float]:
        """使用真实训练图统计量做 sigmoid-zscore 归一化，输出限定在 [0, 1]。"""
        x = np.asarray(raw_desc, dtype=np.float64)
        mean = np.asarray(stats["mean"], dtype=np.float64)
        std = np.asarray(stats["std"], dtype=np.float64)
        z = (x - mean) / np.maximum(std, 1e-6)
        y = 1.0 / (1.0 + np.exp(-z))
        return y.astype(float).tolist()

    def _compute_real_stats(self, real_paths: List[str]):
        """只用真实训练图的有效原始描述符计算归一化统计量。"""
        real_desc = [
            self.raw_cache[p]
            for p in real_paths
            if self.raw_cache.get(p) is not None
        ]
        if not real_desc:
            raise RuntimeError("真实训练图没有有效 ROI 描述符，无法计算归一化统计量。")
        arr = np.asarray(real_desc, dtype=np.float64)
        std = np.std(arr, axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        self.normalize_stats = {
            "mean": np.mean(arr, axis=0).astype(float).tolist(),
            "std": std.astype(float).tolist(),
            "normalization": "sigmoid_zscore",
            "source": "first_image_dir",
            "n_real_valid": int(arr.shape[0]),
        }

    def build(self, image_dirs: List[str]):
        """
        遍历 image_dirs 下所有图像，提取描述符并保存。
        已存在缓存的图像跳过；提取失败的图像记录 null。
        """
        if not image_dirs:
            raise ValueError("image_dirs 不能为空。")

        image_dirs = [os.path.abspath(d) for d in image_dirs]
        all_paths_by_dir = [self._image_paths(d) for d in image_dirs]
        real_paths = all_paths_by_dir[0]

        for image_dir, paths in zip(image_dirs, all_paths_by_dir):
            extracted = 0
            skipped = 0
            failed = 0
            print(f"\n[ROI] Processing {image_dir}")
            for img_path in tqdm(paths, desc=os.path.basename(image_dir) or image_dir):
                if img_path in self.raw_cache:
                    skipped += 1
                    continue
                raw_desc = self._extract_raw_descriptor(img_path)
                self.raw_cache[img_path] = raw_desc
                if raw_desc is None:
                    self.cache[img_path] = None
                    failed += 1
                else:
                    extracted += 1

            print(
                f"[ROI] dir={image_dir} total={len(paths)} "
                f"extracted={extracted} skipped={skipped} failed={failed}"
            )
            self._save()

            if image_dir == image_dirs[0] or self.normalize_stats is None:
                self._compute_real_stats(real_paths)
                # 真实统计量确定后，重新归一化所有已提取成功的描述符。
                for path, raw_desc in self.raw_cache.items():
                    self.cache[path] = (
                        None if raw_desc is None else self._normalize(raw_desc, self.normalize_stats)
                    )
                self._save()

        if self.normalize_stats is None:
            self._compute_real_stats(real_paths)

        for path, raw_desc in self.raw_cache.items():
            self.cache[path] = None if raw_desc is None else self._normalize(raw_desc, self.normalize_stats)
        self._save()

        total = len(self.cache)
        failures = sum(1 for v in self.cache.values() if v is None)
        print(f"\n[ROI] cache saved: {self.cache_path}")
        print(f"[ROI] total_entries={total} valid={total - failures} failed={failures}")
        print(f"[ROI] normalize_stats={self.normalize_stats}")

    def get(self, img_path: str) -> Optional[List[float]]:
        """返回 [s1, s2, s3, s4] 或 None（提取失败时）。"""
        return self.cache.get(os.path.abspath(img_path))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dirs", nargs="+", required=True, help="需要构建缓存的图像目录列表")
    parser.add_argument("--cache_path", required=True, help="描述符缓存 JSON 路径")
    parser.add_argument("--model_path", required=True, help="MediaPipe face_landmarker.task 路径")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cache = ROIDescriptorCache(cache_path=args.cache_path, model_path=args.model_path)
    cache.build(args.image_dirs)
