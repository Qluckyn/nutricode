"""C2-QC 的第一背侧骨间肌肌腹 ROI 几何定义。

ROI 对准拇指与食指掌骨之间、略偏向拇指根部的手背肌腹区，而非狭义的
V 字皮肤缝隙。所有坐标均为 [0, 1] 归一化坐标。
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class RoiSpec:
    """单手 ROI 的中心、朝向及椭圆半轴。"""
    center: tuple[float, float]
    angle_deg: float
    long_radius: float
    short_radius: float


def first_dorsal_interosseous_roi(hand: np.ndarray) -> RoiSpec | None:
    """由 MediaPipe 21 点定位拇指根部—第一背侧骨间肌肌腹 ROI。

    1 为拇指 CMC、2 为拇指 MCP。圆心从 CMC 向 MCP 移动 45%，
    长轴取该近端拇指节段的法向方向，以覆盖第一、二掌骨间的局部肌腹。
    """
    hand = np.asarray(hand, dtype=np.float32)
    if hand.shape != (21, 2) or not np.isfinite(hand).all():
        return None
    thumb_cmc, thumb_mcp = hand[1], hand[2]
    thumb_axis = thumb_mcp - thumb_cmc
    thumb_base_length = float(np.linalg.norm(thumb_axis))
    # 肌腹横跨第一、二掌骨间隙，长轴应近似垂直于拇指近端节段。
    axis = np.asarray((-thumb_axis[1], thumb_axis[0]), dtype=np.float32)
    if thumb_base_length < 1e-3:
        return None
    center = thumb_cmc + 0.45 * thumb_axis
    # 半轴相对拇指近端节段固定，覆盖局部肌腹而不扩张至腕部或手背中央。
    return RoiSpec(
        center=(float(center[0]), float(center[1])),
        angle_deg=math.degrees(math.atan2(float(axis[1]), float(axis[0]))),
        long_radius=0.56 * thumb_base_length,
        short_radius=0.38 * thumb_base_length,
    )


def roi_specs_for_two_hands(keypoints: np.ndarray) -> list[RoiSpec] | None:
    """对按画面 x 坐标排序的 42 点双手关键点分别建立 ROI。"""
    points = np.asarray(keypoints, dtype=np.float32)
    if points.shape != (42, 2):
        return None
    specs = [first_dorsal_interosseous_roi(points[:21]), first_dorsal_interosseous_roi(points[21:])]
    return specs if all(spec is not None for spec in specs) else None
