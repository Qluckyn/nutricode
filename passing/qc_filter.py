import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import glob
from tqdm import tqdm
import shutil
import json
import multiprocessing
from functools import partial
from typing import Optional, Tuple
import argparse
import warnings

# ================= Configuration =================

# ================= 3-view (paper) subset =================
# Default to the 3-view subset for BOTH labels (normal + malnourished).
# You can still override this at runtime via --only-groups.
DEFAULT_ONLY_GROUPS = [
    "normal_front_face",
    "normal_left_three-quarter_face",
    "normal_right_three-quarter_face",
    "malnourished_front_face",
    "malnourished_left_three-quarter_face",
    "malnourished_right_three-quarter_face",
]

# DEFAULT_ONLY_GROUPS = [
#     "normal_face",
#     "malnourished_face",
# ]

# Paths provided by user
# REAL_DATA_ROOTS = [
#     "/root/autodl-tmp/datadream/data/malnutrition/real_train_fewshot/seed0",
#     "/root/autodl-tmp/datadream/data/normal_train_fewshot/seed0"  # Assuming similar structure
# ]
# REAL_DATA_ROOTS = [
#     "/root/autodl-tmp/datadream/data/my_dataset_binary/seed0"
# ]
REAL_DATA_ROOTS = [
    "/root/autodl-tmp/runs/cv/fold_4/real_train_groups/seed0"
]

SYNTHETIC_DATA_ROOT = "/root/autodl-tmp/datadream_outputs/generated_images/my_dataset/sd2.1/gs3.5_nis50/shot20_seed0_template1_lr0.0001_ep240/train"
# SYNTHETIC_DATA_ROOT = "/root/autodl-tmp/datadream_outputs/generated_images/my_dataset_binary/sd2.1/gs3.5_nis50/shot20_seed0_template1_lr0.0001_ep240/train"
# SYNTHETIC_DATA_ROOT = "/root/autodl-tmp/runs/cv/fold_4/synth_raw/my_dataset/sd2.1/gs3.5_nis50/shot20_seed0_template1_lr0.0001_ep240/train"

OUTPUT_DIR = "/root/autodl-tmp/datadream_outputs/generated_images/my_dataset/sd2.1/gs3.5_nis50/shot20_seed0_template1_lr0.0001_ep240/filtered_train"
# OUTPUT_DIR = "/root/autodl-tmp/datadream_outputs/generated_images/my_dataset_binary/sd2.1/gs3.5_nis50/shot20_seed0_template1_lr0.0001_ep240/filtered_train"
# OUTPUT_DIR = "/root/autodl-tmp/runs/cv/fold_4/synth_raw/my_dataset/sd2.1/gs3.5_nis50/shot20_seed0_template1_lr0.0001_ep240/filtered_train"


MODEL_PATH = '/root/autodl-tmp/face_landmarker.task' 

# ================= Optional: InsightFace bbox fallback (Strategy A) =================
# If FaceLandmarker returns 0 faces (common in extreme profile views), we can
# use InsightFace to detect a face bbox, crop with padding, and retry FaceLandmarker.
# This keeps the existing MediaPipe-468 ROI indices unchanged.
USE_INSIGHTFACE_FALLBACK = True
INSIGHTFACE_CTX_ID = 0  # 0 for GPU0 by default; will auto-fallback to CPU (-1) if GPU init fails
# Default InsightFace detector input size. For profile views, we will try a larger size too.
INSIGHTFACE_DET_SIZE = (640, 640)
INSIGHTFACE_DET_SIZES = (INSIGHTFACE_DET_SIZE, (960, 960))

# Crop padding factors (will be overridden per-view in DataFilter)
INSIGHTFACE_CROP_EXPANDS = (1.30, 1.60, 1.90)

# ================= MediaPipe FaceLandmarker knobs (loosened for higher recall) =================
# NOTE: Different mediapipe versions expose different option fields. We will
# attempt to pass these options, and gracefully fall back if unsupported.
MP_NUM_FACES = 2
MP_MIN_FACE_DETECTION_CONFIDENCE = 0.30
MP_MIN_FACE_PRESENCE_CONFIDENCE = 0.30
MP_MIN_TRACKING_CONFIDENCE = 0.30
MP_OUTPUT_FACIAL_TRANSFORMATION_MATRIXES = False

# ================= Stage2: SGA-guided clinical filtering =================
# We model the 4D phenotype descriptor s(x) on REAL data for each
# view-attribute folder (e.g., malnourished_front_face), then filter synthetic
# samples by Mahalanobis distance with a quantile threshold.
MAHALANOBIS_TAU_QUANTILE = 95  # keep samples within this real-distance quantile
# Base covariance regularization for stable inverse. We will increase this when n is small.
COV_REG_EPS = 1e-6

# Shrinkage covariance options: 'ledoit_wolf' (default), 'oas', 'ridge', 'none'
DEFAULT_COV_SHRINKAGE = "ledoit_wolf"
# Top-K selection after P95. If k_abs is None, we use k_beta * n_real.
DEFAULT_K_BETA = 8.0

try:
    from sklearn.covariance import LedoitWolf, OAS
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False


def _estimate_covariance(X: np.ndarray, method: str = "ledoit_wolf", reg_eps: float = 1e-6) -> np.ndarray:
    method = (method or "none").lower()
    if method in ("lw", "ledoitwolf", "ledoit_wolf", "ledoit-wolf"):
        if _HAS_SKLEARN:
            cov = LedoitWolf().fit(X).covariance_
        else:
            warnings.warn("scikit-learn not available; falling back to ridge covariance.")
            cov = np.cov(X, rowvar=False)
    elif method == "oas":
        if _HAS_SKLEARN:
            cov = OAS().fit(X).covariance_
        else:
            warnings.warn("scikit-learn not available; falling back to ridge covariance.")
            cov = np.cov(X, rowvar=False)
    elif method in ("ridge", "diag", "regularized"):
        cov = np.cov(X, rowvar=False)
    elif method in ("none", "empirical"):
        cov = np.cov(X, rowvar=False)
    else:
        raise ValueError(f"Unsupported covariance method: {method}")

    cov = np.asarray(cov, dtype=np.float64)
    if reg_eps and reg_eps > 0:
        cov = cov + (float(reg_eps) * np.eye(cov.shape[0], dtype=np.float64))
    return cov

# Minimum samples to learn per-group model (will be relaxed per-view).
MIN_REAL_SAMPLES_PER_GROUP = 10

# Mask/metric robustness knobs (especially important for profile views)
FULL_FACE_MIN_PIXELS = 120
TEMPORAL_MIN_PIXELS = 20
ORBITAL_MIN_PIXELS = 20
MALAR_MIN_PIXELS = 30
JAW_MIN_PIXELS = 10

# Avoid eroding already-small masks (erosion can delete thin/partial-profile ROIs)
ERODE_ONLY_IF_AREA_AT_LEAST = 300

# ================= ROI Definitions (from visualize_landmarks.py) =================

SYMMETRY_MAP = {
    10: 10, 151: 151, 9: 9, 8: 8, 168: 168, 6: 6, 197: 197, 195: 195, 5: 5, 4: 4, 1: 1, 19: 19, 94: 94, 2: 2, 164: 164, 0: 0, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18, 200: 200, 199: 199, 175: 175, 152: 152,
    103: 332, 104: 333, 63: 293, 46: 276, 124: 353, 35: 265, 143: 372, 34: 264, 127: 356,
    52: 282, 53: 283, 65: 295, 55: 285, 70: 300, 107: 336, 66: 296, 105: 334,
    33: 263, 246: 466, 161: 388, 160: 387, 159: 386, 158: 385, 157: 384, 173: 398,
    133: 362, 155: 382, 154: 381, 153: 380, 145: 374, 144: 373, 163: 390, 7: 249,
    193: 417, 245: 465, 128: 357, 232: 452, 231: 451, 230: 450, 229: 449, 228: 448, 31: 261,
    122: 351, 188: 412, 114: 343, 47: 277, 100: 329, 131: 360, 119: 348, 117: 346, 111: 340,
    21: 251, 54: 284, 67: 297, 109: 338, 162: 389, 127: 356, 234: 454, 93: 323, 132: 361, 58: 288, 172: 397, 136: 365, 150: 379, 149: 378, 176: 400, 148: 377,
    116: 345, 227: 447, 
    118: 347, 101: 330, 36: 266, 203: 423, 165: 391, 92: 322, 186: 410, 212: 432, 210: 430, 169: 394, 135: 364, 138: 367, 215: 435, 177: 401, 137: 366,
    172: 397, 136: 365, 150: 379, 149: 378, 176: 400, 148: 377, 152: 152, 377: 148, 400: 176, 378: 149, 379: 150, 365: 136, 397: 172,
    215: 435, 138: 367, 135: 364
}

def get_symmetric_indices(left_indices):
    right_indices = []
    for idx in left_indices:
        if idx in SYMMETRY_MAP:
            right_indices.append(SYMMETRY_MAP[idx])
        else:
            found = False
            for k, v in SYMMETRY_MAP.items():
                if v == idx and k != v:
                    right_indices.append(k)
                    found = True
                    break
            if not found:
                right_indices.append(idx)
    return right_indices

# 4个ROI区域的关键点索引（论文图2）
LEFT_TEMPORAL = [103, 104, 63, 46, 124, 35, 143, 34, 127, 162, 21, 54]
LEFT_ORBITAL = [35, 124, 46, 53, 52, 65, 55, 193, 245, 128, 232, 231, 230, 229, 228, 31]
LEFT_MALAR = [227, 116, 117, 36, 203, 165, 92, 186, 212, 210, 169, 135, 138, 215, 177, 137]
LEFT_JAW_LINE = [172, 136, 150, 149, 176, 148, 152]

RIGHT_TEMPORAL = get_symmetric_indices(LEFT_TEMPORAL)
RIGHT_ORBITAL = get_symmetric_indices(LEFT_ORBITAL)
RIGHT_MALAR = get_symmetric_indices(LEFT_MALAR)
RIGHT_JAW_LINE = get_symmetric_indices(LEFT_JAW_LINE)
# FULL_JAW_LINE = LEFT_JAW_LINE[:-1] + [152] + get_symmetric_indices(LEFT_JAW_LINE[:-1])[::-1]

# Face Oval for Full Face Skin (Standard MediaPipe indices)
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# ================= Classes =================

class ClinicalFeatureExtractor:
    def __init__(
        self,
        model_path,
        use_insightface_fallback: bool = USE_INSIGHTFACE_FALLBACK,
        insightface_ctx_id: int = INSIGHTFACE_CTX_ID,
        insightface_det_size: tuple = INSIGHTFACE_DET_SIZE,
        insightface_det_sizes: tuple = INSIGHTFACE_DET_SIZES,
        insightface_crop_expands: tuple = INSIGHTFACE_CROP_EXPANDS,
        insightface_det_thresh: float = 0.35,
        insightface_primary: bool = False,
        mp_num_faces: int = MP_NUM_FACES,
        mp_min_face_detection_confidence: float = MP_MIN_FACE_DETECTION_CONFIDENCE,
        mp_min_face_presence_confidence: float = MP_MIN_FACE_PRESENCE_CONFIDENCE,
        mp_min_tracking_confidence: float = MP_MIN_TRACKING_CONFIDENCE,
        mp_output_facial_transformation_matrixes: bool = MP_OUTPUT_FACIAL_TRANSFORMATION_MATRIXES,
        view_hint: str = "front",
        side_hint: Optional[str] = None,
    ):
        base_options = python.BaseOptions(model_asset_path=model_path)
        # mediapipe API compatibility: some fields may not exist in older versions.
        options_kwargs = dict(
            base_options=base_options,
            output_face_blendshapes=False,
            num_faces=int(mp_num_faces),
        )
        # Best-effort relaxed thresholds
        options_kwargs.update(
            dict(
                min_face_detection_confidence=float(mp_min_face_detection_confidence),
                min_face_presence_confidence=float(mp_min_face_presence_confidence),
                min_tracking_confidence=float(mp_min_tracking_confidence),
                output_facial_transformation_matrixes=bool(mp_output_facial_transformation_matrixes),
            )
        )
        try:
            options = vision.FaceLandmarkerOptions(**options_kwargs)
        except TypeError:
            # Fallback to minimal supported options
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=False,
                num_faces=int(mp_num_faces),
            )
        self.detector = vision.FaceLandmarker.create_from_options(options)

        self.use_insightface_fallback = bool(use_insightface_fallback)
        self.insightface_ctx_id = insightface_ctx_id
        self.insightface_det_size = insightface_det_size
        self.insightface_det_sizes = tuple(insightface_det_sizes) if insightface_det_sizes else (insightface_det_size,)
        self.insightface_crop_expands = insightface_crop_expands
        self.insightface_det_thresh = float(insightface_det_thresh) if insightface_det_thresh is not None else 0.35
        self.insightface_primary = bool(insightface_primary)

        self.view_hint = (view_hint or "front").lower()
        self.side_hint = (side_hint.lower() if isinstance(side_hint, str) else None)
        self._face_analyzer = None
        self._insightface_init_error = None
        self._insightface_warned = False
        self._insightface_prepared_det_size = None

        if self.use_insightface_fallback:
            self._init_insightface()

    def _init_insightface(self):
        try:
            from insightface.app import FaceAnalysis

            # Prefer GPU if available; fall back to CPU.
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self._face_analyzer = FaceAnalysis(name="buffalo_l", providers=providers)
            # Prefer requested ctx_id, but auto-fallback to CPU if GPU init fails.
            try:
                self._face_analyzer.prepare(ctx_id=self.insightface_ctx_id, det_size=self.insightface_det_size)
            except Exception:
                if int(self.insightface_ctx_id) != -1:
                    self._face_analyzer.prepare(ctx_id=-1, det_size=self.insightface_det_size)
                    self.insightface_ctx_id = -1
                else:
                    raise
            self._insightface_prepared_det_size = tuple(self.insightface_det_size)
            # If available, lower detector threshold a bit for recall.
            if hasattr(self._face_analyzer, "det_thresh"):
                try:
                    target = float(self.insightface_det_thresh)
                    self._face_analyzer.det_thresh = min(float(getattr(self._face_analyzer, "det_thresh")), target)
                except Exception:
                    pass
        except Exception as e:
            self._face_analyzer = None
            self._insightface_init_error = str(e)

    def _warn_insightface_once(self):
        if self._insightface_warned:
            return
        self._insightface_warned = True
        if self._insightface_init_error:
            print(
                "Warning: InsightFace fallback is enabled but could not be initialized. "
                f"Reason: {self._insightface_init_error}. Falling back to MediaPipe-only."
            )

    @staticmethod
    def _select_largest_bbox(faces) -> np.ndarray:
        # faces: list of insightface Face objects
        best = None
        best_area = -1.0
        for f in faces:
            if not hasattr(f, "bbox"):
                continue
            b = np.array(f.bbox, dtype=np.float32)  # [x1,y1,x2,y2]
            area = float(max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]))
            if area > best_area:
                best_area = area
                best = b
        return best

    @staticmethod
    def _crop_with_expand(image_bgr: np.ndarray, bbox_xyxy: np.ndarray, expand: float) -> Optional[Tuple[np.ndarray, Tuple[int, int]]]:
        h, w = image_bgr.shape[:2]
        x1, y1, x2, y2 = bbox_xyxy
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        bw = (x2 - x1) * expand
        bh = (y2 - y1) * expand

        x0 = int(np.floor(cx - bw / 2.0))
        y0 = int(np.floor(cy - bh / 2.0))
        x3 = int(np.ceil(cx + bw / 2.0))
        y3 = int(np.ceil(cy + bh / 2.0))

        x0 = max(0, x0)
        y0 = max(0, y0)
        x3 = min(w, x3)
        y3 = min(h, y3)

        if x3 - x0 < 32 or y3 - y0 < 32:
            return None

        crop = image_bgr[y0:y3, x0:x3]
        return crop, (x0, y0)

    def _insightface_get_faces(self, image_bgr: np.ndarray):
        if self._face_analyzer is None:
            return []
        # Try multiple det_size values for robustness
        for det_size in self.insightface_det_sizes:
            try:
                det_size = tuple(det_size)
                if self._insightface_prepared_det_size != det_size:
                    self._face_analyzer.prepare(ctx_id=self.insightface_ctx_id, det_size=det_size)
                    self._insightface_prepared_det_size = det_size
                faces = self._face_analyzer.get(image_bgr)
            except Exception:
                faces = []
            if faces:
                return faces
        return []

    def _detect_landmarks_rgb(self, image_rgb: np.ndarray) -> Optional[np.ndarray]:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        try:
            detection_result = self.detector.detect(mp_image)
        except Exception:
            return None

        if not detection_result.face_landmarks:
            return None

        # If multiple faces are returned, pick the one with largest landmark bbox.
        best_coords = None
        best_area = -1.0
        h, w = image_rgb.shape[:2]
        for landmarks in detection_result.face_landmarks:
            coords = np.array([(lm.x * w, lm.y * h) for lm in landmarks], dtype=np.float32)
            x1 = float(np.min(coords[:, 0])); x2 = float(np.max(coords[:, 0]))
            y1 = float(np.min(coords[:, 1])); y2 = float(np.max(coords[:, 1]))
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if area > best_area:
                best_area = area
                best_coords = coords
        return best_coords

    def get_landmarks(self, image_np):
        # Two strategies:
        # - default: MediaPipe full-image first, then InsightFace bbox->crop retry
        # - primary crop: InsightFace bbox->crop first (better for profile/extreme pose)

        if self.insightface_primary and self.use_insightface_fallback and self._face_analyzer is not None:
            try:
                image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                faces = self._insightface_get_faces(image_bgr)
            except Exception:
                faces = []
            if faces:
                bbox = self._select_largest_bbox(faces)
                if bbox is not None:
                    for expand in self.insightface_crop_expands:
                        cropped = self._crop_with_expand(image_bgr, bbox, float(expand))
                        if cropped is None:
                            continue
                        crop_bgr, (x0, y0) = cropped
                        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                        # 策略1：直接对全图用MediaPipe检测
                        coords_crop = self._detect_landmarks_rgb(crop_rgb)
                        if coords_crop is None:
                            continue
                        coords_crop[:, 0] += float(x0)
                        coords_crop[:, 1] += float(y0)
                        return coords_crop

        # 1) Try directly on the full image.
        coords = self._detect_landmarks_rgb(image_np)
        if coords is not None:
            return coords

        # 2) Fallback: use InsightFace bbox -> crop -> retry FaceLandmarker.
        if not self.use_insightface_fallback:
            return None
        if self._face_analyzer is None:
            self._warn_insightface_once()
            return None

        try:
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            faces = self._insightface_get_faces(image_bgr)
        except Exception:
            faces = []

        if not faces:
            return None

        # # 策略2：MediaPipe失败 → InsightFace先定位人脸框
        #        → 裁剪人脸区域 → 再用MediaPipe检测
        # 侧脸视角（three-quarter）MediaPipe容易失败，所以需要InsightFace兜底。
        bbox = self._select_largest_bbox(faces)
        if bbox is None:
            return None

        for expand in self.insightface_crop_expands: # 尝试不同扩展比例
            cropped = self._crop_with_expand(image_bgr, bbox, float(expand))
            if cropped is None:
                continue
            crop_bgr, (x0, y0) = cropped
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            coords_crop = self._detect_landmarks_rgb(crop_rgb)
            if coords_crop is None:
                continue
            coords_crop[:, 0] += float(x0)
            coords_crop[:, 1] += float(y0)
            return coords_crop

        return None

    def get_mask_from_points(self, shape, points, mode='hull', thickness=1):
        mask = np.zeros(shape[:2], dtype=np.uint8)
        h, w = shape[:2]
        pts = points.astype(np.int32)

        # Clip to image bounds to avoid empty/out-of-frame hulls on extreme poses.
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)

        # Remove duplicate points (convexHull/plotting can misbehave with degenerate sets)
        pts = np.unique(pts, axis=0)
        if mode in ('hull', 'poly') and pts.shape[0] < 3:
            return mask
        if mode == 'line' and pts.shape[0] < 2:
            return mask
        
        if mode == 'hull':
            hull = cv2.convexHull(pts)
            cv2.fillPoly(mask, [hull], 1)
        elif mode == 'poly':
            cv2.fillPoly(mask, [pts], 1)
        elif mode == 'line':
            cv2.polylines(mask, [pts], isClosed=False, color=1, thickness=thickness)
            
        return mask

    def get_full_face_mask(self, shape, landmarks):
        # Create face oval mask
        oval_pts = landmarks[FACE_OVAL].astype(np.int32)
        mask = np.zeros(shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [cv2.convexHull(oval_pts)], 1)

        face_area = int(np.sum(mask))
        if face_area <= 0:
            return mask
        
        # Subtract eyes (Orbital)
        left_eye_mask = self.get_mask_from_points(shape, landmarks[LEFT_ORBITAL], mode='hull')
        right_eye_mask = self.get_mask_from_points(shape, landmarks[RIGHT_ORBITAL], mode='hull')
        
        # Subtract mouth (Lips) - using a simple hull of lips
        # MediaPipe lips indices (outer)
        lips_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
        lips_mask = self.get_mask_from_points(shape, landmarks[lips_indices], mode='hull')

        def subtract_if_reasonable(base_mask, sub_mask, min_ratio=0.0005, max_ratio=0.30):
            sub_area = int(np.sum(sub_mask))
            if sub_area <= 0:
                return base_mask
            ratio = sub_area / max(1, face_area)
            # Skip subtracting if the sub-mask is implausibly tiny or huge (can happen on extreme poses)
            if ratio < min_ratio or ratio > max_ratio:
                return base_mask
            return base_mask - sub_mask

        mask = subtract_if_reasonable(mask, left_eye_mask)
        mask = subtract_if_reasonable(mask, right_eye_mask)
        mask = subtract_if_reasonable(mask, lips_mask, max_ratio=0.40)
        mask = np.clip(mask, 0, 1)
        return mask

    def calculate_metrics(self, image_path, return_debug=False):
        image = cv2.imread(image_path)
        if image is None:
            return (None, "imread_failed") if return_debug else None
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        landmarks = self.get_landmarks(image_rgb)
        if landmarks is None:
            return (None, "landmark_failed") if return_debug else None

        h, w = image.shape[:2]
        
        # 1. Generate Masks
        mask_l_temp = self.get_mask_from_points(image.shape, landmarks[LEFT_TEMPORAL], mode='hull')
        mask_r_temp = self.get_mask_from_points(image.shape, landmarks[RIGHT_TEMPORAL], mode='hull')
        
        mask_l_orb = self.get_mask_from_points(image.shape, landmarks[LEFT_ORBITAL], mode='hull')
        mask_r_orb = self.get_mask_from_points(image.shape, landmarks[RIGHT_ORBITAL], mode='hull')
        
        mask_l_malar = self.get_mask_from_points(image.shape, landmarks[LEFT_MALAR], mode='hull')
        mask_r_malar = self.get_mask_from_points(image.shape, landmarks[RIGHT_MALAR], mode='hull')
        
        # Split Jawline into Left and Right
        mask_l_jaw = self.get_mask_from_points(image.shape, landmarks[LEFT_JAW_LINE], mode='line', thickness=15)
        mask_r_jaw = self.get_mask_from_points(image.shape, landmarks[RIGHT_JAW_LINE], mode='line', thickness=15)
        
        mask_full_face = self.get_full_face_mask(image.shape, landmarks)

        # 2. Apply Erosion (3x3 kernel)
        kernel = np.ones((3, 3), np.uint8)
        
        def erode_mask(m):
            # Erode only if mask has enough area; otherwise erosion may delete it.
            if int(np.sum(m)) < ERODE_ONLY_IF_AREA_AT_LEAST:
                return m
            return cv2.erode(m, kernel, iterations=1)

        mask_l_temp = erode_mask(mask_l_temp)
        mask_r_temp = erode_mask(mask_r_temp)
        mask_l_orb = erode_mask(mask_l_orb)
        mask_r_orb = erode_mask(mask_r_orb)
        mask_l_malar = erode_mask(mask_l_malar)
        mask_r_malar = erode_mask(mask_r_malar)
        mask_l_jaw = erode_mask(mask_l_jaw)
        mask_r_jaw = erode_mask(mask_r_jaw)
        # Full-face mask is the normalization reference; do NOT erode it.

        # 3. Calculate Metrics
        metrics = {}

        # Helper to get mean intensity safely
        def get_mean_intensity(img_gray, mask, min_pixels=100):
            if np.sum(mask) < min_pixels: return None
            return np.mean(img_gray[mask > 0])

        # For profile groups, prefer using only the visible side metrics.
        use_left = True
        use_right = True
        if self.view_hint == "profile" and self.side_hint in ("left", "right"):
            if self.side_hint == "left":
                use_right = False
            else:
                use_left = False

        full_face_mean = get_mean_intensity(image_gray, mask_full_face, min_pixels=FULL_FACE_MIN_PIXELS)
        if full_face_mean is None or full_face_mean == 0: 
            return (None, "full_face_invalid") if return_debug else None # Cannot normalize if full face is invalid

        # Helper to combine Left/Right metrics
        def combine_metrics(val_l, val_r):
            valid_vals = [v for v in [val_l, val_r] if v is not None]
            if not valid_vals: return None
            return sum(valid_vals) / len(valid_vals)

        # Metric 1: Relative Intensity Ratio (Hollowness)
        temp_l = get_mean_intensity(image_gray, mask_l_temp, min_pixels=TEMPORAL_MIN_PIXELS) if use_left else None
        temp_r = get_mean_intensity(image_gray, mask_r_temp, min_pixels=TEMPORAL_MIN_PIXELS) if use_right else None
        temp_mean = combine_metrics(temp_l, temp_r)
        
        orb_l = get_mean_intensity(image_gray, mask_l_orb, min_pixels=ORBITAL_MIN_PIXELS) if use_left else None
        orb_r = get_mean_intensity(image_gray, mask_r_orb, min_pixels=ORBITAL_MIN_PIXELS) if use_right else None
        orb_mean = combine_metrics(orb_l, orb_r)
        
        if temp_mean is not None:
            metrics['temporal_ratio'] = temp_mean / full_face_mean
        
        if orb_mean is not None:
            metrics['orbital_ratio'] = orb_mean / full_face_mean

        # Metric 2: Texture Energy (Cheek) - Variance of Laplacian
        # Calculate variance for each valid mask separately, then average?
        # Or combine masks? Combining masks is risky if one is small/garbage.
        # Better to calculate variance on valid masks individually.
        laplacian = cv2.Laplacian(image_gray, cv2.CV_64F)
        
        def get_variance(lap, mask, min_pixels=100):
            if np.sum(mask) < min_pixels: return None
            return np.var(lap[mask > 0])

        cheek_l = get_variance(laplacian, mask_l_malar, min_pixels=MALAR_MIN_PIXELS) if use_left else None
        cheek_r = get_variance(laplacian, mask_r_malar, min_pixels=MALAR_MIN_PIXELS) if use_right else None
        metrics['cheek_texture'] = combine_metrics(cheek_l, cheek_r)

        # Metric 3: Gradient Magnitude (Jawline)
        sobelx = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        def get_mean_magnitude(mag, mask, min_pixels=JAW_MIN_PIXELS): # Jawline mask is thin, lower threshold
            if np.sum(mask) < min_pixels: return None
            return np.mean(mag[mask > 0])

        jaw_l = get_mean_magnitude(magnitude, mask_l_jaw) if use_left else None
        jaw_r = get_mean_magnitude(magnitude, mask_r_jaw) if use_right else None
        metrics['jawline_sharpness'] = combine_metrics(jaw_l, jaw_r)

        # If any metric is missing (e.g. face too far/small), return None or partial?
        # For filtering, we need all metrics. If some are missing due to occlusion, 
        # but we have at least one side, it's fine.
        # If BOTH sides are missing for a metric, we can't evaluate it.
        required_keys = ['temporal_ratio', 'orbital_ratio', 'cheek_texture', 'jawline_sharpness']
        for k in required_keys:
            if k not in metrics or metrics[k] is None:
                return (None, f"metric_missing:{k}") if return_debug else None

        return (metrics, None) if return_debug else metrics

def process_image_wrapper(args):
    extractor, img_path = args
    return extractor.calculate_metrics(img_path)

class DataFilter:
    def __init__(
        self,
        real_data_roots: Optional[list] = None,
        synthetic_root: Optional[str] = None,
        output_dir: Optional[str] = None,
        tau_quantile: float = MAHALANOBIS_TAU_QUANTILE,
        cov_shrinkage: str = DEFAULT_COV_SHRINKAGE,
        cov_reg_eps: float = COV_REG_EPS,
        k_beta: Optional[float] = DEFAULT_K_BETA,
        k_abs: Optional[int] = None,
    ):
        self.group_models = {}
        # Cache extractors by (view_hint, side_hint). Side hint matters for profile views.
        self._extractor_cache = {}

        self.real_data_roots = real_data_roots or list(REAL_DATA_ROOTS)
        self.synthetic_root = synthetic_root or SYNTHETIC_DATA_ROOT
        self.output_dir = output_dir or OUTPUT_DIR
        self.tau_quantile = float(tau_quantile)
        self.cov_shrinkage = cov_shrinkage or DEFAULT_COV_SHRINKAGE
        self.cov_reg_eps = float(cov_reg_eps)
        self.k_beta = k_beta
        self.k_abs = k_abs

    @staticmethod
    def _view_hint_from_group(group_name: str) -> str:
        n = (group_name or "").lower()
        if "profile" in n:
            return "profile"
        if "three-quarter" in n or "three_quarter" in n or "threequarter" in n:
            return "three_quarter"
        return "front"

    @staticmethod
    def _side_hint_from_group(group_name: str) -> Optional[str]:
        n = (group_name or "").lower()
        if "left_" in n:
            return "left"
        if "right_" in n:
            return "right"
        return None

    @staticmethod
    def _min_samples_for_view(view_hint: str) -> int:
        # Loosened to let profile groups build models more often.
        if view_hint == "profile":
            return 5
        if view_hint == "three_quarter":
            return 8
        return MIN_REAL_SAMPLES_PER_GROUP

    def _get_extractor_for_group(self, group_name: str) -> ClinicalFeatureExtractor:
        view_hint = self._view_hint_from_group(group_name)
        side_hint = self._side_hint_from_group(group_name)
        cache_key = (view_hint, side_hint)
        if cache_key in self._extractor_cache:
            return self._extractor_cache[cache_key]

        # Per-view parameters: be more aggressive for profile.
        if view_hint == "profile":
            extractor = ClinicalFeatureExtractor(
                MODEL_PATH,
                use_insightface_fallback=True,
                insightface_det_sizes=INSIGHTFACE_DET_SIZES,
                insightface_crop_expands=(1.60, 1.90, 2.20),
                insightface_det_thresh=0.25,
                insightface_primary=True,
                mp_num_faces=2,
                mp_min_face_detection_confidence=0.20,
                mp_min_face_presence_confidence=0.20,
                mp_min_tracking_confidence=0.20,
                view_hint=view_hint,
                side_hint=side_hint,
            )
        elif view_hint == "three_quarter":
            extractor = ClinicalFeatureExtractor(
                MODEL_PATH,
                use_insightface_fallback=True,
                insightface_det_sizes=(INSIGHTFACE_DET_SIZE,),
                insightface_crop_expands=(1.45, 1.70, 2.00),
                insightface_det_thresh=0.30,
                insightface_primary=True,
                mp_num_faces=2,
                mp_min_face_detection_confidence=0.25,
                mp_min_face_presence_confidence=0.25,
                mp_min_tracking_confidence=0.25,
                view_hint=view_hint,
                side_hint=side_hint,
            )
        else:
            extractor = ClinicalFeatureExtractor(
                MODEL_PATH,
                use_insightface_fallback=True,
                insightface_det_sizes=(INSIGHTFACE_DET_SIZE,),
                insightface_crop_expands=(1.30, 1.60, 1.90),
                insightface_det_thresh=0.35,
                insightface_primary=False,
                mp_num_faces=2,
                mp_min_face_detection_confidence=0.30,
                mp_min_face_presence_confidence=0.30,
                mp_min_tracking_confidence=0.30,
                view_hint=view_hint,
                side_hint=side_hint,
            )

        self._extractor_cache[cache_key] = extractor
        return extractor

    @staticmethod
    def _metrics_to_vector(metrics_dict):
        # Fixed order for s(x) = [s1, s2, s3, s4]
        return np.array([
            metrics_dict['temporal_ratio'],
            metrics_dict['orbital_ratio'],
            metrics_dict['cheek_texture'],
            metrics_dict['jawline_sharpness'],
        ], dtype=np.float64)

    @staticmethod
    def _mahalanobis_d2(x, mu, inv_cov):
        diff = x - mu
        return float(diff.T @ inv_cov @ diff)

    # learn_real_models 学习真实图像分布
    def learn_real_models(self, only_groups: Optional[set] = None):
        print("Step 1: Learning Real Distributions (Mahalanobis models)...")
        
        # Find all subfolders in real data roots
        real_folders = []
        for root in self.real_data_roots:
            if os.path.exists(root):
                subdirs = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
                real_folders.extend(subdirs)
        
        for folder in real_folders:
            folder_name = os.path.basename(folder)
            if only_groups is not None and folder_name not in only_groups:
                continue
            print(f"Processing real folder: {folder_name}")

            extractor = self._get_extractor_for_group(folder_name)
            view_hint = self._view_hint_from_group(folder_name)
            min_needed = self._min_samples_for_view(view_hint)
            
            image_paths = glob.glob(os.path.join(folder, "*.jpg")) + glob.glob(os.path.join(folder, "*.png"))
            if not image_paths:
                continue

            # Collect 4D descriptor s(x)
            descriptor_list = []
            fail_reasons = {}

            # Serial processing for calibration (usually fewer images)
            for img_path in tqdm(image_paths, desc=f"Calibrating {folder_name}"):
                m, reason = extractor.calculate_metrics(img_path, return_debug=True)
                if m is not None:
                    descriptor_list.append(self._metrics_to_vector(m))
                else:
                    fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
            
            if len(descriptor_list) < min_needed:
                print(
                    f"Warning: Not enough valid real samples in {folder_name} "
                    f"({len(descriptor_list)} < {min_needed}). Skipping this group."
                )
                if fail_reasons:
                    # Print top few reasons to help diagnose pose-specific failures
                    top = sorted(fail_reasons.items(), key=lambda x: x[1], reverse=True)[:5]
                    top_str = ", ".join([f"{k}:{v}" for k, v in top])
                    print(f"  Failure reasons (top): {top_str}")
                continue

            X = np.stack(descriptor_list, axis=0)  # (n, 4)
            mu = np.mean(X, axis=0)

            # Regularize covariance for numerical stability (more reg when n is small)
            n_real = int(X.shape[0])
            reg_eps = float(self.cov_reg_eps)
            if n_real < 10:
                reg_eps = max(reg_eps, 1e-4)
            if n_real < 7:
                reg_eps = max(reg_eps, 1e-3)

            cov = _estimate_covariance(X, method=self.cov_shrinkage, reg_eps=reg_eps)

            try:
                inv_cov = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                inv_cov = np.linalg.pinv(cov)

            diff = X - mu
            d2_real = np.einsum('ni,ij,nj->n', diff, inv_cov, diff)
            tau = float(np.percentile(d2_real, self.tau_quantile))

            self.group_models[folder_name] = {
                "mu": mu.tolist(),
                "cov": cov.tolist(),
                "inv_cov": inv_cov.tolist(),
                "tau": tau,
                "n_real": int(X.shape[0]),
                "tau_quantile": self.tau_quantile,
                "d2_real_summary": {
                    "min": float(np.min(d2_real)),
                    "median": float(np.median(d2_real)),
                    "max": float(np.max(d2_real)),
                },
            }

            print(
                f"Model for {folder_name}: n_real={X.shape[0]}, "
                f"tau(q{self.tau_quantile})={tau:.4f}"
            )

    # filter_synthetic 过滤合成图像
    def filter_synthetic(self, only_groups: Optional[set] = None, max_images_per_group: Optional[int] = None):
        print("\nStep 2: Filtering Synthetic Data...")

        if not os.path.exists(self.synthetic_root):
            raise FileNotFoundError(
                f"Synthetic root not found: {self.synthetic_root}. "
                "Please update SYNTHETIC_DATA_ROOT to an existing 'train' folder."
            )
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        synthetic_folders = [d for d in os.listdir(self.synthetic_root) if os.path.isdir(os.path.join(self.synthetic_root, d))]
        synthetic_folders = sorted(synthetic_folders)
        
        for folder_name in synthetic_folders:
            if only_groups is not None and folder_name not in only_groups:
                continue
            src_folder = os.path.join(self.synthetic_root, folder_name)
            dst_folder = os.path.join(self.output_dir, folder_name)
            
            if not os.path.exists(dst_folder):
                os.makedirs(dst_folder)
            
            # Check if we have thresholds for this class
            if folder_name not in self.group_models:
                print(f"Warning: No real model found for {folder_name}. Skipping.")
                continue

            extractor = self._get_extractor_for_group(folder_name)

            model = self.group_models[folder_name]
            mu = np.array(model["mu"], dtype=np.float64)
            inv_cov = np.array(model["inv_cov"], dtype=np.float64)
            tau = float(model["tau"])
            image_paths = glob.glob(os.path.join(src_folder, "*.png")) + glob.glob(os.path.join(src_folder, "*.jpg"))
            image_paths = sorted(image_paths)
            if max_images_per_group is not None and max_images_per_group > 0 and len(image_paths) > max_images_per_group:
                # deterministic subset for smoke tests
                image_paths = image_paths[: int(max_images_per_group)]

            # Determine Top-K target (per group)
            k_target = None
            if self.k_abs is not None and self.k_abs > 0:
                k_target = int(self.k_abs)
            elif self.k_beta is not None and self.k_beta > 0 and model.get("n_real") is not None:
                k_target = int(np.ceil(float(self.k_beta) * float(model.get("n_real"))))
            
            stats = {
                "total": len(image_paths),
                "kept": 0,
                "rejected": 0,
                "candidates": 0,
                "reasons": {},
                "tau": tau,
                "tau_quantile": model.get("tau_quantile", MAHALANOBIS_TAU_QUANTILE),
                "n_real": model.get("n_real", None),
                "k_target": k_target,
                "k_shortage": 0,
                "d2_kept": {"count": 0, "mean": None, "max": None},
                "d2_rejected": {"count": 0, "mean": None, "min": None},
            }
            
            print(f"Filtering {folder_name} ({len(image_paths)} images)...")
            
            # Use multiprocessing for filtering
            # Note: We need to re-instantiate extractor inside worker if using spawn, 
            # but fork is default on Linux so passing self.extractor might work if it's picklable.
            # MediaPipe objects are often not picklable. Better to init inside worker or run serial.
            # For simplicity and stability, let's run serial first. If slow, we optimize.
            
            candidates = []
            d2_rejected = []

            for img_path in tqdm(image_paths):
                metrics, reason = extractor.calculate_metrics(img_path, return_debug=True)

                if metrics is None:
                    stats["rejected"] += 1
                    reason_key = reason or "detection_failed"
                    stats["reasons"][reason_key] = stats["reasons"].get(reason_key, 0) + 1
                    continue

                x = self._metrics_to_vector(metrics)
                d2 = self._mahalanobis_d2(x, mu, inv_cov)

                if d2 <= tau:
                    candidates.append((img_path, d2))
                else:
                    stats["rejected"] += 1
                    reason_str = "mahalanobis_outlier"
                    stats["reasons"][reason_str] = stats["reasons"].get(reason_str, 0) + 1
                    d2_rejected.append(d2)

            # Top-K selection within candidates
            candidates.sort(key=lambda x: x[1])
            stats["candidates"] = len(candidates)
            if k_target is not None:
                selected = candidates[: min(len(candidates), k_target)]
                stats["k_shortage"] = max(0, k_target - len(candidates))
            else:
                selected = candidates

            for img_path, d2 in selected:
                shutil.copy(img_path, os.path.join(dst_folder, os.path.basename(img_path)))
                stats["kept"] += 1
                prev_count = stats["d2_kept"]["count"]
                prev_mean = stats["d2_kept"]["mean"]
                stats["d2_kept"]["count"] = prev_count + 1
                stats["d2_kept"]["mean"] = (
                    d2 if prev_mean is None else (prev_mean * prev_count + d2) / (prev_count + 1)
                )
                stats["d2_kept"]["max"] = d2 if stats["d2_kept"]["max"] is None else max(stats["d2_kept"]["max"], d2)

            if d2_rejected:
                stats["d2_rejected"]["count"] = len(d2_rejected)
                stats["d2_rejected"]["mean"] = float(np.mean(d2_rejected))
                stats["d2_rejected"]["min"] = float(np.min(d2_rejected))
            
            # Save stats
            with open(os.path.join(dst_folder, "filter_stats.json"), "w") as f:
                json.dump(stats, f, indent=4)
            
            print(f"Finished {folder_name}: Kept {stats['kept']}/{stats['total']}")

if __name__ == "__main__":
    # Ensure model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        exit(1)

    parser = argparse.ArgumentParser(description="Stage2: SGA-guided clinical filtering (learn real Mahalanobis models, then filter synthetic).")
    parser.add_argument(
        "--real-roots",
        nargs="+",
        default=None,
        help="One or more real-data root folders containing group subdirectories.",
    )
    parser.add_argument(
        "--synthetic-root",
        type=str,
        default=None,
        help="Synthetic 'train' root folder containing group subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output folder for filtered synthetic groups.",
    )
    parser.add_argument(
        "--tau-quantile",
        type=float,
        default=MAHALANOBIS_TAU_QUANTILE,
        help="Quantile for tau threshold (e.g., 95 for P95).",
    )
    parser.add_argument(
        "--cov-shrinkage",
        type=str,
        default=DEFAULT_COV_SHRINKAGE,
        choices=["ledoit_wolf", "lw", "oas", "ridge", "none", "empirical"],
        help="Covariance estimator: ledoit_wolf (default), oas, ridge, none.",
    )
    parser.add_argument(
        "--cov-reg-eps",
        type=float,
        default=COV_REG_EPS,
        help="Diagonal regularization added to covariance after shrinkage.",
    )
    parser.add_argument(
        "--k-beta",
        type=float,
        default=DEFAULT_K_BETA,
        help="Top-K multiplier: K = ceil(beta * n_real) per group (ignored if --k-abs is set).",
    )
    parser.add_argument(
        "--k-abs",
        type=int,
        default=None,
        help="Absolute Top-K per group. If set, overrides --k-beta.",
    )
    parser.add_argument("--learn-only", action="store_true", help="Only learn real-group Mahalanobis models; skip synthetic filtering.")
    parser.add_argument("--filter-only", action="store_true", help="Only filter synthetic using previously learned models in memory (requires running learn in same process).")
    parser.add_argument(
        "--only-groups",
        nargs="*",
        default=DEFAULT_ONLY_GROUPS,
        help=(
            "Optional list of group folder names to run (learn + filter). "
            "Default: 3-view normal + malnourished (6 groups)."
        ),
    )
    parser.add_argument(
        "--max-synth-images",
        type=int,
        default=None,
        help="Optional cap on number of synthetic images processed per group (for smoke tests).",
    )
    args = parser.parse_args()

    filter_system = DataFilter(
        real_data_roots=args.real_roots,
        synthetic_root=args.synthetic_root,
        output_dir=args.output_dir,
        tau_quantile=args.tau_quantile,
        cov_shrinkage=args.cov_shrinkage,
        cov_reg_eps=args.cov_reg_eps,
        k_beta=args.k_beta,
        k_abs=args.k_abs,
    )
    only = set(args.only_groups) if args.only_groups else None
    if not args.filter_only:
        filter_system.learn_real_models(only_groups=only)
    if not args.learn_only:
        filter_system.filter_synthetic(only_groups=only, max_images_per_group=args.max_synth_images)
        print(f"\nDone! Filtered dataset saved to {filter_system.output_dir}")
