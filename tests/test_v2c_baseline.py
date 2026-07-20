import importlib.util
import sys
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "hand_synthesis_v2" / "build_v2c_baseline.py"
SPEC = importlib.util.spec_from_file_location("build_v2c_baseline", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_derived_seed_is_stable_and_path_specific():
    first = MODULE.derived_seed(22, "train/a/1_01.png", 0)
    assert first == MODULE.derived_seed(22, "train/a/1_01.png", 0)
    assert first != MODULE.derived_seed(22, "train/a/1_02.png", 0)


def test_low_risk_augmentation_is_square_and_never_flips():
    config = {
        "augmentation": {
            "horizontal_flip": False,
            "rotation_degrees": [-4.0, 4.0],
            "translate_fraction": [-0.02, 0.02],
            "scale": [0.97, 1.03],
            "exposure_gain": [0.94, 1.06],
            "white_balance_red_gain": [0.97, 1.03],
            "white_balance_green_gain": [1.0, 1.0],
            "white_balance_blue_gain": [0.97, 1.03],
            "sensor_noise_sigma_8bit": [0.0, 2.0],
        }
    }
    params = MODULE.sample_params(config, 123)
    image = Image.fromarray(np.full((80, 160, 3), 127, dtype=np.uint8), "RGB")
    output, margin, _ = MODULE.augment_image(image, params, 256, 0.12)
    assert params["horizontal_flip"] is False
    assert output.mode == "RGB"
    assert output.size == (256, 256)
    assert margin >= 20
