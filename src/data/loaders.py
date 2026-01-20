"""Utilities for listing available stimulus images for activation extraction."""
from __future__ import annotations

import os
from typing import List


def get_image_set(image_folder: str) -> List[str]:
    """Return sorted image base names (without extension) for .png/.jpg files."""
    if not os.path.isdir(image_folder):
        raise FileNotFoundError(f"Image folder not found: {image_folder}")
    return sorted(os.path.splitext(f)[0] for f in os.listdir(image_folder) if f.lower().endswith((".png", ".jpg")))
