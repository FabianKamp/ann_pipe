"""Image processing utilities.

This module contains helper functions for common image processing tasks used
across the project.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

__all__: Tuple[str, ...] = ("image_to_rgb_matrix",)


def image_to_rgb_matrix(image_path: str | Path, flatten: bool = True) -> np.ndarray:
    """Load an image and return its RGB values as a NumPy matrix.

    Parameters
    ----------
    image_path : str or pathlib.Path
        Path to the image file that should be converted.
    flatten : bool, default=True
        If *True*, the returned array is two-dimensional with shape
        ``(height * width, 3)`` where each row corresponds to one pixel and the
        three columns store the *R*, *G*, and *B* channel values respectively.
        If *False*, the array keeps the spatial structure and has shape
        ``(height, width, 3)``.

    Returns
    -------
    numpy.ndarray
        A floating-point NumPy array in the requested shape whose values lie in
        the interval [0, 255]. The dtype is ``np.uint8``.

    Notes
    -----
    * Grayscale or single-channel images are automatically converted to RGB by
      duplicating the channel across R, G, and B. Images with an alpha channel
      are returned without the alpha component.
    * No additional preprocessing (e.g., normalization) is performed so that
      downstream functions can decide the correct scaling.
    """

    # --- Input validation -------------------------------------------------
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image file not found: {img_path}")

    # --- Image loading ----------------------------------------------------
    # Pillow loads images in RGB by default; convert explicitly to be safe.
    with Image.open(img_path) as img:
        img = img.convert("RGB")
        rgb_array = np.asarray(img, dtype=np.uint8)  # (H, W, 3)

    if flatten:
        rgb_array = rgb_array.reshape(-1, 3)  # (H*W, 3)

    return rgb_array 