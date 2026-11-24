from __future__ import annotations

import cv2
import numpy as np

from .remover import BackgroundRemover


def jpeg_to_transparent_png(data: bytes, remover: BackgroundRemover) -> bytes:
    """Decode a JPEG frame, calculate alpha, and return a transparent PNG."""
    encoded = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("unable to decode input image")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    alpha = remover.mask(rgb)
    if alpha.shape != rgb.shape[:2]:
        raise ValueError("alpha dimensions do not match input frame")

    refine_rgb = getattr(remover, "refine_rgb", None)
    if callable(refine_rgb):
        rgb = refine_rgb(rgb, alpha)

    rgba = np.dstack((rgb, alpha))
    bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
    ok, png = cv2.imencode(".png", bgra, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        raise RuntimeError("unable to encode output PNG")
    return png.tobytes()

