"""Pre-OCR image quality checks (resolution, blur, glare).

Running OCR on a bad capture just produces confident-looking garbage — it's
cheaper and more honest to flag the capture and let the caller ask the user
to retake the photo. These are warnings, not a hard block: the service still
returns whatever it can extract.
"""
from __future__ import annotations

import cv2
import numpy as np

MIN_CARD_WIDTH_PX = 600
BLUR_VARIANCE_THRESHOLD = 60.0
GLARE_PIXEL_RATIO_THRESHOLD = 0.08


def _blur_score(gray: np.ndarray) -> float:
    """Variance of the Laplacian — sharp edges produce high variance;
    a blurred image's edges are smeared out, producing low variance."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _glare_ratio(bgr: np.ndarray) -> float:
    """Fraction of pixels that are blown-out highlights (near-white, low
    saturation) — a proxy for glare off a laminated/plastic card surface."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    saturation, value = hsv[:, :, 1], hsv[:, :, 2]
    blown_out = (value > 240) & (saturation < 30)
    return float(np.count_nonzero(blown_out)) / blown_out.size


def assess_quality(cropped_bgr: np.ndarray) -> dict:
    """Run cheap heuristics on the perspective-corrected card image.

    Returns {"ok": bool, "issues": [...], "metrics": {...}}.
    """
    gray = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    blur = _blur_score(gray)
    glare = _glare_ratio(cropped_bgr)

    issues = []
    if w < MIN_CARD_WIDTH_PX:
        issues.append("low_resolution")
    if blur < BLUR_VARIANCE_THRESHOLD:
        issues.append("blurry")
    if glare > GLARE_PIXEL_RATIO_THRESHOLD:
        issues.append("glare")

    return {
        "ok": not issues,
        "issues": issues,
        "metrics": {
            "width": w,
            "height": h,
            "blur_score": round(blur, 1),
            "glare_ratio": round(glare, 4),
        },
    }
