"""Image pipeline: crop card edges, enhance contrast, run OCR."""
from __future__ import annotations

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

from app.roi_extract import lines_from_ocr_data

# Cards smaller than this fraction of the frame are assumed to already
# fill the shot (selfie-style crop), so we skip perspective correction.
MIN_CARD_AREA_RATIO = 0.15


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _find_card_contour(gray: np.ndarray) -> np.ndarray | None:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    edged = cv2.dilate(edged, np.ones((5, 5), np.uint8), iterations=2)
    edged = cv2.erode(edged, np.ones((5, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    frame_area = gray.shape[0] * gray.shape[1]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    contours = [c for c in contours if cv2.contourArea(c) >= frame_area * MIN_CARD_AREA_RATIO]
    if not contours:
        return None

    # Ideal case: a clean 4-point polygon (sharp-cornered card, good contrast).
    for c in contours:
        peri = cv2.arcLength(c, True)
        for eps_frac in (0.02, 0.03, 0.04, 0.05):
            approx = cv2.approxPolyDP(c, eps_frac * peri, True)
            if len(approx) == 4:
                return approx.reshape(4, 2)

    # Real ID cards have rounded corners, which often defeats approxPolyDP
    # entirely. Fall back to a rotated bounding box around the largest
    # candidate contour instead of giving up and using the whole frame.
    largest = contours[0]
    rot_rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rot_rect)
    return box


def crop_card(image: np.ndarray) -> np.ndarray:
    """Detect the ID card in the frame and perspective-warp it to a flat rectangle.

    Falls back to the original image untouched if no confident 4-point
    contour is found (e.g. the card already fills the frame).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pts = _find_card_contour(gray)
    if pts is None:
        return image

    rect = _order_points(pts.astype("float32"))
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = int(max(width_a, width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = int(max(height_a, height_b))

    if max_width < 50 or max_height < 50:
        return image

    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
    return warped


def enhance(image: np.ndarray) -> np.ndarray:
    """Grayscale + CLAHE contrast boost + light sharpening for cleaner OCR input."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrasted = clahe.apply(gray)

    denoised = cv2.fastNlMeansDenoising(contrasted, h=10)

    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(denoised, -1, sharpen_kernel)

    return sharpened


def upscale_if_small(image: np.ndarray, min_width: int = 1400) -> np.ndarray:
    """OCR accuracy drops on small images; upscale if below a minimum width."""
    h, w = image.shape[:2]
    if w >= min_width:
        return image
    scale = min_width / w
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)


def get_ocr_data(image: np.ndarray, lang: str = "eng", psm: int = 6) -> dict:
    """Word-level OCR output (text + bounding box + confidence per word).

    Used both to build the whole-card text (so we don't pay for a second
    Tesseract call just to get plain text) and, by the ROI extractor, to
    locate field labels so it can re-OCR just the value next to them.
    """
    config = f"--oem 3 --psm {psm}"
    return pytesseract.image_to_data(image, lang=lang, config=config, output_type=Output.DICT)


def text_from_ocr_data(data: dict) -> str:
    """Reconstruct line-broken text from word-level OCR data — the regex
    parser relies on line boundaries (e.g. "did this line have a label")."""
    lines = lines_from_ocr_data(data)
    return "\n".join(line["text"] for line in lines)


def process_image(raw_image: np.ndarray) -> tuple[str, np.ndarray, dict, np.ndarray]:
    """Full pipeline: crop -> upscale -> enhance -> OCR.

    Returns (text, enhanced_image, ocr_data_block, cropped_image).
    `ocr_data_block` and `cropped_image` are exposed so callers can run
    label-anchored ROI re-OCR (app.roi_extract) and pre-OCR quality checks
    (app.quality) without redoing the crop/enhance work.

    A single PSM 6 (uniform block) pass reliably catches the large-print
    fields (name, ID number, gender) at whole-card scale; small-print fields
    (DOB, expiry) are handled separately by app.roi_extract, which re-OCRs
    just the region next to their label. An earlier version also ran a
    second whole-card pass (PSM 11, sparse text) to catch what PSM 6 missed,
    but once ROI extraction covers the fields that actually needed it, that
    second pass stopped changing any result — just ~650ms of dead latency —
    so it was dropped.
    """
    cropped = crop_card(raw_image)
    upscaled = upscale_if_small(cropped)
    enhanced = enhance(upscaled)

    data_block = get_ocr_data(enhanced, psm=6)
    text = text_from_ocr_data(data_block)

    return text, enhanced, data_block, cropped
