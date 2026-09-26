"""Label-anchored region-of-interest (ROI) OCR for date fields.

Whole-card OCR + regex already reads high-contrast, large-print fields (name,
ID number, gender) reliably — tested and confirmed on a real ID card photo.
What it consistently misses is small printed digits (date of birth, expiry)
sitting next to a security seal/watermark graphic: at whole-card scale
they're a handful of pixels and get lost in the layout noise. This module
finds where a date label sits (from word-level OCR data already computed for
the whole-card pass) and re-OCRs *just* the small region next to it — its
own upscale, its own page-segmentation-mode, its own digit whitelist. That
isolation is what fixes the misreads; it has nothing to do with guessing a
better global config.

Falls back cleanly (returns None) if a label isn't found or the crop doesn't
parse as a date, so callers should treat these as overrides on top of the
whole-card regex parse, not a replacement for it.
"""
from __future__ import annotations

import re

import cv2
import numpy as np
import pytesseract

from app.text_utils import DATE_PATTERN, normalize_date

BIRTH_LABEL = re.compile(r"BIRTH", re.IGNORECASE)
EXPIRY_LABEL = re.compile(r"EXPIR", re.IGNORECASE)


def lines_from_ocr_data(data: dict) -> list[dict]:
    """Group Tesseract's word-level image_to_data output into lines, each
    carrying the union bounding box of its words plus the joined text."""
    lines: dict[tuple, dict] = {}
    for i in range(len(data["text"])):
        word = data["text"][i].strip()
        if not word:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        line = lines.setdefault(key, {"words": [], "left": x, "top": y, "right": x + w, "bottom": y + h})
        line["words"].append({"text": word, "left": x, "top": y, "width": w, "height": h})
        line["left"] = min(line["left"], x)
        line["top"] = min(line["top"], y)
        line["right"] = max(line["right"], x + w)
        line["bottom"] = max(line["bottom"], y + h)

    ordered = sorted(lines.values(), key=lambda l: l["top"])
    for line in ordered:
        line["text"] = " ".join(w["text"] for w in line["words"])
    return ordered


def _find_label_line(lines: list[dict], label_pattern: re.Pattern) -> dict | None:
    for line in lines:
        if label_pattern.search(line["text"]):
            return line
    return None


def _label_word_right_edge(line: dict, label_pattern: re.Pattern) -> int:
    """x just past the matched label word — so the crop starts at the value,
    not the label itself."""
    for word in line["words"]:
        if label_pattern.search(word["text"]):
            return word["left"] + word["width"]
    return line["left"]


def _trim_trailing_graphic(binary: np.ndarray, gap_ratio: float = 0.6) -> np.ndarray:
    """A label's value crop is bounded on the right by the card edge, not the
    end of the text — which often runs into a seal/watermark graphic further
    along the same row. Cut the crop at the first wide blank gap after the
    text starts, so that graphic doesn't get fed to Tesseract as if it were
    more characters."""
    ink_cols = (binary < 128).any(axis=0)
    if not ink_cols.any():
        return binary

    min_gap_px = max(20, int(binary.shape[0] * gap_ratio))
    started = False
    gap = 0
    last_ink_x = 0
    for x, has_ink in enumerate(ink_cols):
        if has_ink:
            started = True
            gap = 0
            last_ink_x = x
        elif started:
            gap += 1
            if gap >= min_gap_px:
                return binary[:, : last_ink_x + 1]
    return binary


def _crop_field(gray: np.ndarray, box: tuple[int, int, int, int], target_height: int = 140) -> np.ndarray | None:
    """Upscale the label-value region and trim it to just the text run.

    Tesseract's LSTM engine (--oem 3) reads clean upscaled grayscale better
    than a hard-thresholded binary image — it was trained on antialiased
    text, so forcing pure black/white here measurably hurt accuracy. Otsu
    binarization is used only internally, to decide where the text run ends
    so the crop can be trimmed before a neighboring seal/watermark graphic.
    """
    x0, y0, x1, y1 = box
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(gray.shape[1], x1), min(gray.shape[0], y1)
    if x1 - x0 < 10 or y1 - y0 < 6:
        return None

    crop = gray[y0:y1, x0:x1]
    h, w = crop.shape[:2]
    if h < target_height:
        scale = target_height / h
        crop = cv2.resize(crop, (int(w * scale), target_height), interpolation=cv2.INTER_CUBIC)

    _, otsu = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    trimmed_otsu = _trim_trailing_graphic(otsu)
    return crop[:, : trimmed_otsu.shape[1]]


def _ocr_date_near_label(gray: np.ndarray, lines: list[dict], label_pattern: re.Pattern, pad: int = 6) -> str | None:
    line = _find_label_line(lines, label_pattern)
    if line is None:
        return None

    x_start = _label_word_right_edge(line, label_pattern) + pad
    x_end = gray.shape[1]
    box = (x_start, line["top"] - pad, x_end, line["bottom"] + pad)

    crop = _crop_field(gray, box)
    if crop is None:
        return None

    config = "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789/-."
    text = pytesseract.image_to_string(crop, config=config).strip()
    if not text:
        return None

    match = DATE_PATTERN.search(text)
    if not match:
        return None
    return normalize_date(match.group(0))


def extract_fields_via_roi(gray: np.ndarray, ocr_data: dict) -> dict:
    """Returns override candidates for fields that benefit from a dedicated,
    isolated re-OCR pass. Values are None where the label wasn't found or the
    crop didn't yield a parseable date — callers should only use a value
    when it's non-None, keeping the whole-card regex parse as the fallback."""
    lines = lines_from_ocr_data(ocr_data)
    return {
        "date_of_birth": _ocr_date_near_label(gray, lines, BIRTH_LABEL),
        "expiry_date": _ocr_date_near_label(gray, lines, EXPIRY_LABEL),
    }
