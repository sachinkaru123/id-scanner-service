"""Text-cleanup helpers shared by the whole-card regex parser and the
label-anchored ROI extractor — kept in one place so date/name normalization
rules don't drift between the two."""
from __future__ import annotations

import re
from datetime import datetime

DATE_PATTERN = re.compile(
    r"\b(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}|\d{4}[/\-.]\d{1,2}[/\-.]\d{1,2}|"
    r"\d{1,2}\s?(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*\s?\d{2,4})\b",
    re.IGNORECASE,
)

DATE_FORMATS = [
    "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y",
    "%m/%d/%Y", "%m-%d-%Y",
    "%Y/%m/%d", "%Y-%m-%d",
    "%d/%m/%y", "%d-%m-%y",
    "%d %b %Y", "%d %B %Y", "%d%b%Y",
]


def normalize_date(raw: str) -> str | None:
    cleaned = raw.strip().upper().replace("  ", " ")
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def strip_border_noise(s: str) -> str:
    """OCR often reads card borders as stray |, _, or box-drawing chars."""
    return s.strip(" |_-—–~[](){}").strip()


def letters_and_spaces(s: str) -> str:
    """Collapse to just letters/spaces — strips stray OCR punctuation noise
    (curly quotes, dashes, box-drawing chars) without discarding the line."""
    return re.sub(r"\s+", " ", re.sub(r"[^A-Za-z\s]", " ", s)).strip()
