"""Regex/heuristic extraction of structured fields from raw OCR text.

ID card layouts vary a lot by country/issuer, so this favors label-driven
heuristics (look near words like "DOB", "Name", "ID No") with generic
fallback patterns, rather than hard-coding one specific card format.
"""
from __future__ import annotations

import re

from app.text_utils import DATE_PATTERN, letters_and_spaces, normalize_date, strip_border_noise

ID_NUMBER_PATTERN = re.compile(r"\b[A-Z0-9]{0,3}[-\s]?\d{4,}[-\s]?[A-Z0-9]{0,4}\b")

DOB_LABELS = re.compile(r"(DATE\s*OF\s*BIRTH|DOB|BIRTH\s*DATE|BORN)", re.IGNORECASE)
EXPIRY_LABELS = re.compile(r"(EXP(?:IRY|IRES)?|VALID\s*(?:UNTIL|THRU|TO))", re.IGNORECASE)
ID_LABELS = re.compile(
    r"(ID\s*NO\.?|ID\s*NUMBER|LICENSE\s*NO\.?|LICENCE\s*NO\.?|CARD\s*NO\.?|DL\s*NO\.?|PASSPORT\s*NO\.?|"
    r"DOCUMENT\s*NO\.?)",
    re.IGNORECASE,
)
NAME_LABELS = re.compile(r"\bNAME\s*[:\-]?\s*(.+)", re.IGNORECASE)
GENDER_LABELS = re.compile(r"\b(SEX|GENDER)\b", re.IGNORECASE)


def _find_dates_near(text: str, label_pattern: re.Pattern) -> str | None:
    """Find a date on the same line as (or immediately after) a matching label."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if label_pattern.search(line):
            window = line
            if i + 1 < len(lines):
                window += " " + lines[i + 1]
            match = DATE_PATTERN.search(window)
            if match:
                normalized = normalize_date(match.group(0))
                if normalized:
                    return normalized
    return None


def extract_dob(text: str) -> str | None:
    return _find_dates_near(text, DOB_LABELS)


def extract_expiry(text: str) -> str | None:
    return _find_dates_near(text, EXPIRY_LABELS)


def extract_all_dates(text: str) -> list[str]:
    dates = []
    for match in DATE_PATTERN.finditer(text):
        normalized = normalize_date(match.group(0))
        if normalized and normalized not in dates:
            dates.append(normalized)
    return dates


def extract_id_number(text: str) -> str | None:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if ID_LABELS.search(line):
            window = line
            if i + 1 < len(lines):
                window += " " + lines[i + 1]
            without_label = ID_LABELS.sub("", window)
            match = ID_NUMBER_PATTERN.search(without_label)
            if match:
                return match.group(0).strip(" -")
    # Fallback: longest standalone alphanumeric token with digits, common on cards
    candidates = re.findall(r"\b[A-Z0-9][A-Z0-9\-]{5,20}\b", text.upper())
    candidates = [c for c in candidates if any(ch.isdigit() for ch in c)]
    if candidates:
        return max(candidates, key=len)
    return None


# Words that signal the line after a name label has moved on to a different
# field, so we know where a wrapped (multi-line) name value ends.
_OTHER_LABEL_HINT = re.compile(
    r"(SEX|GENDER|DATE|BIRTH|ADDRESS|SIGNATURE|ID\s*NO|NIC|VALID|EXPIR|NATIONALITY|ISSUE)",
    re.IGNORECASE,
)


def extract_name(text: str) -> str | None:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        match = re.search(r"\bNAME\s*[:\-]?\s*(.*)", line, re.IGNORECASE)
        if not match:
            continue

        first = re.split(r"\s{2,}|\||/", match.group(1).strip())[0].strip()
        parts = [first]

        # Printed names often wrap onto the next line(s) with no label of
        # their own — keep consuming lines until one looks like a new field.
        # OCR noise (stray quotes/dashes from card borders/holograms) is
        # tolerated as long as the line is still mostly letters.
        j = i + 1
        max_continuation_lines = 1
        while j < len(lines) and (j - i) <= max_continuation_lines:
            raw_nxt = lines[j]
            stripped_nxt = raw_nxt.strip()
            if (
                not stripped_nxt
                or ":" in stripped_nxt
                or any(ch.isdigit() for ch in stripped_nxt)
                or _OTHER_LABEL_HINT.search(stripped_nxt)
            ):
                break
            alpha_ratio = sum(ch.isalpha() or ch.isspace() for ch in stripped_nxt) / len(stripped_nxt)
            if alpha_ratio < 0.6:
                break
            parts.append(stripped_nxt)
            j += 1

        candidate = letters_and_spaces(" ".join(parts))
        if candidate:
            return candidate

    # Fallback: first all-caps line of 2+ words with no digits (typical printed name)
    for line in lines:
        stripped = strip_border_noise(line.strip())
        if (
            len(stripped) > 4
            and stripped == stripped.upper()
            and not any(ch.isdigit() for ch in stripped)
            and len(stripped.split()) >= 2
            and stripped.isascii()
        ):
            return stripped
    return None


def extract_gender(text: str) -> str | None:
    # The value doesn't always sit immediately after the label — bilingual
    # cards interleave other-language words between "Sex:" and "Male", so
    # scan the whole label line for the value instead of anchoring to it.
    for line in text.splitlines():
        if not GENDER_LABELS.search(line):
            continue
        if re.search(r"\bFEMALE\b", line, re.IGNORECASE):
            return "F"
        if re.search(r"\bMALE\b", line, re.IGNORECASE):
            return "M"
        match = re.search(r"\b([MF])\b", line)
        if match:
            return match.group(1).upper()
    return None


def parse_id_card(text: str) -> dict:
    """Extract all known fields from raw OCR text into a structured dict."""
    return {
        "name": extract_name(text),
        "id_number": extract_id_number(text),
        "date_of_birth": extract_dob(text),
        "expiry_date": extract_expiry(text),
        "gender": extract_gender(text),
        "all_dates_found": extract_all_dates(text),
        "raw_text": text.strip(),
    }
