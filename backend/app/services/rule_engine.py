"""Rule-based verification engine."""
import re
from datetime import datetime
from typing import Optional, Sequence, Tuple


# Date regexes for extraction
DATE_REGEXES: Sequence[re.Pattern[str]] = (
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b"),
    re.compile(
        r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
        r"sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?,\s+\d{4}\b",
        re.IGNORECASE
    ),
    re.compile(
        r"\b\d{1,2}\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
        r"sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{4}\b",
        re.IGNORECASE
    ),
)

# Date formats for parsing
DATE_FORMATS: Sequence[str] = (
    "%Y-%m-%d",
    "%m/%d/%Y",
    "%d/%m/%Y",
    "%B %d, %Y",
    "%b %d, %Y",
    "%d %B %Y",
    "%d %b %Y",
)


def _parse_date_token(token: str) -> Optional[datetime]:
    """Try parsing a date token with multiple formats."""
    cleaned = re.sub(r"(?:st|nd|rd|th),", ",", token, flags=re.IGNORECASE)
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
    return None


def _extract_date(sentence: Optional[str]) -> Optional[datetime]:
    """Extract date from a sentence using regex patterns."""
    if not sentence:
        return None
    for pattern in DATE_REGEXES:
        match = pattern.search(sentence)
        if not match:
            continue
        parsed = _parse_date_token(match.group(0))
        if parsed:
            return parsed
    return None


def _find_sentence(text: str, keywords: Sequence[str]) -> Optional[str]:
    """Find the first sentence containing all keywords."""
    fragments = re.split(r"(?<=[.!?])\s+|\n+", text)
    for fragment in fragments:
        lowered = fragment.lower()
        if all(keyword in lowered for keyword in keywords):
            return fragment.strip()
    return None


def detect_date_inconsistency(text: str) -> Tuple[str, str]:
    """
    Rule-based comparison of anticipated vs signed dates.

    Returns:
        Tuple of (status, message) where status is "PASS" or "FAIL"
    """
    anticipated_sentence = _find_sentence(text, ["anticipated submission date"])
    signed_sentence = _find_sentence(text, ["signed", "sealed"])

    anticipated_date = _extract_date(anticipated_sentence)
    signed_date = _extract_date(signed_sentence)

    if anticipated_date and signed_date:
        anticipated_str = anticipated_date.date().isoformat()
        signed_str = signed_date.date().isoformat()
        if signed_date.date() > anticipated_date.date():
            return ("FAIL", f"Signed {signed_str} after anticipated {anticipated_str}")
        return ("PASS", f"Signed {signed_str} on/before anticipated {anticipated_str}")

    return ("PASS", "Insufficient date context detected")
