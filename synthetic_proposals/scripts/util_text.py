"""
Utilities for normalising proposal text and generating embedding-friendly chunks.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Sequence


WHITESPACE_RE = re.compile(r"\s+")
SECTION_WORD_LIMITS = {
    "letter": 70,
    "qualifications": 90,
    "staffing": 80,
    "subconsultants": 60,
    "work_approach": 120,
    "schedule": 80,
    "references": 50,
}
MAX_TOTAL_WORDS = 512
MAX_TOTAL_WORDS_WITH_TOLERANCE = int(MAX_TOTAL_WORDS * 1.05)

def _truncate_words(text: str, max_words: int) -> str:
    words = normalize_text(text).split(" ")
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words])


def normalize_text(value: str) -> str:
    """Collapse whitespace and strip leading/trailing spaces."""
    return WHITESPACE_RE.sub(" ", value).strip()


def ensure_list(value: Any) -> List[str]:
    """Force a value into a list of strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return [normalize_text(str(item)) for item in value if item is not None]
    return [normalize_text(str(value))]


def flatten_paragraphs(paragraphs: Sequence[str]) -> str:
    """Join paragraphs into a single body of text."""
    return "\n\n".join(normalize_text(p) for p in paragraphs if p)


def chunk_text(text: str, *, max_words: int = 180, overlap: int = 25) -> List[str]:
    """
    Split text into overlapping word chunks while preserving natural breaks.

    Args:
        text: Source text to chunk.
        max_words: Max words per chunk.
        overlap: Number of words to carry over between chunks.
    """
    words = normalize_text(text).split(" ")
    if not words:
        return []
    if len(words) <= max_words:
        return [" ".join(words)]

    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end == len(words):
            break
        start = end
    return chunks


def estimate_page_hints(chunk_lengths: Sequence[int], *, words_per_page: int = 430) -> List[int]:
    """
    Convert chunk word lengths into 1-indexed page hints using cumulative words.
    """
    hints: List[int] = []
    cumulative = 0
    for length in chunk_lengths:
        cumulative += length
        page = max(1, math.ceil(cumulative / words_per_page))
        hints.append(page)
    return hints


def gather_section_text(proposal: Dict[str, Any]) -> Dict[str, str]:
    """
    Extract primary narrative sections from a proposal YAML payload.

    Returns a mapping from section identifier to flattened text.
    """
    sections: Dict[str, str] = {}
    letter = proposal.get("letter", {})
    if letter:
        text = flatten_paragraphs(ensure_list(letter.get("body", [])))
        sections["letter"] = _truncate_words(text, SECTION_WORD_LIMITS.get("letter", 150))

    qualifications = proposal.get("qualifications", [])
    if qualifications:
        parts: List[str] = []
        for qual in qualifications:
            title = qual.get("title")
            if title:
                parts.append(f"{title}: {qual.get('description', '')}")
            highlights = ensure_list(qual.get("highlights", []))
            if highlights:
                parts.append("Highlights: " + "; ".join(highlights))
        text = flatten_paragraphs(parts)
        sections["qualifications"] = _truncate_words(text, SECTION_WORD_LIMITS.get("qualifications", 150))

    staffing = proposal.get("staffing_plan", {})
    if staffing:
        staffing_parts = ensure_list(staffing.get("overview", [])) + ensure_list(staffing.get("availability", []))
        matrix = staffing.get("resource_matrix", [])
        for entry in matrix:
            staffing_parts.append(
                f"{entry.get('role')}: {entry.get('name')} ({entry.get('allocation_percent', 0)}% availability)"
            )
        text = flatten_paragraphs(staffing_parts)
        sections["staffing"] = _truncate_words(text, SECTION_WORD_LIMITS.get("staffing", 150))

    subs_section = proposal.get("subs_summary", {})
    if subs_section:
        subs_parts = ensure_list(subs_section.get("approach", [])) + ensure_list(subs_section.get("dbe_strategy", []))
        text = flatten_paragraphs(subs_parts)
        sections["subconsultants"] = _truncate_words(text, SECTION_WORD_LIMITS.get("subconsultants", 100))

    work_approach = proposal.get("work_approach", {})
    if work_approach:
        parts: List[str] = [work_approach.get("executive_summary", "")]
        parts.extend(ensure_list(work_approach.get("objectives", [])))
        parts.extend(ensure_list(work_approach.get("constraints", [])))
        parts.extend(ensure_list(work_approach.get("success_factors", [])))
        for element in work_approach.get("elements", []):
            title = element.get("title")
            context = element.get("context", "")
            activities = ensure_list(element.get("activities", []))
            snippet = f"{title}: {context}. Activities: {'; '.join(activities)}" if title else context
            parts.append(snippet)
        text = flatten_paragraphs(parts)
        sections["work_approach"] = _truncate_words(text, SECTION_WORD_LIMITS.get("work_approach", 150))

    schedule = proposal.get("schedule", {})
    if schedule:
        milestone_lines = [
            f"{item.get('name')} ({item.get('date')}): {item.get('description')}"
            for item in schedule.get("milestones", [])
        ]
        drivers = ensure_list(schedule.get("critical_path", {}).get("drivers", []))
        schedule_parts = [
            f"Period: {schedule.get('start_date')} to {schedule.get('end_date')}",
            schedule.get("critical_path", {}).get("narrative", ""),
            "Milestones: " + "; ".join(milestone_lines),
            "Drivers: " + "; ".join(drivers),
        ]
        text = flatten_paragraphs(schedule_parts)
        sections["schedule"] = _truncate_words(text, SECTION_WORD_LIMITS.get("schedule", 150))

    references = proposal.get("appendices", {}).get("E", {}).get("references", [])
    if references:
        ref_lines = [
            f"{ref.get('client')} ({ref.get('contact')} - {ref.get('email')}): {ref.get('description')}"
            for ref in references
        ]
        text = flatten_paragraphs(ref_lines)
        sections["references"] = _truncate_words(text, SECTION_WORD_LIMITS.get("references", 80))

    return sections


def generate_chunks(proposal: Dict[str, Any], *, max_words: int = 180, overlap: int = 25) -> List[Dict[str, Any]]:
    """
    Produce embedding-ready chunks derived purely from structured YAML content.
    Total words are capped to ~512 (5% tolerance) to reduce transformer truncation.
    """
    sections = gather_section_text(proposal)
    output: List[Dict[str, Any]] = []
    chunk_lengths: List[int] = []
    total_words = 0
    for section, text in sections.items():
        splits = chunk_text(text, max_words=max_words, overlap=overlap)
        for chunk in splits:
            chunk_words = len(chunk.split())
            if total_words >= MAX_TOTAL_WORDS and total_words + chunk_words > MAX_TOTAL_WORDS_WITH_TOLERANCE:
                break
            output.append({"section": section, "text": chunk})
            chunk_lengths.append(chunk_words)
            total_words += chunk_words
        else:
            continue
        break

    if output:
        page_hints = estimate_page_hints(chunk_lengths)
        for idx, hint in enumerate(page_hints):
            output[idx]["page_hint"] = hint
    return output
