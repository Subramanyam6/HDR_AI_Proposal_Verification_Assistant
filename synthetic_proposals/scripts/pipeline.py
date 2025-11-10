"""
Core orchestration logic for the synthetic proposal dataset pipeline.
"""

from __future__ import annotations

import copy
import itertools
import json
import logging
import math
import string
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from . import common
from .util_text import generate_chunks


logger = logging.getLogger("synthetic_proposals.pipeline")
TARGET_LABELS = ("crosswalk_error", "banned_phrase", "pm_name_variant")
VARIANT_SUFFIX_MAP = {
    "clean": "",
    "crosswalk_error": "_crosswalk",
    "banned_phrase": "_banned",
    "pm_name_variant": "_name",
}


@dataclass
class ProposalPaths:
    """File layout for a single proposal instance."""

    proposal_id: str
    split: str
    base_dir: Path
    yaml_path: Path
    json_path: Path
    meta_path: Path
    labels_path: Path
    chunks_path: Path
    meta_history_path: Path

    @classmethod
    def from_dir(cls, split: str, proposal_dir: Path) -> "ProposalPaths":
        pid = proposal_dir.name
        return cls(
            proposal_id=pid,
            split=split,
            base_dir=proposal_dir,
            yaml_path=proposal_dir / "proposal.yaml",
            json_path=proposal_dir / "proposal.json",
            meta_path=proposal_dir / "meta.json",
            labels_path=proposal_dir / "labels.json",
            chunks_path=proposal_dir / "chunks.json",
            meta_history_path=proposal_dir / "meta_history.jsonl",
        )


@dataclass
class GenerationContext:
    """RNG, configuration, dictionaries, and seeds for generation stage."""

    config: Dict[str, Any]
    dictionaries: Dict[str, Any]
    seeds_by_sector: Dict[str, List[Dict[str, Any]]]
    rng: common.RandomSource


def _mix_phrase_into_sentence(sentence: str, phrase: str, rng: common.RandomSource) -> str:
    """Inject a banned phrase into an existing sentence without introducing new structure."""
    stripped = sentence.strip()
    if not stripped:
        return phrase

    tokens = stripped.split()
    if len(tokens) <= 3:
        connector = rng.choice(["and", "including", "plus"])
        return f"{stripped.rstrip()} {connector} {phrase}"

    insertion = rng.randint(1, len(tokens) - 1)
    bridge_options = ["", "and", "including", "plus", "with"]
    bridge = rng.choice(bridge_options)
    injected = phrase if not bridge else f"{bridge} {phrase}"
    tokens.insert(insertion, injected)
    mixed = " ".join(tokens)
    return mixed


def _gather_work_text_slots(work: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Collect mutable text slots within the work approach section."""
    slots: List[Dict[str, Any]] = []
    for key, value in work.items():
        base_path = f"work_approach.{key}"
        if isinstance(value, str) and value.strip():
            slots.append({"kind": "field", "container": work, "key": key, "path": base_path})
        elif isinstance(value, list):
            for idx, item in enumerate(value):
                path = f"{base_path}[{idx}]"
                if isinstance(item, str) and item.strip():
                    slots.append({"kind": "list", "container": value, "index": idx, "path": path})
                elif isinstance(item, dict):
                    for sub_key in ("text", "description", "summary", "content", "narrative"):
                        sub_val = item.get(sub_key)
                        if isinstance(sub_val, str) and sub_val.strip():
                            slots.append(
                                {
                                    "kind": "dict",
                                    "container": item,
                                    "key": sub_key,
                                    "path": f"{path}.{sub_key}",
                                }
                            )
    return slots


def _sprinkle_banned_phrase(work: Dict[str, Any], phrase: str, rng: common.RandomSource) -> Optional[str]:
    """Inline a banned phrase anywhere inside the work approach section."""
    slots = _gather_work_text_slots(work)
    if not slots:
        return None
    slot = rng.choice(slots)
    if slot["kind"] == "field":
        current = slot["container"][slot["key"]]
        slot["container"][slot["key"]] = _mix_phrase_into_sentence(current, phrase, rng)
    elif slot["kind"] == "list":
        idx = slot["index"]
        current = slot["container"][idx]
        slot["container"][idx] = _mix_phrase_into_sentence(current, phrase, rng)
    elif slot["kind"] == "dict":
        current = slot["container"][slot["key"]]
        slot["container"][slot["key"]] = _mix_phrase_into_sentence(current, phrase, rng)
    return slot["path"]


def _requirement_reference_sentence(req_id: str, rng: common.RandomSource) -> Optional[str]:
    """Create a varied sentence pointing to the work-approach requirement."""
    templates = [
        "Section 5 (Work Approach) details how we will satisfy requirement {req}.",
        "Requirement {req} is mapped directly to our Work Approach deliverables.",
        "See the Work Approach chapter for our full response to requirement {req}.",
        "Our Work Approach narrative fully addresses requirement {req}.",
    ]
    if rng.random.random() < 0.2:
        return None
    sentence = rng.choice(templates)
    return sentence.format(req=req_id)


def _executive_summary_text(req_id: str, rng: common.RandomSource, include_req: bool = True) -> str:
    """Return a base executive-summary sentence with optional explicit requirement mention."""
    with_req = [
        "Requirement {req} is addressed through phased reviews, targeted workshops, and integrated risk checkpoints.",
        "Our plan for requirement {req} combines collaborative design sprints with progressive assurance gates.",
        "To meet requirement {req}, we run iterative coordination cycles paired with on-call technical support.",
    ]
    generic = [
        "Our plan combines phased reviews, targeted workshops, and integrated risk checkpoints.",
        "The work approach blends collaborative design sprints with progressive assurance gates.",
        "We run iterative coordination cycles paired with on-call technical support.",
    ]
    templates = with_req if include_req else generic
    return rng.choice(templates).format(req=req_id)


def _remove_requirement_reference(proposal: Dict[str, Any], req_id: str, rng: common.RandomSource) -> bool:
    """Strip explicit requirement mentions to simulate missing citations."""
    pattern = f"Requirement {req_id}"
    removed = False
    letter = proposal.setdefault("letter", {})
    body = letter.setdefault("body", [])
    for idx in range(len(body) - 1, -1, -1):
        if pattern in body[idx]:
            if len(body[idx].strip()) <= len(pattern) + 20 or rng.random.random() < 0.5:
                body.pop(idx)
            else:
                body[idx] = body[idx].replace(pattern, "The requirement", 1)
            removed = True
    work = proposal.setdefault("work_approach", {})
    exec_summary = work.get("executive_summary", "")
    if isinstance(exec_summary, str) and pattern in exec_summary:
        work["executive_summary"] = _executive_summary_text(req_id, rng, include_req=False)
        removed = True
    success_factors = work.get("success_factors", [])
    for idx, line in enumerate(success_factors):
        if isinstance(line, str) and pattern in line:
            success_factors[idx] = line.replace(pattern, "Key requirement", 1)
            removed = True
    return removed


def _pick_name_variant(pm_name: str, rng: common.RandomSource) -> str:
    """Generate a plausible alternate PM name including spelling noise."""
    parts = pm_name.split()

    def init_last() -> str:
        if len(parts) >= 2:
            return f"{parts[0]} {parts[-1][0]}."
        return parts[0]

    def first_initial_full_last() -> str:
        if len(parts) >= 2:
            return f"{parts[0][0]}. {parts[-1]}"
        return pm_name

    def suffix_variant() -> str:
        suffix = rng.choice(["Jr.", "Sr.", "III"])
        return f"{pm_name} {suffix}"

    def uppercase() -> str:
        return pm_name.upper()

    strategies = [init_last, first_initial_full_last, suffix_variant, uppercase]
    strategies.append(lambda: _apply_spelling_noise(pm_name, rng))
    variant = rng.choice(strategies)()
    if variant == pm_name:
        variant = _apply_spelling_noise(pm_name, rng)
    return variant


def _apply_spelling_noise(name: str, rng: common.RandomSource) -> str:
    """Introduce letter-level typos across the name tokens."""
    tokens = name.split()
    if not tokens:
        return name
    edits = rng.randint(1, min(2, len(tokens)))
    indices = rng.sample(range(len(tokens)), edits)
    for idx in indices:
        tokens[idx] = _mutate_token(tokens[idx], rng)
    return " ".join(tokens)


def _mutate_token(token: str, rng: common.RandomSource) -> str:
    """Apply a random mutation (delete, duplicate, swap, substitute) to a token."""
    if len(token) <= 1:
        return token
    letters = list(token)
    pos = rng.randint(0, len(letters) - 1)
    operations = ["delete", "duplicate", "swap", "substitute"]
    op = rng.choice(operations)
    if op == "delete" and len(letters) > 1:
        letters.pop(pos)
    elif op == "duplicate":
        letters.insert(pos, letters[pos])
    elif op == "swap" and pos < len(letters) - 1:
        letters[pos], letters[pos + 1] = letters[pos + 1], letters[pos]
    else:  # substitute
        replacement = rng.choice(string.ascii_lowercase)
        letters[pos] = replacement.upper() if letters[pos].isupper() else replacement
    return "".join(letters)


def _collect_pm_name_slots(proposal: Dict[str, Any], pm_name: str) -> List[Dict[str, Any]]:
    """Find all sections that currently contain the PM name."""
    slots: List[Dict[str, Any]] = []
    letter = proposal.setdefault("letter", {})
    body = letter.setdefault("body", [])
    if any(pm_name in line for line in body):
        slots.append({"kind": "list", "container": body, "allow_multiple": True})

    staffing = proposal.setdefault("staffing_plan", {})
    availability = staffing.setdefault("availability", [])
    if any(pm_name in line for line in availability):
        slots.append({"kind": "list", "container": availability, "allow_multiple": False})

    work = proposal.setdefault("work_approach", {})
    exec_summary = work.get("executive_summary")
    if isinstance(exec_summary, str) and pm_name in exec_summary:
        slots.append({"kind": "field", "container": work, "key": "executive_summary"})
    success_factors = work.setdefault("success_factors", [])
    if any(isinstance(line, str) and pm_name in line for line in success_factors):
        slots.append({"kind": "list", "container": success_factors, "allow_multiple": True})

    subs = proposal.setdefault("subs_summary", {})
    for key in ("approach", "dbe_strategy"):
        entries = subs.setdefault(key, [])
        if any(pm_name in line for line in entries):
            slots.append({"kind": "list", "container": entries, "allow_multiple": False})

    schedule = proposal.setdefault("schedule", {})
    for milestone in schedule.get("milestones", []):
        desc = milestone.get("description")
        if isinstance(desc, str) and pm_name in desc:
            slots.append({"kind": "dict", "container": milestone, "key": "description"})

    return slots


def _replace_name_in_slot(slot: Dict[str, Any], pm_name: str, variant: str) -> bool:
    """Swap a PM name occurrence inside a collected slot."""
    if slot["kind"] == "field":
        text = slot["container"].get(slot["key"], "")
        if pm_name in text:
            slot["container"][slot["key"]] = text.replace(pm_name, variant, 1)
            return True
    elif slot["kind"] == "dict":
        text = slot["container"].get(slot["key"], "")
        if pm_name in text:
            slot["container"][slot["key"]] = text.replace(pm_name, variant, 1)
            return True
    elif slot["kind"] == "list":
        lines = slot["container"]
        replaced = False
        for idx, line in enumerate(lines):
            if pm_name in line:
                lines[idx] = line.replace(pm_name, variant, 1)
                replaced = True
                if not slot.get("allow_multiple", False):
                    break
        return replaced
    return False


def _load_seeds_by_sector() -> Dict[str, List[Dict[str, Any]]]:
    seeds: Dict[str, List[Dict[str, Any]]] = {}
    for seed in common.load_seeds():
        sector = seed.get("metadata", {}).get("sector", "General")
        seeds.setdefault(sector, []).append(common.safe_copy_dict(seed))
    return seeds


def _allocate_split_counts(total: int, ratios: Dict[str, float]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    assigned = 0
    ordered = list(ratios.items())
    for split, ratio in ordered:
        count = int(total * ratio)
        counts[split] = count
        assigned += count
    remainder = total - assigned
    if remainder > 0:
        # Distribute remainder based on descending ratio order
        for split, _ in sorted(ordered, key=lambda item: item[1], reverse=True):
            if remainder <= 0:
                break
            counts[split] += 1
            remainder -= 1
    return counts


def _resolve_fraction(
    counts_cfg: Dict[str, Any],
    fraction_key: str,
    legacy_key: str,
    total: int,
) -> float:
    """Resolve a minimum coverage fraction from config, supporting legacy absolute knobs."""

    if total <= 0:
        return 0.0
    if fraction_key in counts_cfg:
        try:
            value = float(counts_cfg[fraction_key])
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(value, 1.0))
    legacy_value = counts_cfg.get(legacy_key)
    if legacy_value is None:
        return 0.0
    try:
        legacy_float = float(legacy_value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(legacy_float / float(total), 1.0))


def _build_attribute_sequence(
    items: Sequence[str],
    total: int,
    min_fraction: float,
    rng: common.RandomSource,
) -> List[str]:
    """Allocate items across `total` slots while honouring minimum coverage targets."""

    items = [item for item in items if item]
    if not items:
        return ["" for _ in range(max(total, 0))]
    if total <= 0:
        return []

    counts: Dict[str, int] = {item: 0 for item in items}
    remaining = total

    # Step 1: ensure each item appears at least once when possible.
    if total >= len(items):
        for item in items:
            if remaining <= 0:
                break
            counts[item] += 1
            remaining -= 1

    # Step 2: honour fractional minimums (converted to absolute target per item).
    if min_fraction > 0.0:
        target_per_item = max(1, math.ceil(total * min_fraction))
        for item in items:
            while counts[item] < target_per_item and remaining > 0:
                counts[item] += 1
                remaining -= 1
            if remaining <= 0:
                break

    # Step 3: distribute remaining slots randomly.
    while remaining > 0:
        counts[rng.choice(items)] += 1
        remaining -= 1

    sequence: List[str] = []
    for item in items:
        sequence.extend([item] * counts[item])

    # Guard against rounding artefacts.
    if len(sequence) < total:
        sequence.extend(rng.choice(items) for _ in range(total - len(sequence)))
    elif len(sequence) > total:
        sequence = sequence[:total]

    rng.shuffle(sequence)
    return sequence


def _random_date_within(rng: common.RandomSource, year: int) -> date:
    start = date(year, 1, 1)
    end = date(year, 12, 31)
    delta = (end - start).days
    return start + timedelta(days=rng.randint(0, delta))


def _generate_addenda(rng: common.RandomSource, count: int) -> List[Dict[str, Any]]:
    addenda: List[Dict[str, Any]] = []
    for idx in range(count):
        addenda.append(
            {
                "number": idx + 1,
                "title": f"Addendum {idx + 1} clarification on scope item {rng.randint(1, 6)}",
                "date": str(_random_date_within(rng, 2024)),
            }
        )
    return addenda


def _generate_team(seed_team: Dict[str, Any], ctx: GenerationContext) -> Dict[str, Any]:
    rng = ctx.rng
    faker = rng.faker
    team = common.safe_copy_dict(seed_team) if seed_team else {}
    def random_contact() -> Dict[str, str]:
        name = faker.name()
        email = common.slugify(name) + "@example.com"
        return {"name": name, "email": email, "phone": common.random_phone(rng.random)}

    pm = random_contact()
    qm = random_contact()
    team["pm"] = pm
    team["quality_manager"] = qm
    team["pm_title"] = "Program Manager"
    team["pm_summary"] = faker.sentence(nb_words=6)
    team["qm_title"] = "Quality Assurance Manager"
    team["qm_summary"] = faker.sentence(nb_words=6)
    role_traits = ctx.dictionaries.get("people_roles", {})
    tech_leads_vocab = role_traits.get("technical_leads", [])
    resume_snippets = role_traits.get("resume_snippets", [])
    tech_leads: List[Dict[str, Any]] = []
    lead_count = max(3, min(4, rng.randint(3, 4)))
    rng.shuffle(tech_leads_vocab)
    for idx in range(lead_count):
        trait = tech_leads_vocab[idx % len(tech_leads_vocab)] if tech_leads_vocab else f"Discipline Lead {idx+1}"
        tech_leads.append(
            {
                "role": trait.title(),
                "name": faker.name(),
                "bio": resume_snippets[idx % len(resume_snippets)] if resume_snippets else faker.sentence(nb_words=6),
            }
        )
    team["technical_leads"] = tech_leads
    support_titles = role_traits.get("support_titles", [])
    team["subs"] = []
    sub_count = rng.randint(3, 4)
    for _ in range(sub_count):
        company = faker.company()
        team["subs"].append(
            {
                "name": company,
                "role": rng.choice(support_titles) if support_titles else "Specialty Support",
                "dbe": rng.boolean(0.5),
                "scope": faker.sentence(nb_words=6),
            }
        )
    return team


def _generate_letter(metadata: Dict[str, Any], team: Dict[str, Any], ctx: GenerationContext, tone: str) -> Dict[str, Any]:
    pm_name = team.get("pm", {}).get("name", "Project Manager")
    return {
        "tone": tone,
        "body": [],
        "closing": "Respectfully submitted,",
        "signature_block": {
            "name": pm_name,
            "title": "Project Manager",
            "date": metadata.get("submission_date"),
        },
    }


def _generate_qualifications(ctx: GenerationContext, sector: str) -> List[Dict[str, Any]]:
    base_entries = [
        {
            "title": "Final Design and Documentation",
            "description": "30/60/90 percent design submittals",
            "highlights": [
                "Support across commercial write.",
                "Long age west seek.",
                "Enjoy happy military agent significant hotel.",
            ],
        },
        {
            "title": "Alternatives Development and Evaluation",
            "description": "Screening matrix",
            "highlights": [
                "Talk want tell cause national often real.",
                "Special wind pay his.",
                "Risk data room consumer I away.",
            ],
        },
    ]
    return base_entries


def _generate_staffing_plan(team: Dict[str, Any], ctx: GenerationContext) -> Dict[str, Any]:
    rng = ctx.rng
    overview: List[str] = []
    availability: List[str] = []
    resource_matrix: List[Dict[str, Any]] = [
        {"role": "Project Manager", "name": team["pm"]["name"], "allocation_percent": rng.randint(35, 50)},
        {"role": "Quality Manager", "name": team["quality_manager"]["name"], "allocation_percent": rng.randint(20, 30)},
    ]
    for lead in team.get("technical_leads", []):
        resource_matrix.append(
            {"role": lead["role"], "name": lead["name"], "allocation_percent": rng.randint(15, 30)}
        )
    return {"overview": overview, "availability": availability, "resource_matrix": resource_matrix}


def _generate_subs_summary(team: Dict[str, Any], ctx: GenerationContext) -> Dict[str, Any]:
    faker = ctx.rng.faker
    approach = [
        "Subconsultants support workstreams.",
    ]
    dbe_strategy = [
        "DBE partners get mentorship.",
    ]
    return {"approach": approach, "dbe_strategy": dbe_strategy}


def _generate_schedule(seed_schedule: Dict[str, Any], ctx: GenerationContext) -> Dict[str, Any]:
    rng = ctx.rng
    schedule = {}
    start_dt = _random_date_within(rng, 2024)
    end_dt = start_dt + timedelta(days=365)
    schedule["start_date"] = str(start_dt)
    schedule["end_date"] = str(end_dt)
    schedule["milestones"] = [
        {"name": "Program Kickoff", "date": str(start_dt), "description": "Governance charter approval."},
        {"name": "Final Delivery", "date": str(end_dt), "description": "Final design package submitted."},
    ]
    schedule["critical_path"] = {
        "narrative": "Approvals drive schedule.",
        "drivers": ["Agency approvals", "Coordination"],
    }
    schedule["questions_due"] = str(start_dt - timedelta(days=30))
    return schedule


def _generate_appendices(seed_appendices: Dict[str, Any], team: Dict[str, Any], ctx: GenerationContext) -> Dict[str, Any]:
    rng = ctx.rng
    faker = rng.faker
    appendices = common.safe_copy_dict(seed_appendices) if seed_appendices else {}
    acknowledgments = appendices.get("A", {}).get("addenda_acknowledgment", [])
    if not acknowledgments:
        acknowledgments = []
    appendices.setdefault("A", {})["addenda_acknowledgment"] = acknowledgments

    dbe_plan = appendices.get("B", {}).get("dbe_plan", [])
    if not dbe_plan:
        dbe_plan = [
            {
                "firm": faker.company(),
                "role": "Community Engagement",
                "commitment_percent": rng.randint(8, 20),
            }
        ]
    appendices.setdefault("B", {})["dbe_plan"] = dbe_plan

    resumes = appendices.get("C", {}).get("resumes", [])
    if not resumes:
        resumes = []
        resumes.append(
            {
                "name": team["pm"]["name"],
                "role": "Project Manager",
                "years_experience": rng.randint(12, 22),
                "summary": faker.sentence(nb_words=6),
            }
        )
        resumes.append(
            {
                "name": team["quality_manager"]["name"],
                "role": "Quality Manager",
                "years_experience": rng.randint(10, 20),
                "summary": faker.sentence(nb_words=6),
            }
        )
        for lead in team.get("technical_leads", [])[:2]:
            resumes.append(
                {
                    "name": lead["name"],
                    "role": lead["role"],
                    "years_experience": rng.randint(8, 18),
                    "summary": lead.get("bio", faker.sentence(nb_words=6)),
                }
            )
    appendices.setdefault("C", {})["resumes"] = resumes

    forms = appendices.get("D", {}).get(
        "forms", {"non_collusion": True, "insurance_ack": True, "certification": True}
    )
    appendices.setdefault("D", {})["forms"] = forms

    references = appendices.get("E", {}).get("references", [])
    if not references:
        references = [
            {
                "client": faker.company(),
                "contact": faker.name(),
                "phone": common.random_phone(rng.random),
                "email": f"{common.slugify(faker.name())}@example.com",
                "description": faker.sentence(nb_words=6),
            }
            for _ in range(1)
        ]
    appendices.setdefault("E", {})["references"] = references
    return appendices


def _generate_work_approach(seed_work: Dict[str, Any], ctx: GenerationContext, sector: str) -> Dict[str, Any]:
    rng = ctx.rng
    faker = rng.faker
    work = common.safe_copy_dict(seed_work) if seed_work else {}
    work["executive_summary"] = ""
    work["objectives"] = []
    work["constraints"] = []
    work["success_factors"] = []
    work["elements"] = []
    return work


def _select_primary_requirement(matrix: List[Dict[str, Any]]) -> Dict[str, Any]:
    for entry in matrix:
        if entry.get("expected_section") == "Work Approach":
            return entry
    return matrix[0] if matrix else {"req_id": "R1", "expected_section": "Work Approach", "cited_section": "Work Approach"}


def _apply_base_narrative(
    proposal: Dict[str, Any],
    metadata: Dict[str, Any],
    team: Dict[str, Any],
    requirement: Dict[str, Any],
    ctx: GenerationContext,
) -> None:
    pm_name = team.get("pm", {}).get("name", "Project Manager")
    project_name = metadata.get("rfp_title") or metadata.get("project_name") or "the project"
    submission_date = metadata.get("submission_date", "2024-12-31")
    req_id = requirement.get("req_id", "R1")
    rng = ctx.rng
    letter = proposal.setdefault("letter", {})
    letter_body = [
        f"We are pleased to submit our proposal for {project_name}.",
        f"{pm_name} will serve as the primary contact and Project Manager for this engagement.",
        f"Our anticipated submission date remains {submission_date}.",
        f"This proposal was signed and sealed on {submission_date} by {pm_name}.",
        "Our team brings extensive experience in transit design and environmental compliance.",
    ]
    requirement_sentence = _requirement_reference_sentence(req_id, rng)
    if requirement_sentence:
        letter_body.append(requirement_sentence)
    letter["body"] = letter_body

    staffing = proposal.setdefault("staffing_plan", {})
    staffing["overview"] = [
        "Our project team consists of highly qualified professionals with proven track records.",
    ]
    staffing["availability"] = [
        f"{pm_name} will remain available 40 hours per week as the dedicated project lead.",
    ]

    work = proposal.setdefault("work_approach", {})
    work["executive_summary"] = _executive_summary_text(req_id, rng, include_req=True)
    work["success_factors"] = [
        "This provides an assurance that our delivery method avoids scope drift.",
        "We avoid scope drift by using compliant language throughout our program plan.",
    ]


def _generate_compliance_matrix(ctx: GenerationContext) -> List[Dict[str, Any]]:
    requirements = ctx.dictionaries.get("compliance", {}).get("crosswalk_requirements", [])
    matrix: List[Dict[str, Any]] = []
    sections = [
        "Letter of Transmittal",
        "Qualifications",
        "Staffing",
        "Work Approach",
        "Schedule",
        "Appendices",
    ]
    for idx, req in enumerate(requirements):
        matrix.append(
            {
                "req_id": req.get("id", f"R{idx+1}"),
                "description": req.get("description", ""),
                "expected_section": sections[idx % len(sections)],
                "cited_section": sections[idx % len(sections)],
            }
        )
    return matrix


def _select_style_plan(ctx: GenerationContext, mistakes: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    rng = ctx.rng
    mistakes = mistakes or []
    variants = ctx.config.get("content_variation", {}).get("margin_variants", ["base"])
    variant_choice = rng.choice(variants)
    font_families = ctx.dictionaries.get("misc", {}).get("fonts", ["Arial"])
    font_family = rng.choice(font_families)
    base_font_size = 11.5
    font_violation = "font_size_violation" in mistakes
    if font_violation:
        base_font_size = 9.5
    margin_profiles = {
        "base": {"top": 1.0, "bottom": 1.0, "left": 1.0, "right": 1.0},
        "classic": {"top": 1.1, "bottom": 1.1, "left": 1.0, "right": 1.0},
        "modern": {"top": 0.85, "bottom": 0.85, "left": 0.9, "right": 0.9},
        "technical": {"top": 0.75, "bottom": 0.75, "left": 0.85, "right": 0.85},
    }
    margins = margin_profiles.get(variant_choice, margin_profiles["base"]).copy()
    margin_violation = "margin_violation" in mistakes
    if margin_violation:
        margins["left"] = 0.6
        margins["right"] = 0.6
        margins["top"] = 0.65
    header_cfg = ctx.config.get("content_variation", {}).get("header_footer_variants", {})
    if isinstance(header_cfg, dict) and header_cfg:
        total = sum(header_cfg.values())
        pick = rng.uniform(0, total)
        cumulative = 0.0
        header_variant = list(header_cfg.keys())[0]
        for key, weight in header_cfg.items():
            cumulative += weight
            if pick <= cumulative:
                header_variant = key
                break
    else:
        header_variant = "classic"

    return {
        "variant_code": variant_choice,
        "font_family": font_family,
        "base_font_size": base_font_size,
        "font_size_violation": font_violation,
        "margin_violation": margin_violation,
        "margins": margins,
        "header_variant": header_variant,
    }


def _prepare_noise_plan(ctx: GenerationContext) -> Dict[str, Any]:
    rng = ctx.rng
    noise_cfg = ctx.config.get("noise", {})
    apply_noise = rng.boolean(noise_cfg.get("apply_probability", 0.0))
    plan = {"apply": apply_noise, "operations": []}
    if not apply_noise:
        return plan
    operations: List[Dict[str, Any]] = []
    if rng.boolean(noise_cfg.get("skew_probability", 0.0)):
        operations.append({"type": "skew", "degrees": rng.uniform(-1.2, 1.2)})
    if rng.boolean(noise_cfg.get("stamp_probability", 0.0)):
        operations.append({"type": "stamp", "text": rng.choice(noise_cfg.get("stamp_texts", ["RECEIVED"]))})
    if rng.boolean(noise_cfg.get("ocr_probability", 0.0)):
        operations.append({"type": "ocr_noise"})
    if not operations:
        operations.append({"type": "stamp", "text": rng.choice(noise_cfg.get("stamp_texts", ["RECEIVED"]))})
    plan["operations"] = operations
    return plan


def _build_clean_payload(
    proposal_id: str,
    split: str,
    sector: str,
    agency_type: str,
    ctx: GenerationContext,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    rng = ctx.rng
    seeds = ctx.seeds_by_sector.get(sector) or list(itertools.chain.from_iterable(ctx.seeds_by_sector.values()))
    seed = common.safe_copy_dict(rng.choice(seeds)) if seeds else {}
    metadata = _generate_metadata(seed.get("metadata", {}), sector, agency_type, ctx)
    team = _generate_team(seed.get("team", {}), ctx)
    tone_choices = ctx.config.get("content_variation", {}).get("letter_styles", ["formal"])
    tone_weights = ctx.config.get("content_variation", {}).get("tone_weights", {})
    tone = rng.choice(tone_choices)
    if tone_weights:
        weighted = [(style, tone_weights.get(style, 0.2)) for style in tone_choices]
        total = sum(weight for _, weight in weighted)
        if total > 0:
            pick = rng.uniform(0, total)
            cumulative = 0.0
            for style, weight in weighted:
                cumulative += weight
                if pick <= cumulative:
                    tone = style
                    break
    letter = _generate_letter(metadata, team, ctx, tone)
    qualifications = _generate_qualifications(ctx, sector)
    staffing_plan = _generate_staffing_plan(team, ctx)
    subs_summary = _generate_subs_summary(team, ctx)
    work_approach = _generate_work_approach(seed.get("work_approach", {}), ctx, sector)
    schedule = _generate_schedule(seed.get("schedule", {}), ctx)
    appendices = _generate_appendices(seed.get("appendices", {}), team, ctx)
    checklist_seed = common.safe_copy_dict(seed.get("checklist", {}))
    content_cfg = ctx.config.get("content_variation", {})
    page_limit = rng.choice(content_cfg.get("page_limit_options", [40, 45]))
    compliance_cfg = ctx.config.get("compliance", {})
    dbe_goal = rng.randint(*compliance_cfg.get("dbe_goal_range", [12, 25]))
    checklist = {
        "page_limit_required": page_limit,
        "addenda_required": bool(metadata.get("addenda")),
        "dbe_goal_percent": dbe_goal,
        "resumes_required": True,
        "insurance_certificate": True,
    }
    checklist.update(checklist_seed)
    compliance_matrix = _generate_compliance_matrix(ctx)
    primary_requirement = _select_primary_requirement(compliance_matrix)
    style_plan = _select_style_plan(ctx, [])
    noise_plan = _prepare_noise_plan(ctx)
    proposal = {
        "proposal_id": proposal_id,
        "split": split,
        "metadata": metadata,
        "team": team,
        "letter": letter,
        "qualifications": qualifications,
        "staffing_plan": staffing_plan,
        "subs_summary": subs_summary,
        "work_approach": work_approach,
        "schedule": schedule,
        "appendices": appendices,
        "checklist": checklist,
        "compliance_matrix": compliance_matrix,
    }

    meta = {
        "id": proposal_id,
        "split": split,
        "sector": sector,
        "agency_type": metadata.get("agency_type"),
        "style": style_plan,
        "noise_plan": noise_plan,
        "mistakes": {label: False for label in TARGET_LABELS},
        "content": {
            "tone": tone,
            "page_limit_required": page_limit,
            "banned_phrases": [],
        },
        "consistency": {},
        "compliance": {
            "dbe_goal_percent": dbe_goal,
            "dbe_commit_percent": sum(item.get("commitment_percent", 0) for item in appendices.get("B", {}).get("dbe_plan", [])),
            "primary_requirement": primary_requirement,
            "crosswalk_errors": [],
        },
        "render": {},
        "noise": {},
        "timestamp": datetime.utcnow().isoformat(),
    }
    _apply_base_narrative(proposal, metadata, team, primary_requirement, ctx)
    meta["compliance"]["dbe_commit_percent"] = sum(
        item.get("commitment_percent", 0) for item in proposal.get("appendices", {}).get("B", {}).get("dbe_plan", [])
    )
    return proposal, meta


def _build_variant_payloads(
    base_proposal: Dict[str, Any],
    base_meta: Dict[str, Any],
    ctx: GenerationContext,
) -> Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]]:
    variants: Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]] = {"clean": (base_proposal, base_meta)}
    for label in TARGET_LABELS:
        proposal_clone = copy.deepcopy(base_proposal)
        meta_clone = copy.deepcopy(base_meta)
        applied = False
        if label == "crosswalk_error":
            applied = _apply_crosswalk_variant(proposal_clone, meta_clone, ctx)
        elif label == "banned_phrase":
            applied = _apply_banned_phrase_variant(proposal_clone, meta_clone, ctx)
        elif label == "pm_name_variant":
            applied = _apply_name_variant(proposal_clone, meta_clone, ctx)
        if applied:
            meta_clone["mistakes"][label] = True
            variants[label] = (proposal_clone, meta_clone)
    return variants


def _apply_crosswalk_variant(proposal: Dict[str, Any], meta: Dict[str, Any], ctx: GenerationContext) -> bool:
    rng = ctx.rng
    primary = meta.get("compliance", {}).get("primary_requirement", {})
    req_id = primary.get("req_id")
    if not req_id:
        return False
    work = proposal.setdefault("work_approach", {})
    alt_req = "R2" if req_id != "R2" else "R3"
    strategy_missing = rng.random.random() < 0.4
    cited_section = "Work Approach (wrong content)"
    detail = f"Cited {alt_req} instead of {req_id}"
    if strategy_missing:
        removed = _remove_requirement_reference(proposal, req_id, rng)
        if removed:
            cited_section = "Not cited"
            detail = "Requirement reference removed"
        else:
            strategy_missing = False
    if not strategy_missing:
        sentence = work.get("executive_summary", "")
        if sentence:
            work["executive_summary"] = sentence.replace(req_id, alt_req, 1)
        else:
            work["executive_summary"] = _executive_summary_text(alt_req, rng, include_req=True)
        for row in proposal.get("compliance_matrix", []):
            if row.get("req_id") == req_id:
                row["cited_section"] = cited_section
                break
    meta.setdefault("compliance", {})["crosswalk_errors"] = [
        {
            "req_id": req_id,
            "expected_section": "Work Approach",
            "cited_section": cited_section,
            "detail": detail,
        }
    ]
    return True


def _apply_banned_phrase_variant(proposal: Dict[str, Any], meta: Dict[str, Any], ctx: GenerationContext) -> bool:
    banned_phrases = ctx.dictionaries.get("compliance", {}).get("banned_phrases", [])
    if not banned_phrases:
        return False
    rng = ctx.rng
    phrase = rng.choice(banned_phrases)
    work = proposal.setdefault("work_approach", {})
    section_path = _sprinkle_banned_phrase(work, phrase, rng)
    if not section_path:
        success_factors = work.setdefault("success_factors", [])
        insert_at = rng.randint(0, len(success_factors)) if success_factors else 0
        success_factors.insert(insert_at, phrase)
        section_path = "work_approach.success_factors"
    meta.setdefault("content", {})["banned_phrases"] = [{"phrase": phrase, "section": section_path}]
    return True


def _apply_name_variant(proposal: Dict[str, Any], meta: Dict[str, Any], ctx: GenerationContext) -> bool:
    rng = ctx.rng
    pm_name = proposal.get("team", {}).get("pm", {}).get("name")
    if not pm_name:
        return False
    slots = _collect_pm_name_slots(proposal, pm_name)
    if not slots:
        return False
    variant = _pick_name_variant(pm_name, rng)
    rng.shuffle(slots)
    count = rng.randint(1, len(slots))
    for slot in slots[:count]:
        _replace_name_in_slot(slot, pm_name, variant)
    meta.setdefault("consistency", {})["pm_name_variant"] = {"original": pm_name, "variant": variant}
    return True


def generate_yaml_batch(
    *,
    config_path: Optional[Path] = None,
    dataset_root: Optional[Path] = None,
) -> List[ProposalPaths]:
    config = common.load_knobs(config_path)
    dictionaries = common.load_dictionaries()
    seeds_by_sector = _load_seeds_by_sector()
    rng = common.RandomSource(config.get("seed", 1234))
    ctx = GenerationContext(config=config, dictionaries=dictionaries, seeds_by_sector=seeds_by_sector, rng=rng)
    root = dataset_root or common.DATASET_ROOT
    common.ensure_dir(root)

    counts_cfg = config.get("counts", {})
    total = counts_cfg.get("total", 250)
    split_ratios = counts_cfg.get("split", {"train": 0.72, "dev": 0.14, "test": 0.14})
    split_counts = _allocate_split_counts(total, split_ratios)
    split_sequence: List[str] = list(itertools.chain.from_iterable([[split] * count for split, count in split_counts.items()]))
    rng.shuffle(split_sequence)

    sectors = list(dictionaries.get("agency_sector", {}).get("sectors", {}).keys())
    if not sectors:
        sectors = ["Transportation", "Water", "Energy", "Buildings", "Environmental"]
    sector_fraction = _resolve_fraction(counts_cfg, "min_by_sector_fraction", "min_by_sector", total)
    sector_sequence = _build_attribute_sequence(sectors, total, sector_fraction, rng)

    agency_types = dictionaries.get("agency_sector", {}).get("agency_types", [])
    if not agency_types:
        agency_types = ["City", "County", "State", "Transit", "University", "Port Authority"]
    agency_fraction = _resolve_fraction(counts_cfg, "min_by_agency_type_fraction", "min_by_agency_type", total)
    agency_sequence = _build_attribute_sequence(agency_types, total, agency_fraction, rng)

    proposals: List[ProposalPaths] = []
    for index in range(total):
        proposal_id = f"proposal_{index + 1:04d}"
        split = split_sequence[index]
        sector = sector_sequence[index] if index < len(sector_sequence) else rng.choice(sectors)
        agency_type = agency_sequence[index] if index < len(agency_sequence) else rng.choice(agency_types)
        base_proposal, base_meta = _build_clean_payload(proposal_id, split, sector, agency_type, ctx)
        variant_payloads = _build_variant_payloads(base_proposal, base_meta, ctx)

        for variant_key, (variant_proposal, variant_meta) in variant_payloads.items():
            suffix = VARIANT_SUFFIX_MAP.get(variant_key, f"_{variant_key}")
            variant_id = proposal_id if suffix == "" else f"{proposal_id}{suffix}"

            variant_proposal["proposal_id"] = variant_id
            variant_proposal["split"] = split
            variant_meta["id"] = variant_id
            variant_meta["split"] = split
            variant_meta.setdefault("paths", {})

            proposal_dir = root / split / variant_id
            common.ensure_dir(proposal_dir)
            paths = ProposalPaths.from_dir(split, proposal_dir)

            common.dump_yaml_file(variant_proposal, paths.yaml_path)
            common.dump_json_file(variant_proposal, paths.json_path)

            variant_meta["paths"]["proposal_yaml"] = str(paths.yaml_path)
            variant_meta["paths"]["proposal_json"] = str(paths.json_path)
            common.dump_json_file(variant_meta, paths.meta_path)
            with paths.meta_history_path.open("a", encoding="utf-8") as history_file:
                history_file.write(json.dumps(variant_meta) + "\n")

            proposals.append(paths)
    logger.info("Generated %d proposal YAML manifests", len(proposals))
    return proposals


def make_chunks(
    proposals: Optional[List[ProposalPaths]] = None,
    *,
    dataset_root: Optional[Path] = None,
) -> List[ProposalPaths]:
    root = dataset_root or common.DATASET_ROOT
    if proposals is None:
        proposals = [ProposalPaths.from_dir(split, path) for split, path in common.iter_proposal_dirs(root)]

    processed: List[ProposalPaths] = []
    for paths in proposals:
        if not paths.yaml_path.exists():
            continue
        data = common.load_yaml_file(paths.yaml_path)
        chunks = generate_chunks(data)
        common.dump_json_file(chunks, paths.chunks_path)
        if paths.meta_path.exists():
            meta = common.load_json_file(paths.meta_path)
            meta.setdefault("content", {})["chunk_count"] = len(chunks)
            if chunks:
                estimated_pages = max(chunk.get("page_hint", 1) for chunk in chunks if chunk.get("page_hint"))
            else:
                estimated_pages = None
            meta.setdefault("render", {})["page_count_estimate"] = estimated_pages
            common.dump_json_file(meta, paths.meta_path)
            with paths.meta_history_path.open("a", encoding="utf-8") as history_file:
                history_file.write(json.dumps(meta) + "\n")
        processed.append(paths)
    logger.info("Generated chunks for %d proposals", len(processed))
    return processed


def _summarize_labels(proposal: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    checklist = proposal.get("checklist", {})
    appendices = proposal.get("appendices", {})
    addenda_ack = appendices.get("A", {}).get("addenda_acknowledgment", [])
    forms = appendices.get("D", {}).get(
        "forms", {"non_collusion": False, "insurance_ack": False, "certification": False}
    )
    dbe_plan = appendices.get("B", {}).get("dbe_plan", [])
    render_meta = meta.get("render", {})
    page_count = render_meta.get("page_count")
    if page_count is None:
        page_count = render_meta.get("page_count_estimate")
    page_limit = checklist.get("page_limit_required")
    page_limit_violation = meta.get("content", {}).get("page_limit_violation", False)
    if page_count is not None and page_limit is not None:
        page_limit_violation = page_limit_violation or (page_count > page_limit)

    dbe_commit_percent = sum(entry.get("commitment_percent", 0) for entry in dbe_plan)
    dbe_goal_percent = checklist.get("dbe_goal_percent", meta.get("compliance", {}).get("dbe_goal_percent"))
    dbe_gap_flag = meta.get("compliance", {}).get("dbe_gap", False) or (
        dbe_goal_percent is not None and dbe_commit_percent < dbe_goal_percent
    )

    crosswalk_errors = meta.get("compliance", {}).get("crosswalk_errors")
    if not crosswalk_errors:
        crosswalk_errors = []
        for row in proposal.get("compliance_matrix", []):
            if row.get("expected_section") != row.get("cited_section"):
                crosswalk_errors.append(
                    {
                        "req_id": row.get("req_id"),
                        "expected_section": row.get("expected_section"),
                        "cited_section": row.get("cited_section"),
                    }
                )

    banned_phrases = meta.get("content", {}).get("banned_phrases", [])

    labels = {
        "page_limit_required": page_limit,
        "page_count_actual": page_count,
        "page_limit_violation": page_limit_violation,
        "addenda_required": checklist.get("addenda_required", bool(proposal.get("metadata", {}).get("addenda"))),
        "addenda_acknowledged": bool(addenda_ack) and all(item.get("acknowledged", False) for item in addenda_ack),
        "forms_present": {
            "non_collusion": forms.get("non_collusion", False),
            "insurance_ack": forms.get("insurance_ack", False),
            "certification": forms.get("certification", False),
        },
        "dbe_goal_percent": dbe_goal_percent,
        "dbe_commit_percent": dbe_commit_percent,
        "dbe_gap_flag": dbe_gap_flag,
        "name_consistency_flag": "pm_name_variant" not in meta.get("consistency", {}),
        "crosswalk_errors": crosswalk_errors,
        "banned_phrases_found": banned_phrases,
        "font_size_violation": meta.get("style", {}).get("font_size_violation", False),
        "margin_violation": meta.get("style", {}).get("margin_violation", False),
    }
    return labels


def label_dataset(
    proposals: Optional[List[ProposalPaths]] = None,
    *,
    dataset_root: Optional[Path] = None,
) -> List[ProposalPaths]:
    root = dataset_root or common.DATASET_ROOT
    if proposals is None:
        proposals = [ProposalPaths.from_dir(split, path) for split, path in common.iter_proposal_dirs(root)]

    labeled: List[ProposalPaths] = []
    for paths in proposals:
        if not paths.yaml_path.exists() or not paths.meta_path.exists():
            continue
        data = common.load_yaml_file(paths.yaml_path)
        meta = common.load_json_file(paths.meta_path)
        meta.setdefault("paths", {})["proposal_json"] = str(paths.json_path)
        meta["paths"]["labels_json"] = str(paths.labels_path)
        meta["paths"]["chunks_json"] = str(paths.chunks_path)
        labels = _summarize_labels(data, meta)
        common.dump_json_file(labels, paths.labels_path)
        common.dump_json_file(meta, paths.meta_path)
        with paths.meta_history_path.open("a", encoding="utf-8") as history_file:
            history_file.write(json.dumps(meta) + "\n")
        labeled.append(paths)
    logger.info("Generated labels for %d proposals", len(labeled))
    return labeled


def export_dataset_json(
    proposals: Optional[List[ProposalPaths]] = None,
    *,
    dataset_root: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    root = dataset_root or common.DATASET_ROOT
    if proposals is None:
        proposals = [ProposalPaths.from_dir(split, path) for split, path in common.iter_proposal_dirs(root)]

    exported: List[Dict[str, Any]] = []
    for paths in proposals:
        if not paths.json_path.exists():
            continue
        proposal_data = common.load_json_file(paths.json_path)
        meta = common.load_json_file(paths.meta_path) if paths.meta_path.exists() else {}
        labels = common.load_json_file(paths.labels_path) if paths.labels_path.exists() else {}
        chunks = common.load_json_file(paths.chunks_path) if paths.chunks_path.exists() else []
        exported.append(
            {
                "id": paths.proposal_id,
                "split": paths.split,
                "proposal": proposal_data,
                "meta": meta,
                "labels": labels,
                "chunks": chunks,
            }
        )

    output_path = root / "synthetic_proposals.json"
    common.dump_json_file(exported, output_path)
    logger.info("Exported %d proposals to %s", len(exported), output_path)
    return exported


def build_dataset(
    *,
    config_path: Optional[Path] = None,
    dataset_root: Optional[Path] = None,
) -> List[ProposalPaths]:
    proposals = generate_yaml_batch(config_path=config_path, dataset_root=dataset_root)
    make_chunks(proposals=proposals, dataset_root=dataset_root)
    label_dataset(proposals=proposals, dataset_root=dataset_root)
    export_dataset_json(proposals=proposals, dataset_root=dataset_root)
    return proposals


def _generate_metadata(
    seed_meta: Dict[str, Any],
    sector: str,
    agency_type: Optional[str],
    ctx: GenerationContext,
) -> Dict[str, Any]:
    rng = ctx.rng
    faker = rng.faker
    metadata = common.safe_copy_dict(seed_meta) if seed_meta else {}
    agency_vocab = ctx.dictionaries.get("agency_sector", {}).get("sectors", {})
    sector_info = agency_vocab.get(sector, {})
    metadata["sector"] = sector
    project_types = sector_info.get("project_types") or ["Program Client"]
    metadata["client"] = rng.choice(project_types)
    metadata["agency"] = rng.choice(sector_info.get("agencies", [metadata.get("agency", "Regional Agency")]))
    if agency_type:
        metadata["agency_type"] = agency_type
    else:
        metadata["agency_type"] = rng.choice(
            ctx.dictionaries.get("agency_sector", {}).get("agency_types", ["Agency"])
        )
    metadata["prime_firm"] = faker.company() + " " + rng.choice(["Group", "Partners", "Collective", "Consulting"])
    metadata["submission_method"] = rng.choice(ctx.dictionaries.get("misc", {}).get("submission_methods", ["Electronic via Procurement Portal"]))
    metadata["submission_date"] = str(_random_date_within(rng, rng.randint(2024, 2025)))
    rfp_prefix = sector[:3].upper()
    metadata["rfp_number"] = f"{rfp_prefix}-{rng.randint(1000, 9999)}-{rng.randint(10, 99)}"
    metadata["rfp_title"] = metadata.get("rfp_title") or f"{sector} Program Professional Services"
    addenda_range = ctx.config.get("content_variation", {}).get("addenda_range", [0, 2])
    addenda_count = rng.randint(addenda_range[0], addenda_range[1])
    metadata["addenda"] = _generate_addenda(rng, addenda_count)
    return metadata
