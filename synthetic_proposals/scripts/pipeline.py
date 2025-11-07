"""
Core orchestration logic for the synthetic proposal dataset pipeline.
"""

from __future__ import annotations

import itertools
import json
import logging
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from . import common
from .util_text import generate_chunks


logger = logging.getLogger("synthetic_proposals.pipeline")


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


def _build_mistake_plan(total: int, mistakes_config: Dict[str, float], rng: common.RandomSource) -> List[List[str]]:
    """Assign mistakes to each proposal index while hitting minimum target counts."""
    mistake_list = list(mistakes_config.keys())
    targets: Dict[str, int] = {}
    for mistake_key, probability in mistakes_config.items():
        targets[mistake_key] = max(int(total * probability), 20)

    plan: List[List[str]] = []
    for index in range(total):
        assigned: List[str] = []
        proposals_remaining = total - index
        for mistake_key in mistake_list:
            remaining = targets[mistake_key]
            if remaining <= 0:
                continue
            probability = remaining / proposals_remaining
            if rng.random.random() <= probability:
                assigned.append(mistake_key)
                targets[mistake_key] -= 1
        plan.append(assigned)
    # If any target remains due to rounding, assign to final proposals
    for mistake_key, remaining in targets.items():
        while remaining > 0:
            plan[-remaining].append(mistake_key)
            remaining -= 1
    return plan


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


def _apply_base_narrative(proposal: Dict[str, Any], metadata: Dict[str, Any], team: Dict[str, Any], requirement: Dict[str, Any]) -> None:
    pm_name = team.get("pm", {}).get("name", "Project Manager")
    project_name = metadata.get("rfp_title") or metadata.get("project_name") or "the project"
    submission_date = metadata.get("submission_date", "2024-12-31")
    req_id = requirement.get("req_id", "R1")
    letter = proposal.setdefault("letter", {})
    letter["body"] = [
        f"We are pleased to submit our proposal for {project_name}.",
        f"{pm_name} will serve as the primary contact and Project Manager for this engagement.",
        f"Our anticipated submission date remains {submission_date}.",
        f"This proposal was signed and sealed on {submission_date} by {pm_name}.",
        "Our team brings extensive experience in transit design and environmental compliance.",
        f"Requirement {req_id} is detailed in Section 5: Work Approach.",
    ]

    staffing = proposal.setdefault("staffing_plan", {})
    staffing["overview"] = [
        "Our project team consists of highly qualified professionals with proven track records.",
    ]
    staffing["availability"] = [
        f"{pm_name} will remain available 40 hours per week as the dedicated project lead.",
    ]

    work = proposal.setdefault("work_approach", {})
    work["executive_summary"] = (
        f"Requirement {req_id} response is detailed through staged reviews, targeted stakeholder workshops, and integrated risk checks to manage scope."
    )
    work["success_factors"] = [
        "This provides an assurance that our delivery method avoids scope drift.",
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


def _select_style_plan(ctx: GenerationContext, mistakes: Sequence[str]) -> Dict[str, Any]:
    rng = ctx.rng
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


def _build_proposal_payload(
    proposal_id: str,
    split: str,
    sector: str,
    agency_type: str,
    mistakes: Sequence[str],
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
    style_plan = _select_style_plan(ctx, mistakes)
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
        "mistakes": {key: (key in mistakes) for key in ctx.config.get("mistakes", {}).keys()},
        "content": {
            "tone": tone,
            "page_limit_required": page_limit,
        },
        "consistency": {},
        "compliance": {
            "dbe_goal_percent": dbe_goal,
            "dbe_commit_percent": sum(item.get("commitment_percent", 0) for item in appendices.get("B", {}).get("dbe_plan", [])),
            "primary_requirement": primary_requirement,
        },
        "render": {},
        "noise": {},
        "timestamp": datetime.utcnow().isoformat(),
    }
    _apply_base_narrative(proposal, metadata, team, primary_requirement)
    _apply_mistakes(proposal, meta, mistakes, ctx)
    meta["compliance"]["dbe_commit_percent"] = sum(
        item.get("commitment_percent", 0) for item in proposal.get("appendices", {}).get("B", {}).get("dbe_plan", [])
    )
    return proposal, meta


def _apply_mistakes(proposal: Dict[str, Any], meta: Dict[str, Any], mistakes: Sequence[str], ctx: GenerationContext) -> None:
    rng = ctx.rng
    faker = rng.faker
    appendices = proposal.get("appendices", {})
    checklist = proposal.get("checklist", {})
    metadata = proposal.get("metadata", {})
    compliance_matrix = proposal.get("compliance_matrix", [])
    banned_phrases = ctx.dictionaries.get("compliance", {}).get("banned_phrases", [])

    for mistake in mistakes:
        if mistake == "page_limit_violation":
            extra_paragraphs = [faker.paragraph(nb_sentences=1) for _ in range(1)]
            proposal.setdefault("letter", {}).setdefault("body", []).extend(extra_paragraphs)
            meta.setdefault("content", {})["page_limit_violation"] = True

        elif mistake == "missing_appendix_a":
            appendices.setdefault("A", {})["addenda_acknowledgment"] = []
            meta.setdefault("compliance", {})["missing_appendix_a"] = True

        elif mistake == "missing_appendix_b":
            appendices.setdefault("B", {})["dbe_plan"] = []
            meta.setdefault("compliance", {})["missing_appendix_b"] = True

        elif mistake == "missing_appendix_c":
            appendices.setdefault("C", {})["resumes"] = []
            meta.setdefault("compliance", {})["missing_appendix_c"] = True

        elif mistake == "missing_signature_block":
            proposal.setdefault("letter", {})["signature_block"] = None
            meta.setdefault("content", {})["missing_signature_block"] = True

        elif mistake == "addendum_unacknowledged":
            ack_list = appendices.setdefault("A", {}).setdefault("addenda_acknowledgment", [])
            if metadata.get("addenda") and ack_list:
                ack_list[0]["acknowledged"] = False
                meta.setdefault("compliance", {})["addenda_unacknowledged"] = True
            else:
                meta.setdefault("compliance", {})["addenda_unacknowledged"] = False

        elif mistake == "dbe_mismatch":
            dbe_plan = appendices.setdefault("B", {}).setdefault("dbe_plan", [])
            for entry in dbe_plan:
                entry["commitment_percent"] = max(0, entry.get("commitment_percent", 0) - rng.randint(3, 8))
            meta.setdefault("compliance", {})["dbe_gap"] = True

        elif mistake == "insurance_missing":
            forms = appendices.setdefault("D", {}).setdefault(
                "forms", {"non_collusion": True, "insurance_ack": True, "certification": True}
            )
            forms["insurance_ack"] = False
            meta.setdefault("compliance", {})["insurance_missing"] = True

        elif mistake == "pm_name_variant":
            pm_name = proposal.get("team", {}).get("pm", {}).get("name", "Project Manager")
            parts = pm_name.split()
            if len(parts) >= 2 and parts[-1]:
                variant = f"{parts[0]} {parts[-1][0]}."
            else:
                variant = pm_name + " Jr."

            letter = proposal.setdefault("letter", {})
            body = letter.setdefault("body", [])
            for idx, paragraph in enumerate(body):
                if pm_name in paragraph:
                    body[idx] = paragraph.replace(pm_name, variant)

            staffing = proposal.get("staffing_plan", {})
            availability = staffing.setdefault("availability", [])
            for idx, line in enumerate(availability):
                if pm_name in line:
                    availability[idx] = line.replace(pm_name, variant)

            meta.setdefault("consistency", {})["pm_name_variant"] = {"original": pm_name, "variant": variant}

        elif mistake == "date_mismatch":
            letter = proposal.setdefault("letter", {})
            signature_block = letter.get("signature_block")
            if not isinstance(signature_block, dict):
                signature_block = {}
                letter["signature_block"] = signature_block
            orig_date = signature_block.get("date") or metadata.get("submission_date")
            if orig_date:
                try:
                    expected_dt = datetime.fromisoformat(orig_date)
                    new_date = str((expected_dt + timedelta(days=rng.randint(3, 14))).date())
                except ValueError:
                    new_date = orig_date
                signature_block["date"] = new_date
                body = letter.setdefault("body", [])
                for idx, paragraph in enumerate(body):
                    if "signed and sealed" in paragraph.lower():
                        body[idx] = paragraph.replace(orig_date, new_date)
                        break

        elif mistake == "project_number_drift":
            original = metadata.get("rfp_number")
            drifted = original + "A" if original else f"ALT-{rng.randint(100,999)}"
            metadata["rfp_number_alias"] = drifted
            body = proposal.setdefault("letter", {}).setdefault("body", [])
            if body:
                body[0] = body[0] + f" Reference: {drifted}."
            meta.setdefault("consistency", {})["project_number_drift"] = {"expected": original, "cited": drifted}

        elif mistake == "crosswalk_error":
            primary = meta.get("compliance", {}).get("primary_requirement", {})
            req_id = primary.get("req_id", "R1")
            work = proposal.setdefault("work_approach", {})
            sentence = work.get("executive_summary", "")
            alt_req = "R2" if req_id != "R2" else "R3"
            if sentence:
                work["executive_summary"] = sentence.replace(req_id, alt_req)
            meta.setdefault("compliance", {}).setdefault("crosswalk_errors", []).append(
                {
                    "req_id": req_id,
                    "expected_section": "Work Approach",
                    "cited_section": "Work Approach (wrong content)",
                }
            )

        elif mistake == "banned_phrase":
            if banned_phrases:
                phrase = rng.choice(banned_phrases)
                work = proposal.setdefault("work_approach", {})
                success_factors = work.setdefault("success_factors", [])
                if success_factors:
                    success_factors[0] = f"This provides an {phrase} that our delivery method avoids scope drift."
                meta.setdefault("content", {}).setdefault("banned_phrases", []).append(
                    {"phrase": phrase, "section": "work_approach"}
                )

        elif mistake == "font_size_violation":
            meta.setdefault("style", {})["font_size_violation"] = True

        elif mistake == "margin_violation":
            meta.setdefault("style", {})["margin_violation"] = True

    proposal["appendices"] = appendices
    proposal["checklist"] = checklist


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

    mistake_plan = _build_mistake_plan(total, config.get("mistakes", {}), rng)

    proposals: List[ProposalPaths] = []
    for index in range(total):
        proposal_id = f"proposal_{index + 1:04d}"
        split = split_sequence[index]
        sector = sector_sequence[index] if index < len(sector_sequence) else rng.choice(sectors)
        agency_type = agency_sequence[index] if index < len(agency_sequence) else rng.choice(agency_types)
        mistakes = mistake_plan[index]
        proposal, meta = _build_proposal_payload(proposal_id, split, sector, agency_type, mistakes, ctx)
        proposal_dir = root / split / proposal_id
        common.ensure_dir(proposal_dir)
        paths = ProposalPaths.from_dir(split, proposal_dir)
        common.dump_yaml_file(proposal, paths.yaml_path)
        common.dump_json_file(proposal, paths.json_path)
        meta.setdefault("paths", {})
        meta["paths"]["proposal_yaml"] = str(paths.yaml_path)
        meta["paths"]["proposal_json"] = str(paths.json_path)
        common.dump_json_file(meta, paths.meta_path)
        with paths.meta_history_path.open("a", encoding="utf-8") as history_file:
            history_file.write(json.dumps(meta) + "\n")
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
