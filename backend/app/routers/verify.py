"""Verification API endpoints."""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional, Dict, List
from pathlib import Path
import tempfile

from ..config import settings
from ..services.pdf_extractor import extract_text_from_pdf
from ..services.tfidf_model import tfidf_service
from ..services.distilbert_model import distilbert_service
from ..services.nb_model import nb_service
from ..services.rule_engine import detect_date_inconsistency
from ..services.rag_service import generate_rag_suggestions


router = APIRouter()


@router.post("/verify")
async def verify_proposal(
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
):
    """
    Verify a proposal for compliance issues.

    Accepts either raw text or a PDF file upload.
    Runs all ML models + rule-based checks in parallel.
    Returns results and AI-generated suggestions.
    """
    # 1. Extract text from input
    try:
        proposal_text = await _resolve_input_text(text, file)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 2. Run all models
    try:
        transformer_results = distilbert_service.predict(proposal_text)
        tfidf_results = tfidf_service.predict(proposal_text)
        nb_results = nb_service.predict(proposal_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")

    # 3. Run rule-based check
    date_status, date_message = detect_date_inconsistency(proposal_text)
    rule_results = {
        "date_inconsistency": date_status == "FAIL"
    }

    # 4. Identify failed checks for RAG
    failed_checks = _identify_failed_checks(transformer_results, tfidf_results)

    # 5. Generate RAG suggestions
    try:
        suggestions = generate_rag_suggestions(proposal_text, failed_checks)
    except Exception as e:
        suggestions = f"⚠ Error generating suggestions: {str(e)}"

    # 6. Return combined results
    return {
        "ml_results": {
            "transformer": transformer_results,
            "tfidf": tfidf_results,
            "naive_bayes": nb_results,
        },
        "rule_results": rule_results,
        "suggestions": suggestions,
    }


@router.get("/samples")
async def get_samples() -> Dict[str, str]:
    """
    Get all available sample proposal texts.

    Returns:
        Dictionary mapping sample names to full text content
    """
    samples = {}
    samples_dir = settings.samples_dir

    if not samples_dir.exists():
        return samples

    # Load all sample text files
    sample_files = {
        "Clean proposal": "sample_clean.txt",
        "Crosswalk error": "sample_crosswalk.txt",
        "Banned phrases": "sample_banned.txt",
        "Name inconsistency": "sample_name.txt",
        "Date inconsistency (rule)": "sample_date.txt",
    }

    for label, filename in sample_files.items():
        sample_path = samples_dir / filename
        if sample_path.exists():
            try:
                samples[label] = sample_path.read_text(encoding="utf-8")
            except Exception as e:
                print(f"⚠ Failed to read {filename}: {e}")

    return samples


# Helper functions

async def _resolve_input_text(text: Optional[str], file: Optional[UploadFile]) -> str:
    """Extract text from either text input or PDF upload."""
    if text and text.strip():
        return text.strip()

    if file is not None:
        # Save uploaded file temporarily
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                contents = await file.read()
                tmp_file.write(contents)
                tmp_path = tmp_file.name

            # Extract text
            extracted_text = extract_text_from_pdf(tmp_path)

            # Clean up temp file
            Path(tmp_path).unlink()

            if not extracted_text:
                raise ValueError("PDF contained no extractable text")

            return extracted_text

        except Exception as e:
            raise ValueError(f"Failed to process PDF: {str(e)}")

    raise ValueError("Please provide either text or a PDF file")


def _identify_failed_checks(
    transformer_results: Dict[str, bool],
    tfidf_results: Dict[str, bool],
) -> List[str]:
    """
    Identify which checks failed (using OR logic across models).

    Returns list of failed check names for RAG system.
    """
    failed = []

    # Check Crosswalk Error
    if transformer_results.get("Crosswalk Error", False) or tfidf_results.get("Crosswalk Error", False):
        failed.append("Crosswalk Error")

    # Check Banned Phrases
    if transformer_results.get("Banned Phrases", False) or tfidf_results.get("Banned Phrases", False):
        failed.append("Banned Phrases")

    # Check Name Inconsistency
    if transformer_results.get("Name Inconsistency", False) or tfidf_results.get("Name Inconsistency", False):
        failed.append("Name Inconsistency")

    return failed
