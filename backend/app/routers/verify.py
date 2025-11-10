"""Verification API endpoints."""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from typing import Optional, Dict, List
from pathlib import Path
import tempfile

from ..config import settings
from ..services.pdf_extractor import extract_text_from_pdf
from ..services.pdf_generator import create_pdf_from_text
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

    # 2. Run all models (fast - return immediately)
    try:
        transformer_results = distilbert_service.predict(proposal_text)
        tfidf_results = tfidf_service.predict(proposal_text)
        nb_results = nb_service.predict(proposal_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")

    # 3. Run rule-based check (fast)
    date_status, date_message = detect_date_inconsistency(proposal_text)
    rule_results = {
        "date_inconsistency": date_status == "FAIL"
    }

    # 4. Identify failed checks for RAG (only from DistilBERT)
    failed_checks = _identify_failed_checks_from_transformer(transformer_results)

    # 5. Return ML and rule results immediately (suggestions will be added async)
    base_response = {
        "ml_results": {
            "transformer": transformer_results,
            "tfidf": tfidf_results,
            "naive_bayes": nb_results,
        },
        "rule_results": rule_results,
        "suggestions": None,  # Will be populated below
    }

    # 6. Generate RAG suggestions (slow - can take time)
    try:
        suggestions = generate_rag_suggestions(proposal_text, failed_checks)
        base_response["suggestions"] = suggestions
    except Exception as e:
        base_response["suggestions"] = f"⚠ Error generating suggestions: {str(e)}"

    return base_response


@router.post("/suggestions")
async def regenerate_suggestions(
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    failed_checks: Optional[str] = Form(None),
):
    """
    Regenerate AI suggestions only (without running ML models).
    
    Accepts either raw text or a PDF file upload, and a comma-separated list of failed checks.
    Returns only the AI-generated suggestions.
    """
    # 1. Extract text from input
    try:
        proposal_text = await _resolve_input_text(text, file)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # 2. Parse failed checks
    if failed_checks:
        failed_checks_list = [check.strip() for check in failed_checks.split(",") if check.strip()]
    else:
        # If no failed checks provided, determine from DistilBERT transformer results only
        transformer_results = distilbert_service.predict(proposal_text)
        failed_checks_list = _identify_failed_checks_from_transformer(transformer_results)
    
    # 3. Generate RAG suggestions
    try:
        suggestions = generate_rag_suggestions(proposal_text, failed_checks_list)
        return {"suggestions": suggestions}
    except Exception as e:
        return {"suggestions": f"⚠ Error generating suggestions: {str(e)}"}


@router.get("/samples")
async def get_samples() -> Dict[str, str]:
    """
    Get all available sample proposal texts.
    Also generates PDFs from text samples if they don't exist.

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

    # Create PDFs directory if it doesn't exist
    pdfs_dir = samples_dir.parent / "pdfs"
    pdfs_dir.mkdir(exist_ok=True)

    for label, filename in sample_files.items():
        sample_path = samples_dir / filename
        if sample_path.exists():
            try:
                text_content = sample_path.read_text(encoding="utf-8")
                samples[label] = text_content
                
                # Generate PDF if it doesn't exist
                pdf_filename = filename.replace(".txt", ".pdf")
                pdf_path = pdfs_dir / pdf_filename
                if not pdf_path.exists():
                    create_pdf_from_text(text_content, pdf_path)
            except Exception as e:
                print(f"⚠ Failed to read {filename}: {e}")

    return samples


@router.get("/samples/pdf/{sample_name}")
async def get_sample_pdf(sample_name: str):
    """
    Get a PDF file for a specific sample.
    
    Args:
        sample_name: Name of the sample (e.g., "Clean proposal")
    
    Returns:
        PDF file response
    """
    samples_dir = settings.samples_dir
    pdfs_dir = samples_dir.parent / "pdfs"
    
    # Map sample names to filenames
    sample_files = {
        "Clean proposal": "sample_clean.pdf",
        "Crosswalk error": "sample_crosswalk.pdf",
        "Banned phrases": "sample_banned.pdf",
        "Name inconsistency": "sample_name.pdf",
        "Date inconsistency (rule)": "sample_date.pdf",
    }
    
    pdf_filename = sample_files.get(sample_name)
    if not pdf_filename:
        raise HTTPException(status_code=404, detail=f"Sample '{sample_name}' not found")
    
    pdf_path = pdfs_dir / pdf_filename
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"PDF for '{sample_name}' not found")
    
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=pdf_filename
    )


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


def _identify_failed_checks_from_transformer(
    transformer_results: Dict[str, bool],
) -> List[str]:
    """
    Identify which checks failed (using only DistilBERT transformer results).

    Returns list of failed check names for RAG system.
    """
    failed = []

    # Check Crosswalk Error (only from DistilBERT)
    if transformer_results.get("Crosswalk Error", False):
        failed.append("Crosswalk Error")

    # Check Banned Phrases (only from DistilBERT)
    if transformer_results.get("Banned Phrases", False):
        failed.append("Banned Phrases")

    # Check Name Inconsistency (only from DistilBERT)
    if transformer_results.get("Name Inconsistency", False):
        failed.append("Name Inconsistency")

    return failed


def _identify_failed_checks(
    transformer_results: Dict[str, bool],
    tfidf_results: Dict[str, bool],
) -> List[str]:
    """
    Identify which checks failed (using only DistilBERT transformer results).

    Returns list of failed check names for RAG system.
    """
    failed = []

    # Check Crosswalk Error (only from DistilBERT)
    if transformer_results.get("Crosswalk Error", False):
        failed.append("Crosswalk Error")

    # Check Banned Phrases (only from DistilBERT)
    if transformer_results.get("Banned Phrases", False):
        failed.append("Banned Phrases")

    # Check Name Inconsistency (only from DistilBERT)
    if transformer_results.get("Name Inconsistency", False):
        failed.append("Name Inconsistency")

    return failed
