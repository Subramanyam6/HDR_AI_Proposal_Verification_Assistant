"""PDF text extraction service."""
import PyPDF2
from pathlib import Path
from typing import Optional


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """
    Extract text from a PDF file.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Extracted text as a string

    Raises:
        Exception: If PDF extraction fails
    """
    try:
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text_parts = []

            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)

            full_text = "\n".join(text_parts)
            return full_text.strip()

    except Exception as e:
        raise Exception(f"Failed to extract text from PDF: {str(e)}")
