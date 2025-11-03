"""HDR Proposal Verification Assistant - Minimal UI."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gradio as gr
import numpy as np
import PyPDF2
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Paths and configuration
BASE_DIR = Path(__file__).resolve().parent
TFIDF_MODEL_DIR = BASE_DIR / "model"
DISTILBERT_MODEL_DIR = TFIDF_MODEL_DIR / "distilbert"
SYNTHETIC_DIR = BASE_DIR.parent / "synthetic_proposals"

MODEL_THRESHOLDS = {"transformer": 0.50, "tfidf": 0.00}

# Sample library (text files only)
SAMPLE_LIBRARY: Dict[str, Path] = {}
if SYNTHETIC_DIR.exists():
    sample_candidates = {
        "Clean proposal": "scratch_clean.txt",
        "Crosswalk error": "scratch_crosswalk.txt",
        "Banned phrases": "scratch_banned.txt",
        "Name inconsistency": "scratch_name.txt",
        "Date inconsistency": "scratch_date.txt",
    }
    for label, filename in sample_candidates.items():
        sample_path = SYNTHETIC_DIR / filename
        if sample_path.exists():
            SAMPLE_LIBRARY[label] = sample_path

HAS_SAMPLE_DROPDOWN = bool(SAMPLE_LIBRARY)

# Load TF-IDF model
with open(TFIDF_MODEL_DIR / "vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open(TFIDF_MODEL_DIR / "classifier.pkl", "rb") as f:
    tfidf_classifier = pickle.load(f)

with open(TFIDF_MODEL_DIR / "config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

# Load DistilBERT model
print("Loading DistilBERT model...")
distilbert_tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_MODEL_DIR)
distilbert_model = AutoModelForSequenceClassification.from_pretrained(DISTILBERT_MODEL_DIR)
device = torch.device("cpu")
distilbert_model = distilbert_model.to(device)
distilbert_model.eval()

LABELS: List[str] = config["labels"]
LABEL_DISPLAY: Dict[str, str] = {
    "crosswalk_error": "Crosswalk Error",
    "banned_phrases": "Banned Phrases",
    "name_inconsistency": "Name Inconsistency",
    "date_inconsistency": "Date Inconsistency",
}

def extract_text_from_pdf(pdf_file: Any) -> str:
    """Extract text from PDF."""
    try:
        if isinstance(pdf_file, str):
            pdf_path = Path(pdf_file)
        elif hasattr(pdf_file, "name"):
            pdf_path = Path(pdf_file.name)
        else:
            pdf_path = Path(pdf_file)

        text_parts: List[str] = []
        with pdf_path.open("rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text_parts.append(page_text)

        extracted = "\n".join(part.strip() for part in text_parts if part).strip()
        if not extracted:
            raise ValueError("PDF contained no extractable text.")
        return extracted
    except Exception as exc:
        raise ValueError(f"Error extracting text: {exc}") from exc


def resolve_inputs(pdf_file: Any, raw_text: str | None) -> str:
    """Get text from PDF or text input."""
    if raw_text and raw_text.strip():
        return raw_text.strip()
    if pdf_file is not None:
        return extract_text_from_pdf(pdf_file)
    raise ValueError("Upload a PDF file or paste text to verify.")


def run_tfidf_model(text: str) -> Dict[str, bool]:
    """Run TF-IDF model and return predictions."""
    features = vectorizer.transform([text])
    predictions = tfidf_classifier.predict(features)

    results = {}
    for idx, label in enumerate(LABELS):
        results[LABEL_DISPLAY[label]] = bool(predictions[0][idx])
    return results


def run_distilbert_model(text: str) -> Dict[str, bool]:
    """Run DistilBERT model and return predictions."""
    inputs = distilbert_tokenizer(
        text,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        logits = distilbert_model(**inputs).logits

    probabilities = torch.sigmoid(logits).cpu().numpy()[0]
    predictions = (probabilities > MODEL_THRESHOLDS["transformer"]).astype(int)

    results = {}
    for idx, label in enumerate(LABELS):
        results[LABEL_DISPLAY[label]] = bool(predictions[idx])
    return results


def generate_results_table_html(
    transformer_results: Dict[str, bool],
    tfidf_results: Dict[str, bool]
) -> str:
    """Generate clean HTML table from verification results."""
    html_rows = []
    for label in LABELS:
        display_label = LABEL_DISPLAY[label]
        transformer_status = "PASS" if not transformer_results[display_label] else "FAIL"
        tfidf_status = "PASS" if not tfidf_results[display_label] else "FAIL"

        transformer_color = "#16a34a" if transformer_status == "PASS" else "#dc2626"
        tfidf_color = "#16a34a" if tfidf_status == "PASS" else "#dc2626"

        html_rows.append(
            f'<tr>'
            f'<td class="check-cell">{display_label}</td>'
            f'<td class="status-cell" style="color: {transformer_color}; font-weight: 600;">{transformer_status}</td>'
            f'<td class="status-cell" style="color: {tfidf_color}; font-weight: 600;">{tfidf_status}</td>'
            f'</tr>'
        )

    html_table = f'''<table class="results-table">
<thead>
<tr>
<th class="check-header">Check</th>
<th class="status-header">Transformer</th>
<th class="status-header">TF-IDF</th>
</tr>
</thead>
<tbody>
{''.join(html_rows)}
</tbody>
</table>'''
    return html_table


def get_empty_table_html() -> str:
    """Get empty results table HTML with headers visible."""
    return '''<table class="results-table">
<thead>
<tr>
<th class="check-header">Check</th>
<th class="status-header">Transformer</th>
<th class="status-header">TF-IDF</th>
</tr>
</thead>
<tbody>
</tbody>
</table>'''


def verify_proposal(pdf_file: Any, raw_text: str | None) -> str:
    """Run verification and return results table as HTML."""
    try:
        text = resolve_inputs(pdf_file, raw_text)
        transformer_results = run_distilbert_model(text)
        tfidf_results = run_tfidf_model(text)

        return generate_results_table_html(transformer_results, tfidf_results)
    except Exception as exc:
        gr.Warning(str(exc))
        return '<p style="color: var(--foreground);">Error generating table.</p>'


def load_sample(sample_key: str | None) -> str:
    """Load sample text when dropdown changes."""
    if not sample_key or sample_key == "Select a sample...":
        return ""

    sample_path = SAMPLE_LIBRARY.get(sample_key)
    if sample_path and sample_path.exists():
        return sample_path.read_text(encoding="utf-8").strip()
    return ""




def clear_all() -> Tuple[Any, ...]:
    """Clear all inputs and outputs."""
    empty_table = get_empty_table_html()
    if HAS_SAMPLE_DROPDOWN:
        return None, "", "Select a sample...", empty_table
    return None, "", empty_table


def handle_pdf_upload(pdf_file: Any | None, current_text: str) -> str:
    """Clear text input when a PDF is uploaded."""
    if pdf_file is not None and current_text.strip():
        return ""
    return current_text


def handle_text_change(new_text: str, current_pdf: Any | None):
    """Clear PDF upload when text is provided."""
    if new_text.strip() and current_pdf is not None:
        return gr.update(value=None)
    return gr.update()


# Load CSS
css_path = BASE_DIR / "assets" / "styles.css"
css_styles = css_path.read_text(encoding="utf-8") if css_path.exists() else ""

# Build UI
with gr.Blocks(css=css_styles, title="HDR Proposal Verification") as demo:

    with gr.Row(elem_classes="hero-section"):
        with gr.Column():
            gr.Markdown("# HDR Proposal Verification Assistant", elem_classes="hero-title")
            gr.Markdown("Upload a proposal PDF or paste text to verify compliance.", elem_classes="hero-subtitle")

    with gr.Row():
        with gr.Column(scale=1, elem_classes="input-panel"):
            gr.Markdown("### Input")

            sample_dropdown: gr.Dropdown | None = None

            pdf_input = gr.File(
                label="PDF file",
                show_label=False,
                file_types=[".pdf"],
                elem_id="pdf-upload",
                file_count="single",
            )

            text_input = gr.Textbox(
                label="Or paste text",
                lines=15,
                placeholder="Paste proposal text here...",
                elem_id="text-input"
            )

            if SAMPLE_LIBRARY:
                sample_dropdown = gr.Dropdown(
                    label="Load sample",
                    choices=["Select a sample..."] + list(SAMPLE_LIBRARY.keys()),
                    value="Select a sample...",
                    elem_id="sample-dropdown"
                )

            run_button = gr.Button(
                "Run Verification",
                variant="primary",
                size="lg",
                elem_id="run-button"
            )

        # Right panel - Results
        with gr.Column(scale=1, elem_classes="results-panel"):
            gr.Markdown("### Results")

            results_table = gr.HTML(
                value=get_empty_table_html(),
                elem_id="results-table",
            )

            clear_button = gr.Button(
                "Clear",
                size="sm",
                elem_id="clear-button"
            )

    # Event handlers
    run_button.click(
        fn=verify_proposal,
        inputs=[pdf_input, text_input],
        outputs=results_table,
    )

    pdf_input.upload(
        fn=handle_pdf_upload,
        inputs=[pdf_input, text_input],
        outputs=text_input,
    )

    text_input.change(
        fn=handle_text_change,
        inputs=[text_input, pdf_input],
        outputs=pdf_input,
    )

    if SAMPLE_LIBRARY:
        sample_dropdown.change(
            fn=load_sample,
            inputs=sample_dropdown,
            outputs=text_input,
        )

    clear_outputs = [pdf_input, text_input]
    if HAS_SAMPLE_DROPDOWN and sample_dropdown is not None:
        clear_outputs.append(sample_dropdown)
    clear_outputs.append(results_table)

    clear_button.click(
        fn=clear_all,
        outputs=clear_outputs,
    )

if __name__ == "__main__":
    demo.launch(show_api=False, show_error=True)
