"""HDR Proposal Verification Assistant - Minimal UI."""

from __future__ import annotations

import importlib.util
import json
import os
import pickle
import re
import socket
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple
import html
try:
    import markdown  # type: ignore
except Exception:  # pragma: no cover
    markdown = None

import gradio as gr
import numpy as np
import PyPDF2
import torch
from dotenv import load_dotenv
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Workaround for Gradio bug: handle boolean additionalProperties in JSON schema
try:
    from gradio_client import utils as client_utils
    
    original_get_type = client_utils.get_type
    
    def patched_get_type(schema):
        """Patched version that handles boolean schema values."""
        # Gradio's get_type() tries to check "const" in schema, but schema might be a bool
        if isinstance(schema, bool):
            return "Any"  # Return a safe default type
        return original_get_type(schema)
    
    client_utils.get_type = patched_get_type
    
    original_json_schema_to_python_type = client_utils._json_schema_to_python_type
    
    def patched_json_schema_to_python_type(schema, defs=None):
        """Patched version that handles boolean additionalProperties."""
        # Handle the case where schema itself is a boolean (shouldn't happen, but be safe)
        if isinstance(schema, bool):
            return "Any"
        # Handle boolean additionalProperties
        if isinstance(schema, dict) and "additionalProperties" in schema:
            additional_props = schema["additionalProperties"]
            if isinstance(additional_props, bool):
                # If additionalProperties is boolean, treat it as allowing any type
                schema = schema.copy()
                schema["additionalProperties"] = {}
        return original_json_schema_to_python_type(schema, defs)
    
    client_utils._json_schema_to_python_type = patched_json_schema_to_python_type
except (ImportError, AttributeError):
    # If patching fails, continue anyway - might work without it
    pass

# Load environment variables from .env file
load_dotenv()

# Paths and configuration
BASE_DIR = Path(__file__).resolve().parent
TFIDF_MODEL_DIR = BASE_DIR / "model"
DISTILBERT_MODEL_DIR = TFIDF_MODEL_DIR / "distilbert"
NB_MODEL_DIR = TFIDF_MODEL_DIR / "nb_baseline"
SYNTHETIC_DIR = BASE_DIR

MODEL_THRESHOLDS = {"transformer": 0.50, "tfidf": 0.00}
DATE_RULE_LABEL = "Date Inconsistency (Rule-based)"
RULE_PLACEHOLDER = "—"
RULE_METHOD_LABEL = "Python & Regex"
NB_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
DATE_REGEXES: Sequence[re.Pattern[str]] = (
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b"),
    re.compile(r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
               r"sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?,\s+\d{4}\b",
               re.IGNORECASE),
    re.compile(r"\b\d{1,2}\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
               r"sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{4}\b",
               re.IGNORECASE),
)
DATE_FORMATS: Sequence[str] = (
    "%Y-%m-%d",
    "%m/%d/%Y",
    "%d/%m/%Y",
    "%B %d, %Y",
    "%b %d, %Y",
    "%d %B %Y",
    "%d %b %Y",
)

# Sample library (text & PDF)
SAMPLE_LIBRARY: Dict[str, Path] = {}
PDF_SAMPLE_LIBRARY: Dict[str, Path] = {}
if SYNTHETIC_DIR.exists():
    sample_candidates = {
        "Clean proposal": "sample_clean.txt",
        "Crosswalk error": "sample_crosswalk.txt",
        "Banned phrases": "sample_banned.txt",
        "Name inconsistency": "sample_name.txt",
        "Date inconsistency (rule)": "sample_date.txt",
    }
    for label, filename in sample_candidates.items():
        sample_path = SYNTHETIC_DIR / filename
        if sample_path.exists():
            SAMPLE_LIBRARY[label] = sample_path
            pdf_path = sample_path.with_suffix(".pdf")
            if pdf_path.exists():
                PDF_SAMPLE_LIBRARY[label] = pdf_path

HAS_SAMPLE_DROPDOWN = bool(SAMPLE_LIBRARY)
HAS_PDF_SAMPLE_DROPDOWN = bool(PDF_SAMPLE_LIBRARY)

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
MODEL_LABEL_DISPLAY: Dict[str, str] = {
    "crosswalk_error": "Crosswalk Error",
    "banned_phrases": "Banned Phrases",
    "name_inconsistency": "Name Inconsistency",
}
NB_CLASS = None
NB_MODELS: Dict[str, Any] = {}


def _load_nb_class() -> Optional[Any]:
    script_path = BASE_DIR.parent / "synthetic_proposals" / "scripts" / "07_baseline_ml.py"
    if not script_path.exists():
        return None
    spec = importlib.util.spec_from_file_location("nb_runtime_module", script_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules["nb_runtime_module"] = module
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return getattr(module, "NB", None)


class _NBUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "NB" and NB_CLASS is not None:
            return NB_CLASS
        return super().find_class(module, name)


def _load_nb_models() -> Dict[str, Any]:
    models: Dict[str, Any] = {}
    metadata_path = NB_MODEL_DIR / "metadata.json"
    if not metadata_path.exists():
        return models
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for label in metadata.get("targets", []):
        model_path = NB_MODEL_DIR / f"{label}_nb.pkl"
        if model_path.exists():
            with model_path.open("rb") as f:
                models[label] = _NBUnpickler(f).load()
    return models


if NB_MODEL_DIR.exists():
    NB_CLASS = _load_nb_class()
    if NB_CLASS is not None:
        NB_MODELS = _load_nb_models()


def _normalize_ordinal_suffix(token: str) -> str:
    """Remove ordinal suffixes (1st -> 1) for parsing."""
    return re.sub(r"(\d)(st|nd|rd|th)", r"\1", token, flags=re.IGNORECASE)


def _parse_date_token(token: str) -> Optional[datetime]:
    """Attempt to parse a date string using common formats."""
    cleaned = _normalize_ordinal_suffix(token.strip())
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
    return None


def _extract_date(sentence: Optional[str]) -> Optional[datetime]:
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
    """Rule-based comparison of anticipated vs signed dates."""
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

def extract_text_from_pdf(pdf_file: Optional[str]) -> str:
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


def resolve_inputs(pdf_file: Optional[str], raw_text: Optional[str]) -> str:
    """Get text from PDF or text input."""
    if raw_text and raw_text.strip():
        return raw_text.strip()
    if pdf_file is not None:
        return extract_text_from_pdf(pdf_file)
    raise ValueError("Upload a PDF file or paste text to verify.")


def run_tfidf_model(text: str) -> Dict[str, bool]:
    """Run TF-IDF model and return predictions."""
    features = vectorizer.transform([text])
    probabilities = tfidf_classifier.predict_proba(features)

    results = {}
    for idx, label in enumerate(LABELS):
        display_label = MODEL_LABEL_DISPLAY.get(label, label)
        results[display_label] = bool(probabilities[0][idx] >= 0.5)
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
        display_label = MODEL_LABEL_DISPLAY.get(label, label)
        results[display_label] = bool(predictions[idx])
    return results


def _tokenize_nb(text: str) -> List[str]:
    tokens = [token.lower() for token in NB_TOKEN_RE.findall(text)]
    normalized: List[str] = []
    pm_tokens = {"sarah", "martinez", "lee", "davis", "jennifer", "thompson", "kevin", "vazquez", "lisa", "dana"}
    for tok in tokens:
        if tok.startswith("requirement"):
            normalized.append("requirement_token")
        elif tok in pm_tokens:
            normalized.append("pm_name_token")
        else:
            normalized.append(tok)
    return normalized


def run_nb_models(text: str) -> Dict[str, bool]:
    """Run saved Naive Bayes classifiers (if available)."""
    if not NB_MODELS:
        return {MODEL_LABEL_DISPLAY.get(label, label): False for label in LABELS}
    tokens = _tokenize_nb(text)
    results: Dict[str, bool] = {}
    for label in LABELS:
        display_label = MODEL_LABEL_DISPLAY.get(label, label)
        model = NB_MODELS.get(label)
        if model is None:
            results[display_label] = False
        else:
            results[display_label] = bool(model.predict_one(tokens))
    return results


# ============================================================================
# RAG SUGGESTION SYSTEM (GPT-4o Integration)
# ============================================================================

def _call_openai_chat_completion(
    api_key: str,
    messages: Sequence[Dict[str, str]],
    model: str = "gpt-4o",
    temperature: float = 0.3,
    max_tokens: int = 1000,
) -> str:
    """Call OpenAI chat completion supporting both >=1.0 and legacy clients."""
    try:
        from openai import OpenAI  # type: ignore import-not-found

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=list(messages),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        if isinstance(content, list):
            return "".join(
                getattr(part, "text", str(part)) for part in content if part is not None
            )
        return content or ""
    except (ModuleNotFoundError, AttributeError):
        # Fall back to legacy client below
        pass

    # Legacy fallback (openai<1.0)
    import openai  # type: ignore import-not-found

    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model=model,
        messages=list(messages),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    choice = response["choices"][0]
    message = choice.get("message") if isinstance(choice, dict) else None
    if isinstance(message, dict):
        return message.get("content", "") or ""
    return ""


def get_failed_checks_from_distilbert(transformer_results: Dict[str, bool]) -> List[str]:
    """Extract failed check names from DistilBERT results."""
    failed = []
    if transformer_results.get("Crosswalk Error", False):
        failed.append("Crosswalk Error")
    if transformer_results.get("Banned Phrases", False):
        failed.append("Banned Phrases")
    if transformer_results.get("Name Inconsistency", False):
        failed.append("Name Inconsistency")
    return failed


def generate_rag_suggestions(text: str, failed_checks: List[str]) -> str:
    """Generate AI-powered fix suggestions using GPT-4o (or mock response)."""

    # If no failures, return success message
    if not failed_checks:
        return "✓ All compliance checks passed. No fixes needed."

    # Read API key from environment variable (set in .env file)
    api_key = os.getenv("OPENAI_API_KEY", "sk-dummy-key-replace-me")

    if api_key.startswith("sk-dummy"):
        # Return dummy suggestions
        mock_output = ""
        for check in failed_checks:
            if check == "Crosswalk Error":
                mock_output += "**✗ Crosswalk Error**\n"
                mock_output += "- **Issue Found:** Requirement citations may be incorrect\n"
                mock_output += "- **Recommended Fix:** Verify that R1-R5 citations match the correct RFP sections\n\n"
            elif check == "Banned Phrases":
                mock_output += "**✗ Banned Phrases**\n"
                mock_output += "- **Issue Found:** Prohibited language detected\n"
                mock_output += "- **Recommended Fix:** Replace with compliant phrasing (e.g., 'We will make every effort to...')\n\n"
            elif check == "Name Inconsistency":
                mock_output += "**✗ Name Inconsistency**\n"
                mock_output += "- **Issue Found:** PM name appears in different formats\n"
                mock_output += "- **Recommended Fix:** Use consistent full name throughout the proposal\n\n"

        mock_output += "*Note: Using mock suggestions. Configure GPT-4o API key for detailed analysis.*"
        return mock_output.strip()

    # Real GPT-4o implementation (when API key configured)
    try:
        system_prompt = """You are an HDR proposal compliance expert. Analyze proposals for these issues:

1. Crosswalk Errors: Wrong requirement IDs (R1-R5) cited in wrong sections
2. Banned Phrases: "unconditional guarantee", "guaranteed savings", "risk-free delivery", "absolute assurance"
3. Name Inconsistency: PM name in different formats (e.g., "John Smith" vs "John S.")

For each failed check, provide:
- Specific location where issue appears (quote the text)
- Clear explanation of the problem
- Actionable fix recommendation

Format output with Markdown headers and bullet points."""

        user_prompt = f"""This proposal failed these compliance checks: {', '.join(failed_checks)}

Analyze the full proposal text below and provide specific fix suggestions:

{text}"""

        response_text = _call_openai_chat_completion(
            api_key=api_key,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        return response_text.strip() or "No suggestions returned by the language model."

    except Exception as e:
        return f"⚠ Error generating suggestions: {str(e)}. Please review manually."


def generate_results_table_html(
    transformer_results: Dict[str, bool],
    tfidf_results: Dict[str, bool],
    nb_results: Dict[str, bool],
) -> str:
    """Generate clean HTML table from ML verification results."""

    def format_status(flag: bool) -> str:
        status_text = "FAIL" if flag else "PASS"
        color = "#dc2626" if flag else "#16a34a"
        return f'<span style="color:{color};font-weight:600;">{status_text}</span>'

    rows = [
        [
            display_label,
            format_status(transformer_results.get(display_label, False)),
            format_status(tfidf_results.get(display_label, False)),
            format_status(nb_results.get(display_label, False)),
        ]
        for display_label in (MODEL_LABEL_DISPLAY.get(label, label) for label in LABELS)
    ]
    return _render_results_table(
        "Machine Learning-Based Checks",
        rows,
        ["Check", "Transformer", "TF-IDF", "Naive Bayes"],
    )


def generate_rule_table_html(rule_result: Tuple[str, str]) -> str:
    """Render the rule-based date check table."""
    rule_status, _ = rule_result

    status_text = rule_status if rule_status in {"PASS", "FAIL"} else RULE_PLACEHOLDER
    status_color = "#16a34a" if status_text == "PASS" else "#dc2626"
    if status_text == RULE_PLACEHOLDER:
        status_color = "#6b7280"
    status_cell = (
        f'<span style="color:{status_color};font-weight:600;">'
        f'{status_text}</span>'
    )
    html_row = (
        f'<tr>'
        f'<td class="check-cell">{DATE_RULE_LABEL}</td>'
        f'<td class="status-cell" style="font-weight:600;">{status_cell}</td>'
        f'</tr>'
    )

    return _render_results_table(
        "Rule-Based Checks",
        [[DATE_RULE_LABEL, status_cell]],
        ["Check", RULE_METHOD_LABEL],
    )


def get_empty_table_html() -> str:
    """Render placeholder ML table with em-dash cells."""
    placeholder = f'<span style="opacity:0.6;">{RULE_PLACEHOLDER}</span>'
    rows = [
        [
            MODEL_LABEL_DISPLAY.get(label, label),
            placeholder,
            placeholder,
            placeholder,
        ]
        for label in LABELS
    ]
    return _render_results_table(
        "Machine Learning-Based Checks",
        rows,
        ["Check", "Transformer", "TF-IDF", "Naive Bayes"],
    )


def get_empty_rule_table_html() -> str:
    """Placeholder table for rule-based check."""
    placeholder = f'<span style="opacity:0.6;">{RULE_PLACEHOLDER}</span>'
    return _render_results_table(
        "Rule-Based Checks",
        [[DATE_RULE_LABEL, placeholder]],
        ["Check", RULE_METHOD_LABEL],
    )


def _render_results_table(title: str, rows: List[List[str]], headers: List[str]) -> str:
    header_row = "".join(f"<th>{h}</th>" for h in headers)
    body_rows = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows
    )
    colspan = len(headers)
    return f"""
<table class="results-table">
  <thead>
    <tr class="results-table__section"><th colspan="{colspan}">{title.upper()}</th></tr>
    <tr class="results-table__header-row">{header_row}</tr>
  </thead>
  <tbody>{body_rows}</tbody>
</table>
""".strip()


def render_ai_block(content: str | None) -> str:
    text = (content or "").strip()
    if not text:
        body = '<p class="ai-placeholder">Awaiting analysis...</p>'
    else:
        if markdown is not None:
            body = markdown.markdown(text)
        else:
            body = "<p>" + html.escape(text).replace("\n", "<br>") + "</p>"

    return f"""
<div class="ai-table">
  <div class="ai-table__header">AI-POWERED SUGGESTIONS (GPT-4o + RAG)</div>
  <div class="ai-table__body">{body}</div>
</div>
""".strip()


def verify_proposal(pdf_file: Optional[str], raw_text: Optional[str]) -> Generator[Tuple[str, str, str], None, None]:
    """Run verification and stream RAG suggestions after core results are ready."""
    try:
        text = resolve_inputs(pdf_file, raw_text)
        transformer_results = run_distilbert_model(text)
        tfidf_results = run_tfidf_model(text)
        nb_results = run_nb_models(text)
        rule_result = detect_date_inconsistency(text)

        ml_table = generate_results_table_html(transformer_results, tfidf_results, nb_results)
        rule_table = generate_rule_table_html(rule_result)

        # Immediately show tables while AI suggestions load.
        yield ml_table, rule_table, render_ai_block("_Generating AI suggestions..._")

        failed_checks = get_failed_checks_from_distilbert(transformer_results)
        suggestions = generate_rag_suggestions(text, failed_checks)
        yield ml_table, rule_table, render_ai_block(suggestions)

    except Exception as exc:
        gr.Warning(str(exc))
        error_html = '<p style="color: var(--foreground);">Error generating table.</p>'
        error_msg = f"⚠ Error\n\nCould not generate suggestions: {exc}"
        yield error_html, error_html, error_msg


def load_sample(sample_key: Optional[str]) -> str:
    """Load sample text when dropdown changes."""
    if not sample_key or sample_key == "Select a sample...":
        return ""

    sample_path = SAMPLE_LIBRARY.get(sample_key)
    if sample_path and sample_path.exists():
        return sample_path.read_text(encoding="utf-8").strip()
    return ""


def load_pdf_sample(sample_key: Optional[str]):
    """Load a placeholder PDF into the uploader."""
    if not sample_key or sample_key == "Select a PDF sample...":
        return gr.update(value=None)
    sample_path = PDF_SAMPLE_LIBRARY.get(sample_key)
    if sample_path and sample_path.exists():
        return gr.update(value=str(sample_path))
    gr.Warning("Sample PDF not found.")
    return gr.update(value=None)



def clear_all() -> Tuple[Any, ...]:
    """Clear all inputs and outputs."""
    empty_table = get_empty_table_html()
    empty_rule_table = get_empty_rule_table_html()
    empty_suggestions = render_ai_block("")

    outputs: List[Any] = [gr.update(value=None), ""]
    if HAS_SAMPLE_DROPDOWN:
        outputs.append("Select a text sample...")
    if HAS_PDF_SAMPLE_DROPDOWN:
        outputs.append("Select a PDF sample...")
    outputs.extend([empty_table, empty_rule_table, empty_suggestions])
    return tuple(outputs)


def handle_pdf_upload(pdf_file: Optional[str], current_text: str) -> str:
    """Clear text input when a PDF is uploaded."""
    if pdf_file is not None and current_text.strip():
        return ""
    return current_text


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
            gr.Markdown("### INPUT")

            gr.Markdown("**Upload a Proposal PDF:**", elem_classes="input-label")
            pdf_input = gr.File(
                label="PDF file",
                show_label=False,
                file_types=[".pdf"],
                elem_id="pdf-upload",
                type="filepath",
            )

            pdf_sample_dropdown = gr.Dropdown(
                label="Load sample PDF",
                choices=["Select a PDF sample..."] + (list(PDF_SAMPLE_LIBRARY.keys()) if PDF_SAMPLE_LIBRARY else []),
                value="Select a PDF sample...",
                elem_id="pdf-sample-dropdown",
                visible=bool(PDF_SAMPLE_LIBRARY),
            )

            gr.HTML(
                '<div class="input-separator"><span>OR</span></div>',
                elem_id="input-separator",
            )

            gr.Markdown("**Copy-paste the proposal text:**", elem_classes="input-label")
            text_input = gr.Textbox(
                label="Or paste text",
                lines=15,
                placeholder="Paste proposal text here...",
                elem_id="text-input"
            )

            sample_dropdown = gr.Dropdown(
                label="Load sample text",
                choices=["Select a text sample..."] + (list(SAMPLE_LIBRARY.keys()) if SAMPLE_LIBRARY else []),
                value="Select a text sample...",
                elem_id="sample-dropdown",
                visible=bool(SAMPLE_LIBRARY),
            )

            run_button = gr.Button(
                "Run Verification",
                variant="primary",
                size="lg",
                elem_id="run-button"
            )

        # Right panel - Results
        with gr.Column(scale=1, elem_classes="results-panel"):
            gr.Markdown("### RESULTS")

            results_table = gr.HTML(
                value=get_empty_table_html(),
                elem_id="results-table",
            )

            rule_table = gr.HTML(
                value=get_empty_rule_table_html(),
                elem_id="rule-table",
            )

            suggestions_box = gr.HTML(
                value=render_ai_block(""),
                elem_id="rag-suggestions",
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
        outputs=[results_table, rule_table, suggestions_box],
        show_progress="full",  # Show loading spinner
    )

    pdf_input.upload(
        fn=handle_pdf_upload,
        inputs=[pdf_input, text_input],
        outputs=text_input,
    )

    if HAS_SAMPLE_DROPDOWN:
        sample_dropdown.change(
            fn=load_sample,
            inputs=sample_dropdown,
            outputs=text_input,
        )
    if HAS_PDF_SAMPLE_DROPDOWN:
        pdf_sample_dropdown.change(
            fn=load_pdf_sample,
            inputs=pdf_sample_dropdown,
            outputs=pdf_input,
        )

    clear_outputs: List[Any] = [pdf_input, text_input]
    if HAS_SAMPLE_DROPDOWN:
        clear_outputs.append(sample_dropdown)
    if HAS_PDF_SAMPLE_DROPDOWN:
        clear_outputs.append(pdf_sample_dropdown)
    clear_outputs.extend([results_table, rule_table, suggestions_box])

    clear_button.click(
        fn=clear_all,
        outputs=clear_outputs,
    )


def _port_is_available(port: int) -> bool:
    """Return True if the port can be bound on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _choose_server_port() -> int:
    """Pick a server port honoring env overrides and falling back to any free port."""
    explicit = os.getenv("HDR_SERVER_PORT") or os.getenv("GRADIO_SERVER_PORT")
    if explicit:
        try:
            port = int(explicit)
            if port > 0:
                return port
        except ValueError:
            pass

    for port_range in (range(7860, 7960), range(8000, 8200), range(9000, 9100)):
        for port in port_range:
            if _port_is_available(port):
                return port

    # If no ports were free, raise a clear error so the caller can decide.
    raise OSError("Unable to find an open TCP port for Gradio.")


if __name__ == "__main__":
    # On Hugging Face Spaces, don't override server_port (Gradio handles it)
    # Only use custom port selection for local development
    launch_kwargs = {
        "show_api": False,
        "show_error": True,
    }
    # Only set server_port if not running on HF Spaces
    if not os.getenv("SPACE_ID"):
        launch_kwargs["server_port"] = _choose_server_port()
    
    demo.launch(**launch_kwargs)
