"""HDR Proposal Verification Assistant - Hugging Face Space
Clean, enterprise-grade interface for proposal compliance verification.
"""
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root { color-scheme: dark; }

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

body,
html {
    margin: 0 !important;
    padding: 0 !important;
    width: 100% !important;
}

body,
.gradio-container,
.gradio-container .main {
    background: radial-gradient(circle at top left, #191b22 0%, #0b0c10 45%, #050607 100%) !important;
    color: #f9fafb !important;
}

.gradio-container,
.gradio-container .main,
.main,
.main .container,
.main .wrap,
.gradio-container .wrap,
.padded,
.component-set,
.page,
.component-wrap,
.component-body,
.panel-body {
    max-width: 100% !important;
    width: 100% !important;
    margin-left: 0 !important;
    margin-right: 0 !important;
}

.gradio-container {
    margin: 0 auto !important;
    padding: 2.75rem 0 3.5rem !important;
}

.hero {
    padding: 0 0 1.8rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    margin-bottom: 1.5rem;
}

.hero-title {
    font-size: 2.6rem;
    font-weight: 600;
    letter-spacing: -0.035em;
    color: #ffffff;
    margin-bottom: 0.45rem;
}

.hero-subtext {
    max-width: 620px;
    color: rgba(255, 255, 255, 0.6);
    font-size: 0.98rem;
    line-height: 1.65;
}

.shell {
    max-width: 1220px;
    margin: 0 auto;
    width: 100%;
    padding: 0 1.75rem;
}

.top-nav {
    width: 100%;
    display: flex;
    justify-content: center;
    backdrop-filter: blur(22px);
    background: linear-gradient(90deg, rgba(12, 14, 21, 0.82) 0%, rgba(12, 14, 21, 0.55) 100%);
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    padding: 0.85rem 2.5rem;
    margin-bottom: 0.5rem;
}

.top-nav-inner {
    width: 100%;
    max-width: 100%;
    padding: 0 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    color: rgba(255, 255, 255, 0.72);
    font-size: 0.9rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.top-nav-links {
    display: flex;
    gap: 1.8rem;
    font-size: 0.8rem;
}

.top-nav-links a {
    color: rgba(255, 255, 255, 0.55);
    text-decoration: none;
    letter-spacing: 0.14em;
    transition: color 0.2s ease;
}

.top-nav-links a:hover {
    color: rgba(255, 255, 255, 0.9);
}

.metrics {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 1rem;
    margin-bottom: 1.75rem;
    margin-top: 2rem;
}

.metric-card {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 1rem 1.1rem;
}

.metric-card h5 {
    margin: 0;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    color: rgba(255, 255, 255, 0.46);
}

.metric-card span {
    display: block;
    font-size: 1.15rem;
    font-weight: 600;
    margin-top: 0.4rem;
    color: rgba(255, 255, 255, 0.88);
}

.workspace {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1.12fr);
    gap: 1.75rem;
    margin-top: 2rem;
}

.panel {
    background: rgba(10, 12, 18, 0.92) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 18px !important;
    padding: 1.9rem !important;
    backdrop-filter: blur(14px);
    box-shadow: 0 32px 56px rgba(5, 6, 7, 0.45);
}

.section-title {
    color: #ffffff !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    margin-bottom: 1.3rem !important;
}

.subsection-title {
    color: rgba(255, 255, 255, 0.55) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    margin-top: 1.6rem !important;
    margin-bottom: 0.5rem !important;
}

.upload-box {
    border: 1px dashed rgba(255, 255, 255, 0.15) !important;
    border-radius: 14px !important;
    padding: 2.2rem 1.4rem !important;
    background: rgba(255, 255, 255, 0.04) !important;
    transition: border 0.2s ease, background 0.2s ease;
}

.upload-box:hover {
    border-color: rgba(255, 255, 255, 0.34) !important;
    background: rgba(255, 255, 255, 0.07) !important;
}

/* Accordion styling with custom arrows */
.gr-accordion,
.gr-box.gr-accordion,
details.gr-accordion {
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    background: rgba(255, 255, 255, 0.03) !important;
}

.gr-accordion summary,
.gr-box summary,
summary {
    padding: 1rem !important;
    cursor: pointer !important;
    position: relative !important;
}

.gr-accordion summary::-webkit-details-marker,
.gr-box summary::-webkit-details-marker,
summary::-webkit-details-marker {
    display: none !important;
}

.gr-accordion summary::marker,
.gr-box summary::marker,
summary::marker {
    display: none !important;
}

/* Custom arrow for closed state */
.gr-accordion summary::before,
.gr-box summary::before {
    content: "▶" !important;
    position: absolute !important;
    right: 1rem !important;
    top: 50% !important;
    transform: translateY(-50%) !important;
    color: rgba(255, 255, 255, 0.6) !important;
    font-size: 0.75rem !important;
    display: inline-block !important;
    transition: all 0.2s ease !important;
}

/* Arrow for open state */
.gr-accordion[open] summary::before,
.gr-box[open] summary::before,
.gr-accordion.open summary::before,
details.gr-accordion[open] summary::before {
    content: "▼" !important;
}

.input-area textarea {
    min-height: 220px !important;
    line-height: 1.65 !important;
}

.primary-btn {
    width: 100%;
    margin-top: 1.35rem;
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: #fff !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
    padding: 0.85rem 1.4rem !important;
    letter-spacing: 0.015em;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.primary-btn:hover {
    transform: translateY(-1.5px);
    box-shadow: 0 18px 30px rgba(99, 102, 241, 0.32) !important;
}

textarea,
input,
.gr-textbox textarea {
    background: rgba(12, 14, 20, 0.82) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 12px !important;
    color: #f9fafb !important;
    font-size: 0.95rem !important;
    padding: 0.9rem 1rem !important;
}

.status-box {
    background: rgba(99, 102, 241, 0.12) !important;
    border: 1px solid rgba(99, 102, 241, 0.28) !important;
    border-radius: 14px !important;
    padding: 1rem 1.2rem !important;
    font-size: 0.98rem !important;
    line-height: 1.65 !important;
    white-space: pre-line;
}

.info-card {
    background: rgba(15, 17, 23, 0.78) !important;
    border: 1px solid rgba(255, 255, 255, 0.09) !important;
    border-radius: 14px !important;
    padding: 1rem 1.2rem !important;
    font-size: 0.9rem !important;
    color: rgba(255, 255, 255, 0.72) !important;
    margin-bottom: 1.2rem;
}

.info-card .meta-label {
    color: rgba(255, 255, 255, 0.48);
    text-transform: uppercase;
    letter-spacing: 0.18em;
    font-size: 0.75rem;
}

.preview-box textarea {
    min-height: 260px !important;
    font-size: 0.93rem !important;
    line-height: 1.65 !important;
    color: rgba(255, 255, 255, 0.84) !important;
}

.output-box {
    background: rgba(255, 255, 255, 0.04) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 14px !important;
    padding: 1.25rem !important;
    font-size: 0.9rem !important;
}

.result-card {
    min-height: 0;
    padding: 1.1rem !important;
    box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.02);
    display: flex;
    flex-direction: column;
}

.result-card .results-card {
    width: 100%;
    overflow-x: auto;
    padding-bottom: 0.25rem;
}

.results-card h4 {
    margin-bottom: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: rgba(255, 255, 255, 0.7);
    font-size: 0.85rem;
}

.results-table {
    width: 100%;
    border-collapse: collapse;
    table-layout: fixed;
}

.results-table th,
.results-table td {
    padding: 0.5rem 0.35rem;
    text-align: left;
    word-break: break-word;
}

.results-table thead tr {
    border-bottom: 1px solid rgba(255, 255, 255, 0.12);
    color: rgba(255, 255, 255, 0.45);
    font-size: 0.75rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
}

.cell-label {
    width: 38%;
    color: rgba(255, 255, 255, 0.85);
    font-weight: 500;
    letter-spacing: 0.02em;
}

.cell-status {
    width: 32%;
}

.cell-confidence {
    width: 18%;
    color: rgba(255, 255, 255, 0.55);
    letter-spacing: 0.04em;
}

.confidence-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    margin-left: 0.6rem;
    padding: 0.18rem 0.5rem;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.08);
    color: rgba(255, 255, 255, 0.6);
    font-size: 0.75rem;
    letter-spacing: 0.08em;
}

.badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.25rem 0.6rem;
    border-radius: 999px;
    font-weight: 500;
    letter-spacing: 0.02em;
    font-size: 0.82rem;
}

.badge-ok {
    background: rgba(34, 197, 94, 0.18);
    color: #4ade80;
    border: 1px solid rgba(74, 222, 128, 0.36);
}

.badge-issue {
    background: rgba(239, 68, 68, 0.18);
    color: #f87171;
    border: 1px solid rgba(248, 113, 113, 0.36);
}

@media (max-width: 1080px) {
    .workspace {
        grid-template-columns: minmax(0, 1fr);
    }
}

@media (max-width: 900px) {
    .metrics {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}

@media (max-width: 640px) {
    .hero {
        padding: 1.6rem 0 1.2rem;
    }
    .workspace {
        gap: 1.1rem;
    }
    .metrics {
        grid-template-columns: minmax(0, 1fr);
    }
    .top-nav-links {
        display: none;
    }
    .top-nav-inner {
        justify-content: center;
    }
}

.footer-card {
    text-align: center;
    color: rgba(255, 255, 255, 0.38);
    padding: 1rem 1rem;
    border-top: 1px solid rgba(255, 255, 255, 0.05);
    margin-top: 2.4rem;
}

.gradio-container footer,
.gradio-container .built-with,
.gradio-container .share-btn,
.gradio-container .preview-wrapper,
.gradio-container .copy-api,
.gradio-container .icon-button,
.gradio-container .floating {
    display: none !important;
}

.status-box-text {
    font-size: 0.95rem;
    line-height: 1.6;
    color: rgba(255, 255, 255, 0.78);
}

.model-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.2rem 0.65rem;
    border-radius: 999px;
    font-size: 0.78rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.model-chip-ok {
    background: rgba(34, 197, 94, 0.18);
    color: #4ade80;
    border: 1px solid rgba(74, 222, 128, 0.32);
}

.model-chip-alert {
    background: rgba(239, 68, 68, 0.18);
    color: #f87171;
    border: 1px solid rgba(248, 113, 113, 0.32);
}

.gradio-container footer,
.gradio-container .built-with,
.gradio-container .share-btn,
.gradio-container .preview-wrapper,
.gradio-container .copy-api,
.gradio-container .icon-button,
.gradio-container .floating {
    display: none !important;
}

"""
import gradio as gr
import pickle
import json
from pathlib import Path
import PyPDF2
from typing import Dict
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load TF-IDF model artifacts
TFIDF_MODEL_DIR = Path(__file__).parent / "model"

with open(TFIDF_MODEL_DIR / "vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open(TFIDF_MODEL_DIR / "classifier.pkl", "rb") as f:
    tfidf_classifier = pickle.load(f)

with open(TFIDF_MODEL_DIR / "config.json", "r") as f:
    config = json.load(f)

# Load DistilBERT model
DISTILBERT_MODEL_DIR = Path(__file__).parent / "model" / "distilbert"

print("Loading DistilBERT model...")
distilbert_tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_MODEL_DIR)
distilbert_model = AutoModelForSequenceClassification.from_pretrained(DISTILBERT_MODEL_DIR)

# Set device (CPU for HF Spaces free tier)
device = torch.device("cpu")
distilbert_model = distilbert_model.to(device)
distilbert_model.eval()

LABELS = config["labels"]
LABEL_DISPLAY = {
    "crosswalk_error": "Crosswalk Error",
    "banned_phrases": "Banned Phrases Found",
    "name_inconsistency": "Name Inconsistency",
    "date_inconsistency": "Date Inconsistency"
}

DIAGNOSTICS_HEADER_MD = "#### Model diagnostics"
SCAN_SUMMARY_HEADER_MD = "#### Scan summary"


def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from uploaded PDF file."""
    try:
        # Gradio passes file path as string
        if isinstance(pdf_file, str):
            pdf_path = pdf_file
        elif hasattr(pdf_file, 'name'):
            pdf_path = pdf_file.name
        else:
            pdf_path = pdf_file
        
        # Read PDF from file path
        with open(pdf_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            text = []
            
            for page in pdf_reader.pages:
                text.append(page.extract_text())
        
        return "\n".join(text)
    except Exception as e:
        return f"Error extracting text: {str(e)}"


def run_tfidf_model(text: str) -> Dict[str, Dict[str, object]]:
    """Run TF-IDF + LogReg model."""
    X = vectorizer.transform([text])
    predictions = tfidf_classifier.predict(X)
    decision_scores = tfidf_classifier.decision_function(X)
    
    results = {}
    for idx, label in enumerate(LABELS):
        predicted = bool(predictions[0][idx])
        score = decision_scores[0][idx]
        confidence = 100 / (1 + np.exp(-score))
        label_display = LABEL_DISPLAY[label]
        results[label_display] = {
            "is_issue": predicted,
            "status": "Issue Detected" if predicted else "OK",
            "confidence": confidence,
        }
    
    return results


def run_distilbert_model(text: str) -> Dict[str, Dict[str, object]]:
    """Run DistilBERT transformer model."""
    # Tokenize
    inputs = distilbert_tokenizer(
        text,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Predict
    with torch.no_grad():
        outputs = distilbert_model(**inputs)
        logits = outputs.logits
    
    # Apply sigmoid for multi-label classification
    probs = torch.sigmoid(logits).cpu().numpy()[0]
    predictions = (probs > 0.5).astype(int)
    
    results: Dict[str, Dict[str, object]] = {}
    for idx, label in enumerate(LABELS):
        predicted = bool(predictions[idx])
        confidence = float(probs[idx]) * 100
        label_display = LABEL_DISPLAY[label]
        results[label_display] = {
            "is_issue": predicted,
            "status": "Issue Detected" if predicted else "OK",
            "confidence": confidence,
        }
    
    return results


def build_single_model_html(title: str, results: Dict[str, Dict[str, object]]) -> str:
    rows = []
    for label, info in results.items():
        badge_class = "badge badge-issue" if info["is_issue"] else "badge badge-ok"
        icon = "❌" if info["is_issue"] else "✅"
        rows.append(
            f"<tr>"
            f"<td class='cell-label'>{label}</td>"
            f"<td class='cell-status'><span class='{badge_class}'>{icon} {info['status']}</span></td>"
            f"<td class='cell-confidence'>{info['confidence']:.1f}%</td>"
            f"</tr>"
        )
    return (
        f"<div class='results-card'>"
        f"<h4>{title}</h4>"
        f"<table class='results-table'>"
        f"<thead><tr><th>Check</th><th>Status</th><th>Confidence</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        f"</table>"
        f"</div>"
    )


def verify_proposal(pdf_file, text_input):
    """Run verification and return Gradio update payloads for UI components."""
    # Determine input source
    if text_input and text_input.strip():
        text = text_input.strip()
        input_source = "text input"
    elif pdf_file is not None:
        text = extract_text_from_pdf(pdf_file)
        input_source = "PDF file"
        if text.startswith("Error"):
            error_html = f"<div class='status-box-text model-chip-alert'>❌ {text}</div>"
            return (
                gr.update(value=DIAGNOSTICS_HEADER_MD, visible=False),
                gr.update(value="", visible=False),
                gr.update(value="", visible=False),
                gr.update(value=SCAN_SUMMARY_HEADER_MD, visible=True),
                gr.update(value=error_html, visible=True),
                gr.update(value="", visible=False),
            )
    else:
        error_html = "<div class='status-box-text model-chip-alert'>⚠️ Please upload a PDF file or enter text</div>"
        return (
            gr.update(value=DIAGNOSTICS_HEADER_MD, visible=False),
            gr.update(value="", visible=False),
            gr.update(value="", visible=False),
            gr.update(value=SCAN_SUMMARY_HEADER_MD, visible=True),
            gr.update(value=error_html, visible=True),
            gr.update(value="", visible=False),
        )
    
    # Prepare snapshot for UI
    word_count = len(text.split())
    char_count = len(text)
    document_info = (
        f"<div><span class='meta-label'>Source:</span> {input_source.title()} &middot; "
        f"<span class='meta-label'>Words:</span> {word_count:,} &middot; "
        f"<span class='meta-label'>Characters:</span> {char_count:,}</div>"
    )
    
    # Run both models
    tfidf_results = run_tfidf_model(text)
    distilbert_results = run_distilbert_model(text)
    
    distilbert_issues = [label for label, info in distilbert_results.items() if info["is_issue"]]
    tfidf_issues = [label for label, info in tfidf_results.items() if info["is_issue"]]
    
    # Status message (using DistilBERT as primary)
    if distilbert_issues:
        status_distilbert = (
            f"<span class='model-chip model-chip-alert'>DistilBERT</span> "
            f"Flagged {len(distilbert_issues)} issue(s): {', '.join(distilbert_issues)}"
        )
    else:
        status_distilbert = (
            "<span class='model-chip model-chip-ok'>DistilBERT</span> All checks passed"
        )

    if tfidf_issues:
        status_tfidf = (
            f"<span class='model-chip model-chip-alert'>TF-IDF</span> "
            f"Flagged {len(tfidf_issues)} issue(s): {', '.join(tfidf_issues)}"
        )
    else:
        status_tfidf = (
            "<span class='model-chip model-chip-ok'>TF-IDF</span> No additional issues detected"
        )

    status_html = (
        f"<div class='status-box-text'>{status_distilbert}<br>{status_tfidf}<br><span class='meta-label'>Source:</span> {input_source.title()}</div>"
    )

    distilbert_html = build_single_model_html("DistilBERT (Transformer)", distilbert_results)
    tfidf_html = build_single_model_html("TF-IDF + Logistic Regression", tfidf_results)
    return (
        gr.update(value=DIAGNOSTICS_HEADER_MD, visible=True),
        gr.update(value=distilbert_html, visible=True),
        gr.update(value=tfidf_html, visible=True),
        gr.update(value=SCAN_SUMMARY_HEADER_MD, visible=True),
        gr.update(value=status_html, visible=True),
        gr.update(value=document_info, visible=True),
    )


# Enhanced status handling function
def enhanced_status_update(pdf_file, text_input):
    return verify_proposal(pdf_file, text_input)



# Build Gradio interface
with gr.Blocks(css=custom_css, title="HDR Proposal Verification") as demo:
    # Cache-busting meta tags
    gr.HTML("""
        <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
        <meta http-equiv="Pragma" content="no-cache">
        <meta http-equiv="Expires" content="0">
        <script>
            // Force reload on every page load
            window.addEventListener('load', function() {
                var links = document.querySelectorAll('link[rel="stylesheet"]');
                links.forEach(function(link) {
                    var href = link.href;
                    var separator = href.indexOf('?') !== -1 ? '&' : '?';
                    link.href = href + separator + 'v=' + new Date().getTime();
                });
            });
        </script>
    """)
    
    with gr.Column(elem_classes=["shell"]):
        gr.HTML(
            """
            <section class="hero">
                <h1 class="hero-title">HDR Proposal Verification Assistant</h1>
                <p class="hero-subtext">
                    Enterprise-grade document intelligence that blends transformer-based deep learning with classical ML
                    to surface risks, compliance gaps, and redline candidates in seconds.
                </p>
            </section>
            """
        )

        gr.HTML(
            """
            <section class="metrics">
                <div class="metric-card">
                    <h5>Transformer Model</h5>
                    <span>DistilBERT · 66M parameters</span>
                </div>
                <div class="metric-card">
                    <h5>Classical Model</h5>
                    <span>TF-IDF · 10k features</span>
                </div>
                <div class="metric-card">
                    <h5>Supported Flags</h5>
                    <span>Crosswalk · Claims · Consistency</span>
                </div>
            </section>
            """
        )

        with gr.Row(elem_classes=["workspace"]):
            with gr.Column(elem_classes=["panel"]):
                gr.Markdown("#### Intake", elem_classes=["section-title"])
                pdf_input = gr.File(
                    label="Upload proposal PDF",
                    file_types=[".pdf"],
                    elem_classes=["upload-box"],
                )
                gr.Markdown("##### or paste raw text", elem_classes=["subsection-title"])
                text_input = gr.Textbox(
                    label="Paste proposal text",
                    lines=10,
                    placeholder="Drop in a section or full proposal to analyze…",
                    elem_classes=["input-area"],
                )
                verify_btn = gr.Button("Run verification", elem_classes=["primary-btn"])

            with gr.Column(elem_classes=["panel"]):
                diagnostics_header = gr.Markdown(DIAGNOSTICS_HEADER_MD, elem_classes=["section-title"], visible=False)
                with gr.Row():
                    distilbert_output = gr.HTML(elem_classes=["output-box", "result-card"], visible=False)
                    tfidf_output = gr.HTML(elem_classes=["output-box", "result-card"], visible=False)
                summary_header = gr.Markdown(SCAN_SUMMARY_HEADER_MD, elem_classes=["section-title"], visible=False)
                status_output = gr.HTML(elem_classes=["status-box"], visible=False)
                document_meta = gr.HTML(elem_classes=["info-card"], visible=False)

    verify_btn.click(
        fn=enhanced_status_update,
        inputs=[pdf_input, text_input],
        outputs=[
            diagnostics_header,
            distilbert_output,
            tfidf_output,
            summary_header,
            status_output,
            document_meta,
        ]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()
