#!/usr/bin/env python3
"""Generate colorful PDFs from sample text files."""

from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from datetime import datetime

# Color palette
HEADER_COLOR = colors.HexColor("#2563eb")  # Blue
ACCENT_COLOR = colors.HexColor("#7c3aed")  # Purple
SECTION_BG = colors.HexColor("#f3f4f6")    # Light grey
TEXT_COLOR = colors.HexColor("#1f2937")    # Dark grey

def create_colorful_pdf(text_file: Path, output_file: Path):
    """Create a colorful PDF from a text file."""

    # Read the text file
    with open(text_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    # Create PDF document
    doc = SimpleDocTemplate(
        str(output_file),
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch,
    )

    # Container for the 'Flowable' objects
    elements = []

    # Define styles
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.white,
        spaceAfter=12,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold',
    )

    section_style = ParagraphStyle(
        'Section',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=HEADER_COLOR,
        spaceAfter=10,
        spaceBefore=10,
        fontName='Helvetica-Bold',
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['BodyText'],
        fontSize=11,
        textColor=TEXT_COLOR,
        leading=16,
        alignment=TA_LEFT,
    )

    # Extract title from filename
    title = text_file.stem.replace('scratch_', '').replace('_', ' ').title()

    # Add colored header
    header_data = [[Paragraph(f"<b>{title}</b>", title_style)]]
    header_table = Table(header_data, colWidths=[7.5*inch])
    header_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), HEADER_COLOR),
        ('PADDING', (0, 0), (-1, -1), 15),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 20),
        ('RIGHTPADDING', (0, 0), (-1, -1), 20),
    ]))
    elements.append(header_table)
    elements.append(Spacer(1, 0.3*inch))

    # Add metadata
    metadata = f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}"
    meta_style = ParagraphStyle(
        'Meta',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.grey,
        alignment=TA_CENTER,
    )
    elements.append(Paragraph(metadata, meta_style))
    elements.append(Spacer(1, 0.2*inch))

    # Add content with background
    content_data = [[Paragraph(content, body_style)]]
    content_table = Table(content_data, colWidths=[7.5*inch])
    content_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), SECTION_BG),
        ('PADDING', (0, 0), (-1, -1), 15),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BORDER', (0, 0), (-1, -1), 2, ACCENT_COLOR),
        ('LEFTPADDING', (0, 0), (-1, -1), 20),
        ('RIGHTPADDING', (0, 0), (-1, -1), 20),
        ('TOPPADDING', (0, 0), (-1, -1), 20),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 20),
    ]))
    elements.append(content_table)

    elements.append(Spacer(1, 0.3*inch))

    # Add footer with accent color
    footer_text = "HDR Proposal Verification Assistant - Sample Document"
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=ACCENT_COLOR,
        alignment=TA_CENTER,
    )
    elements.append(Paragraph(footer_text, footer_style))

    # Build PDF
    doc.build(elements)
    print(f"✓ Created: {output_file.name}")

def main():
    """Generate PDFs for all sample text files."""
    current_dir = Path(__file__).parent

    sample_files = sorted(current_dir.glob("scratch_*.txt"))

    if not sample_files:
        print("No sample files found!")
        return

    print(f"Found {len(sample_files)} sample files\n")

    for text_file in sample_files:
        pdf_file = text_file.with_suffix(".pdf")
        create_colorful_pdf(text_file, pdf_file)

    print(f"\n✓ All PDFs generated successfully!")

if __name__ == "__main__":
    main()
