"""Generate PDFs from scratch text files for demo purposes."""

from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_LEFT, TA_CENTER

def create_pdf_from_text(text_file: Path, pdf_file: Path):
    """Convert a text file to a formatted PDF."""

    # Read the text
    text = text_file.read_text(encoding='utf-8')

    # Create PDF
    doc = SimpleDocTemplate(
        str(pdf_file),
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )

    # Styles
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        textColor='#1a1d29',
        spaceAfter=12,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=12,
        textColor='#1a1d29',
        spaceAfter=6,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )

    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=10,
        textColor='#333333',
        spaceAfter=6,
        leading=14,
        fontName='Helvetica'
    )

    # Build content
    story = []

    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            story.append(Spacer(1, 0.1*inch))
            continue

        # Detect section types
        if line.startswith('HDR Engineering'):
            story.append(Paragraph(line, title_style))
        elif line.startswith('Technical Proposal') or line.startswith('Project Reference'):
            story.append(Paragraph(line, heading_style))
        elif line.isupper() and len(line.split()) <= 5:
            # All caps short lines are headings
            story.append(Paragraph(line, heading_style))
        elif line.startswith(('1.', '2.', '3.', '4.', '-')):
            # List items
            story.append(Paragraph(line, body_style))
        else:
            # Normal paragraphs
            story.append(Paragraph(line, body_style))

    # Build PDF
    doc.build(story)
    print(f"✓ Created: {pdf_file.name}")


def main():
    """Generate PDFs for all scratch text files."""
    base_dir = Path(__file__).parent

    scratch_files = [
        'scratch_clean.txt',
        'scratch_crosswalk.txt',
        'scratch_banned.txt',
        'scratch_name.txt',
        'scratch_date.txt'
    ]

    print("Generating PDFs from scratch text files...")
    print("="*60)

    for filename in scratch_files:
        text_file = base_dir / filename
        if not text_file.exists():
            print(f"⚠ Skipping {filename} (not found)")
            continue

        pdf_file = base_dir / filename.replace('.txt', '.pdf')
        create_pdf_from_text(text_file, pdf_file)

    print("="*60)
    print("✅ PDF generation complete!")


if __name__ == "__main__":
    main()
