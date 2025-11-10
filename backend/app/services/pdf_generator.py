"""PDF generation service."""
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.colors import black, HexColor
from io import BytesIO
from pathlib import Path


def create_pdf_from_text(text: str, output_path: str | Path) -> None:
    """
    Create a PDF file from text content with colors preserved.
    
    Args:
        text: Text content to convert to PDF
        output_path: Path where PDF will be saved
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.75*inch, bottomMargin=0.75*inch)
    
    styles = getSampleStyleSheet()
    # Create a custom style that preserves formatting
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=11,
        leading=14,
        textColor=black,
        spaceAfter=6,
    )
    
    story = []
    
    # Split text into paragraphs and create PDF content
    paragraphs = text.split('\n\n')
    for para in paragraphs:
        if para.strip():
            # Replace newlines within paragraphs with <br/> tags
            para_formatted = para.replace('\n', '<br/>')
            p = Paragraph(para_formatted, normal_style)
            story.append(p)
            story.append(Spacer(1, 0.1*inch))
    
    doc.build(story)
    
    # Write to file
    with open(output_path, 'wb') as f:
        f.write(buffer.getvalue())
    buffer.close()

