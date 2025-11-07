#!/usr/bin/env python3
"""
Convert Markdown Research Report to PDF

This script converts the XAU_USD_Research_Report.md file into a professional PDF document.
It handles image paths, formatting, and styling to produce a publication-ready PDF.

Usage:
    python scripts/convert_pdf.py
    python scripts/convert_pdf.py --input reports/XAU_USD_Research_Report.md --output reports/XAU_USD_Research_Report.pdf
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config_manager import load_config, validate_config, get_setting

def convert_markdown_to_pdf(md_file: str, pdf_file: str = None) -> str:
    """
    Convert markdown file to PDF using available libraries.
    
    Args:
        md_file: Path to input markdown file
        pdf_file: Path to output PDF file (optional, auto-generated if None)
    
    Returns:
        Path to generated PDF file
    """
    md_path = Path(md_file)
    if not md_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {md_file}")
    
    # Auto-generate PDF filename if not provided
    if pdf_file is None:
        pdf_file = md_path.with_suffix('.pdf')
    pdf_path = Path(pdf_file)
    
    # Ensure output directory exists
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Read markdown content
    with open(md_path, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Try different conversion methods in order of preference
    conversion_methods = [
        _convert_with_weasyprint,
        _convert_with_pypandoc,
        _convert_with_markdown_pdf,
        _convert_with_reportlab,
    ]
    
    for method in conversion_methods:
        try:
            print(f"Attempting conversion with {method.__name__}...")
            method(md_content, md_path, pdf_path)
            print(f"✅ Successfully converted to PDF: {pdf_path}")
            return str(pdf_path)
        except ImportError:
            continue
        except Exception as e:
            print(f"⚠️  {method.__name__} failed: {e}")
            continue
    
    raise RuntimeError(
        "No PDF conversion library available. Please install one of:\n"
        "  - weasyprint: pip install weasyprint\n"
        "  - pypandoc: pip install pypandoc (requires pandoc)\n"
        "  - markdown-pdf: pip install markdown-pdf\n"
        "  - reportlab: pip install reportlab"
    )


def _convert_with_weasyprint(md_content: str, md_path: Path, pdf_path: Path):
    """Convert using weasyprint (markdown -> HTML -> PDF)."""
    try:
        import markdown
        from weasyprint import HTML, CSS
        from weasyprint.text.fonts import FontConfiguration
    except ImportError:
        raise ImportError("weasyprint or markdown not installed")
    
    # Convert markdown to HTML
    html_content = markdown.markdown(
        md_content,
        extensions=['extra', 'codehilite', 'tables', 'toc']
    )
    
    # Fix image paths - convert relative paths to absolute
    html_content = _fix_image_paths(html_content, md_path)
    
    # Add CSS styling for professional PDF
    css_string = """
    @page {
        size: A4;
        margin: 2.5cm 2cm;
        @top-center {
            content: "Gold (XAU/USD) Quantitative Trading Research Report";
            font-size: 10pt;
            color: #666;
        }
        @bottom-center {
            content: "Page " counter(page) " of " counter(pages);
            font-size: 10pt;
            color: #666;
        }
    }
    
    body {
        font-family: 'Georgia', 'Times New Roman', serif;
        font-size: 11pt;
        line-height: 1.6;
        color: #333;
    }
    
    h1 {
        font-size: 24pt;
        font-weight: bold;
        color: #1a1a1a;
        margin-top: 2em;
        margin-bottom: 1em;
        page-break-after: avoid;
    }
    
    h2 {
        font-size: 18pt;
        font-weight: bold;
        color: #2a2a2a;
        margin-top: 1.5em;
        margin-bottom: 0.8em;
        page-break-after: avoid;
        border-bottom: 2px solid #ddd;
        padding-bottom: 0.3em;
    }
    
    h3 {
        font-size: 14pt;
        font-weight: bold;
        color: #3a3a3a;
        margin-top: 1.2em;
        margin-bottom: 0.6em;
        page-break-after: avoid;
    }
    
    h4 {
        font-size: 12pt;
        font-weight: bold;
        color: #4a4a4a;
        margin-top: 1em;
        margin-bottom: 0.5em;
    }
    
    p {
        margin-top: 0.5em;
        margin-bottom: 0.5em;
        text-align: justify;
    }
    
    code {
        font-family: 'Courier New', monospace;
        font-size: 10pt;
        background-color: #f5f5f5;
        padding: 2px 4px;
        border-radius: 3px;
    }
    
    pre {
        background-color: #f5f5f5;
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 1em;
        overflow-x: auto;
        page-break-inside: avoid;
    }
    
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 1em 0;
        page-break-inside: avoid;
    }
    
    th, td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: left;
    }
    
    th {
        background-color: #4a4a4a;
        color: white;
        font-weight: bold;
    }
    
    tr:nth-child(even) {
        background-color: #f9f9f9;
    }
    
    img {
        max-width: 100%;
        height: auto;
        display: block;
        margin: 1em auto;
        page-break-inside: avoid;
    }
    
    blockquote {
        border-left: 4px solid #4a4a4a;
        margin: 1em 0;
        padding-left: 1em;
        color: #555;
        font-style: italic;
    }
    
    ul, ol {
        margin: 0.5em 0;
        padding-left: 2em;
    }
    
    li {
        margin: 0.3em 0;
    }
    
    strong {
        font-weight: bold;
        color: #2a2a2a;
    }
    
    em {
        font-style: italic;
    }
    
    hr {
        border: none;
        border-top: 2px solid #ddd;
        margin: 2em 0;
    }
    """
    
    # Wrap HTML content with proper structure
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Gold (XAU/USD) Quantitative Trading Research Report</title>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    # Generate PDF
    font_config = FontConfiguration()
    HTML(string=full_html, base_url=str(md_path.parent)).write_pdf(
        pdf_path,
        stylesheets=[CSS(string=css_string)],
        font_config=font_config
    )


def _convert_with_pypandoc(md_content: str, md_path: Path, pdf_path: Path):
    """Convert using pypandoc (requires pandoc installed)."""
    try:
        import pypandoc
    except ImportError:
        raise ImportError("pypandoc not installed")
    
    # Convert markdown to PDF using pandoc
    pypandoc.convert_file(
        str(md_path),
        'pdf',
        outputfile=str(pdf_path),
        extra_args=[
            '--pdf-engine=xelatex',
            '--variable', 'mainfont=Helvetica',
            '--resource-path=./'
        ]
    )


def _convert_with_markdown_pdf(md_content: str, md_path: Path, pdf_path: Path):
    """Convert using markdown-pdf library."""
    try:
        import markdown2
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
    except ImportError:
        raise ImportError("markdown-pdf dependencies not installed")
    
    # This is a simplified version - full implementation would require more parsing
    raise NotImplementedError("markdown-pdf conversion requires more complex implementation")


def _convert_with_reportlab(md_content: str, md_path: Path, pdf_path: Path):
    """Convert using reportlab (basic implementation)."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch
        import markdown2
    except ImportError:
        raise ImportError("reportlab or markdown2 not installed")
    
    # This is a simplified version - full implementation would require more parsing
    raise NotImplementedError("reportlab conversion requires more complex markdown parsing")


def _fix_image_paths(html_content: str, md_path: Path) -> str:
    """Fix relative image paths in HTML to absolute paths."""
    import re
    
    # Find all image tags with relative paths
    def replace_path(match):
        full_match = match.group(0)
        img_path = match.group(1)
        # If it's a relative path, convert to absolute
        if not img_path.startswith('/') and '://' not in img_path:
            # Try to find the image relative to the markdown file
            img_full_path = (md_path.parent / img_path).resolve()
            if img_full_path.exists():
                return full_match.replace(img_path, str(img_full_path))
        return full_match
    
    # Handle HTML image tags <img src="path">
    html_content = re.sub(
        r'<img[^>]+src=["\']([^"\']+)["\']',
        replace_path,
        html_content
    )
    
    return html_content


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Convert Markdown Research Report to PDF'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Optional asset configuration YAML file (used to determine report paths)'
    )
    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='Input markdown file path (overrides configuration)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output PDF file path (default: same as input with .pdf extension)'
    )
    
    args = parser.parse_args()
    
    asset_config = None
    if args.config:
        asset_config = load_config(args.config)
        validate_config(asset_config)
    
    if args.input:
        md_file = project_root / args.input
    elif asset_config:
        report_name = get_setting(asset_config, 'output.report_name')
        md_file = project_root / 'reports' / f"{report_name}.md"
    else:
        md_file = project_root / 'reports' / 'XAU_USD_Research_Report.md'
    
    if args.output:
        pdf_file = project_root / args.output
    elif asset_config:
        report_name = get_setting(asset_config, 'output.report_name')
        pdf_file = project_root / 'reports' / f"{report_name}.pdf"
    else:
        pdf_file = None
    
    try:
        pdf_path = convert_markdown_to_pdf(str(md_file), str(pdf_file) if pdf_file else None)
        print(f"\n✅ PDF conversion successful!")
        print(f"   Input:  {md_file}")
        print(f"   Output: {pdf_path}")
        print(f"\n📄 PDF saved to: {Path(pdf_path).absolute()}")
        return 0
    except Exception as e:
        print(f"\n❌ Error converting to PDF: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())

