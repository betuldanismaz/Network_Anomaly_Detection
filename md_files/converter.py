"""
Markdown to PDF Converter for Network Detection Technical Reports
Converts markdown documentation to professionally styled PDF reports.
Supports Mermaid diagrams by converting them to placeholder boxes.

Usage:
    python converter.py                           # Convert default file
    python converter.py --input file.md           # Convert specific file
    python converter.py --all                     # Convert all MD files in md_files/
    python converter.py --help                    # Show help
"""

import argparse
import os
import sys
import re
from pathlib import Path
from typing import Optional, List
import markdown
from xhtml2pdf import pisa


class MarkdownToPDFConverter:
    """Handles conversion of Markdown files to professionally styled PDFs."""
    
    # Professional CSS styling for technical reports
    CSS_TEMPLATE = """
    <style>
        @page {
            size: A4;
            margin: 2cm;
        }
        body {
            font-family: Helvetica, Arial, sans-serif;
            font-size: 11pt;
            line-height: 1.5;
            color: #333;
        }
        h1 {
            color: #1a4e8a;
            font-size: 24pt;
            border-bottom: 2px solid #1a4e8a;
            padding-bottom: 5px;
            margin-top: 0;
        }
        h2 {
            color: #1a4e8a;
            font-size: 18pt;
            margin-top: 25px;
            border-bottom: 1px solid #ddd;
        }
        h3 {
            color: #2c3e50;
            font-size: 14pt;
            margin-top: 20px;
            font-weight: bold;
        }
        h4, h5, h6 {
            color: #2c3e50;
            margin-top: 15px;
        }
        pre {
            background-color: #f4f4f4;
            border: 1px solid #ddd;
            padding: 10px;
            font-family: "Courier New", Courier, monospace;
            font-size: 9pt;
            white-space: pre-wrap;
            border-radius: 4px;
            overflow-x: auto;
        }
        code {
            font-family: "Courier New", Courier, monospace;
            background-color: #f4f4f4;
            padding: 2px 4px;
            border-radius: 2px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 10pt;
        }
        th {
            background-color: #1a4e8a;
            color: white;
            padding: 8px;
            text-align: left;
            font-weight: bold;
        }
        td {
            border: 1px solid #ddd;
            padding: 8px;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        blockquote {
            background-color: #eef6ff;
            border-left: 5px solid #1a4e8a;
            padding: 10px;
            margin: 10px 0;
            font-style: italic;
        }
        hr {
            border: 0;
            height: 1px;
            background: #ccc;
            margin: 20px 0;
        }
        ul, ol {
            margin: 10px 0;
            padding-left: 30px;
        }
        li {
            margin: 5px 0;
        }
        img {
            max-width: 100%;
            height: auto;
        }
        a {
            color: #1a4e8a;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
    </style>
    """
    
    def __init__(self, project_root: Optional[Path] = None):
        """Initialize converter with project root directory."""
        self.project_root = project_root or Path(__file__).parent
        self.md_files_dir = self.project_root / "networkdetection" / "md_files"
        
    def validate_input(self, input_file: Path) -> bool:
        """Validate that input file exists and is a markdown file."""
        if not input_file.exists():
            print(f"❌ Error: File not found: {input_file}")
            return False
        if input_file.suffix.lower() not in ['.md', '.markdown']:
            print(f"❌ Error: File is not a markdown file: {input_file}")
            return False
        return True
    
    def read_markdown(self, input_file: Path) -> str:
        """Read and return markdown content from file."""
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise IOError(f"Failed to read file {input_file}: {e}")
    
    def preprocess_mermaid_diagrams(self, md_content: str) -> str:
        """Convert Mermaid diagrams to styled HTML boxes with diagram content."""
        def replace_mermaid(match):
            mermaid_code = match.group(1).strip()
            # Create a styled box with the mermaid code displayed as formatted text
            html_box = f'''
<div style="border: 2px solid #1a4e8a; padding: 15px; margin: 20px 0; background-color: #f8f9fa; border-radius: 5px;">
    <h4 style="color: #1a4e8a; margin-top: 0;">📊 Diagram:</h4>
    <pre style="background-color: #ffffff; padding: 10px; border: 1px solid #ddd; overflow-x: auto; font-size: 8pt; line-height: 1.4;">
{mermaid_code}
    </pre>
</div>
'''
            return html_box
        
        # Match mermaid code blocks
        pattern = r'```mermaid\n(.*?)```'
        result = re.sub(pattern, replace_mermaid, md_content, flags=re.DOTALL)
        return result
    
    def markdown_to_html(self, md_content: str) -> str:
        """Convert markdown content to HTML with extended features."""
        # Preprocess Mermaid diagrams first
        md_content = self.preprocess_mermaid_diagrams(md_content)
        
        return markdown.markdown(
            md_content,
            extensions=[
                'extra',          # Tables, fenced code blocks, etc.
                'codehilite',     # Syntax highlighting for code
                'tables',         # Table support
                'sane_lists',     # Better list handling
                'nl2br',          # Newline to <br>
                'toc'             # Table of contents
            ]
        )
    
    def generate_pdf(self, html_content: str, output_file: Path) -> bool:
        """Generate PDF from HTML content."""
        full_html = f"<html><head>{self.CSS_TEMPLATE}</head><body>{html_content}</body></html>"
        
        try:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, "wb") as pdf_file:
                pisa_status = pisa.CreatePDF(full_html, dest=pdf_file)
                
            if pisa_status.err:
                print(f"❌ Error: PDF generation failed for {output_file}")
                return False
            
            return True
        except Exception as e:
            print(f"❌ Error writing PDF: {e}")
            return False
    
    def convert(self, input_file: Path, output_file: Optional[Path] = None) -> bool:
        """
        Convert a markdown file to PDF.
        
        Args:
            input_file: Path to input markdown file
            output_file: Path to output PDF file (auto-generated if None)
            
        Returns:
            True if conversion successful, False otherwise
        """
        # Validate input
        if not self.validate_input(input_file):
            return False
        
        # Generate output filename if not provided
        if output_file is None:
            output_file = self.project_root / f"{input_file.stem}.pdf"
        
        print(f"📄 Converting: {input_file.name}")
        
        try:
            # Read markdown
            md_content = self.read_markdown(input_file)
            
            # Convert to HTML
            html_content = self.markdown_to_html(md_content)
            
            # Generate PDF
            if self.generate_pdf(html_content, output_file):
                print(f"✅ Successfully created: {output_file}")
                print(f"   Size: {output_file.stat().st_size / 1024:.1f} KB")
                return True
            else:
                return False
                
        except Exception as e:
            print(f"❌ Conversion failed: {e}")
            return False
    
    def convert_all(self) -> int:
        """Convert all markdown files in the md_files directory."""
        if not self.md_files_dir.exists():
            print(f"❌ Directory not found: {self.md_files_dir}")
            return 0
        
        md_files = list(self.md_files_dir.glob("*.md"))
        
        if not md_files:
            print(f"❌ No markdown files found in {self.md_files_dir}")
            return 0
        
        print(f"📚 Found {len(md_files)} markdown file(s)")
        print("-" * 60)
        
        success_count = 0
        for md_file in md_files:
            output_file = self.project_root / f"{md_file.stem}.pdf"
            if self.convert(md_file, output_file):
                success_count += 1
            print("-" * 60)
        
        print(f"\n✨ Converted {success_count}/{len(md_files)} file(s) successfully")
        return success_count


def main():
    """Main entry point for the converter."""
    parser = argparse.ArgumentParser(
        description="Convert Markdown technical reports to professionally styled PDFs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Convert default report
  %(prog)s --input custom.md                  # Convert specific file
  %(prog)s --input report.md --output out.pdf # Custom output name
  %(prog)s --all                              # Convert all MD files
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        help='Input markdown file path'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Output PDF file path (default: <input_name>.pdf)'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Convert all markdown files in networkdetection/md_files/'
    )
    
    args = parser.parse_args()
    
    # Initialize converter
    converter = MarkdownToPDFConverter()
    
    # Handle --all flag
    if args.all:
        converter.convert_all()
        return
    
    # Determine input file
    if args.input:
        input_path = Path(args.input)
        if not input_path.is_absolute():
            # Try relative to script, then relative to md_files
            if not input_path.exists():
                input_path = converter.md_files_dir / args.input
    else:
        # Default file
        input_path = converter.md_files_dir / "DATA_SCIENCE_TECHNICAL_REPORT.md"
        if not input_path.exists():
            print("❌ Default file not found. Use --input to specify a file or --all to convert all files.")
            print("   Or use --help for more options.")
            sys.exit(1)
    
    # Determine output file
    output_path = Path(args.output) if args.output else None
    
    # Convert
    success = converter.convert(input_path, output_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()