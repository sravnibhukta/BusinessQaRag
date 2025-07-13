#!/usr/bin/env python3
"""
Convert markdown files to PDF format for submission.
"""

import markdown
import weasyprint
from pathlib import Path
import sys

def convert_markdown_to_pdf(markdown_file, output_file):
    """Convert markdown file to PDF"""
    try:
        # Read markdown file
        with open(markdown_file, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        # Convert markdown to HTML
        html = markdown.markdown(
            markdown_content,
            extensions=['codehilite', 'tables', 'toc', 'fenced_code']
        )
        
        # Add CSS styling for professional appearance
        css_style = """
        <style>
        body {
            font-family: 'Times New Roman', serif;
            line-height: 1.6;
            margin: 2cm;
            color: #333;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #2c3e50;
            margin-top: 1.5em;
            margin-bottom: 0.5em;
        }
        h1 {
            font-size: 2.2em;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            font-size: 1.8em;
            border-bottom: 1px solid #bdc3c7;
            padding-bottom: 5px;
        }
        h3 {
            font-size: 1.4em;
            color: #34495e;
        }
        code {
            background-color: #f8f9fa;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }
        pre {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #3498db;
            overflow-x: auto;
            font-family: 'Courier New', monospace;
            font-size: 0.85em;
        }
        pre code {
            background-color: transparent;
            padding: 0;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 1em 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        blockquote {
            border-left: 4px solid #3498db;
            padding-left: 15px;
            margin: 1em 0;
            color: #555;
            font-style: italic;
        }
        ul, ol {
            margin: 1em 0;
            padding-left: 2em;
        }
        li {
            margin: 0.5em 0;
        }
        .page-break {
            page-break-before: always;
        }
        @page {
            size: A4;
            margin: 2cm;
        }
        </style>
        """
        
        # Create complete HTML document
        html_document = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Document</title>
            {css_style}
        </head>
        <body>
            {html}
        </body>
        </html>
        """
        
        # Convert HTML to PDF
        pdf_document = weasyprint.HTML(string=html_document)
        pdf_document.write_pdf(output_file)
        
        print(f"Successfully converted {markdown_file} to {output_file}")
        return True
        
    except Exception as e:
        print(f"Error converting {markdown_file}: {e}")
        return False

def main():
    """Main function to convert markdown files to PDF"""
    
    # Files to convert
    files_to_convert = [
        ("Task2_RAG_Optimization_Techniques.md", "Task2_RAG_Optimization_Techniques.pdf"),
        ("Task3_Dataset_Preparation_and_Fine_Tuning.md", "Task3_Dataset_Preparation_and_Fine_Tuning.pdf")
    ]
    
    print("Converting markdown files to PDF...")
    
    success_count = 0
    for markdown_file, pdf_file in files_to_convert:
        if Path(markdown_file).exists():
            if convert_markdown_to_pdf(markdown_file, pdf_file):
                success_count += 1
        else:
            print(f"Warning: {markdown_file} not found")
    
    print(f"\nConversion complete. Successfully converted {success_count} files.")
    
    # List generated files
    print("\nGenerated files:")
    for _, pdf_file in files_to_convert:
        if Path(pdf_file).exists():
            print(f"  ✓ {pdf_file}")
        else:
            print(f"  ✗ {pdf_file} (failed)")

if __name__ == "__main__":
    main()