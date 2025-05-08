import streamlit as st
import re
import os
import markdown2
from weasyprint import HTML
from config import PDF_CSS_STYLE

def _preprocess_markdown_for_pdf(markdown_content):
    """Modifies markdown for better PDF heading structure."""
    processed_md = re.sub(r"^\s*###\s*(Report for Batch.*)", r"# \1", markdown_content, flags=re.MULTILINE)
    sections = [
        "Executive Summary", "Overall Sentiment Analysis",
        "Key Themes and Issues", "Actionable Recommendations"
    ]
    for i, section_title_base in enumerate(sections, 1):
        pattern = re.compile(r"^\s*\*\*{}\s*[\.\)]?\s*{}\s*\*\*".format(i, re.escape(section_title_base)), re.IGNORECASE | re.MULTILINE)
        replacement = r"## {}. {}".format(i, section_title_base)
        processed_md = pattern.sub(replacement, processed_md)
    return processed_md

def generate_pdf_from_markdown_bytes(markdown_content):
    try:
        processed_markdown = _preprocess_markdown_for_pdf(markdown_content)
        html_body = markdown2.markdown(
            processed_markdown,
            extras=["fenced-code-blocks", "tables", "header-ids", "smarty-pants", "break-on-newline"]
        )
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Customer Review Report</title>
            <style>
                {PDF_CSS_STYLE}
            </style>
        </head>
        <body>
            <div class="report-content">
                {html_body}
            </div>
        </body>
        </html>
        """
        pdf_bytes = HTML(string=full_html, base_url=os.getcwd()).write_pdf()
        return pdf_bytes
    except Exception as e:
        st.error(f"Error generating PDF: {e}. Please ensure WeasyPrint and its dependencies are correctly installed.")
        return None

def extract_batch_range_for_filename(markdown_content):
    match = re.search(r"Report for Batch\s+([\w\d\s-]+)", markdown_content, re.IGNORECASE)
    if match:
        batch_name = match.group(1).strip().replace(" ", "_").replace("/", "-")
        return f"Report_Batch_{batch_name}"
    return "Customer_Review_Report"