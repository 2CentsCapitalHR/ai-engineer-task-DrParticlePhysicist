# doc_parser.py
# src/doc_parser.py
from docx import Document
from pathlib import Path

def extract_text_from_docx(file_path: Path) -> str:
    """
    Reads a .docx file and returns its full text.
    """
    doc = Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text.strip())
    return "\n".join([p for p in full_text if p])
