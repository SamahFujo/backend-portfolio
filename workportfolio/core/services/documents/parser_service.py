"""
Document parser service.

Supports:
- PDF
- DOCX
- TXT
"""

from pathlib import Path
import re
import fitz  # PyMuPDF
from docx import Document


class ParserService:
    """
    Extract raw text from uploaded files.
    """

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Normalize extracted text for downstream chunking and embeddings.
        """
        text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def extract_text(file_path: str) -> str:
        """
        Extract text based on file extension.

        Args:
            file_path (str): Absolute or relative file path.

        Returns:
            str: Extracted text.
        """
        suffix = Path(file_path).suffix.lower()

        if suffix == ".pdf":
            text = ParserService._extract_from_pdf(file_path)
        elif suffix == ".docx":
            text = ParserService._extract_from_docx(file_path)
        elif suffix == ".txt":
            text = ParserService._extract_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        return ParserService._clean_text(text)

    @staticmethod
    def _extract_from_pdf(file_path: str) -> str:
        """
        Extract text from a PDF using PyMuPDF.
        """
        text_parts = []

        with fitz.open(file_path) as pdf:
            for page in pdf:
                page_text = page.get_text("text") or ""
                if page_text.strip():
                    text_parts.append(page_text)

        return "\n".join(text_parts).strip()

    @staticmethod
    def _extract_from_docx(file_path: str) -> str:
        """
        Extract text from a DOCX, including both paragraphs and tables.
        """
        doc = Document(file_path)
        parts = []

        # Paragraphs
        for p in doc.paragraphs:
            if p.text and p.text.strip():
                parts.append(p.text.strip())

        # Tables
        for table in doc.tables:
            for row in table.rows:
                cells = []
                for cell in row.cells:
                    cell_text = cell.text.strip()
                    if cell_text:
                        cells.append(cell_text)
                if cells:
                    parts.append(" | ".join(cells))

        return "\n".join(parts).strip()

    @staticmethod
    def _extract_from_txt(file_path: str) -> str:
        """
        Extract text from TXT with simple encoding fallbacks.
        """
        encodings = ["utf-8", "utf-8-sig", "latin-1"]

        for enc in encodings:
            try:
                with open(file_path, "r", encoding=enc) as f:
                    return f.read().strip()
            except UnicodeDecodeError:
                continue

        raise ValueError("Could not decode TXT file with supported encodings.")
