"""
Document parser service.

Supports:
- PDF
- DOCX
- TXT
"""

from pathlib import Path
import fitz  # PyMuPDF
from docx import Document


class ParserService:
    """
    Extract raw text from uploaded files.
    """

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
            return ParserService._extract_from_pdf(file_path)
        elif suffix == ".docx":
            return ParserService._extract_from_docx(file_path)
        elif suffix == ".txt":
            return ParserService._extract_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    @staticmethod
    def _extract_from_pdf(file_path: str) -> str:
        text_parts = []
        with fitz.open(file_path) as pdf:
            for page in pdf:
                text_parts.append(page.get_text())
        return "\n".join(text_parts).strip()

    @staticmethod
    def _extract_from_docx(file_path: str) -> str:
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs]).strip()

    @staticmethod
    def _extract_from_txt(file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
