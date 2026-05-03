"""
Chunking service with document-type-specific strategies.

Goals:
- Preserve structure when documents are clearly sectioned
- Improve retrieval quality for FAQ, projects, certificates, preferences, etc.
- Still provide a safe generic fallback for unknown document types
"""

from __future__ import annotations

import re
from typing import List, Dict, Optional


class ChunkService:
    """
    Provides multiple chunking strategies based on document type.
    """

    DEFAULT_MAX_CHARS = 900
    DEFAULT_OVERLAP = 120

    RESUME_SECTION_HEADINGS = {
        "skills",
        "about me",
        "professional summary",
        "summary",
        "work experience",
        "professional experience",
        "experience",
        "projects",
        "selected ai projects",
        "selected ai projects & independent initiatives",
        "education",
        "personal details",
        "extra-curricular activities",
        "extracurricular activities",
        "certifications",
        "certificates",
        "publications",
        "research",
        "additional information",
        "contact",
    }
    
    @classmethod
    def chunk_capabilities(cls, text: str) -> List[str]:
        """
        Chunk the 'What I Can Help With' document.

        This document has two heading levels:
        1. Main capability sections
        2. Technology sub-sections under 'Technologies I Use Professionally'

        This custom chunker keeps the technology block intact so tools like
        Hugging Face Transformers and embeddings/vector search are not lost.
        """

        full_text = (text or "").strip()
        if not full_text:
            return []

        # Main headings only. Do not include technology subheadings here.
        main_headings = {
            "overview",
            "what i can do confidently",
            "ai and llm solutions",
            "backend development",
            "full-stack web applications",
            "data and ai-powered business systems",
            "frontend and user experience",
            "technical leadership and delivery",
            "types of projects i can build",
            "technologies i use professionally",
            "work i can do with some ramp-up",
            "areas i do not primarily specialize in",
            "the kind of problems i am best at solving",
            "summary",
        }

        lines = full_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

        sections = []
        current_heading = None
        current_body = []

        for line in lines:
            clean_line = line.strip()
            if not clean_line:
                continue

            normalized = " ".join(clean_line.lower().split())

            if normalized in main_headings:
                if current_heading:
                    sections.append({
                        "heading": current_heading,
                        "body": "\n".join(current_body).strip(),
                    })

                current_heading = clean_line
                current_body = []
            else:
                if current_heading:
                    current_body.append(clean_line)

        if current_heading:
            sections.append({
                "heading": current_heading,
                "body": "\n".join(current_body).strip(),
            })

        chunks = []

        for section in sections:
            heading = section["heading"].strip()
            body = section["body"].strip()

            if not body:
                continue

            max_chars = 1300 if heading.lower() == "technologies i use professionally" else 850

            chunks.extend(
                cls._chunk_long_text(
                    body,
                    max_chars=max_chars,
                    overlap=100,
                    prefix=f"Section: {heading}",
                )
            )

        return cls._dedupe_chunks(chunks)

    @classmethod
    def _clean_resume_section_body(cls, heading: str, body: str) -> str:
        """
        Clean section-specific extraction noise caused by two-column CV layouts.
        For example, sidebar skills/languages may appear inside Work Experience.
        """

        heading_key = (heading or "").strip().lower()
        body = body or ""

        if heading_key == "work experience":
            sidebar_markers = [
                "Python",
                "Next.js",
                "RBAC, JWT",
                "SQL Server",
                "MySQL",
                "LangChain",
                "FastAPI",
                "JavaScript",
                "Postman",
                "Node.js",
                "MongoDB",
                "Docker",
                "Streamlit",
                "Langfuse",
                "Gunicorn",
                "NGINX",
                "Ollama",
                "Flask",
                ".NET",
                "Java",
                "C++",
                "PHP",
                "LANGUAGES",
                "Arabic",
                "English",
            ]

            lines = body.splitlines()
            cleaned_lines = []

            for line in lines:
                clean = line.strip()

                if clean in sidebar_markers:
                    continue

                cleaned_lines.append(line)

            return "\n".join(cleaned_lines).strip()

        return body.strip()

    @classmethod
    def chunk_document(
        cls,
        raw_text: str,
        document_type: Optional[str] = None,
        title: Optional[str] = None,
    ) -> List[str]:
        """
        Main entry point.
        Routes chunking strategy based on document type.
        """
        text = cls._normalize_text(raw_text)
        if not text:
            return []

        doc_type = (document_type or "").strip().lower()

        if doc_type == "projects":
            return cls.chunk_projects(text)

        if doc_type == "faq":
            return cls.chunk_faq(text)

        if doc_type == "certificates":
            return cls.chunk_certificates(text)

        if doc_type == "capabilities":
            return cls.chunk_capabilities(text)

        if doc_type in {
            "preferences",
            "compensation",
            "achievements",
            "career_timeline",
        }:
            return cls.chunk_by_headings(text)

        if doc_type == "cv":
            return cls.chunk_resume(text)

        if doc_type in {"experience_letter", "recommendation"}:
            return cls.chunk_official_letter(text)

        return cls.chunk_generic(text)

    # ---------------------------------------------------------------------
    # Generic helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _normalize_text(text: str) -> str:
        """
        Normalize extracted text while preserving line breaks for structure detection.
        """
        if not text:
            return ""

        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace("\t", " ")

        # Trim trailing spaces per line
        lines = [line.strip() for line in text.split("\n")]

        # Collapse excessive blank lines
        cleaned_lines: List[str] = []
        blank_count = 0
        for line in lines:
            if not line:
                blank_count += 1
                if blank_count <= 1:
                    cleaned_lines.append("")
            else:
                blank_count = 0
                cleaned_lines.append(line)

        return "\n".join(cleaned_lines).strip()

    @staticmethod
    def _is_heading(line: str) -> bool:
        """
        Heuristic heading detector.
        Works for docs like:
        - Overview
        - Technology Stack
        - Preferred Backend Framework
        - Summary
        """
        if not line:
            return False

        lower = line.lower().strip()

        blocked = {
            "prepared for",
            "prepared on",
            "document purpose",
            "date",
            "sincerely,",
            "summary",
        }

        if lower in blocked:
            return True

        # Short title-like line
        if len(line) <= 80 and not line.endswith("."):
            # Few words and title-ish
            words = line.split()
            if 1 <= len(words) <= 8:
                return True

        return False

    @classmethod
    def _chunk_long_text(
        cls,
        text: str,
        max_chars: Optional[int] = None,
        overlap: Optional[int] = None,
        prefix: str = "",
    ) -> List[str]:
        """
        Generic sliding chunker for long text blocks.
        """
        text = (text or "").strip()
        if not text:
            return []

        max_chars = max_chars or cls.DEFAULT_MAX_CHARS
        overlap = overlap or cls.DEFAULT_OVERLAP

        if len(text) <= max_chars:
            return [f"{prefix}\n{text}".strip() if prefix else text]

        chunks: List[str] = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + max_chars, text_len)

            # Prefer breaking near sentence or newline boundaries
            if end < text_len:
                backtrack_zone = text[start:end]
                split_candidates = [
                    backtrack_zone.rfind("\n\n"),
                    backtrack_zone.rfind("\n"),
                    backtrack_zone.rfind(". "),
                    backtrack_zone.rfind("; "),
                ]
                best = max(split_candidates)
                if best > max_chars * 0.55:
                    end = start + best + 1

            piece = text[start:end].strip()
            if piece:
                chunks.append(f"{prefix}\n{piece}".strip()
                              if prefix else piece)

            if end >= text_len:
                break

            start = max(0, end - overlap)

        return chunks

    @staticmethod
    def _split_lines(text: str) -> List[str]:
        return [line.strip() for line in text.split("\n")]

    @classmethod
    def _extract_heading_sections(cls, text: str) -> List[dict]:
        """
        Extract resume/CV sections using known resume headings only.

        This avoids treating dates, company names, languages, universities,
        or job titles as top-level CV sections.
        """

        full_text = (text or "").strip()
        if not full_text:
            return []

        def normalize_line(value: str) -> str:
            value = (value or "").strip()
            value = value.replace("&amp;", "&")
            value = " ".join(value.split())
            return value.lower()

        lines = full_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

        sections: List[dict] = []
        current_heading = None
        current_body: List[str] = []

        for line in lines:
            clean_line = line.strip()

            if not clean_line:
                continue

            normalized = normalize_line(clean_line)

            is_resume_heading = normalized in cls.RESUME_SECTION_HEADINGS

            if is_resume_heading:
                if current_heading:
                    sections.append({
                        "heading": current_heading,
                        "body": "\n".join(current_body).strip(),
                    })

                current_heading = normalized.upper()
                current_body = []
            else:
                if current_heading:
                    current_body.append(clean_line)

        if current_heading:
            sections.append({
                "heading": current_heading,
                "body": "\n".join(current_body).strip(),
            })

        return sections

    # ---------------------------------------------------------------------
    # Generic fallback
    # ---------------------------------------------------------------------

    @classmethod
    def chunk_generic(cls, text: str) -> List[str]:
        """
        Generic fallback chunking.
        """
        return cls._chunk_long_text(text)
    
    
    @classmethod
    def _extract_generic_heading_sections(cls, text: str) -> List[dict]:
        """
        Extract sections for normal heading-based profile documents.

        Used for:
        - achievements
        - preferences
        - compensation
        - career_timeline
        - capabilities

        This is different from _extract_heading_sections(),
        which is CV-specific.
        """

        full_text = (text or "").strip()
        if not full_text:
            return []

        lines = full_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

        sections: List[dict] = []
        current_heading = None
        current_body: List[str] = []

        for line in lines:
            clean_line = line.strip()

            if not clean_line:
                continue

            if cls._is_heading(clean_line):
                if current_heading:
                    sections.append({
                        "heading": current_heading,
                        "body": "\n".join(current_body).strip(),
                    })

                current_heading = clean_line
                current_body = []
            else:
                if current_heading:
                    current_body.append(clean_line)

        if current_heading:
            sections.append({
                "heading": current_heading,
                "body": "\n".join(current_body).strip(),
            })

        return sections

    # ---------------------------------------------------------------------
    # Projects
    # ---------------------------------------------------------------------

    @classmethod
    def chunk_projects(cls, text: str) -> List[str]:
        """
        Chunk a project portfolio into project-aware and section-aware chunks.

        Strategy:
        - detect project starts from numbered titles like:
          '1. Live Smart Electricity Dashboard for Power Consumption'
        - split each project block
        - within each project block, try to extract sections
        - keep project title embedded into every chunk for better retrieval
        """
        project_pattern = re.compile(r"(?m)^\s*\d+\.\s+.+$")

        matches = list(project_pattern.finditer(text))
        if not matches:
            # fallback to heading sections if numbering is absent
            return cls.chunk_by_headings(text)

        chunks: List[str] = []

        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

            project_block = text[start:end].strip()
            if not project_block:
                continue

            first_line = project_block.split("\n", 1)[0].strip()
            project_title = re.sub(r"^\d+\.\s*", "", first_line).strip()

            # Remove title from body for cleaner section parsing
            remainder = project_block[len(first_line):].strip()

            sections = cls._extract_heading_sections(remainder)
            if not sections:
                chunks.extend(
                    cls._chunk_long_text(
                        project_block,
                        prefix=f"Project: {project_title}"
                    )
                )
                continue

            for section in sections:
                heading = section["heading"].strip()
                body = section["body"].strip()

                prefix = f"Project: {project_title}\nSection: {heading}"
                section_chunks = cls._chunk_long_text(
                    body,
                    max_chars=750,
                    overlap=80,
                    prefix=prefix,
                )
                chunks.extend(section_chunks)

        return cls._dedupe_chunks(chunks)

    # ---------------------------------------------------------------------
    # FAQ
    # ---------------------------------------------------------------------
    @classmethod
    def chunk_faq(cls, text: str) -> List[str]:
        """
        Chunk FAQ docs as question-answer pairs.

        Expected pattern:
        Question?
        Answer...

        This format is ideal for retrieval.
        """
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        chunks: List[str] = []

        current_question: Optional[str] = None
        current_answer: List[str] = []

        def flush():
            if current_question and current_answer:
                answer_text = " ".join(current_answer).strip()
                chunk = f"FAQ Question: {current_question}\nFAQ Answer: {answer_text}"
                chunks.append(chunk)

        for line in lines:
            if line.endswith("?"):
                flush()
                current_question = line
                current_answer = []
            else:
                if current_question:
                    current_answer.append(line)

        flush()
        return cls._dedupe_chunks(chunks) if chunks else cls.chunk_by_headings(text)

    # ---------------------------------------------------------------------
    # Certificates
    # ---------------------------------------------------------------------

    @classmethod
    def chunk_certificates(cls, text: str) -> List[str]:
        """
        Chunk certificates as exactly one rich chunk per certificate.

        Strategy:
        1. Prefer detailed certificate sections:
        - Title
        - Issuer / Date
        - Focus
        - Why it matters
        2. Ignore the register/table rows if detailed sections exist
        3. Fall back to register rows only if detailed sections cannot be found
        """
        lines = [line.strip() for line in cls._split_lines(text)]
        lines = [line for line in lines if line]

        chunks: List[str] = []

        def is_noise(line: str) -> bool:
            lower = line.lower().strip()
            return lower in {
                "certificates portfolio",
                "samah fujo",
                "certificate register",
                "note",
                "document purpose: this portfolio groups the certifications currently identified from previously shared information. it can be used as a base document for cvs, portfolios, client proposals, or linkedin updates.",
            }

        def is_certificate_title(line: str) -> bool:
            """
            Detect real certificate titles, not field labels or table rows.
            """
            lower = line.lower().strip()

            if not line or is_noise(line):
                return False

            # Reject field lines
            if lower.startswith("issuer:") or lower.startswith("focus:") or lower.startswith("why it matters:"):
                return False

            # Reject register row style lines containing many separators
            if line.count("|") >= 2:
                return False

            # Good certificate title heuristics
            title_markers = [
                "master of",
                "react & django",
                "fundamentals of",
                "certificate",
                "certification",
            ]

            return any(marker in lower for marker in title_markers)

        i = 0
        while i < len(lines):
            line = lines[i]

            if not is_certificate_title(line):
                i += 1
                continue

            title = line
            issuer_line = ""
            focus_line = ""
            why_line = ""

            j = i + 1
            while j < len(lines):
                current = lines[j]
                lower = current.lower().strip()

                # Stop if next certificate starts
                if j > i + 1 and is_certificate_title(current):
                    break

                if lower.startswith("issuer:"):
                    issuer_line = current
                elif lower.startswith("focus:"):
                    focus_line = current
                elif lower.startswith("why it matters:"):
                    why_line = current

                j += 1

            # Only keep well-formed detailed certificate blocks
            if issuer_line:
                block_parts = [f"Certificate: {title}", title, issuer_line]

                if focus_line:
                    block_parts.append(focus_line)

                if why_line:
                    block_parts.append(why_line)

                block_text = "\n".join(block_parts).strip()
                chunks.append(block_text)

                i = j
            else:
                i += 1

        # If detailed sections were found, use them only
        if chunks:
            return cls._dedupe_chunks(chunks)

        # Fallback: parse certificate register rows
        fallback_chunks: List[str] = []
        for line in lines:
            # Example row:
            # 1 | Master of ChatGPT | Coursiv | 5 February 2026 | Supports ...
            if line.count("|") >= 4:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 5 and parts[0].isdigit():
                    _, cert_name, issuer, date, relevance = parts[:5]
                    fallback_chunks.append(
                        "\n".join([
                            f"Certificate: {cert_name}",
                            f"Issuer: {issuer}",
                            f"Date: {date}",
                            f"Why it matters: {relevance}",
                        ])
                    )

        return cls._dedupe_chunks(fallback_chunks) if fallback_chunks else cls.chunk_by_headings(text)

    # ---------------------------------------------------------------------
    # Heading-based docs
    # ---------------------------------------------------------------------

    @classmethod
    def chunk_by_headings(cls, text: str) -> List[str]:
        """
        Chunk documents that are mainly section-based:
        - preferences
        - compensation
        - achievements
        - career timeline
        - capabilities

        Uses generic heading detection, not CV-specific heading detection.
        """

        sections = cls._extract_generic_heading_sections(text)

        if not sections:
            return cls.chunk_generic(text)

        chunks: List[str] = []

        for section in sections:
            heading = section["heading"].strip()
            body = section["body"].strip()

            if not heading and not body:
                continue

            prefix = f"Section: {heading}"

            section_chunks = cls._chunk_long_text(
                body,
                max_chars=850,
                overlap=100,
                prefix=prefix,
            )

            chunks.extend(section_chunks)

        return cls._dedupe_chunks(chunks)

    # ---------------------------------------------------------------------
    # Resume / CV
    # ---------------------------------------------------------------------
    @classmethod
    def chunk_resume(cls, text: str) -> List[str]:
        """
        Chunk resume/CV by real resume sections.

        This version:
        1. Preserves the CV header/preamble for contact questions.
        2. Splits the CV by trusted section headings only.
        3. Prevents Education from being swallowed into Work Experience.
        4. Keeps each section independently retrievable.
        5. Splits long sections only inside that same section.
        """

        full_text = (text or "").strip()
        if not full_text:
            return []

        sections = cls._extract_heading_sections(full_text)

        if not sections:
            return cls.chunk_generic(full_text)

        chunks: List[str] = []

        # ------------------------------------------------------------
        # 1. Preserve header / preamble before the first detected section
        # Example:
        # Samah Fujo
        # +971...
        # s.fujo@hotmail.com
        # Dubai, United Arab Emirates
        # ------------------------------------------------------------
        first_heading = sections[0].get("heading") if sections else None

        if first_heading:
            lower_text = full_text.lower()
            first_pos = lower_text.find(first_heading.lower())

            if first_pos > 0:
                preamble = full_text[:first_pos].strip()

                if preamble:
                    chunks.extend(
                        cls._chunk_long_text(
                            preamble,
                            max_chars=850,
                            overlap=100,
                            prefix="Resume Header",
                        )
                    )

        # ------------------------------------------------------------
        # 2. Chunk each real resume section independently
        # ------------------------------------------------------------
        for section in sections:
            heading = (section.get("heading") or "").strip()
            body = cls._clean_resume_section_body(
                heading=heading,
                body=(section.get("body") or "").strip(),
            )

            if not heading and not body:
                continue

            section_text = body if body else heading

            section_chunks = cls._chunk_long_text(
                section_text,
                max_chars=1000,
                overlap=120,
                prefix=f"Resume Section: {heading}",
            )

            chunks.extend(section_chunks)

        return cls._dedupe_chunks(chunks)

    # ---------------------------------------------------------------------
    # Official letters
    # ---------------------------------------------------------------------

    @classmethod
    def chunk_official_letter(cls, text: str) -> List[str]:
        """
        Keep official short letters mostly intact to avoid losing important facts.
        """
        text = text.strip()
        if not text:
            return []

        if len(text) <= 1400:
            return [text]

        return cls._chunk_long_text(text, max_chars=1000, overlap=100)

    # ---------------------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------------------

    @staticmethod
    def _dedupe_chunks(chunks: List[str]) -> List[str]:
        seen = set()
        output: List[str] = []

        for chunk in chunks:
            normalized = " ".join(chunk.split()).strip().lower()
            if normalized and normalized not in seen:
                seen.add(normalized)
                output.append(chunk.strip())

        return output
