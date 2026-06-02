"""
Document quality service.

This service deeply inspects extracted document text before the document
is allowed to move to chunking, embedding, or approval.
"""

import re
from collections import Counter
from typing import Any

from django.utils import timezone

from core.models import ProfileDocument, DocumentQualityCheck
from core.services.knowledge_quality.processing_log_service import (
    log_info,
    log_warning,
    log_error,
)


class DocumentQualityService:
    """
    Runs document-level quality checks.

    Checks include:
    - Extracted text exists
    - Text length is acceptable
    - Word count is acceptable
    - Symbol ratio is not too high
    - Repeated lines are not excessive
    - OCR garbage patterns are not present
    - Duplicate content by file hash
    """

    MIN_TEXT_LENGTH = 300
    MIN_WORD_COUNT = 80
    MAX_SYMBOL_RATIO = 0.35
    MAX_REPEATED_LINE_RATIO = 0.30

    def __init__(self, document: ProfileDocument):
        self.document = document
        self.issues: list[dict[str, Any]] = []
        self.score = 100

    def run_all_checks(self) -> dict[str, Any]:
        """
        Run all quality checks and save the result into the document.
        """

        log_info(
            document=self.document,
            step="document_validation_started",
            message="Document quality validation started.",
        )

        self.document.quality_checks.all().delete()

        text = self.document.raw_text or ""

        self.check_text_exists(text)
        self.check_text_length(text)
        self.check_word_count(text)
        self.check_symbol_ratio(text)
        self.check_repeated_lines(text)
        self.check_garbage_patterns(text)
        self.check_duplicate_file_hash()
        self.check_document_type_valid()
        self.check_tags_valid()

        final_score = max(self.score, 0)
        has_critical_issue = self.has_critical_issue()

        if has_critical_issue:
            quality_status = "failed"
            status = "validation_failed"
        elif self.issues:
            quality_status = "warning"
            status = "validation_warning"
        else:
            quality_status = "passed"
            status = "validating"

        passed = final_score >= 75 and not has_critical_issue

        validation_summary = {
            "score": final_score,
            "passed": passed,
            "issues": self.issues,
            "total_issues": len(self.issues),
            "critical_issues": len(
                [issue for issue in self.issues if issue["severity"] == "critical"]
            ),
            "warning_issues": len(
                [issue for issue in self.issues if issue["severity"] == "warning"]
            ),
            "validated_at": timezone.now().isoformat(),
        }

        self.document.extraction_score = final_score
        self.document.quality_status = quality_status
        self.document.status = "validation_failed" if not passed else status
        self.document.validation_summary = validation_summary
        self.document.extracted_text_preview = text[:1000] if text else ""
        self.document.save(
            update_fields=[
                "extraction_score",
                "quality_status",
                "status",
                "validation_summary",
                "extracted_text_preview",
                "updated_at",
            ]
        )

        if passed:
            log_info(
                document=self.document,
                step="document_validation_completed",
                message="Document quality validation completed successfully.",
                metadata=validation_summary,
            )
        else:
            log_warning(
                document=self.document,
                step="document_validation_completed_with_issues",
                message="Document validation completed with quality issues.",
                metadata=validation_summary,
            )

        return validation_summary

    def add_issue(
        self,
        code: str,
        message: str,
        severity: str = "warning",
        penalty: int = 5,
        details: dict[str, Any] | None = None,
    ) -> None:
        """
        Add quality issue and create DocumentQualityCheck record.
        """

        details = details or {}

        issue = {
            "code": code,
            "message": message,
            "severity": severity,
            "details": details,
        }

        self.issues.append(issue)
        self.score -= penalty

        check_status = "failed" if severity in [
            "error", "critical"] else "warning"

        DocumentQualityCheck.objects.create(
            document=self.document,
            check_name=code,
            check_status=check_status,
            severity=severity,
            message=message,
            details=details,
        )

    def add_passed_check(
        self,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """
        Store a passed quality check.
        """

        DocumentQualityCheck.objects.create(
            document=self.document,
            check_name=code,
            check_status="passed",
            severity="info",
            message=message,
            details=details or {},
        )

    def has_critical_issue(self) -> bool:
        """
        Return True if any critical issue exists.
        """

        return any(issue["severity"] == "critical" for issue in self.issues)

    def check_text_exists(self, text: str) -> None:
        """
        Check whether extracted text exists.
        """

        if not text.strip():
            self.add_issue(
                code="empty_extracted_text",
                message="No text was extracted from the document.",
                severity="critical",
                penalty=100,
            )
            return

        self.add_passed_check(
            code="text_exists",
            message="Extracted text exists.",
            details={"text_length": len(text)},
        )

    def check_text_length(self, text: str) -> None:
        """
        Check whether extracted text length is acceptable.
        """

        text_length = len(text.strip())

        if text_length < self.MIN_TEXT_LENGTH:
            self.add_issue(
                code="short_extracted_text",
                message="Extracted text is too short. The document may not have been read correctly.",
                severity="critical",
                penalty=40,
                details={
                    "text_length": text_length,
                    "minimum_required": self.MIN_TEXT_LENGTH,
                },
            )
            return

        self.add_passed_check(
            code="text_length_valid",
            message="Extracted text length is acceptable.",
            details={"text_length": text_length},
        )

    def check_word_count(self, text: str) -> None:
        """
        Check whether extracted text has enough meaningful words.
        """

        words = re.findall(r"\b\w+\b", text)
        word_count = len(words)

        if word_count < self.MIN_WORD_COUNT:
            self.add_issue(
                code="low_word_count",
                message="Extracted text has too few meaningful words.",
                severity="critical",
                penalty=35,
                details={
                    "word_count": word_count,
                    "minimum_required": self.MIN_WORD_COUNT,
                },
            )
            return

        self.add_passed_check(
            code="word_count_valid",
            message="Extracted text has enough meaningful words.",
            details={"word_count": word_count},
        )

    def check_symbol_ratio(self, text: str) -> None:
        """
        Check whether text contains too many symbols or unreadable characters.
        """

        if not text:
            return

        symbols = re.findall(r"[^a-zA-Z0-9\s.,;:!?()\-_/]", text)
        symbol_ratio = len(symbols) / max(len(text), 1)

        if symbol_ratio > self.MAX_SYMBOL_RATIO:
            self.add_issue(
                code="high_symbol_ratio",
                message="Extracted text contains too many symbols or unreadable characters.",
                severity="critical",
                penalty=40,
                details={
                    "symbol_count": len(symbols),
                    "symbol_ratio": round(symbol_ratio, 4),
                    "maximum_allowed": self.MAX_SYMBOL_RATIO,
                },
            )
            return

        self.add_passed_check(
            code="symbol_ratio_valid",
            message="Symbol ratio is acceptable.",
            details={
                "symbol_count": len(symbols),
                "symbol_ratio": round(symbol_ratio, 4),
            },
        )

    def check_repeated_lines(self, text: str) -> None:
        """
        Check whether text contains too many repeated lines.
        """

        lines = [line.strip() for line in text.splitlines() if line.strip()]

        if len(lines) < 5:
            return

        counts = Counter(lines)
        repeated_lines = {
            line: count for line, count in counts.items() if count > 1
        }

        repeated_count = sum(repeated_lines.values())
        repeated_ratio = repeated_count / max(len(lines), 1)

        if repeated_ratio > self.MAX_REPEATED_LINE_RATIO:
            self.add_issue(
                code="repeated_lines",
                message="Extracted text contains too many repeated lines.",
                severity="warning",
                penalty=20,
                details={
                    "line_count": len(lines),
                    "repeated_count": repeated_count,
                    "repeated_ratio": round(repeated_ratio, 4),
                    "sample_repeated_lines": list(repeated_lines.keys())[:10],
                },
            )
            return

        self.add_passed_check(
            code="repeated_lines_valid",
            message="Repeated line ratio is acceptable.",
            details={
                "line_count": len(lines),
                "repeated_ratio": round(repeated_ratio, 4),
            },
        )

    def check_garbage_patterns(self, text: str) -> None:
        """
        Detect common OCR/extraction garbage patterns.
        """

        garbage_patterns = [
            r"(.)\1{15,}",
            r"[|]{5,}",
            r"[_]{10,}",
            r"[�]{2,}",
            r"\b[a-zA-Z]\s[a-zA-Z]\s[a-zA-Z]\s[a-zA-Z]\b",
        ]

        detected_patterns = []

        for pattern in garbage_patterns:
            if re.search(pattern, text):
                detected_patterns.append(pattern)

        if detected_patterns:
            self.add_issue(
                code="ocr_or_extraction_garbage",
                message="Extracted text appears to contain OCR or extraction garbage.",
                severity="warning",
                penalty=15,
                details={"patterns": detected_patterns},
            )
            return

        self.add_passed_check(
            code="garbage_patterns_valid",
            message="No obvious OCR garbage patterns detected.",
        )

    def check_duplicate_file_hash(self) -> None:
        """
        Check if another active/non-archived document has the same file hash.
        """

        if not self.document.file_hash:
            return

        duplicate_exists = ProfileDocument.objects.filter(
            file_hash=self.document.file_hash
        ).exclude(
            pk=self.document.pk
        ).exclude(
            status__in=["rejected", "archived"]
        ).exists()

        if duplicate_exists:
            self.add_issue(
                code="duplicate_file_hash",
                message="Another active or pending document has the same file hash.",
                severity="critical",
                penalty=50,
                details={"file_hash": self.document.file_hash},
            )
            return

        self.add_passed_check(
            code="duplicate_file_hash_valid",
            message="No duplicate document file hash detected.",
        )
        
        
        
        
        
    def check_document_type_valid(self) -> None:
        """
        Check whether the document type is one of the supported chatbot knowledge types.

        This should align with DocumentTypeClassifier.ALLOWED_TYPES.
        """

        allowed_types = {
            "cv",
            "projects",
            "certificates",
            "recommendation",
            "experience_letter",
            "capabilities",
            "security_deployment",
            "preferences",
            "compensation",
            "faq",
            "achievements",
            "career_timeline",
            "other",
        }

        document_type = (self.document.document_type or "").strip().lower()

        if not document_type:
            self.add_issue(
                code="missing_document_type",
                message="Document type was not detected.",
                severity="warning",
                penalty=10,
            )
            return

        if document_type not in allowed_types:
            self.add_issue(
                code="unsupported_document_type",
                message=f"Unsupported document type detected: {document_type}.",
                severity="warning",
                penalty=10,
                details={
                    "document_type": document_type,
                    "allowed_types": sorted(allowed_types),
                },
            )
            return

        self.add_passed_check(
            code="document_type_valid",
            message="Document type is supported.",
            details={"document_type": document_type},
        )


    def check_tags_valid(self) -> None:
        """
        Validate tags generated by the classifier.
        """

        tags = self.document.tags or []

        if tags is None:
            tags = []

        if not isinstance(tags, list):
            self.add_issue(
                code="invalid_tags_format",
                message="Document tags must be stored as a list.",
                severity="warning",
                penalty=10,
                details={"tags": tags},
            )
            return

        cleaned_tags = []

        for tag in tags:
            if not isinstance(tag, str):
                self.add_issue(
                    code="invalid_tag_value",
                    message="One or more tags are not valid text values.",
                    severity="warning",
                    penalty=5,
                    details={"tag": str(tag)},
                )
                continue

            clean_tag = tag.strip().lower()

            if clean_tag:
                cleaned_tags.append(clean_tag)

        if not cleaned_tags:
            self.add_issue(
                code="missing_tags",
                message="No useful tags were generated for this document.",
                severity="warning",
                penalty=5,
            )
            return

        self.add_passed_check(
            code="tags_valid",
            message="Document tags are valid.",
            details={"tags": cleaned_tags},
        )
