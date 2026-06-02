"""
Embedding quality service.

This service validates whether document chunks have usable embeddings.
"""

from typing import Any

from core.models import ProfileDocument, DocumentQualityCheck
from core.services.knowledge_quality.processing_log_service import (
    log_info,
    log_warning,
)


class EmbeddingQualityService:
    """
    Validates embedding completion and consistency.
    """
    
    EXPECTED_EMBEDDING_MODEL = "jina-embeddings-v3"
    EXPECTED_EMBEDDING_DIMENSION = 1024

    MIN_COMPLETION_RATE = 95

    def __init__(self, document: ProfileDocument):
        self.document = document

    def run_all_checks(self) -> dict[str, Any]:
        """
        Run embedding checks for all chunks in a document.
        """

        log_info(
            document=self.document,
            step="embedding_validation_started",
            message="Embedding quality validation started.",
        )

        chunks = self.document.chunks.all()
        total_chunks = chunks.count()
        
        

        if total_chunks == 0:
            DocumentQualityCheck.objects.create(
                document=self.document,
                check_name="embedding_no_chunks",
                check_status="failed",
                severity="critical",
                message="No chunks found. Embedding validation cannot run.",
                details={},
            )

            self.document.embedding_quality_score = 0
            self.document.status = "embedding_failed"
            self.document.save(
                update_fields=["embedding_quality_score",
                               "status", "updated_at"]
            )

            return {
                "score": 0,
                "passed": False,
                "total_chunks": 0,
                "embedded_chunks": 0,
                "missing_embeddings": 0,
                "issues": [
                    {
                        "code": "embedding_no_chunks",
                        "message": "No chunks found for embedding validation.",
                        "severity": "critical",
                    }
                ],
            }

        embedded_chunks = chunks.filter(has_embedding=True).count()
        missing_embeddings = total_chunks - embedded_chunks
        completion_rate = round((embedded_chunks / total_chunks) * 100, 2)

        issues = []

        if missing_embeddings > 0:
            severity = "critical" if missing_embeddings == total_chunks else "warning"

            issues.append({
                "code": "missing_embeddings",
                "message": f"{missing_embeddings} chunks are missing embeddings.",
                "severity": severity,
                "details": {
                    "missing_embeddings": missing_embeddings,
                    "total_chunks": total_chunks,
                },
            })

            DocumentQualityCheck.objects.create(
                document=self.document,
                check_name="missing_embeddings",
                check_status="failed" if severity == "critical" else "warning",
                severity=severity,
                message=f"{missing_embeddings} chunks are missing embeddings.",
                details={
                    "missing_embeddings": missing_embeddings,
                    "total_chunks": total_chunks,
                },
            )
        else:
            DocumentQualityCheck.objects.create(
                document=self.document,
                check_name="all_chunks_embedded",
                check_status="passed",
                severity="info",
                message="All chunks have embeddings.",
                details={
                    "embedded_chunks": embedded_chunks,
                    "total_chunks": total_chunks,
                },
            )

        dimension_issues = self.check_embedding_dimensions()
        issues.extend(dimension_issues)

        model_issues = self.check_embedding_models()
        issues.extend(model_issues)

        score = completion_rate

        if dimension_issues:
            score -= 20

        if model_issues:
            score -= 10

        score = max(round(score, 2), 0)

        has_critical_issue = any(
            issue["severity"] == "critical" for issue in issues)
        passed = score >= self.MIN_COMPLETION_RATE and not has_critical_issue

        self.document.embedding_quality_score = score
        self.document.status = "embedded" if passed else "embedding_failed"
        self.document.save(
            update_fields=["embedding_quality_score", "status", "updated_at"]
        )

        summary = {
            "score": score,
            "passed": passed,
            "total_chunks": total_chunks,
            "embedded_chunks": embedded_chunks,
            "missing_embeddings": missing_embeddings,
            "completion_rate": completion_rate,
            "issues": issues,
        }

        if passed:
            log_info(
                document=self.document,
                step="embedding_validation_completed",
                message="Embedding validation completed successfully.",
                metadata=summary,
            )
        else:
            log_warning(
                document=self.document,
                step="embedding_validation_completed_with_issues",
                message="Embedding validation completed with issues.",
                metadata=summary,
            )

        return summary

    def check_embedding_dimensions(self) -> list[dict[str, Any]]:
        """
        Check whether embedding dimensions are available, consistent,
        and aligned with the expected embedding dimension.
        """

        chunks = self.document.chunks.filter(has_embedding=True)

        if not chunks.exists():
            issue = {
                "code": "no_embedded_chunks_for_dimension_check",
                "message": "No embedded chunks found for embedding dimension validation.",
                "severity": "critical",
                "details": {},
            }

            DocumentQualityCheck.objects.create(
                document=self.document,
                check_name="no_embedded_chunks_for_dimension_check",
                check_status="failed",
                severity="critical",
                message=issue["message"],
                details=issue["details"],
            )

            return [issue]

        dimensions = set(
            chunks.exclude(embedding_dimension__isnull=True)
            .exclude(embedding_dimension=0)
            .values_list("embedding_dimension", flat=True)
        )

        if not dimensions:
            issue = {
                "code": "missing_embedding_dimensions",
                "message": "Embedded chunks do not have stored embedding dimensions.",
                "severity": "warning",
                "details": {},
            }

            DocumentQualityCheck.objects.create(
                document=self.document,
                check_name="missing_embedding_dimensions",
                check_status="warning",
                severity="warning",
                message=issue["message"],
                details=issue["details"],
            )

            return [issue]

        if len(dimensions) > 1:
            issue = {
                "code": "inconsistent_embedding_dimensions",
                "message": "Chunks have inconsistent embedding dimensions.",
                "severity": "critical",
                "details": {
                    "dimensions": sorted(list(dimensions)),
                },
            }

            DocumentQualityCheck.objects.create(
                document=self.document,
                check_name="inconsistent_embedding_dimensions",
                check_status="failed",
                severity="critical",
                message=issue["message"],
                details=issue["details"],
            )

            return [issue]

        dimension = list(dimensions)[0]

        if dimension != self.EXPECTED_EMBEDDING_DIMENSION:
            issue = {
                "code": "unexpected_embedding_dimension",
                "message": (
                    f"Expected embedding dimension {self.EXPECTED_EMBEDDING_DIMENSION}, "
                    f"but found {dimension}."
                ),
                "severity": "warning",
                "details": {
                    "expected_dimension": self.EXPECTED_EMBEDDING_DIMENSION,
                    "actual_dimension": dimension,
                },
            }

            DocumentQualityCheck.objects.create(
                document=self.document,
                check_name="unexpected_embedding_dimension",
                check_status="warning",
                severity="warning",
                message=issue["message"],
                details=issue["details"],
            )

            return [issue]

        DocumentQualityCheck.objects.create(
            document=self.document,
            check_name="embedding_dimensions_valid",
            check_status="passed",
            severity="info",
            message="Embedding dimensions are valid and consistent.",
            details={
                "dimension": dimension,
                "expected_dimension": self.EXPECTED_EMBEDDING_DIMENSION,
            },
        )

        return []

    def check_embedding_models(self) -> list[dict[str, Any]]:
        """
        Check whether embedding model values are consistent.
        """

        chunks = self.document.chunks.filter(has_embedding=True)
        
        embedded_chunks = self.document.chunks.filter(has_embedding=True)
        missing_model_count = embedded_chunks.filter(embedding_model="").count()

        if missing_model_count > 0:
            DocumentQualityCheck.objects.create(
                document=self.document,
                check_name="missing_embedding_model_name",
                check_status="warning",
                severity="warning",
                message=f"{missing_model_count} embedded chunks do not store the embedding model name.",
                details={"missing_model_count": missing_model_count},
            )

        models = set(
            model for model in chunks.values_list("embedding_model", flat=True) if model
        )

        if len(models) > 1:
            issue = {
                "code": "inconsistent_embedding_models",
                "message": "Chunks were embedded using different embedding models.",
                "severity": "warning",
                "details": {
                    "models": list(models),
                },
            }

            DocumentQualityCheck.objects.create(
                document=self.document,
                check_name="inconsistent_embedding_models",
                check_status="warning",
                severity="warning",
                message="Chunks were embedded using different embedding models.",
                details={"models": list(models)},
            )

            return [issue]

        DocumentQualityCheck.objects.create(
            document=self.document,
            check_name="embedding_models_valid",
            check_status="passed",
            severity="info",
            message="Embedding model values are consistent.",
            details={"models": list(models)},
        )

        return []
