"""
Processing log service for chatbot knowledge documents.

This service creates a clean audit trail for every document processing step.
"""

import logging
from typing import Any

from core.models import ProfileDocument, DocumentProcessingLog

logger = logging.getLogger(__name__)


def create_processing_log(
    document: ProfileDocument,
    step: str,
    message: str,
    level: str = "info",
    metadata: dict[str, Any] | None = None,
) -> DocumentProcessingLog:
    """
    Create a database log entry and also write to Django logger.

    Args:
        document: ProfileDocument instance.
        step: Processing step name, for example extraction_started.
        message: Human-readable log message.
        level: info, warning, error, or critical.
        metadata: Optional structured metadata.

    Returns:
        Created DocumentProcessingLog instance.
    """

    metadata = metadata or {}

    log = DocumentProcessingLog.objects.create(
        document=document,
        step=step,
        level=level,
        message=message,
        metadata=metadata,
    )

    logger_message = f"[Document: {document.id}] [{step}] {message}"

    if level == "critical":
        logger.critical(logger_message, extra={"metadata": metadata})
    elif level == "error":
        logger.error(logger_message, extra={"metadata": metadata})
    elif level == "warning":
        logger.warning(logger_message, extra={"metadata": metadata})
    else:
        logger.info(logger_message, extra={"metadata": metadata})

    return log


def log_info(
    document: ProfileDocument,
    step: str,
    message: str,
    metadata: dict[str, Any] | None = None,
) -> DocumentProcessingLog:
    """
    Create an info-level processing log.
    """

    return create_processing_log(
        document=document,
        step=step,
        message=message,
        level="info",
        metadata=metadata,
    )


def log_warning(
    document: ProfileDocument,
    step: str,
    message: str,
    metadata: dict[str, Any] | None = None,
) -> DocumentProcessingLog:
    """
    Create a warning-level processing log.
    """

    return create_processing_log(
        document=document,
        step=step,
        message=message,
        level="warning",
        metadata=metadata,
    )


def log_error(
    document: ProfileDocument,
    step: str,
    message: str,
    metadata: dict[str, Any] | None = None,
) -> DocumentProcessingLog:
    """
    Create an error-level processing log.
    """

    return create_processing_log(
        document=document,
        step=step,
        message=message,
        level="error",
        metadata=metadata,
    )


def log_critical(
    document: ProfileDocument,
    step: str,
    message: str,
    metadata: dict[str, Any] | None = None,
) -> DocumentProcessingLog:
    """
    Create a critical-level processing log.
    """

    return create_processing_log(
        document=document,
        step=step,
        message=message,
        level="critical",
        metadata=metadata,
    )
