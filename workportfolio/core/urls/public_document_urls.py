"""
Basic document URLs.

These are the existing document upload/list/stats APIs.
The new admin quality-control document APIs are separated in admin_document_urls.py.
"""

from django.urls import path

from core.views import (
    ProfileDocumentStatsAPIView,
    ProfileDocumentUploadAPIView,
    ProfileDocumentListAPIView,
)


urlpatterns = [
    path("upload/", ProfileDocumentUploadAPIView.as_view(), name="document-upload"),
    path("", ProfileDocumentListAPIView.as_view(), name="document-list"),
    path("<uuid:doc_id>/stats/",
         ProfileDocumentStatsAPIView.as_view(), name="document-stats"),
]
