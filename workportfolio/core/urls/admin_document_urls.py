"""
Admin document quality-control URLs.
"""

from django.urls import path

from core.views.admin_document_views import (
    AdminDocumentListCreateAPIView,
    AdminDocumentDetailAPIView,
    AdminDocumentProcessAPIView,
    AdminDocumentReprocessAPIView,
    AdminDocumentInspectionAPIView,
    AdminDocumentChunksAPIView,
    AdminDocumentChunkDetailAPIView,
    AdminDocumentQualityChecksAPIView,
    AdminDocumentLogsAPIView,
    AdminDocumentApproveAPIView,
    AdminDocumentRejectAPIView,
    AdminDocumentArchiveAPIView,
    AdminDocumentReplaceAPIView,
)


urlpatterns = [
    path("documents/", AdminDocumentListCreateAPIView.as_view(),
         name="admin-documents-list-create"),
    path("documents/<uuid:document_id>/",
         AdminDocumentDetailAPIView.as_view(), name="admin-documents-detail"),

    path("documents/<uuid:document_id>/process/",
         AdminDocumentProcessAPIView.as_view(), name="admin-documents-process"),
    path("documents/<uuid:document_id>/reprocess/",
         AdminDocumentReprocessAPIView.as_view(), name="admin-documents-reprocess"),

    path("documents/<uuid:document_id>/inspection/",
         AdminDocumentInspectionAPIView.as_view(), name="admin-documents-inspection"),
    path("documents/<uuid:document_id>/chunks/",
         AdminDocumentChunksAPIView.as_view(), name="admin-documents-chunks"),
    path("documents/<uuid:document_id>/quality-checks/",
         AdminDocumentQualityChecksAPIView.as_view(), name="admin-documents-quality-checks"),
    path("documents/<uuid:document_id>/logs/",
         AdminDocumentLogsAPIView.as_view(), name="admin-documents-logs"),

    path("documents/<uuid:document_id>/approve/",
         AdminDocumentApproveAPIView.as_view(), name="admin-documents-approve"),
    path("documents/<uuid:document_id>/reject/",
         AdminDocumentRejectAPIView.as_view(), name="admin-documents-reject"),
    path("documents/<uuid:document_id>/archive/",
         AdminDocumentArchiveAPIView.as_view(), name="admin-documents-archive"),
    path("documents/<uuid:document_id>/replace/",
         AdminDocumentReplaceAPIView.as_view(), name="admin-documents-replace"),

    path("document-chunks/<uuid:chunk_id>/",
         AdminDocumentChunkDetailAPIView.as_view(), name="admin-document-chunk-detail"),
]
