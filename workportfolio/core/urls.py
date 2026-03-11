from django.urls import path
from .views import AskAboutMeAPIView, ProfileDocumentListAPIView, ProfileDocumentStatsAPIView, ProfileDocumentUploadAPIView, StartProjectRequestView

urlpatterns = [
    path("contact/start-project/", StartProjectRequestView.as_view(),
        name="start-project-request"),
    path("chat/ask/", AskAboutMeAPIView.as_view(), name="chat-ask"),
    path("documents/upload/", ProfileDocumentUploadAPIView.as_view(),
        name="documents-upload"),
    path("documents/", ProfileDocumentListAPIView.as_view(), name="documents-list"),
    path("documents/<uuid:doc_id>/stats/", ProfileDocumentStatsAPIView.as_view(), name="documents-stats"),
]
