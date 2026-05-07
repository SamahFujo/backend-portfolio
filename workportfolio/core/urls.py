from django.urls import path

from core.views import (
    StartProjectRequestView,
    GetInTouchView,
    AskAboutMeAPIView,
    SendChatHistoryEmailAPIView,
    ProfileDocumentStatsAPIView,
    ProfileDocumentUploadAPIView,
    ProfileDocumentListAPIView,
    AdminChatSessionListAPIView,
    AdminChatSessionDetailAPIView,
    AdminChatStatsAPIView,
)

urlpatterns = [
    # Public contact APIs
    path("start-project/", StartProjectRequestView.as_view(), name="start-project"),
    path("get-in-touch/", GetInTouchView.as_view(), name="get-in-touch"),

    # Public chatbot APIs
    path("chat/ask/", AskAboutMeAPIView.as_view(), name="chat-ask"),
    path("chat/send-history/", SendChatHistoryEmailAPIView.as_view(),
    name="chat-send-history"),

    # Document APIs
    path("documents/upload/", ProfileDocumentUploadAPIView.as_view(),
    name="document-upload"),
    path("documents/", ProfileDocumentListAPIView.as_view(), name="document-list"),
    path("documents/<uuid:doc_id>/stats/",
    ProfileDocumentStatsAPIView.as_view(), name="document-stats"),

    # Admin chatbot APIs
    path("admin/chat/stats/", AdminChatStatsAPIView.as_view(),
    name="admin-chat-stats"),
    path("admin/chat/sessions/", AdminChatSessionListAPIView.as_view(),
    name="admin-chat-sessions"),
    path(
        "admin/chat/sessions/<uuid:session_id>/",
        AdminChatSessionDetailAPIView.as_view(),
        name="admin-chat-session-detail",
    ),
]
