from django.urls import path

from core.views import (
    StartProjectRequestView,
    GetInTouchView,
    AskAboutMeAPIView,
    SendChatHistoryEmailAPIView,
    ProfileDocumentStatsAPIView,
    ProfileDocumentUploadAPIView,
    ProfileDocumentListAPIView,
    RequestEmailVerificationAPIView,
    VerifyEmailCodeAPIView,
)

from core.views.admin_chat_views import (
    AdminChatAnalyticsAPIView,
    AdminChatSessionListAPIView,
    AdminChatSessionDetailAPIView,
    AdminChatStatsAPIView,
    AdminLeadsAPIView,
    AdminContactMessagesAPIView,
    AdminContactMessageDetailAPIView,
    AdminProjectRequestsAPIView,
    AdminProjectRequestDetailAPIView,
    AdminChatQualityIssuesAPIView,

    AdminDashboardSummaryAPIView,
    AdminNotificationBadgesAPIView,
    AdminChatSessionExportAPIView,

    AdminSystemHealthAPIView,
    AdminRecentActivityAPIView,


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

    # email verification for chat history export
    path(
        "chat/request-email-code/",
        RequestEmailVerificationAPIView.as_view(),
        name="chat-request-email-code",
    ),

    path(
        "chat/verify-email-code/",
        VerifyEmailCodeAPIView.as_view(),
        name="chat-verify-email-code",
    ),


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

    path(
        "admin/chat/analytics/",
        AdminChatAnalyticsAPIView.as_view(),
        name="admin-chat-analytics",
    ),

    path(
        "admin/leads/",
        AdminLeadsAPIView.as_view(),
        name="admin-leads",
    ),

    path(
        "admin/contact-messages/",
        AdminContactMessagesAPIView.as_view(),
        name="admin-contact-messages",
    ),

    path(
        "admin/contact-messages/<uuid:message_id>/",
        AdminContactMessageDetailAPIView.as_view(),
        name="admin-contact-message-detail",
    ),


    path(
        "admin/project-requests/",
        AdminProjectRequestsAPIView.as_view(),
        name="admin-project-requests",
    ),

    path(
        "admin/project-requests/<uuid:request_id>/",
        AdminProjectRequestDetailAPIView.as_view(),
        name="admin-project-request-detail",
    ),


    path(
        "admin/chat/quality-issues/",
        AdminChatQualityIssuesAPIView.as_view(),
        name="admin-chat-quality-issues",
    ),


    path(
        "admin/dashboard/summary/",
        AdminDashboardSummaryAPIView.as_view(),
        name="admin-dashboard-summary",
    ),

    path(
        "admin/notifications/badges/",
        AdminNotificationBadgesAPIView.as_view(),
        name="admin-notification-badges",
    ),

    path(
        "admin/chat/sessions/<uuid:session_id>/export/",
        AdminChatSessionExportAPIView.as_view(),
        name="admin-chat-session-export",
    ),

    path(
        "admin/system/health/",
        AdminSystemHealthAPIView.as_view(),
        name="admin-system-health",
    ),

    path(
        "admin/recent-activity/",
        AdminRecentActivityAPIView.as_view(),
        name="admin-recent-activity",
    ),

]
