"""
Admin chatbot, dashboard, leads, and system URLs.
"""

from django.urls import path

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
    AdminChatMessagesAPIView,
    AdminChatSessionStatusUpdateAPIView,
)


urlpatterns = [
    path("chat/stats/", AdminChatStatsAPIView.as_view(), name="admin-chat-stats"),
    path("chat/sessions/", AdminChatSessionListAPIView.as_view(),
         name="admin-chat-sessions"),
    path("chat/sessions/<uuid:session_id>/",
         AdminChatSessionDetailAPIView.as_view(), name="admin-chat-session-detail"),
    path("chat/sessions/<uuid:session_id>/export/",
         AdminChatSessionExportAPIView.as_view(), name="admin-chat-session-export"),
    path("chat/sessions/<uuid:session_id>/status/",
         AdminChatSessionStatusUpdateAPIView.as_view(), name="admin-chat-session-status-update"),

    path("chat/analytics/", AdminChatAnalyticsAPIView.as_view(),
         name="admin-chat-analytics"),
    path("chat/messages/", AdminChatMessagesAPIView.as_view(),
         name="admin-chat-messages"),
    path("chat/quality-issues/", AdminChatQualityIssuesAPIView.as_view(),
         name="admin-chat-quality-issues"),

    path("leads/", AdminLeadsAPIView.as_view(), name="admin-leads"),

    path("contact-messages/", AdminContactMessagesAPIView.as_view(),
         name="admin-contact-messages"),
    path("contact-messages/<uuid:message_id>/",
         AdminContactMessageDetailAPIView.as_view(), name="admin-contact-message-detail"),

    path("project-requests/", AdminProjectRequestsAPIView.as_view(),
         name="admin-project-requests"),
    path("project-requests/<uuid:request_id>/",
         AdminProjectRequestDetailAPIView.as_view(), name="admin-project-request-detail"),

    path("dashboard/summary/", AdminDashboardSummaryAPIView.as_view(),
         name="admin-dashboard-summary"),
    path("notifications/badges/", AdminNotificationBadgesAPIView.as_view(),
         name="admin-notification-badges"),
    path("system/health/", AdminSystemHealthAPIView.as_view(),
         name="admin-system-health"),
    path("recent-activity/", AdminRecentActivityAPIView.as_view(),
         name="admin-recent-activity"),
]
