"""
Main URL router for the core app.

This file only includes grouped URL modules.
Each feature area has its own URL file to keep the project clean and scalable.
"""

from django.urls import include, path


urlpatterns = [
    # Public portfolio website APIs
    path("site/", include("core.urls.public_site_urls")),

    # Admin website content management APIs
    path("admin/site/", include("core.urls.admin_site_urls")),

    # Public form APIs
    path("", include("core.urls.public_form_urls")),

    # Public chatbot APIs
    path("chat/", include("core.urls.public_chat_urls")),

    # Existing public/basic document APIs
    path("documents/", include("core.urls.public_document_urls")),

    # Admin chatbot/dashboard APIs
    path("admin/", include("core.urls.admin_chat_urls")),

    # Admin document quality-control APIs
    path("admin/", include("core.urls.admin_document_urls")),
]


# from django.urls import path

# from core.views import (
#     StartProjectRequestView,
#     GetInTouchView,
#     AskAboutMeAPIView,
#     SendChatHistoryEmailAPIView,
#     ProfileDocumentStatsAPIView,
#     ProfileDocumentUploadAPIView,
#     ProfileDocumentListAPIView,
#     RequestEmailVerificationAPIView,
#     VerifyEmailCodeAPIView,

#     ActiveHeroSectionAPIView,
#     AdminHeroSectionAPIView,
#     ActiveAboutSectionAPIView,
#     AdminAboutSectionAPIView,
#     ActiveSkillSectionAPIView,
#     AdminSkillSectionAPIView,
#     AdminSkillItemCreateAPIView,
#     AdminSkillItemDetailAPIView,

#     ActiveProjectSectionAPIView,
#     AdminProjectSectionAPIView,
#     AdminProjectItemCreateAPIView,
#     AdminProjectItemDetailAPIView,

#     ActiveCertificateSectionAPIView,
#     AdminCertificateSectionAPIView,
#     AdminCertificateItemCreateAPIView,
#     AdminCertificateItemDetailAPIView,

#     ActiveResearchSectionAPIView,
#     AdminResearchSectionAPIView,
#     AdminResearchItemCreateAPIView,
#     AdminResearchItemDetailAPIView,


#     ActiveFooterSectionAPIView,
#     AdminFooterSectionAPIView,
#     AdminFooterSocialLinkCreateAPIView,
#     AdminFooterSocialLinkDetailAPIView,
#     AdminFooterContactItemCreateAPIView,
#     AdminFooterContactItemDetailAPIView,

# )

# from core.views.admin_chat_views import (
#     AdminChatAnalyticsAPIView,
#     AdminChatSessionListAPIView,
#     AdminChatSessionDetailAPIView,
#     AdminChatStatsAPIView,
#     AdminLeadsAPIView,
#     AdminContactMessagesAPIView,
#     AdminContactMessageDetailAPIView,
#     AdminProjectRequestsAPIView,
#     AdminProjectRequestDetailAPIView,
#     AdminChatQualityIssuesAPIView,

#     AdminDashboardSummaryAPIView,
#     AdminNotificationBadgesAPIView,
#     AdminChatSessionExportAPIView,

#     AdminSystemHealthAPIView,
#     AdminRecentActivityAPIView,
#     AdminChatMessagesAPIView,
#     AdminChatSessionStatusUpdateAPIView,


# )


# urlpatterns = [
#     # Public and admin APIs for the portfolio site (hero section)
#     path("site/hero/", ActiveHeroSectionAPIView.as_view(), name="site-hero"),
#     path("admin/site/hero/", AdminHeroSectionAPIView.as_view(),
#          name="admin-site-hero"),

#     # Public and admin APIs for the portfolio site (about section)
#     path("site/about/", ActiveAboutSectionAPIView.as_view(), name="site-about"),
#     path("admin/site/about/", AdminAboutSectionAPIView.as_view(),
#          name="admin-site-about"),


#     # Public website Skills API
#     path("site/skills/", ActiveSkillSectionAPIView.as_view(), name="site-skills"),

#     # Admin Skills APIs
#     path("admin/site/skills/", AdminSkillSectionAPIView.as_view(),
#          name="admin-site-skills"),
#     path("admin/site/skills/items/", AdminSkillItemCreateAPIView.as_view(),
#          name="admin-site-skills-item-create"),
#     path("admin/site/skills/items/<int:item_id>/",
#          AdminSkillItemDetailAPIView.as_view(), name="admin-site-skills-item-detail"),

#     # Public website Projects API
#     path("site/projects/", ActiveProjectSectionAPIView.as_view(),
#          name="site-projects"),

#     # Admin Projects APIs
#     path("admin/site/projects/", AdminProjectSectionAPIView.as_view(),
#          name="admin-site-projects"),
#     path("admin/site/projects/items/", AdminProjectItemCreateAPIView.as_view(),
#          name="admin-site-projects-item-create"),
#     path("admin/site/projects/items/<int:item_id>/",
#          AdminProjectItemDetailAPIView.as_view(), name="admin-site-projects-item-detail"),

#     # Public website Certificates API
#     path(
#         "site/certificates/",
#         ActiveCertificateSectionAPIView.as_view(),
#         name="site-certificates",
#     ),

#     # Admin Certificates APIs
#     path(
#         "admin/site/certificates/",
#         AdminCertificateSectionAPIView.as_view(),
#         name="admin-site-certificates",
#     ),
#     path(
#         "admin/site/certificates/items/",
#         AdminCertificateItemCreateAPIView.as_view(),
#         name="admin-site-certificates-item-create",
#     ),
#     path(
#         "admin/site/certificates/items/<int:item_id>/",
#         AdminCertificateItemDetailAPIView.as_view(),
#         name="admin-site-certificates-item-detail",
#     ),

#     # Public website Research API
#     path(
#         "site/research/",
#         ActiveResearchSectionAPIView.as_view(),
#         name="site-research",
#     ),

#     # Admin Research APIs
#     path(
#         "admin/site/research/",
#         AdminResearchSectionAPIView.as_view(),
#         name="admin-site-research",
#     ),
#     path(
#         "admin/site/research/items/",
#         AdminResearchItemCreateAPIView.as_view(),
#         name="admin-site-research-item-create",
#     ),
#     path(
#         "admin/site/research/items/<int:item_id>/",
#         AdminResearchItemDetailAPIView.as_view(),
#         name="admin-site-research-item-detail",
#     ),


#     # Public footer API
#     path(
#         "site/footer/",
#         ActiveFooterSectionAPIView.as_view(),
#         name="site-footer",
#     ),

#     # Admin footer section API
#     path(
#         "admin/site/footer/",
#         AdminFooterSectionAPIView.as_view(),
#         name="admin-site-footer",
#     ),

#     # Admin footer social links API
#     path(
#         "admin/site/footer/social-links/",
#         AdminFooterSocialLinkCreateAPIView.as_view(),
#         name="admin-site-footer-social-links-create",
#     ),

#     path(
#         "admin/site/footer/social-links/<int:social_id>/",
#         AdminFooterSocialLinkDetailAPIView.as_view(),
#         name="admin-site-footer-social-links-detail",
#     ),

#     # Admin footer contact items API
#     path(
#         "admin/site/footer/contact-items/",
#         AdminFooterContactItemCreateAPIView.as_view(),
#         name="admin-site-footer-contact-items-create",
#     ),

#     path(
#         "admin/site/footer/contact-items/<int:contact_id>/",
#         AdminFooterContactItemDetailAPIView.as_view(),
#         name="admin-site-footer-contact-items-detail",
#     ),

#     # Public contact APIs
#     path("start-project/", StartProjectRequestView.as_view(), name="start-project"),
#     path("get-in-touch/", GetInTouchView.as_view(), name="get-in-touch"),

#     # Public chatbot APIs
#     path("chat/ask/", AskAboutMeAPIView.as_view(), name="chat-ask"),
#     path("chat/send-history/", SendChatHistoryEmailAPIView.as_view(),
#          name="chat-send-history"),

#     # Document APIs
#     path("documents/upload/", ProfileDocumentUploadAPIView.as_view(),
#          name="document-upload"),
#     path("documents/", ProfileDocumentListAPIView.as_view(), name="document-list"),
#     path("documents/<uuid:doc_id>/stats/",
#          ProfileDocumentStatsAPIView.as_view(), name="document-stats"),

#     # email verification for chat history export
#     path(
#         "chat/request-email-code/",
#         RequestEmailVerificationAPIView.as_view(),
#         name="chat-request-email-code",
#     ),

#     path(
#         "chat/verify-email-code/",
#         VerifyEmailCodeAPIView.as_view(),
#         name="chat-verify-email-code",
#     ),


#     # Admin chatbot APIs
#     path("admin/chat/stats/", AdminChatStatsAPIView.as_view(),
#          name="admin-chat-stats"),
#     path("admin/chat/sessions/", AdminChatSessionListAPIView.as_view(),
#          name="admin-chat-sessions"),
#     path(
#         "admin/chat/sessions/<uuid:session_id>/",
#         AdminChatSessionDetailAPIView.as_view(),
#         name="admin-chat-session-detail",
#     ),

#     path(
#         "admin/chat/analytics/",
#         AdminChatAnalyticsAPIView.as_view(),
#         name="admin-chat-analytics",
#     ),

#     path(
#         "admin/leads/",
#         AdminLeadsAPIView.as_view(),
#         name="admin-leads",
#     ),

#     path(
#         "admin/contact-messages/",
#         AdminContactMessagesAPIView.as_view(),
#         name="admin-contact-messages",
#     ),

#     path(
#         "admin/contact-messages/<uuid:message_id>/",
#         AdminContactMessageDetailAPIView.as_view(),
#         name="admin-contact-message-detail",
#     ),


#     path(
#         "admin/project-requests/",
#         AdminProjectRequestsAPIView.as_view(),
#         name="admin-project-requests",
#     ),

#     path(
#         "admin/project-requests/<uuid:request_id>/",
#         AdminProjectRequestDetailAPIView.as_view(),
#         name="admin-project-request-detail",
#     ),


#     path(
#         "admin/chat/quality-issues/",
#         AdminChatQualityIssuesAPIView.as_view(),
#         name="admin-chat-quality-issues",
#     ),


#     path(
#         "admin/dashboard/summary/",
#         AdminDashboardSummaryAPIView.as_view(),
#         name="admin-dashboard-summary",
#     ),

#     path(
#         "admin/notifications/badges/",
#         AdminNotificationBadgesAPIView.as_view(),
#         name="admin-notification-badges",
#     ),

#     path(
#         "admin/chat/sessions/<uuid:session_id>/export/",
#         AdminChatSessionExportAPIView.as_view(),
#         name="admin-chat-session-export",
#     ),

#     path(
#         "admin/system/health/",
#         AdminSystemHealthAPIView.as_view(),
#         name="admin-system-health",
#     ),

#     path(
#         "admin/recent-activity/",
#         AdminRecentActivityAPIView.as_view(),
#         name="admin-recent-activity",
#     ),


#     path(
#         "admin/chat/messages/",
#         AdminChatMessagesAPIView.as_view(),
#         name="admin-chat-messages",
#     ),

#     path(
#         "admin/chat/sessions/<uuid:session_id>/status/",
#         AdminChatSessionStatusUpdateAPIView.as_view(),
#         name="admin-chat-session-status-update",
#     ),


# ]
