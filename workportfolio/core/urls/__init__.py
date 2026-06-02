"""
Main URL router for the core app.

This package-level URL file includes grouped URL modules.
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

    # Existing basic document APIs
    path("documents/", include("core.urls.public_document_urls")),

    # Admin chatbot/dashboard APIs
    path("admin/", include("core.urls.admin_chat_urls")),

    # Admin document quality-control APIs
    path("admin/", include("core.urls.admin_document_urls")),
]
