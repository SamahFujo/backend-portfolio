"""
Public website URLs.

These endpoints are used by the public portfolio frontend.
"""

from django.urls import path

from core.views import (
    ActiveHeroSectionAPIView,
    ActiveAboutSectionAPIView,
    ActiveSkillSectionAPIView,
    ActiveProjectSectionAPIView,
    ActiveCertificateSectionAPIView,
    ActiveResearchSectionAPIView,
    ActiveFooterSectionAPIView,
    WebsiteVisitTrackAPIView,
)


urlpatterns = [
    path("hero/", ActiveHeroSectionAPIView.as_view(), name="site-hero"),
    path("about/", ActiveAboutSectionAPIView.as_view(), name="site-about"),
    path("skills/", ActiveSkillSectionAPIView.as_view(), name="site-skills"),
    path("projects/", ActiveProjectSectionAPIView.as_view(), name="site-projects"),
    path("certificates/", ActiveCertificateSectionAPIView.as_view(),
         name="site-certificates"),
    path("research/", ActiveResearchSectionAPIView.as_view(), name="site-research"),
    path("footer/", ActiveFooterSectionAPIView.as_view(), name="site-footer"),
    path("analytics/track-visit/", WebsiteVisitTrackAPIView.as_view(),
         name="site-track-visit"),
]
