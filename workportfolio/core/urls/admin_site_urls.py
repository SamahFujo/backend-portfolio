"""
Admin website content management URLs.

These endpoints are used by the custom admin panel to manage
portfolio website sections.
"""

from django.urls import path

from core.views import (
    AdminHeroSectionAPIView,
    AdminAboutSectionAPIView,
    AdminSkillSectionAPIView,
    AdminSkillItemCreateAPIView,
    AdminSkillItemDetailAPIView,
    AdminProjectSectionAPIView,
    AdminProjectItemCreateAPIView,
    AdminProjectItemDetailAPIView,
    AdminCertificateSectionAPIView,
    AdminCertificateItemCreateAPIView,
    AdminCertificateItemDetailAPIView,
    AdminResearchSectionAPIView,
    AdminResearchItemCreateAPIView,
    AdminResearchItemDetailAPIView,
    AdminFooterSectionAPIView,
    AdminFooterSocialLinkCreateAPIView,
    AdminFooterSocialLinkDetailAPIView,
    AdminFooterContactItemCreateAPIView,
    AdminFooterContactItemDetailAPIView,
)


urlpatterns = [
    # Hero
    path("hero/", AdminHeroSectionAPIView.as_view(), name="admin-site-hero"),

    # About
    path("about/", AdminAboutSectionAPIView.as_view(), name="admin-site-about"),

    # Skills
    path("skills/", AdminSkillSectionAPIView.as_view(), name="admin-site-skills"),
    path("skills/items/", AdminSkillItemCreateAPIView.as_view(),
         name="admin-site-skills-item-create"),
    path("skills/items/<int:item_id>/", AdminSkillItemDetailAPIView.as_view(),
         name="admin-site-skills-item-detail"),

    # Projects
    path("projects/", AdminProjectSectionAPIView.as_view(),
         name="admin-site-projects"),
    path("projects/items/", AdminProjectItemCreateAPIView.as_view(),
         name="admin-site-projects-item-create"),
    path("projects/items/<int:item_id>/", AdminProjectItemDetailAPIView.as_view(),
         name="admin-site-projects-item-detail"),

    # Certificates
    path("certificates/", AdminCertificateSectionAPIView.as_view(),
         name="admin-site-certificates"),
    path("certificates/items/", AdminCertificateItemCreateAPIView.as_view(),
         name="admin-site-certificates-item-create"),
    path("certificates/items/<int:item_id>/", AdminCertificateItemDetailAPIView.as_view(),
         name="admin-site-certificates-item-detail"),

    # Research
    path("research/", AdminResearchSectionAPIView.as_view(),
         name="admin-site-research"),
    path("research/items/", AdminResearchItemCreateAPIView.as_view(),
         name="admin-site-research-item-create"),
    path("research/items/<int:item_id>/", AdminResearchItemDetailAPIView.as_view(),
         name="admin-site-research-item-detail"),

    # Footer
    path("footer/", AdminFooterSectionAPIView.as_view(), name="admin-site-footer"),
    path("footer/social-links/", AdminFooterSocialLinkCreateAPIView.as_view(),
         name="admin-site-footer-social-links-create"),
    path("footer/social-links/<int:social_id>/", AdminFooterSocialLinkDetailAPIView.as_view(),
         name="admin-site-footer-social-links-detail"),
    path("footer/contact-items/", AdminFooterContactItemCreateAPIView.as_view(),
         name="admin-site-footer-contact-items-create"),
    path("footer/contact-items/<int:contact_id>/", AdminFooterContactItemDetailAPIView.as_view(),
         name="admin-site-footer-contact-items-detail"),
]
