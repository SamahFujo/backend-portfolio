from core.serializers import AboutSectionSerializer

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.permissions import AllowAny

from core.models import (
    HeroSection,
    AboutSection,
    SkillSection,
    ProjectSection,
    CertificateSection,
    ResearchSection,
    FooterSection,
    WebsiteVisit,
)


from core.serializers import (
    HeroSectionSerializer,
    AboutSectionSerializer,
    SkillSectionSerializer,
    ProjectSectionSerializer,
    CertificateSectionSerializer,
    ResearchSectionSerializer,
    FooterSectionSerializer,
    WebsiteVisitTrackSerializer,
)


def _get_client_ip(request):
    x_forwarded_for = request.META.get("HTTP_X_FORWARDED_FOR")
    if x_forwarded_for:
        return x_forwarded_for.split(",")[0].strip()

    return request.META.get("REMOTE_ADDR")


class ActiveHeroSectionAPIView(APIView):
    """
    Public API used by the website homepage to display the active Hero section.
    """

    authentication_classes = []
    permission_classes = []
    throttle_classes = []

    def get(self, request, *args, **kwargs):
        hero = HeroSection.objects.filter(
            is_active=True).order_by("-updated_at").first()

        if not hero:
            return Response(
                {"detail": "No active hero section found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        serializer = HeroSectionSerializer(hero, context={"request": request})
        return Response(serializer.data, status=status.HTTP_200_OK)


class ActiveAboutSectionAPIView(APIView):
    """
    Public API used by the website frontend to display
    the active About Me section.
    """

    authentication_classes = []
    permission_classes = []
    throttle_classes = []

    def get(self, request, *args, **kwargs):
        about = AboutSection.objects.filter(
            is_active=True).order_by("-updated_at").first()

        if not about:
            return Response(
                {"detail": "No active about section found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        serializer = AboutSectionSerializer(
            about, context={"request": request})
        return Response(serializer.data, status=status.HTTP_200_OK)


class ActiveSkillSectionAPIView(APIView):
    """
    Public API used by the website frontend to display
    the active Skills section and active skill items.
    """

    authentication_classes = []
    permission_classes = []
    throttle_classes = []

    def get(self, request, *args, **kwargs):
        skill_section = SkillSection.objects.filter(
            is_active=True).order_by("-updated_at").first()

        if not skill_section:
            return Response(
                {"detail": "No active skills section found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        serializer = SkillSectionSerializer(
            skill_section, context={"request": request})
        return Response(serializer.data, status=status.HTTP_200_OK)


class ActiveProjectSectionAPIView(APIView):
    """
    Public API used by the website frontend to display
    the active Projects section and active featured project items.
    """

    authentication_classes = []
    permission_classes = []
    throttle_classes = []

    def get(self, request, *args, **kwargs):
        project_section = ProjectSection.objects.filter(
            is_active=True
        ).order_by("-updated_at").first()

        if not project_section:
            return Response(
                {"detail": "No active projects section found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        serializer = ProjectSectionSerializer(
            project_section,
            context={"request": request},
        )

        return Response(serializer.data, status=status.HTTP_200_OK)


class ActiveCertificateSectionAPIView(APIView):
    """
    Public API used by the website frontend to display
    the active Certificates section and active certificate items.
    """

    authentication_classes = []
    permission_classes = []
    throttle_classes = []

    def get(self, request, *args, **kwargs):
        certificate_section = CertificateSection.objects.filter(
            is_active=True
        ).order_by("-updated_at").first()

        if not certificate_section:
            return Response(
                {"detail": "No active certificates section found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        serializer = CertificateSectionSerializer(
            certificate_section,
            context={"request": request},
        )

        return Response(serializer.data, status=status.HTTP_200_OK)


class ActiveResearchSectionAPIView(APIView):
    """
    Public API used by the website frontend to display
    the active Research section and active research items.
    """

    authentication_classes = []
    permission_classes = []
    throttle_classes = []

    def get(self, request, *args, **kwargs):
        research_section = ResearchSection.objects.filter(
            is_active=True
        ).order_by("-updated_at").first()

        if not research_section:
            return Response(
                {"detail": "No active research section found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        serializer = ResearchSectionSerializer(
            research_section,
            context={"request": request},
        )

        return Response(serializer.data, status=status.HTTP_200_OK)


class ActiveFooterSectionAPIView(APIView):
    """
    Public API endpoint for the active footer section.

    This endpoint is used by the website footer.
    It returns:
    - follow title
    - copyright name
    - active social links
    - active contact items

    No authentication is required because this is public website content.
    """

    authentication_classes = []
    permission_classes = []
    throttle_classes = []

    def get(self, request, *args, **kwargs):
        footer = (
            FooterSection.objects.filter(is_active=True)
            .prefetch_related("social_links", "contact_items")
            .first()
        )

        if not footer:
            return Response(
                {
                    "detail": "No active footer section found.",
                    "footer": None,
                },
                status=status.HTTP_404_NOT_FOUND,
            )

        serializer = FooterSectionSerializer(
            footer,
            context={"request": request},
        )

        return Response(
            {
                "footer": serializer.data,
            },
            status=status.HTTP_200_OK,
        )


class WebsiteVisitTrackAPIView(APIView):
    """
    Public analytics endpoint for lightweight visit tracking.

    The frontend should call this on page load or important CTA interactions
    so the admin dashboard can show site-wide visitor analytics.
    """

    authentication_classes = []
    permission_classes = [AllowAny]
    throttle_classes = []

    def post(self, request, *args, **kwargs):
        serializer = WebsiteVisitTrackSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        data = serializer.validated_data

        visit = WebsiteVisit.objects.create(
            visitor_id=data.get("visitor_id", "") or None,
            session_key=data.get("session_key", "") or None,
            path=data["path"],
            page_title=data.get("page_title", ""),
            event_type=data.get("event_type", "page_view"),
            referrer=data.get("referrer", "") or request.META.get("HTTP_REFERER", ""),
            ip_address=_get_client_ip(request),
            user_agent=request.META.get("HTTP_USER_AGENT", ""),
            source_label=data.get("source_label", ""),
            metadata=data.get("metadata") or {},
        )

        return Response(
            {
                "success": True,
                "visit_id": str(visit.id),
            },
            status=status.HTTP_201_CREATED,
        )
