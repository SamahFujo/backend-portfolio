from core.serializers import AboutSectionSerializer

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from core.models import (HeroSection, AboutSection, SkillSection,ProjectSection)


from core.serializers import (
    HeroSectionSerializer,
    AboutSectionSerializer,
    SkillSectionSerializer,
    ProjectSectionSerializer,
)


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
