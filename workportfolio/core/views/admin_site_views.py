from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, parsers

from core.models import HeroSection
from core.permissions import HasInternalAPIKey
from core.serializers import HeroSectionAdminSerializer


class AdminHeroSectionAPIView(APIView):
    """
    Admin API for managing the Hero section from the custom admin dashboard.

    GET:
    - Returns the current active Hero section.

    PUT:
    - Creates or updates the active Hero section.
    - Supports multipart/form-data for image uploads.
    """

    permission_classes = [HasInternalAPIKey]
    admin_api_key = settings.ADMIN_API_KEY

    parser_classes = [
        parsers.MultiPartParser,
        parsers.FormParser,
        parsers.JSONParser,
    ]


    def get(self, request, *args, **kwargs):
        """
        Return the current active Hero section.

        Important:
        If no Hero section exists yet, return 200 with hero = None.
        This allows the custom admin form to open empty and create the first record.
        """

        hero = HeroSection.objects.filter(
            is_active=True).order_by("-updated_at").first()

        if not hero:
            return Response(
                {
                    "detail": "No hero section found yet.",
                    "hero": None,
                },
                status=status.HTTP_200_OK,
            )

        serializer = HeroSectionAdminSerializer(hero, context={"request": request})

        return Response(
            {
                "detail": "Hero section loaded successfully.",
                "hero": serializer.data,
            },
            status=status.HTTP_200_OK,
        )

    def put(self, request, *args, **kwargs):
        hero = HeroSection.objects.filter(
            is_active=True).order_by("-updated_at").first()

        if hero:
            serializer = HeroSectionAdminSerializer(
                hero,
                data=request.data,
                partial=True,
                context={"request": request},
            )
        else:
            serializer = HeroSectionAdminSerializer(
                data=request.data,
                context={"request": request},
            )

        if serializer.is_valid():
            serializer.save(is_active=True)

            return Response(
                {
                    "detail": "Hero section saved successfully.",
                    "hero": serializer.data,
                },
                status=status.HTTP_200_OK,
            )

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
