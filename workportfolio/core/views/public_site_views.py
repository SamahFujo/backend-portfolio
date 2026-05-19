from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from core.models import HeroSection
from core.serializers import HeroSectionSerializer


class ActiveHeroSectionAPIView(APIView):
    """
    Public API used by the website homepage to display the active Hero section.
    """

    authentication_classes = []
    permission_classes = []

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
