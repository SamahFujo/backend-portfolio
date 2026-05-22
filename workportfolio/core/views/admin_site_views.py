from django.conf import settings
from rest_framework import parsers, status
from rest_framework.response import Response
from rest_framework.views import APIView
from core.permissions import HasInternalAPIKey
from core.serializers import (
    HeroSectionAdminSerializer,
    AboutSectionAdminSerializer,
    SkillSectionAdminSerializer,
    SkillItemAdminSerializer,
    ProjectSectionAdminSerializer,
    ProjectItemAdminSerializer,
    CertificateSectionAdminSerializer,
    CertificateItemAdminSerializer,
    ResearchSectionAdminSerializer,
    ResearchItemAdminSerializer,
)

from rest_framework.parsers import FormParser, MultiPartParser, JSONParser

from core.models import (
    HeroSection,
    AboutSection,
    SkillSection,
    SkillItem,
    ProjectSection,
    ProjectItem,
    CertificateSection,
    CertificateItem,
    ResearchSection,
    ResearchItem,
)


class AdminHeroSectionAPIView(APIView):
    """
    Admin API for managing the Hero section from the custom admin dashboard.

    Supports:
    - text fields
    - dark mode hero image
    - light mode hero image
    - optional background image
    """

    permission_classes = [HasInternalAPIKey]
    admin_api_key = settings.ADMIN_API_KEY

    parser_classes = [
        parsers.MultiPartParser,
        parsers.FormParser,
        parsers.JSONParser,
    ]

    def get(self, request, *args, **kwargs):
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

        serializer = HeroSectionAdminSerializer(
            hero, context={"request": request})

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
            hero_instance = serializer.save(is_active=True)

            response_serializer = HeroSectionAdminSerializer(
                hero_instance,
                context={"request": request},
            )

            return Response(
                {
                    "detail": "Hero section saved successfully.",
                    "hero": response_serializer.data,
                },
                status=status.HTTP_200_OK,
            )

        return Response(
            {
                "detail": "Hero section validation failed.",
                "errors": serializer.errors,
            },
            status=status.HTTP_400_BAD_REQUEST,
        )


class AdminAboutSectionAPIView(APIView):
    """
    Admin API for managing the About Me section
    from the custom admin dashboard.

    GET:
    - Returns the current active About section.
    - If no record exists, returns 200 with about = None.

    PUT:
    - Creates or updates the active About section.
    """

    permission_classes = [HasInternalAPIKey]
    admin_api_key = settings.ADMIN_API_KEY

    def get(self, request, *args, **kwargs):
        about = AboutSection.objects.filter(
            is_active=True).order_by("-updated_at").first()

        if not about:
            return Response(
                {
                    "detail": "No about section found yet.",
                    "about": None,
                },
                status=status.HTTP_200_OK,
            )

        serializer = AboutSectionAdminSerializer(
            about, context={"request": request})

        return Response(
            {
                "detail": "About section loaded successfully.",
                "about": serializer.data,
            },
            status=status.HTTP_200_OK,
        )

    def put(self, request, *args, **kwargs):
        about = AboutSection.objects.filter(
            is_active=True).order_by("-updated_at").first()

        if about:
            serializer = AboutSectionAdminSerializer(
                about,
                data=request.data,
                partial=True,
                context={"request": request},
            )
        else:
            serializer = AboutSectionAdminSerializer(
                data=request.data,
                context={"request": request},
            )

        if serializer.is_valid():
            about_instance = serializer.save(is_active=True)

            response_serializer = AboutSectionAdminSerializer(
                about_instance,
                context={"request": request},
            )

            return Response(
                {
                    "detail": "About section saved successfully.",
                    "about": response_serializer.data,
                },
                status=status.HTTP_200_OK,
            )

        return Response(
            {
                "detail": "About section validation failed.",
                "errors": serializer.errors,
            },
            status=status.HTTP_400_BAD_REQUEST,
        )


class AdminSkillSectionAPIView(APIView):
    """
    Admin API for managing the Skills section header.

    GET:
    - Returns the active Skills section with its items.
    - If no section exists, returns 200 with skills = None.

    PUT:
    - Creates or updates the active Skills section header.
    """

    permission_classes = [HasInternalAPIKey]
    admin_api_key = settings.ADMIN_API_KEY

    def get(self, request, *args, **kwargs):
        skill_section = SkillSection.objects.filter(
            is_active=True).order_by("-updated_at").first()

        if not skill_section:
            return Response(
                {
                    "detail": "No skills section found yet.",
                    "skills": None,
                },
                status=status.HTTP_200_OK,
            )

        serializer = SkillSectionAdminSerializer(
            skill_section, context={"request": request})

        return Response(
            {
                "detail": "Skills section loaded successfully.",
                "skills": serializer.data,
            },
            status=status.HTTP_200_OK,
        )

    def put(self, request, *args, **kwargs):
        skill_section = SkillSection.objects.filter(
            is_active=True).order_by("-updated_at").first()

        if skill_section:
            serializer = SkillSectionAdminSerializer(
                skill_section,
                data=request.data,
                partial=True,
                context={"request": request},
            )
        else:
            serializer = SkillSectionAdminSerializer(
                data=request.data,
                context={"request": request},
            )

        if serializer.is_valid():
            skill_section_instance = serializer.save(is_active=True)

            response_serializer = SkillSectionAdminSerializer(
                skill_section_instance,
                context={"request": request},
            )

            return Response(
                {
                    "detail": "Skills section saved successfully.",
                    "skills": response_serializer.data,
                },
                status=status.HTTP_200_OK,
            )

        return Response(
            {
                "detail": "Skills section validation failed.",
                "errors": serializer.errors,
            },
            status=status.HTTP_400_BAD_REQUEST,
        )


class AdminSkillItemCreateAPIView(APIView):
    """
    Admin API for creating a new skill item/card.
    """

    permission_classes = [HasInternalAPIKey]
    admin_api_key = settings.ADMIN_API_KEY

    def post(self, request, *args, **kwargs):
        skill_section = SkillSection.objects.filter(
            is_active=True).order_by("-updated_at").first()

        if not skill_section:
            return Response(
                {"detail": "Create the Skills section before adding skill items."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        data = request.data.copy()
        data["section"] = skill_section.id

        serializer = SkillItemAdminSerializer(
            data=data, context={"request": request})

        if serializer.is_valid():
            item = serializer.save()

            response_serializer = SkillItemAdminSerializer(
                item,
                context={"request": request},
            )

            return Response(
                {
                    "detail": "Skill item created successfully.",
                    "item": response_serializer.data,
                },
                status=status.HTTP_201_CREATED,
            )

        return Response(
            {
                "detail": "Skill item validation failed.",
                "errors": serializer.errors,
            },
            status=status.HTTP_400_BAD_REQUEST,
        )


class AdminSkillItemDetailAPIView(APIView):
    """
    Admin API for updating or deleting one skill item/card.
    """

    permission_classes = [HasInternalAPIKey]
    admin_api_key = settings.ADMIN_API_KEY

    def patch(self, request, item_id, *args, **kwargs):
        item = SkillItem.objects.filter(id=item_id).first()

        if not item:
            return Response(
                {"detail": "Skill item not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        serializer = SkillItemAdminSerializer(
            item,
            data=request.data,
            partial=True,
            context={"request": request},
        )

        if serializer.is_valid():
            updated_item = serializer.save()

            response_serializer = SkillItemAdminSerializer(
                updated_item,
                context={"request": request},
            )

            return Response(
                {
                    "detail": "Skill item updated successfully.",
                    "item": response_serializer.data,
                },
                status=status.HTTP_200_OK,
            )

        return Response(
            {
                "detail": "Skill item validation failed.",
                "errors": serializer.errors,
            },
            status=status.HTTP_400_BAD_REQUEST,
        )

    def delete(self, request, item_id, *args, **kwargs):
        item = SkillItem.objects.filter(id=item_id).first()

        if not item:
            return Response(
                {"detail": "Skill item not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        item.delete()

        return Response(
            {"detail": "Skill item deleted successfully."},
            status=status.HTTP_200_OK,
        )


class AdminProjectSectionAPIView(APIView):
    """
    Admin API for managing the Projects section header.

    GET:
    - Returns the active Projects section with its items.
    - If no section exists, returns 200 with projects = None.

    PUT:
    - Creates or updates the active Projects section header.
    """

    permission_classes = [HasInternalAPIKey]
    admin_api_key = settings.ADMIN_API_KEY
    parser_classes = [JSONParser, MultiPartParser, FormParser]

    def get(self, request, *args, **kwargs):
        project_section = ProjectSection.objects.filter(
            is_active=True
        ).order_by("-updated_at").first()

        if not project_section:
            return Response(
                {
                    "detail": "No projects section found yet.",
                    "projects": None,
                },
                status=status.HTTP_200_OK,
            )

        serializer = ProjectSectionAdminSerializer(
            project_section,
            context={"request": request},
        )

        return Response(
            {
                "detail": "Projects section loaded successfully.",
                "projects": serializer.data,
            },
            status=status.HTTP_200_OK,
        )

    def put(self, request, *args, **kwargs):
        project_section = ProjectSection.objects.filter(
            is_active=True
        ).order_by("-updated_at").first()

        if project_section:
            serializer = ProjectSectionAdminSerializer(
                project_section,
                data=request.data,
                partial=True,
                context={"request": request},
            )
        else:
            serializer = ProjectSectionAdminSerializer(
                data=request.data,
                context={"request": request},
            )

        if serializer.is_valid():
            project_section_instance = serializer.save(is_active=True)

            response_serializer = ProjectSectionAdminSerializer(
                project_section_instance,
                context={"request": request},
            )

            return Response(
                {
                    "detail": "Projects section saved successfully.",
                    "projects": response_serializer.data,
                },
                status=status.HTTP_200_OK,
            )

        return Response(
            {
                "detail": "Projects section validation failed.",
                "errors": serializer.errors,
            },
            status=status.HTTP_400_BAD_REQUEST,
        )


class AdminProjectItemCreateAPIView(APIView):
    """
    Admin API for creating a new project item/card.
    """

    permission_classes = [HasInternalAPIKey]
    admin_api_key = settings.ADMIN_API_KEY
    parser_classes = [JSONParser, MultiPartParser, FormParser]

    def post(self, request, *args, **kwargs):
        project_section = ProjectSection.objects.filter(
            is_active=True
        ).order_by("-updated_at").first()

        if not project_section:
            return Response(
                {"detail": "Create the Projects section before adding project items."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        data = request.data.copy()
        data["section"] = project_section.id

        serializer = ProjectItemAdminSerializer(
            data=data,
            context={"request": request},
        )

        if serializer.is_valid():
            item = serializer.save()

            response_serializer = ProjectItemAdminSerializer(
                item,
                context={"request": request},
            )

            return Response(
                {
                    "detail": "Project item created successfully.",
                    "item": response_serializer.data,
                },
                status=status.HTTP_201_CREATED,
            )

        return Response(
            {
                "detail": "Project item validation failed.",
                "errors": serializer.errors,
            },
            status=status.HTTP_400_BAD_REQUEST,
        )


class AdminProjectItemDetailAPIView(APIView):
    """
    Admin API for updating or deleting one project item.
    """

    permission_classes = [HasInternalAPIKey]
    admin_api_key = settings.ADMIN_API_KEY
    parser_classes = [JSONParser, MultiPartParser, FormParser]

    def patch(self, request, item_id, *args, **kwargs):
        item = ProjectItem.objects.filter(id=item_id).first()

        if not item:
            return Response(
                {"detail": "Project item not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        serializer = ProjectItemAdminSerializer(
            item,
            data=request.data,
            partial=True,
            context={"request": request},
        )

        if serializer.is_valid():
            updated_item = serializer.save()

            response_serializer = ProjectItemAdminSerializer(
                updated_item,
                context={"request": request},
            )

            return Response(
                {
                    "detail": "Project item updated successfully.",
                    "item": response_serializer.data,
                },
                status=status.HTTP_200_OK,
            )

        return Response(
            {
                "detail": "Project item validation failed.",
                "errors": serializer.errors,
            },
            status=status.HTTP_400_BAD_REQUEST,
        )

    def delete(self, request, item_id, *args, **kwargs):
        item = ProjectItem.objects.filter(id=item_id).first()

        if not item:
            return Response(
                {"detail": "Project item not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        item.delete()

        return Response(
            {"detail": "Project item deleted successfully."},
            status=status.HTTP_200_OK,
        )


class AdminCertificateSectionAPIView(APIView):
    """
    Admin API for managing the Certificates section header.

    GET:
    - Returns the active Certificates section with its items.
    - If no section exists, returns 200 with certificates = None.

    PUT:
    - Creates or updates the active Certificates section header.
    """

    permission_classes = [HasInternalAPIKey]
    admin_api_key = settings.ADMIN_API_KEY
    parser_classes = [JSONParser, MultiPartParser, FormParser]

    def get(self, request, *args, **kwargs):
        certificate_section = CertificateSection.objects.filter(
            is_active=True
        ).order_by("-updated_at").first()

        if not certificate_section:
            return Response(
                {
                    "detail": "No certificates section found yet.",
                    "certificates": None,
                },
                status=status.HTTP_200_OK,
            )

        serializer = CertificateSectionAdminSerializer(
            certificate_section,
            context={"request": request},
        )

        return Response(
            {
                "detail": "Certificates section loaded successfully.",
                "certificates": serializer.data,
            },
            status=status.HTTP_200_OK,
        )

    def put(self, request, *args, **kwargs):
        certificate_section = CertificateSection.objects.filter(
            is_active=True
        ).order_by("-updated_at").first()

        if certificate_section:
            serializer = CertificateSectionAdminSerializer(
                certificate_section,
                data=request.data,
                partial=True,
                context={"request": request},
            )
        else:
            serializer = CertificateSectionAdminSerializer(
                data=request.data,
                context={"request": request},
            )

        if serializer.is_valid():
            certificate_section_instance = serializer.save(is_active=True)

            response_serializer = CertificateSectionAdminSerializer(
                certificate_section_instance,
                context={"request": request},
            )

            return Response(
                {
                    "detail": "Certificates section saved successfully.",
                    "certificates": response_serializer.data,
                },
                status=status.HTTP_200_OK,
            )

        return Response(
            {
                "detail": "Certificates section validation failed.",
                "errors": serializer.errors,
            },
            status=status.HTTP_400_BAD_REQUEST,
        )


class AdminCertificateItemCreateAPIView(APIView):
    """
    Admin API for creating a new certificate item.
    """

    permission_classes = [HasInternalAPIKey]
    admin_api_key = settings.ADMIN_API_KEY
    parser_classes = [JSONParser, MultiPartParser, FormParser]

    def post(self, request, *args, **kwargs):
        certificate_section = CertificateSection.objects.filter(
            is_active=True
        ).order_by("-updated_at").first()

        if not certificate_section:
            return Response(
                {"detail": "Create the Certificates section before adding certificate items."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        data = request.data.copy()
        data["section"] = certificate_section.id

        serializer = CertificateItemAdminSerializer(
            data=data,
            context={"request": request},
        )

        if serializer.is_valid():
            item = serializer.save()

            response_serializer = CertificateItemAdminSerializer(
                item,
                context={"request": request},
            )

            return Response(
                {
                    "detail": "Certificate item created successfully.",
                    "item": response_serializer.data,
                },
                status=status.HTTP_201_CREATED,
            )

        return Response(
            {
                "detail": "Certificate item validation failed.",
                "errors": serializer.errors,
            },
            status=status.HTTP_400_BAD_REQUEST,
        )


class AdminCertificateItemDetailAPIView(APIView):
    """
    Admin API for updating or deleting one certificate item.
    """

    permission_classes = [HasInternalAPIKey]
    admin_api_key = settings.ADMIN_API_KEY
    parser_classes = [JSONParser, MultiPartParser, FormParser]

    def patch(self, request, item_id, *args, **kwargs):
        item = CertificateItem.objects.filter(id=item_id).first()

        if not item:
            return Response(
                {"detail": "Certificate item not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        serializer = CertificateItemAdminSerializer(
            item,
            data=request.data,
            partial=True,
            context={"request": request},
        )

        if serializer.is_valid():
            updated_item = serializer.save()

            response_serializer = CertificateItemAdminSerializer(
                updated_item,
                context={"request": request},
            )

            return Response(
                {
                    "detail": "Certificate item updated successfully.",
                    "item": response_serializer.data,
                },
                status=status.HTTP_200_OK,
            )

        return Response(
            {
                "detail": "Certificate item validation failed.",
                "errors": serializer.errors,
            },
            status=status.HTTP_400_BAD_REQUEST,
        )

    def delete(self, request, item_id, *args, **kwargs):
        item = CertificateItem.objects.filter(id=item_id).first()

        if not item:
            return Response(
                {"detail": "Certificate item not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        item.delete()

        return Response(
            {"detail": "Certificate item deleted successfully."},
            status=status.HTTP_200_OK,
        )


class AdminResearchSectionAPIView(APIView):
    """
    Admin API for managing the Research section header.

    GET:
    - Returns the active Research section with its items.
    - If no section exists, returns 200 with research = None.

    PUT:
    - Creates or updates the active Research section header.
    """

    permission_classes = [HasInternalAPIKey]
    admin_api_key = settings.ADMIN_API_KEY
    parser_classes = [JSONParser, MultiPartParser, FormParser]

    def get(self, request, *args, **kwargs):
        research_section = ResearchSection.objects.filter(
            is_active=True
        ).order_by("-updated_at").first()

        if not research_section:
            return Response(
                {
                    "detail": "No research section found yet.",
                    "research": None,
                },
                status=status.HTTP_200_OK,
            )

        serializer = ResearchSectionAdminSerializer(
            research_section,
            context={"request": request},
        )

        return Response(
            {
                "detail": "Research section loaded successfully.",
                "research": serializer.data,
            },
            status=status.HTTP_200_OK,
        )

    def put(self, request, *args, **kwargs):
        research_section = ResearchSection.objects.filter(
            is_active=True
        ).order_by("-updated_at").first()

        if research_section:
            serializer = ResearchSectionAdminSerializer(
                research_section,
                data=request.data,
                partial=True,
                context={"request": request},
            )
        else:
            serializer = ResearchSectionAdminSerializer(
                data=request.data,
                context={"request": request},
            )

        if serializer.is_valid():
            research_section_instance = serializer.save(is_active=True)

            response_serializer = ResearchSectionAdminSerializer(
                research_section_instance,
                context={"request": request},
            )

            return Response(
                {
                    "detail": "Research section saved successfully.",
                    "research": response_serializer.data,
                },
                status=status.HTTP_200_OK,
            )

        return Response(
            {
                "detail": "Research section validation failed.",
                "errors": serializer.errors,
            },
            status=status.HTTP_400_BAD_REQUEST,
        )


class AdminResearchItemCreateAPIView(APIView):
    """
    Admin API for creating a new research item.
    """

    permission_classes = [HasInternalAPIKey]
    admin_api_key = settings.ADMIN_API_KEY
    parser_classes = [JSONParser, MultiPartParser, FormParser]

    def post(self, request, *args, **kwargs):
        research_section = ResearchSection.objects.filter(
            is_active=True
        ).order_by("-updated_at").first()

        if not research_section:
            return Response(
                {"detail": "Create the Research section before adding research items."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        data = request.data.copy()
        data["section"] = research_section.id

        serializer = ResearchItemAdminSerializer(
            data=data,
            context={"request": request},
        )

        if serializer.is_valid():
            item = serializer.save()

            response_serializer = ResearchItemAdminSerializer(
                item,
                context={"request": request},
            )

            return Response(
                {
                    "detail": "Research item created successfully.",
                    "item": response_serializer.data,
                },
                status=status.HTTP_201_CREATED,
            )

        return Response(
            {
                "detail": "Research item validation failed.",
                "errors": serializer.errors,
            },
            status=status.HTTP_400_BAD_REQUEST,
        )


class AdminResearchItemDetailAPIView(APIView):
    """
    Admin API for updating or deleting one research item.
    """

    permission_classes = [HasInternalAPIKey]
    admin_api_key = settings.ADMIN_API_KEY
    parser_classes = [JSONParser, MultiPartParser, FormParser]

    def patch(self, request, item_id, *args, **kwargs):
        item = ResearchItem.objects.filter(id=item_id).first()

        if not item:
            return Response(
                {"detail": "Research item not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        serializer = ResearchItemAdminSerializer(
            item,
            data=request.data,
            partial=True,
            context={"request": request},
        )

        if serializer.is_valid():
            updated_item = serializer.save()

            response_serializer = ResearchItemAdminSerializer(
                updated_item,
                context={"request": request},
            )

            return Response(
                {
                    "detail": "Research item updated successfully.",
                    "item": response_serializer.data,
                },
                status=status.HTTP_200_OK,
            )

        return Response(
            {
                "detail": "Research item validation failed.",
                "errors": serializer.errors,
            },
            status=status.HTTP_400_BAD_REQUEST,
        )

    def delete(self, request, item_id, *args, **kwargs):
        item = ResearchItem.objects.filter(id=item_id).first()

        if not item:
            return Response(
                {"detail": "Research item not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        item.delete()

        return Response(
            {"detail": "Research item deleted successfully."},
            status=status.HTTP_200_OK,
        )
