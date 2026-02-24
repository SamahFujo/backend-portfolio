from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny
from rest_framework.throttling import AnonRateThrottle
from rest_framework.authentication import SessionAuthentication, BasicAuthentication

from .serializers import StartProjectRequestSerializer
from .services.resend_email import send_start_project_email


class CsrfExemptSessionAuthentication(SessionAuthentication):
    """
    Disable CSRF for this API endpoint (public portfolio form).
    """

    def enforce_csrf(self, request):
        return


class StartProjectRequestView(APIView):
    permission_classes = [AllowAny]
    authentication_classes = [
        CsrfExemptSessionAuthentication, BasicAuthentication]
    # optional, but nice if DRF throttling is configured
    throttle_classes = [AnonRateThrottle]

    def post(self, request, *args, **kwargs):
        serializer = StartProjectRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        try:
            resend_result = send_start_project_email(serializer.validated_data)
            return Response(
                {
                    "success": True,
                    "message": "Project request sent successfully.",
                    "provider": "resend",
                    "email_id": resend_result.get("id"),
                },
                status=status.HTTP_200_OK,
            )
        except Exception as e:
            # Keep response user-friendly, log detailed error server-side in production
            return Response(
                {
                    "success": False,
                    "message": "Could not send project request email right now. Please try again later.",
                    "error": str(e),  # You can remove this in production
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    def options(self, request, *args, **kwargs):
        # Usually not needed explicitly, but safe if debugging preflight behavior
        return Response(status=status.HTTP_200_OK)
