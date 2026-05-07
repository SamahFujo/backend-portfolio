from rest_framework.permissions import AllowAny
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.authentication import SessionAuthentication
from ..serializers import GetInTouchSerializer, StartProjectRequestSerializer
from ..throttles import ContactRateThrottle
from ..serializers import StartProjectRequestSerializer
from ..services.resend_email import send_start_project_email
from ..services.resend_contact_email import send_get_in_touch_email
from ..services.resend_email import send_start_project_email
import logging
logger = logging.getLogger(__name__)

class CsrfExemptSessionAuthentication(SessionAuthentication):
    """
    Disable CSRF for this API endpoint (public portfolio form). this is safe because we are not using session authentication for any sensitive operations, and we have other protections in place (throttling, CORS, etc). It allows the public form to submit without needing a CSRF token.
    """

    def enforce_csrf(self, request):
        # To disable CSRF checks for this view, we override this method to do nothing.
        return


"""API views for the public "Start Project" form."""


class StartProjectRequestView(APIView):
    permission_classes = [AllowAny]
    authentication_classes = [CsrfExemptSessionAuthentication]
    throttle_classes = [ContactRateThrottle]

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
            logger.exception("Failed to send project request email")
            return Response(
                {
                    "success": False,
                    "message": "Could not send project request email right now. Please try again later.",
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    def options(self, request, *args, **kwargs):
        # Usually not needed explicitly, but safe if debugging preflight behavior
        return Response(status=status.HTTP_200_OK)


class GetInTouchView(APIView):
    permission_classes = [AllowAny]
    authentication_classes = [CsrfExemptSessionAuthentication]
    throttle_classes = [ContactRateThrottle]

    def post(self, request, *args, **kwargs):
        serializer = GetInTouchSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        try:
            result = send_get_in_touch_email(serializer.validated_data)
            return Response(
                {
                    "success": True,
                    "message": "Message sent successfully.",
                    "provider": "resend",
                    "email_id": result.get("id"),
                },
                status=status.HTTP_200_OK,
            )
        except Exception:
            logger.exception("Failed to send get-in-touch email")
            return Response(
                {
                    "success": False,
                    "message": "Could not send your message right now. Please try again later.",
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
