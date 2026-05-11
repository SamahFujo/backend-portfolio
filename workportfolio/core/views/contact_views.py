import logging

from rest_framework.permissions import AllowAny
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.authentication import SessionAuthentication

from ..models import ContactMessage, ProjectRequest
from ..serializers import GetInTouchSerializer, StartProjectRequestSerializer
from ..throttles import ContactRateThrottle
from ..services.resend_contact_email import send_get_in_touch_email
from ..services.resend_email import send_start_project_email

logger = logging.getLogger(__name__)


def get_client_ip(request):
    """
    Get visitor IP address from request headers.
    Handles proxy headers if available.
    """
    x_forwarded_for = request.META.get("HTTP_X_FORWARDED_FOR")

    if x_forwarded_for:
        return x_forwarded_for.split(",")[0].strip()

    return request.META.get("REMOTE_ADDR")


def get_request_metadata(request):
    """
    Collect lightweight request metadata for admin analytics/security.
    """
    return {
        "ip_address": get_client_ip(request),
        "user_agent": request.META.get("HTTP_USER_AGENT", ""),
        "referrer": request.META.get("HTTP_REFERER", ""),
    }


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

        data = serializer.validated_data
        metadata = get_request_metadata(request)

        # 1. Save request in database first
        project_request = ProjectRequest.objects.create(
            project_name=data.get("projectName", ""),
            project_type=data.get("projectType", ""),
            budget_range=data.get("budgetRange", ""),
            timeline=data.get("timeline", ""),
            project_description=data.get("projectDescription", ""),
            your_name=data.get("yourName", ""),
            your_email=data.get("yourEmail", ""),
            ip_address=metadata["ip_address"],
            user_agent=metadata["user_agent"],
            referrer=metadata["referrer"],
        )

        try:
            # 2. Send email notification
            email_payload = {
                "projectName": project_request.project_name,
                "projectType": project_request.project_type or "",
                "budgetRange": project_request.budget_range or "",
                "timeline": project_request.timeline or "",
                "projectDescription": project_request.project_description,
                "yourName": project_request.your_name,
                "yourEmail": project_request.your_email,
            }

            resend_result = send_start_project_email(email_payload)

            return Response(
                {
                    "success": True,
                    "message": "Project request sent successfully.",
                    "project_request_id": str(project_request.id),
                    "provider": "resend",
                    "email_id": resend_result.get("id"),
                },
                status=status.HTTP_200_OK,
            )

        except Exception:
            logger.exception("Failed to send project request email")

            return Response(
                {
                    "success": False,
                    "message": "Your project request was saved, but the email notification could not be sent right now.",
                    "project_request_id": str(project_request.id),
                    "email_sent": True,
                },
                status=status.HTTP_202_ACCEPTED,
            )

    def options(self, request, *args, **kwargs):
        return Response(status=status.HTTP_200_OK)


class GetInTouchView(APIView):
    permission_classes = [AllowAny]
    authentication_classes = [CsrfExemptSessionAuthentication]
    throttle_classes = [ContactRateThrottle]

    def post(self, request, *args, **kwargs):
        serializer = GetInTouchSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        data = serializer.validated_data
        metadata = get_request_metadata(request)

        # 1. Save message in database first
        contact_message = ContactMessage.objects.create(
            name=data.get("name", ""),
            email=data.get("email", ""),
            subject=data.get("subject", ""),
            message=data.get("message", ""),
            ip_address=metadata["ip_address"],
            user_agent=metadata["user_agent"],
            referrer=metadata["referrer"],
        )

        try:
            # 2. Send email notification
            email_payload = {
                "name": contact_message.name,
                "email": contact_message.email,
                "subject": contact_message.subject or "",
                "message": contact_message.message,
            }

            result = send_get_in_touch_email(email_payload)

            return Response(
                {
                    "success": True,
                    "message": "Message sent successfully.",
                    "contact_message_id": str(contact_message.id),
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
                    "message": "Your message was saved, but the email notification could not be sent right now.",
                    "contact_message_id": str(contact_message.id),
                    "email_sent": True,
                },
                status=status.HTTP_202_ACCEPTED,
            )
