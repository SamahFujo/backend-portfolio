from django.conf import settings
from rest_framework.permissions import BasePermission


class HasInternalAPIKey(BasePermission):
    """
    Protect internal-only endpoints with a shared API key header.
    """

    message = "Invalid or missing admin API key."

    def has_permission(self, request, view):
        expected = getattr(view, "admin_api_key", None) or settings.ADMIN_API_KEY
        if not expected:
            return False

        provided = request.headers.get("X-Admin-API-Key", "")
        return bool(provided) and provided == expected
