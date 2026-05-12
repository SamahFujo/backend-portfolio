"""
Custom permissions for internal/admin API endpoints.
"""

import hmac

from django.conf import settings
from rest_framework.permissions import BasePermission


class HasInternalAPIKey(BasePermission):
    """
    Allows access only when the request includes the correct admin API key.

    Expected frontend header:

        X-API-KEY: <ADMIN_API_KEY>

    Notes:
    - Browser may display this as X-Api-Key.
    - Django request.headers is case-insensitive.
    - request.META fallback is included for extra reliability.
    """

    message = "The admin access Security key is incorrect. Please try again."

    def has_permission(self, request, view):
        expected_key = (getattr(settings, "ADMIN_API_KEY", "") or "").strip()

        provided_key = (
            request.headers.get("X-API-KEY")
            or request.headers.get("X-Api-Key")
            or request.META.get("HTTP_X_API_KEY")
            or ""
        ).strip()

        # Temporary debug while testing. Remove after it works.
        print("===== ADMIN API KEY DEBUG =====")
        print("Provided key:", provided_key)
        print("Expected key:", expected_key)
        print("Keys match:", hmac.compare_digest(provided_key, expected_key))
        print("===============================")

        if not expected_key:
            return False

        if not provided_key:
            return False

        return hmac.compare_digest(provided_key, expected_key)
