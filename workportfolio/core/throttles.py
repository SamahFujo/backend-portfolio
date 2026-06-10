from rest_framework.throttling import AnonRateThrottle,ScopedRateThrottle


class ContactRateThrottle(AnonRateThrottle):
    scope = "contact"


class ChatRateThrottle(AnonRateThrottle):
    scope = "chat"


class UploadRateThrottle(AnonRateThrottle):
    scope = "upload"




class AdminAPIRateThrottle(ScopedRateThrottle):
    """
    Higher rate limit for protected admin APIs.

    Admin APIs are already protected by the internal API key,
    so they should not use the same strict throttle as public endpoints.
    """

    scope = "admin_api"