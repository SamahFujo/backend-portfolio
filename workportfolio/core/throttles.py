from rest_framework.throttling import AnonRateThrottle


class ContactRateThrottle(AnonRateThrottle):
    scope = "contact"


class ChatRateThrottle(AnonRateThrottle):
    scope = "chat"


class UploadRateThrottle(AnonRateThrottle):
    scope = "upload"
