"""
Public chatbot URLs.
"""

from django.urls import path

from core.views import (
    AskAboutMeAPIView,
    SendChatHistoryEmailAPIView,
    RequestEmailVerificationAPIView,
    VerifyEmailCodeAPIView,
)


urlpatterns = [
    path("ask/", AskAboutMeAPIView.as_view(), name="chat-ask"),
    path("send-history/", SendChatHistoryEmailAPIView.as_view(),
         name="chat-send-history"),

    path(
        "request-email-code/",
        RequestEmailVerificationAPIView.as_view(),
        name="chat-request-email-code",
    ),
    path(
        "verify-email-code/",
        VerifyEmailCodeAPIView.as_view(),
        name="chat-verify-email-code",
    ),
]
