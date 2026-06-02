"""
Public form URLs.

These endpoints receive public contact and project request submissions.
"""

from django.urls import path

from core.views import (
    StartProjectRequestView,
    GetInTouchView,
)


urlpatterns = [
    path("start-project/", StartProjectRequestView.as_view(), name="start-project"),
    path("get-in-touch/", GetInTouchView.as_view(), name="get-in-touch"),
]
