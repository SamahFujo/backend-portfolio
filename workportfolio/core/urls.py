from django.urls import path
from .views import StartProjectRequestView

urlpatterns = [
    path("contact/start-project/", StartProjectRequestView.as_view(),
        name="start-project-request"),
]
