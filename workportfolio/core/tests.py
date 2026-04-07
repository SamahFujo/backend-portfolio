from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APIClient

from .models import ProfileDocument


@override_settings(
    ADMIN_API_KEY="test-admin-key",
    ALLOWED_HOSTS=["testserver", "localhost", "127.0.0.1"],
)
class SecurityTests(TestCase):
    def setUp(self):
        self.client = APIClient()

    def test_document_list_requires_admin_api_key(self):
        response = self.client.get(reverse("documents-list"))

        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)

    def test_document_list_accepts_valid_admin_api_key(self):
        ProfileDocument.objects.create(title="Resume", file="profile_documents/resume.txt")

        response = self.client.get(
            reverse("documents-list"),
            HTTP_X_ADMIN_API_KEY="test-admin-key",
        )

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data), 1)

    def test_document_upload_rejects_unsupported_file_type(self):
        upload = SimpleUploadedFile(
            "payload.exe",
            b"binary-data",
            content_type="application/octet-stream",
        )

        response = self.client.post(
            reverse("documents-upload"),
            data={"title": "Bad file", "file": upload},
            format="multipart",
            HTTP_X_ADMIN_API_KEY="test-admin-key",
        )

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("Unsupported file type", str(response.data["file"][0]))
