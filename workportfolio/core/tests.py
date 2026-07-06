from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APIClient
from types import SimpleNamespace

from .models import ProfileDocument
from .services.chatbot.extractors import try_extract_education
from .services.chatbot.question_contracts import evaluate_evidence, infer_question_contract


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


class RetrievalContractTests(TestCase):
    @staticmethod
    def _fake_chunk(
        *,
        chunk_id: str,
        content: str,
        doc_type: str,
        title: str,
        chunk_index: int = 0,
        section_title: str = "",
    ):
        return SimpleNamespace(
            id=chunk_id,
            chunk_index=chunk_index,
            content=content,
            section_title=section_title,
            document=SimpleNamespace(
                title=title,
                document_type=doc_type,
            ),
        )

    def test_infer_question_contract_detects_education(self):
        contract = infer_question_contract(
            question="What is Samah's university GPA?",
            query_plan={"answer_type": "education"},
            question_route="profile_docs_question",
        )

        self.assertIsNotNone(contract)
        self.assertEqual(contract.name, "education")
        self.assertEqual(contract.preferred_document_types, ("cv",))

    def test_evidence_validation_rejects_mismatched_chunks(self):
        contract = infer_question_contract(
            question="What education does Samah have?",
            query_plan={"answer_type": "education"},
            question_route="profile_docs_question",
        )
        chunks = [
            self._fake_chunk(
                chunk_id="faq-1",
                doc_type="faq",
                title="faq",
                content="FAQ Question: What does Samah do? FAQ Answer: Samah is an AI Team Lead.",
            ),
            self._fake_chunk(
                chunk_id="exp-1",
                doc_type="experience_letter",
                title="experience letter",
                content="This is to certify Samah worked as Sr AI & Data Scientist from 2021 to 2025.",
            ),
        ]

        validation = evaluate_evidence(
            chunks=chunks,
            contract=contract,
            question="What education does Samah have?",
        )

        self.assertFalse(validation["is_sufficient"])
        self.assertLess(validation["top_score"], contract.min_chunk_score)

    def test_evidence_validation_accepts_cv_education_chunk(self):
        contract = infer_question_contract(
            question="What education does Samah have?",
            query_plan={"answer_type": "education"},
            question_route="profile_docs_question",
        )
        chunks = [
            self._fake_chunk(
                chunk_id="cv-1",
                doc_type="cv",
                title="cv",
                section_title="Education",
                content="Resume Section: Education\nMaster of Science in Data Science - Ahlia University\nBachelor of Science in Computer Science - Applied Science University",
            )
        ]

        validation = evaluate_evidence(
            chunks=chunks,
            contract=contract,
            question="What education does Samah have?",
        )

        self.assertTrue(validation["is_sufficient"])
        self.assertGreaterEqual(validation["top_score"], contract.min_chunk_score)

    def test_try_extract_education_reads_cv_lines(self):
        chunks = [
            self._fake_chunk(
                chunk_id="cv-1",
                doc_type="cv",
                title="cv",
                section_title="Education",
                content="Resume Section: Education\nMaster of Science in Data Science - Ahlia University\nBachelor of Science in Computer Science - Applied Science University",
            )
        ]

        handled, answer, confidence = try_extract_education(
            "What education does Samah have?",
            chunks,
        )

        self.assertTrue(handled)
        self.assertIn("education includes", answer.lower())
        self.assertGreater(confidence, 0.0)
