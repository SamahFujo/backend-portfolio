from django.core.management.base import BaseCommand
from django.db import transaction

from core.models import SkillSection, SkillItem


SKILLS = [
    {"label": "React", "icon": "react", "level": 9, "category": "Frontend"},
    {"label": "Next.js", "icon": "nextjs", "level": 10, "category": "Frontend"},
    {"label": "Tailwind CSS", "icon": "tailwind",
        "level": 10, "category": "Frontend"},
    {"label": "JavaScript", "icon": "javascript",
        "level": 9, "category": "Frontend"},
    {"label": "TypeScript", "icon": "typescript",
        "level": 8, "category": "Frontend"},

    {"label": "Django REST Framework", "icon": "drf",
        "level": 10, "category": "Backend"},
    {"label": "FastAPI", "icon": "fastapi", "level": 10, "category": "Backend"},
    {"label": "Flask", "icon": "flask", "level": 8, "category": "Backend"},
    {"label": "RESTful APIs", "icon": "rest", "level": 9, "category": "Backend"},
    {"label": "Swagger", "icon": "swagger", "level": 7, "category": "Backend"},
    {"label": "OpenAPI", "icon": "openapi", "level": 8, "category": "Backend"},
    {"label": "RBAC", "icon": "rbac", "level": 8, "category": "Backend"},
    {"label": "JWT", "icon": "jwt", "level": 8, "category": "Backend"},
    {"label": "Node.js", "icon": "nodejs", "level": 7, "category": "Backend"},

    {"label": "Transformers (Hugging Face)", "icon": "transformers",
     "level": 10, "category": "AI / LLM"},
    {"label": "Hugging Face", "icon": "huggingface",
        "level": 10, "category": "AI / LLM"},
    {"label": "LLMs (BERT, RoBERTa, T5)", "icon": "llms",
     "level": 9, "category": "AI / LLM"},
    {"label": "RAG (Embeddings & Vector Search)", "icon": "rag",
     "level": 10, "category": "AI / LLM"},
    {"label": "Prompt Engineering", "icon": "promptengineering",
        "level": 9, "category": "AI / LLM"},
    {"label": "Gemini (Google Generative AI)", "icon": "gemini",
     "level": 8, "category": "AI / LLM"},
    {"label": "PDF/Scanned Parsing (OCR + LLM)", "icon": "pdfparsing",
     "level": 9, "category": "AI / LLM"},
    {"label": "OCR (Tesseract, EasyOCR, OpenCV)", "icon": "ocr",
     "level": 8, "category": "AI / LLM"},
    {"label": "Ollama", "icon": "ollama", "level": 7, "category": "AI / LLM"},
    {"label": "OpenWebUI", "icon": "openwebui",
        "level": 7, "category": "AI / LLM"},
    {"label": "LangChain", "icon": "langchain",
        "level": 9, "category": "AI / LLM"},
    {"label": "Langfuse", "icon": "langfuse", "level": 7, "category": "AI / LLM"},

    {"label": "PostgreSQL", "icon": "postgresql",
        "level": 10, "category": "Database"},
    {"label": "MongoDB", "icon": "mongodb", "level": 8, "category": "Database"},
    {"label": "MySQL", "icon": "mysql", "level": 9, "category": "Database"},
    {"label": "SQL Server", "icon": "sqlserver",
        "level": 9, "category": "Database"},
    {"label": "Oracle", "icon": "oracle", "level": 8, "category": "Database"},

    {"label": "Docker", "icon": "docker", "level": 8, "category": "DevOps"},
    {"label": "NGINX", "icon": "nginx", "level": 7, "category": "DevOps"},
    {"label": "Gunicorn", "icon": "gunicorn", "level": 7, "category": "DevOps"},
    {"label": "Postman", "icon": "postman", "level": 10, "category": "DevOps"},

    {"label": "Python", "icon": "python", "level": 10, "category": "Languages"},
    {"label": ".NET C#", "icon": "dotnet", "level": 7, "category": "Languages"},
    {"label": "Java", "icon": "java", "level": 6, "category": "Languages"},
    {"label": "C++", "icon": "cpp", "level": 5, "category": "Languages"},
    {"label": "PHP", "icon": "php", "level": 8, "category": "Languages"},
    {"label": "Streamlit", "icon": "streamlit",
        "level": 7, "category": "Languages"},

    {"label": "Figma", "icon": "figma", "level": 7, "category": "UI"},
]


SUMMARY_MAP = {
    "react": {
        "heading": "Interactive UI engineering",
        "summary": "React is one of my main frontend foundations for building reusable, scalable, and highly interactive interfaces.",
        "points": [
            "Reusable component-based UI architecture",
            "Strong fit for dynamic dashboards and product interfaces",
            "Comfortable building polished interaction-heavy experiences",
        ],
    },
    "nextjs": {
        "heading": "Production-ready React framework",
        "summary": "I use Next.js to build structured frontend applications with better routing, organization, and deployment readiness.",
        "points": [
            "Great for scalable portfolio and product websites",
            "Supports clean architecture and modular page design",
            "Strong choice for real frontend delivery",
        ],
    },
    "tailwind": {
        "heading": "Fast, modern UI styling",
        "summary": "Tailwind CSS helps me build premium-looking, responsive interfaces quickly while keeping styling consistent.",
        "points": [
            "Speeds up UI development",
            "Excellent for dark/light mode systems",
            "Works well for polished design details and animations",
        ],
    },
    "javascript": {
        "heading": "Core frontend scripting",
        "summary": "JavaScript powers my frontend interactivity and helps connect interface behavior with real user actions.",
        "points": [
            "Essential for dynamic user experiences",
            "Strong for logic-driven UI behavior",
            "Works across frontend and integration tasks",
        ],
    },
    "typescript": {
        "heading": "Safer and cleaner frontend code",
        "summary": "TypeScript helps me build more maintainable applications by improving structure, reducing mistakes, and making components easier to scale.",
        "points": [
            "Improves reliability in larger projects",
            "Makes interfaces and data structures clearer",
            "Useful for cleaner long-term codebases",
        ],
    },
    "docker": {
        "heading": "Portable application delivery",
        "summary": "Docker is part of my deployment and setup workflow for packaging applications and keeping environments more consistent.",
        "points": [
            "Improves environment consistency",
            "Useful for deployment and local setup",
            "Supports modern engineering workflows",
        ],
    },
    "python": {
        "heading": "My strongest engineering language",
        "summary": "Python is the foundation of most of my AI, backend, automation, and data-driven work. It is one of my strongest and most frequently used skills.",
        "points": [
            "Core language for AI and backend development",
            "Used across automation, APIs, and ML workflows",
            "A major strength in my professional stack",
        ],
    },
}


DEFAULT_SUMMARY = {
    "heading": "Practical technical capability",
    "summary": "This skill supports my real project delivery and contributes to building complete, production-oriented solutions.",
    "points": [
        "Used in practical implementation work",
        "Supports end-to-end solution delivery",
        "Relevant to modern product development",
    ],
}


class Command(BaseCommand):
    help = "Seed the Skills section and skill items from the old static website data."

    @transaction.atomic
    def handle(self, *args, **options):
        section, _ = SkillSection.objects.update_or_create(
            is_active=True,
            defaults={
                "badge_text": "Expertise",
                "title_line_1": "Skills &",
                "title_line_2": "Capabilities.",
                "description": "A comprehensive toolkit built through years of hands-on experience and continuous learning.",
            },
        )

        # Optional: remove old placeholder/manual skills before seeding.
        SkillItem.objects.filter(section=section).delete()

        category_order_counter = {}

        created_count = 0

        for skill in SKILLS:
            category = skill["category"]
            category_order_counter[category] = category_order_counter.get(
                category, 0) + 1

            summary = SUMMARY_MAP.get(skill["icon"], DEFAULT_SUMMARY)

            SkillItem.objects.create(
                section=section,
                category=category,
                icon=skill["icon"],
                label=skill["label"],
                level=skill["level"],
                summary_heading=summary["heading"],
                summary_text=summary["summary"],
                summary_points=summary["points"],
                sort_order=category_order_counter[category],
                is_active=True,
            )

            created_count += 1

        self.stdout.write(
            self.style.SUCCESS(
                f"Skills section seeded successfully. Created {created_count} skill items."
            )
        )
