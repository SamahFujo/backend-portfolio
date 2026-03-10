from django.db import migrations
from pgvector.django import VectorExtension


class Migration(migrations.Migration):
    """
        Migration to enable the pgvector extension in PostgreSQL."
    """

    dependencies = [
        ("core", "0001_initial"),
    ]

    operations = [
        VectorExtension(),
    ]
