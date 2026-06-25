FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

COPY workportfolio /app/workportfolio

WORKDIR /app/workportfolio

RUN SECRET_KEY=dummy-build-secret \
    ADMIN_API_KEY=dummy-build-admin-key \
    DEBUG=False \
    DB_NAME=placeholder \
    DB_USER=placeholder \
    DB_PASSWORD=placeholder \
    DB_HOST=localhost \
    DB_PORT=5432 \
    USE_S3_MEDIA=False \
    python manage.py collectstatic --noinput

EXPOSE 8000

CMD ["gunicorn", "workportfolio.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "120"]