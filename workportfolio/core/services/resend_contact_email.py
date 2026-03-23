from __future__ import annotations

from typing import Dict, Any
from django.conf import settings
import requests


def send_get_in_touch_email(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sends a contact form email using Resend.
    """
    if not settings.RESEND_API_KEY:
        raise ValueError("RESEND_API_KEY is not configured.")
    if not settings.CONTACT_TO_EMAIL:
        raise ValueError("CONTACT_TO_EMAIL is not configured.")
    if not settings.CONTACT_FROM_EMAIL:
        raise ValueError("CONTACT_FROM_EMAIL is not configured.")

    name = payload["name"].strip()
    email = payload["email"].strip()
    subject = (payload.get("subject") or "").strip(
    ) or "New message from Samah.ai (Get in touch)"
    message = payload["message"].strip()

    # Plain-text body (safe and clean)
    text_body = (
        f"New contact message from Samah.ai\n\n"
        f"Name: {name}\n"
        f"Email: {email}\n"
        f"Subject: {subject}\n\n"
        f"Message:\n{message}\n"
    )

    # Resend API
    url = "https://api.resend.com/emails"
    headers = {
        "Authorization": f"Bearer {settings.RESEND_API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "from": settings.CONTACT_FROM_EMAIL,
        "to": [settings.CONTACT_TO_EMAIL],
        "subject": subject,
        "text": text_body,
        "reply_to": email,  # ✅ so you can reply directly to the user
    }

    if getattr(settings, "CONTACT_BCC_EMAIL", ""):
        data["bcc"] = [settings.CONTACT_BCC_EMAIL]

    resp = requests.post(url, json=data, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()
