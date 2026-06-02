import requests
from django.conf import settings


RESEND_API_URL = "https://api.resend.com/emails"


def send_email_verification_code(*, recipient_email: str, code: str) -> dict:
    """
    Send a 6-digit verification code to the visitor before starting chatbot.
    """

    api_key = settings.RESEND_API_KEY

    if not api_key:
        raise ValueError("RESEND_API_KEY is missing in settings/.env")

    payload = {
        "from": settings.CONTACT_FROM_EMAIL,
        "to": [recipient_email],
        "subject": "Your Samah.ai verification code",
        "html": f"""
        <div style="font-family: Arial, sans-serif; background: #f8fafc; padding: 24px;">
            <div style="max-width: 520px; margin: auto; background: white; border-radius: 18px; padding: 28px; border: 1px solid #e2e8f0;">
                <h2 style="margin: 0 0 12px; color: #0f172a;">Verify your email</h2>
                <p style="color: #475569; font-size: 15px; line-height: 1.6;">
                    Please use the verification code below to start chatting with Samah.ai.
                </p>

                <div style="margin: 24px 0; text-align: center;">
                    <div style="display: inline-block; letter-spacing: 8px; font-size: 32px; font-weight: 700; color: #0891b2; background: #ecfeff; padding: 16px 24px; border-radius: 14px;">
                        {code}
                    </div>
                </div>

                <p style="color: #64748b; font-size: 13px;">
                    This code will expire in 10 minutes. If you did not request this, you can ignore this email.
                </p>
            </div>
        </div>
        """,
        "text": f"Your Samah.ai verification code is: {code}. This code expires in 10 minutes.",
    }

    response = requests.post(
        RESEND_API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=20,
    )

    response.raise_for_status()
    return response.json()
