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
        "subject": f"{code} is your Samah.ai verification code",
        "html": f"""
        <div style="font-family: Arial, sans-serif; background: #f8fafc; padding: 24px;">
            <div style="max-width: 520px; margin: auto; background: white; border-radius: 18px; padding: 28px; border: 1px solid #e2e8f0;">
                <h2 style="margin: 0 0 12px; color: #0f172a;">Verify your email</h2>

                <p style="color: #475569; font-size: 15px; line-height: 1.6; margin: 0 0 16px;">
                    Use this verification code to continue to Samah.ai:
                </p>

                <div style="margin: 24px 0; text-align: center;">
                    <div style="display: inline-block; letter-spacing: 8px; font-size: 32px; font-weight: 700; color: #0891b2; background: #ecfeff; padding: 16px 24px; border-radius: 14px;">
                        {code}
                    </div>
                </div>

                <p style="color: #0f172a; font-size: 16px; font-weight: 600; margin: 0 0 8px;">
                    Verification code: {code}
                </p>

                <p style="color: #64748b; font-size: 13px; line-height: 1.6; margin: 0;">
                    This code expires in 10 minutes.
                    If you did not request it, you can ignore this email.
                </p>
            </div>
        </div>
        """,
        "text": (
            f"Samah.ai verification code: {code}\n"
            f"Verification code: {code}\n"
            "This code expires in 10 minutes."
        ),
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
