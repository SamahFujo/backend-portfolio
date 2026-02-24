import requests
from django.conf import settings


RESEND_API_URL = "https://api.resend.com/emails"


def send_start_project_email(data: dict) -> dict:
    """
    Sends a project inquiry email to site owner using Resend API.
    Returns parsed JSON response.
    Raises requests.HTTPError on failure.
    """
    api_key = settings.RESEND_API_KEY
    if not api_key:
        raise ValueError("RESEND_API_KEY is missing in settings/.env")

    subject = f"🚀 New Project Request: {data.get('projectName', 'Untitled Project')}"

    # Plain text version (safe + reliable)
    text_content = f"""
New project request submitted from your portfolio website.

Project Name: {data.get('projectName', '')}
Project Type: {data.get('projectType', '')}
Budget Range: {data.get('budgetRange', '')}
Timeline: {data.get('timeline', '')}

Project Description:
{data.get('projectDescription', '')}

Client Name: {data.get('yourName', '')}
Client Email: {data.get('yourEmail', '')}
""".strip()

    # HTML version (nice formatting)
    html_content = f"""
    <div style="font-family: Arial, sans-serif; line-height: 1.6; color: #111;">
      <h2 style="margin-bottom: 12px;">🚀 New Project Request</h2>
      <p>A new project request was submitted from your portfolio website.</p>

      <table style="border-collapse: collapse; width: 100%; margin: 16px 0;">
        <tr><td style="padding: 8px; border: 1px solid #ddd;"><strong>Project Name</strong></td><td style="padding: 8px; border: 1px solid #ddd;">{data.get('projectName', '')}</td></tr>
        <tr><td style="padding: 8px; border: 1px solid #ddd;"><strong>Project Type</strong></td><td style="padding: 8px; border: 1px solid #ddd;">{data.get('projectType', '')}</td></tr>
        <tr><td style="padding: 8px; border: 1px solid #ddd;"><strong>Budget Range</strong></td><td style="padding: 8px; border: 1px solid #ddd;">{data.get('budgetRange', '')}</td></tr>
        <tr><td style="padding: 8px; border: 1px solid #ddd;"><strong>Timeline</strong></td><td style="padding: 8px; border: 1px solid #ddd;">{data.get('timeline', '')}</td></tr>
        <tr><td style="padding: 8px; border: 1px solid #ddd;"><strong>Client Name</strong></td><td style="padding: 8px; border: 1px solid #ddd;">{data.get('yourName', '')}</td></tr>
        <tr><td style="padding: 8px; border: 1px solid #ddd;"><strong>Client Email</strong></td><td style="padding: 8px; border: 1px solid #ddd;">{data.get('yourEmail', '')}</td></tr>
      </table>

      <h3 style="margin-top: 20px;">Project Description</h3>
      <div style="padding: 12px; border: 1px solid #ddd; border-radius: 8px; background: #f8f9fa; white-space: pre-wrap;">
        {data.get('projectDescription', '')}
      </div>
    </div>
    """

    payload = {
        "from": settings.CONTACT_FROM_EMAIL,
        "to": [settings.CONTACT_TO_EMAIL],
        # so when you click Reply, it goes to the client
        "reply_to": data.get("yourEmail"),
        "subject": subject,
        "html": html_content,
        "text": text_content,
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

    # Raise error if Resend returns 4xx/5xx
    response.raise_for_status()
    return response.json()
