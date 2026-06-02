import resend
import html

import requests
from django.conf import settings


RESEND_API_URL = "https://api.resend.com/emails"


def send_start_project_email(data: dict) -> dict:
    """
    Sends a project inquiry email to the site owner using Resend API.
    Returns parsed JSON response and raises requests.HTTPError on failure.
    """
    api_key = settings.RESEND_API_KEY
    if not api_key:
        raise ValueError("RESEND_API_KEY is missing in settings/.env")

    project_name = html.escape(data.get("projectName", ""))
    project_type = html.escape(data.get("projectType", ""))
    budget_range = html.escape(data.get("budgetRange", ""))
    timeline = html.escape(data.get("timeline", ""))
    project_description = html.escape(data.get("projectDescription", ""))
    client_name = html.escape(data.get("yourName", ""))
    client_email = html.escape(data.get("yourEmail", ""))

    subject = f"New Project Request: {data.get('projectName', 'Untitled Project')}"

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

    html_content = f"""
    <div style="font-family: Arial, sans-serif; line-height: 1.6; color: #111;">
      <h2 style="margin-bottom: 12px;">New Project Request</h2>
      <p>A new project request was submitted from your portfolio website.</p>

      <table style="border-collapse: collapse; width: 100%; margin: 16px 0;">
        <tr><td style="padding: 8px; border: 1px solid #ddd;"><strong>Project Name</strong></td><td style="padding: 8px; border: 1px solid #ddd;">{project_name}</td></tr>
        <tr><td style="padding: 8px; border: 1px solid #ddd;"><strong>Project Type</strong></td><td style="padding: 8px; border: 1px solid #ddd;">{project_type}</td></tr>
        <tr><td style="padding: 8px; border: 1px solid #ddd;"><strong>Budget Range</strong></td><td style="padding: 8px; border: 1px solid #ddd;">{budget_range}</td></tr>
        <tr><td style="padding: 8px; border: 1px solid #ddd;"><strong>Timeline</strong></td><td style="padding: 8px; border: 1px solid #ddd;">{timeline}</td></tr>
        <tr><td style="padding: 8px; border: 1px solid #ddd;"><strong>Client Name</strong></td><td style="padding: 8px; border: 1px solid #ddd;">{client_name}</td></tr>
        <tr><td style="padding: 8px; border: 1px solid #ddd;"><strong>Client Email</strong></td><td style="padding: 8px; border: 1px solid #ddd;">{client_email}</td></tr>
      </table>

      <h3 style="margin-top: 20px;">Project Description</h3>
      <div style="padding: 12px; border: 1px solid #ddd; border-radius: 8px; background: #f8f9fa; white-space: pre-wrap;">
        {project_description}
      </div>
    </div>
    """

    payload = {
        "from": settings.CONTACT_FROM_EMAIL,
        "to": [settings.CONTACT_TO_EMAIL],
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
    response.raise_for_status()
    return response.json()


def send_chat_history_email(*, recipient_email: str, history_text: str) -> dict:
    """
    Send chatbot conversation history to the visitor using Resend REST API.
    Sends both plain-text and attractive HTML versions.
    Returns parsed JSON response and raises requests.HTTPError on failure.
    """
    api_key = settings.RESEND_API_KEY
    if not api_key:
        raise ValueError("RESEND_API_KEY is missing in settings/.env")

    safe_history_html = html.escape(history_text).replace("\n", "<br>")

    html_content = f"""
    <!DOCTYPE html>
    <html>
      <body style="margin:0; padding:0; background:#f4f7fb; font-family:Arial, Helvetica, sans-serif; color:#0f172a;">
        <div style="width:100%; padding:16px 16px; background:#f4f7fb;">
          <div style="max-width:720px; margin:0 auto; background:#ffffff; border-radius:20px; overflow:hidden; box-shadow:0 18px 45px rgba(15, 23, 42, 0.10); border:1px solid #e2e8f0;">

            <!-- Header -->
            <div style="background:linear-gradient(135deg, #0891b2 0%, #2563eb 55%, #4f46e5 100%); padding:34px 32px; color:#000000; font-color:#000000;">
              <div style="font-size:13px; letter-spacing:0.18em; text-transform:uppercase; opacity:0.9; font-weight:700;">
                Samah.ai Portfolio Assistant
              </div>
              <h1 style="margin:12px 0 0; font-size:28px; line-height:1.25; font-weight:800;">
                Your Chat Conversation History
              </h1>
              <p style="margin:12px 0 0; font-size:15px; line-height:1.7; opacity:0.95;">
                Here is a copy of the conversation you requested from the Samah.ai chatbot.
              </p>
            </div>

            <!-- Intro -->
            <div style="padding:28px 32px 10px;">
              <p style="margin:0; font-size:15px; line-height:1.8; color:#334155;">
                Hi there,
              </p>
              <p style="margin:10px 0 0; font-size:15px; line-height:1.8; color:#334155;">
                Thank you for chatting with Samah.ai. Your requested conversation transcript is included below for your reference.
              </p>
            </div>

            <!-- Transcript Card -->
            <div style="padding:18px 32px 30px;">
              <div style="border:1px solid #dbeafe; border-radius:18px; overflow:hidden; background:#f8fafc;">
                <div style="padding:14px 18px; background:#eff6ff; border-bottom:1px solid #dbeafe;">
                  <div style="font-size:13px; font-weight:700; color:#1d4ed8; letter-spacing:0.08em; text-transform:uppercase;">
                    Conversation Transcript
                  </div>
                </div>

                <div style="padding:20px; font-size:14px; line-height:1.75; color:#1e293b; white-space:normal;">
                  {safe_history_html}
                </div>
              </div>
            </div>

            <!-- CTA / Footer -->
            <div style="padding:0 32px 32px;">
              <div style="padding:20px; border-radius:18px; background:linear-gradient(135deg, #ecfeff 0%, #eef2ff 100%); border:1px solid #cffafe;">
                <p style="margin:0; font-size:14px; line-height:1.7; color:#334155;">
                  If you would like to discuss a project, collaboration, or opportunity with Samah, you can reply through the portfolio contact form.
                </p>
              </div>
            </div>

            <div style="padding:22px 32px; background:#0f172a; color:#cbd5e1; text-align:center;">
              <div style="font-size:15px; font-weight:700; color:#ffffff;">
                Samah.ai
              </div>
              <p style="margin:8px 0 0; font-size:12px; line-height:1.6;">
                AI Engineer • Full-Stack Developer • Portfolio Assistant
              </p>
              <p style="margin:10px 0 0; font-size:11px; line-height:1.6; color:#94a3b8;">
                This email was generated automatically from the Samah.ai portfolio chatbot.
              </p>
            </div>

          </div>
        </div>
      </body>
    </html>
    """

    payload = {
        "from": settings.CONTACT_FROM_EMAIL,
        "to": [recipient_email],
        "subject": "Your Samah.ai Chat Conversation History",
        "text": history_text,
        "html": html_content,
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
