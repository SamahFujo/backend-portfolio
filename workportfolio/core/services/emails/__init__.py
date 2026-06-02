"""
Email service package.

This file makes `core.services.emails` a Python package and allows clean imports
from the email service modules.
"""

from .resend_email import *
from .resend_contact_email import *
from .resend_verification_email import *