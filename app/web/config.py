"""
Single source of truth for every business-identifying value shown on the
public website. Edit THIS FILE (or override via env vars if you prefer)
before submitting to Razorpay — nothing else in templates/ should hardcode
a business fact.

Plan price/duration default to the same values the bot itself already uses
(app.core.config.Settings.PREMIUM_PLAN_AMOUNT_RUPEES /
PREMIUM_PLAN_DAYS) so the website can never drift out of sync with what
customers are actually charged.
"""

from pydantic import BaseModel
from app.core.config import get_settings

_settings = get_settings()


class SiteConfig(BaseModel):
    BUSINESS_NAME: str = "Steves AI"
    PRODUCT_NAME: str = "AI Health Assistant"
    TAGLINE: str = "AI-powered health support, right on WhatsApp"

    SUPPORT_EMAIL: str = "info@stevesailab.com"
    SUPPORT_PHONE: str = ""  # optional — leave blank to hide on Contact page
    BUSINESS_ADDRESS: str = "Indore"

    # The dialable WhatsApp Business number in international format, no
    # symbols (e.g. "919876543210"), NOT the Meta PHONE_NUMBER_ID. Purchases
    # happen inside WhatsApp chat (see app/services/onboarding.py), so the
    # site's "Subscribe Now" button opens a wa.me chat rather than a web
    # checkout form.
    WHATSAPP_BUSINESS_NUMBER: str = _settings.WHATSAPP_NUMBER
    WHATSAPP_CHAT_PREFILL: str = "Hi! I'd like to start my 21-day premium plan."

    PLAN_NAME: str = "21-Day Premium Plan"
    PLAN_PRICE_RUPEES: int = _settings.PREMIUM_PLAN_AMOUNT_RUPEES
    PLAN_DURATION_DAYS: int = _settings.PREMIUM_PLAN_DAYS

    REFUND_PROCESSING_DAYS: str = "5-7 business days"
    EFFECTIVE_DATE: str = "REPLACE_WITH_DATE"
    GOVERNING_LAW: str = "India"

    #PAYMENT_URL: str ="https://rzp.io/rzp/yFobbTbu"

    @property
    def plan_price_display(self) -> str:
        return f"₹{self.PLAN_PRICE_RUPEES}"

    @property
    def whatsapp_chat_url(self) -> str:
        import urllib.parse
        text = urllib.parse.quote(self.WHATSAPP_CHAT_PREFILL)
        return f"https://wa.me/{self.WHATSAPP_BUSINESS_NUMBER}?text={text}"


site = SiteConfig()
