"""
Updated Pydantic Models with Voice Message Support
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


# ============ AUDIO MESSAGE MODELS ============

class AudioObject(BaseModel):
    """Audio message data."""
    id: str = Field(..., description="Audio media ID")
    mime_type: Optional[str] = "audio/ogg"
    

class AudioMessage(BaseModel):
    """Audio message structure."""
    type: str = "audio"
    audio: AudioObject
    text: Optional[str] = None  # Transcribed text (added by bot)


# ============ EXISTING MESSAGE MODELS (with audio support) ============

class TextObject(BaseModel):
    """Text message data."""
    body: str = Field(..., description="Message text")


class ImageObject(BaseModel):
    """Image message data."""
    id: str = Field(..., description="Image media ID")
    mime_type: Optional[str] = "image/jpeg"


class TextMessage(BaseModel):
    """Text message structure."""
    type: str = "text"
    text: TextObject


class ImageMessage(BaseModel):
    """Image message structure."""
    type: str = "image"
    image: ImageObject


# ============ UNIFIED MESSAGE TYPES ============

class IncomingMessage(BaseModel):
    """
    Unified incoming message from WhatsApp.
    Supports: text, image, audio
    """
    type: str = Field(..., description="Message type: text, image, or audio")
    id: str = Field(..., description="Message ID")
    from_: str = Field(..., alias="from", description="Sender phone number")
    timestamp: str
    text: Optional[TextObject] = None
    image: Optional[ImageObject] = None
    audio: Optional[AudioObject] = None
    # WhatsApp interactive replies (button taps), e.g.
    # {"type": "button_reply", "button_reply": {"id": "...", "title": "..."}}
    interactive: Optional[dict] = None

    class Config:
        populate_by_name = True  # Allow 'from' field
    
    @classmethod
    def from_raw(cls, raw_msg: dict) -> "IncomingMessage":
        """
        Parse incoming message from WhatsApp webhook.
        
        Example raw message:
        {
            "from": "1234567890",
            "id": "wamid.xxx",
            "timestamp": "1234567890",
            "type": "text",
            "text": {"body": "Hello"}
        }
        """
        if isinstance(raw_msg, BaseModel):
            raw_msg = raw_msg.model_dump(by_alias=True)
        return cls(**raw_msg)


# ============ REQUEST/RESPONSE MODELS ============

class TestMessageRequest(BaseModel):
    """Test sending a full message flow."""
    message: str
    to: str


class TestTemplateRequest(BaseModel):
    """Test sending a template."""
    to: str
    template_name: str = "hello_world"


class TestAudioFlowRequest(BaseModel):
    """Test full audio flow."""
    to: str
    use_voice_response: bool = True


class WebhookErrorData(BaseModel):
    """Extra detail Meta sometimes attaches to a status error."""
    details: Optional[str] = None


class WebhookError(BaseModel):
    """
    One entry from a failed status's `errors` array — e.g. code 131047
    ("Re-engagement message" — more than 24h since the customer's last
    reply, so the free-form message was never delivered). Previously
    WebhookStatus had no `errors` field at all, so Pydantic silently
    dropped this on every failed-status webhook and main.py's log line
    only ever showed the bare "failed" with no reason.
    """
    code: Optional[int] = None
    title: Optional[str] = None
    message: Optional[str] = None
    error_data: Optional[WebhookErrorData] = None


class WebhookStatus(BaseModel):
    """Message status from Meta."""
    id: str
    recipient_id: str
    status: str
    timestamp: str
    errors: Optional[List[WebhookError]] = None


class WebhookContact(BaseModel):
    """Contact info from Meta."""
    profile: Optional[dict] = None
    wa_id: Optional[str] = None


class WebhookMessage(BaseModel):
    """Individual message from webhook."""
    from_: str = Field(..., alias="from")
    id: str
    timestamp: str
    type: str
    text: Optional[dict] = None
    image: Optional[dict] = None
    audio: Optional[dict] = None
    interactive: Optional[dict] = None
    
    class Config:
        populate_by_name = True


class WebhookMetadata(BaseModel):
    """Metadata from webhook."""
    display_phone_number: str
    phone_number_id: str


class WebhookValue(BaseModel):
    """Value object from webhook entry."""
    messaging_product: str
    metadata: WebhookMetadata
    statuses: Optional[List[WebhookStatus]] = None
    messages: Optional[List[WebhookMessage]] = None
    contacts: Optional[List[WebhookContact]] = None


class WebhookChange(BaseModel):
    """Change object from webhook."""
    value: WebhookValue
    field: str


class WebhookEntry(BaseModel):
    """Entry object from webhook."""
    id: str
    changes: List[WebhookChange]


class WebhookPayload(BaseModel):
    """Complete webhook payload from Meta."""
    object: str
    entry: List[WebhookEntry]


# ============ HEALTH/DEBUG MODELS ============

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    bot: str
    phone_number_id: str
    active_sessions: int
    supported_media: List[str]


class DebugResponse(BaseModel):
    """Debug information response."""
    config: dict
    token_check: dict
    active_sessions: int
    media_support: str