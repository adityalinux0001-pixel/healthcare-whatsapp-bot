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


class TestRAGRequest(BaseModel):
    """Test RAG retrieval."""
    user_message: str
    top_k: int = 3


class TestAudioFlowRequest(BaseModel):
    """Test full audio flow."""
    to: str
    use_voice_response: bool = True


class IngestTextRequest(BaseModel):
    """Ingest plain text into Pinecone."""
    text: str = Field(..., description="Document text")
    source: str = Field(..., description="Document label (e.g., 'faq', 'team')")
    chunk_tokens: int = Field(300, description="Tokens per chunk")
    overlap_tokens: int = Field(50, description="Overlap between chunks")


class DeleteSourceRequest(BaseModel):
    """Delete all vectors for a source."""
    source: str


class WebhookStatus(BaseModel):
    """Message status from Meta."""
    id: str
    recipient_id: str
    status: str
    timestamp: str


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


# ============ VOICE RESPONSE MODELS ============

class VoiceResponseConfig(BaseModel):
    """Configuration for voice responses."""
    enabled: bool = True
    voice_preset: str = "professional"  # professional, friendly, deep, dynamic, calm
    stability: float = 0.5
    similarity_boost: float = 0.75
    model_id: str = "eleven_monolingual_v1"


class AudioProcessingResult(BaseModel):
    """Result of audio processing."""
    success: bool
    text: Optional[str] = None
    error: Optional[str] = None
    audio_size: str
    processing_time: float  # milliseconds
    confidence: Optional[float] = None


class VoiceMessageResponse(BaseModel):
    """Response containing voice message."""
    status: str
    message_id: Optional[str] = None
    audio_bytes_sent: Optional[int] = None
    fallback_text: Optional[str] = None
    error: Optional[str] = None


# ============ RAG & CONTEXT MODELS ============

class ContextChunk(BaseModel):
    """Retrieved context chunk from Pinecone."""
    id: str
    source: str
    text: str
    score: float
    metadata: Optional[dict] = None


class RAGResponse(BaseModel):
    """RAG retrieval response."""
    chunks_retrieved: int
    chunks: List[ContextChunk]
    llm_reply: str
    rag_used: bool


# ============ CONVERSATION MODELS ============

class ConversationTurn(BaseModel):
    """Single turn in conversation history."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[datetime] = None


class ConversationHistory(BaseModel):
    """Conversation history for a user."""
    user_id: str
    turns: List[ConversationTurn]
    created_at: datetime
    last_message_at: datetime


# ============ HEALTH/DEBUG MODELS ============

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    bot: str
    phone_number_id: str
    pinecone_index: str
    active_sessions: int
    supported_media: List[str]


class DebugResponse(BaseModel):
    """Debug information response."""
    config: dict
    token_check: dict
    pinecone: dict
    active_sessions: int
    media_support: str