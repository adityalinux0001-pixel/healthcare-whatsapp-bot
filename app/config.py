"""
Updated Configuration Module with Eleven Labs Support
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration with voice support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="forbid",
    )

    # WhatsApp Configuration
    phone_number_id: str
    whatsapp_token: str
    verify_token: str

    # LLM Configuration (Gemini)
    gemini_api_key: str

    # OpenAI (optional, for future use)
    openai_api_key: str = ""

    # Pinecone Vector Database
    pinecone_api_key: str
    pinecone_index_name: str
    pinecone_dimension: int = 1536  # Added

    # Eleven Labs Voice Configuration
    eleven_labs_api_key: str
    eleven_labs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"

    # Voice Settings
    enable_voice_mode: bool = False
    voice_preset: str = "professional"
    tts_stability: float = 0.5
    tts_similarity_boost: float = 0.75
    tts_model_id: str = "eleven_monolingual_v1"

    # Speech-to-Text Settings
    stt_enabled: bool = True
    stt_model_id: str = "eleven_flash_v2_latest"
    stt_language: str = "en"

    # Audio Processing
    max_audio_file_size_mb: int = 25
    audio_cache_enabled: bool = True
    audio_cache_expiry_hours: int = 24
    audio_output_format: str = "mp3"

    # RAG Configuration
    rag_top_k: int = 3
    max_history_turns: int = 5

    # Skip embedding + Pinecone query entirely for messages that are
    # obviously just chit-chat/greetings/acks (e.g. "hi", "thanks", "ok")
    # — these never benefit from knowledge-base context, so retrieving it
    # is pure wasted latency (one OpenAI embedding round-trip + one
    # Pinecone query round-trip) on every single such message.
    rag_skip_chitchat: bool = True

    # Cache query embeddings in Redis so a repeated/near-identical
    # question (very common in FAQ-style bots — "pricing?", "pricing",
    # "what's the pricing") doesn't pay for a fresh OpenAI embedding call
    # every time. Cache key is the normalized (lowercased/stripped) query
    # text, so it only helps on close-to-exact repeats — it's a latency
    # optimization, not a semantic cache.
    embedding_cache_enabled: bool = True
    embedding_cache_ttl_seconds: int = 6 * 60 * 60

    # Gemini queueing / concurrency control — caps how many Gemini calls
    # (main reply, language detection, summary, vision, etc. — everything
    # in app/llm.py) are in flight at once. Extra calls wait in an asyncio
    # queue instead of firing all at once and tripping Gemini's own
    # overload/rate limits. Raise this if you're on a higher Gemini quota
    # tier and want more throughput; lower it if you're still seeing
    # 503/UNAVAILABLE errors.
    # Raised from 5 -> 8. With ~20 concurrent users each producing 2-4
    # Gemini calls per turn (language detect, main reply, follow-up,
    # background summary), 5 in-flight was too tight and pushed most
    # calls into the retry path instead of just queueing briefly. 8 gives
    # more real throughput while still protecting lower Gemini quota
    # tiers from bursts. If you're on a paid tier with a higher per-
    # project QPS limit, this can go higher (e.g. 15-20); if you're still
    # seeing 503s on a low quota tier, lower it back down instead.
    gemini_max_concurrent_requests: int = 8

    # Raised from 3 -> 4 retries so a request landing during a brief
    # Gemini overload spike (common when many users message at once) gets
    # more chances to succeed via backoff before being dropped.
    gemini_max_retries: int = 4

    # Backoff between retries, in seconds: base * (2 ** attempt), plus a
    # small random jitter so many queued requests retrying at once don't
    # all hammer Gemini at the exact same instant.
    gemini_retry_base_delay_seconds: float = 1.0

    # Postgres connection (replaces the old SQLite file). Points at the
    # `db` service in docker-compose.yml by default — override via
    # DATABASE_URL in .env for a managed instance (Render/Railway/Supabase/
    # RDS/etc). Standard libpq URL: postgresql://user:pass@host:port/dbname
    database_url: str = "postgresql://whatsapp_bot:whatsapp_bot@db:5432/whatsapp_bot"

    # Min/max size of the Postgres connection pool *per worker process*.
    # With gunicorn running `web_concurrency` workers, total connections
    # to Postgres can be up to workers * db_pool_max — keep this in mind
    # against your Postgres max_connections (default 100).
    db_pool_min_size: int = 1
    db_pool_max_size: int = 5

    # Gunicorn worker concurrency. Used by docker-compose / start.sh.
    web_concurrency: int = 4

    # Redis connection — backs the cross-worker Gemini concurrency
    # semaphore and the message-idempotency guard, both of which need to
    # be shared state once you run more than one worker process. Points
    # at the `redis` service in docker-compose.yml by default.
    redis_url: str = "redis://redis:6379/0"

    # Queue backend selection. Supported values: rq, celery, huey.
    queue_backend: str = "rq"
    queue_name: str = "default"

    # Razorpay — Test Mode keys for now (dashboard toggle "Test/Live", keys
    # start with rzp_test_... in test mode, rzp_live_... in live mode).
    # key_secret and webhook_secret are two DIFFERENT secrets in Razorpay's
    # dashboard — do not confuse them.
    razorpay_key_id: str = ""
    razorpay_key_secret: str = ""
    razorpay_webhook_secret: str = ""

    # Premium plan shown as the upsell at the start of a new conversation.
    premium_plan_amount_rupees: int = 499
    premium_plan_days: int = 21

    # A "new session" for the premium upsell = the user's previous message
    # (if any) was this many seconds ago or older. Default 6 hours.
    # NOTE: no longer used to GATE the premium offer (the offer now checks
    # on every message, not just session starts) — kept only in case
    # something else references it. See premium_reoffer_min_gap_seconds
    # for the throttle that replaced this for premium-offer purposes.
    session_gap_seconds: int = 6 * 60 * 60

    # Minimum time to wait before re-sending a premium payment link to a
    # user who was already sent one but hasn't paid yet. Without this, once
    # the session-gap requirement is removed, a user chatting normally
    # without paying would get a fresh payment link on every single
    # message — this throttle stops that while still checking on every
    # message (not just "new sessions") so a plan that just expired gets
    # re-offered on the user's very next message instead of waiting for a
    # session gap that may never occur.
    premium_reoffer_min_gap_seconds: int = 24 * 60 * 60

    # Logging
    log_level: str = "INFO"


@lru_cache()
def get_settings() -> Settings:
    """Get application settings (cached)."""
    return Settings()