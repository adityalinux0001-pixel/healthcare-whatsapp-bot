"""
Updated Configuration Module with Eleven Labs Support
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="forbid",
    )

    # WhatsApp Configuration
    PHONE_NUMBER_ID: str
    WHATSAPP_TOKEN: str
    VERIFY_TOKEN: str

    # LLM Configuration (Gemini)
    GEMINI_API_KEY: str

    # OpenAI (optional, for future use)
    OPENAI_API_KEY: str = ""

    # Eleven Labs Voice Configuration
    ELEVEN_LABS_API_KEY: str
    ELEVEN_LABS_VOICE_ID: str = "21m00Tcm4TlvDq8ikWAM"

    # Voice Settings
    ENABLE_VOICE_MODE: bool = False
    VOICE_PRESET: str = "professional"
    TTS_STABILITY: float = 0.5
    TTS_SIMILARITY_BOOST: float = 0.75
    TTS_MODEL_ID: str = "eleven_monolingual_v1"

    # Speech-to-Text Settings
    STT_ENABLED: bool = True
    STT_MODEL_ID: str = "eleven_flash_v2_latest"
    STT_LANGUAGE: str = "en"

    # Audio Processing
    MAX_AUDIO_FILE_SIZE_MB: int = 25
    AUDIO_CACHE_ENABLED: bool = True
    AUDIO_CACHE_EXPIRY_HOURS: int = 24
    AUDIO_OUTPUT_FORMAT: str = "mp3"

    MAX_HISTORY_TURNS: int = 5


    GEMINI_MAX_CONCURRENT_REQUESTS: int = 8

    # Raised from 3 -> 4 retries so a request landing during a brief
    # Gemini overload spike (common when many users message at once) gets
    # more chances to succeed via backoff before being dropped.
    GEMINI_MAX_RETRIES: int = 4

    # Backoff between retries, in seconds: base * (2 ** attempt), plus a
    # small random jitter so many queued requests retrying at once don't
    # all hammer Gemini at the exact same instant.
    GEMINI_RETRY_BASE_DELAY_SECONDS: float = 1.0


    DATABASE_URL: str = "postgresql://whatsapp_bot:whatsapp_bot@db:5432/whatsapp_bot"


    DB_POOL_MIN_SIZE: int = 1
    DB_POOL_MAX_SIZE: int = 5

    # Gunicorn worker concurrency. Used by docker-compose / start.sh.
    WEB_CONCURRENCY: int = 4


    REDIS_URL: str = "redis://redis:6379/0"

    # Queue backend selection. Supported values: rq, celery, huey, kafka.
    QUEUE_BACKEND: str = "rq"
    QUEUE_NAME: str = "default"

    # --- Kafka (optional queue_backend="kafka") ---
    # Bootstrap servers, comma-separated (e.g. "kafka:9092" in Docker Compose,
    # or a cluster of "broker1:9092,broker2:9092" in production).
    KAFKA_BOOTSTRAP_SERVERS: str = "kafka:9092"

    # Inbound topic: webhook receivers publish raw WhatsApp messages here,
    # keyed by phone number so all messages from one user land on the same
    # partition (preserves per-user ordering; see kafka_client.py).
    KAFKA_INBOUND_TOPIC: str = "whatsapp.inbound"


    KAFKA_OUTBOUND_TOPIC: str = "whatsapp.outbound"


    KAFKA_NUM_PARTITIONS: int = 6


    KAFKA_INBOUND_GROUP_ID: str = "whatsapp-bot-inbound-workers"
    KAFKA_OUTBOUND_GROUP_ID: str = "whatsapp-bot-outbound-workers"

    # librdkafka producer/consumer tuning. acks=all is the safe default
    # (wait for all in-sync replicas) — drop to "1" only if you've decided
    # you can tolerate occasionally losing a message on broker failover.
    KAFKA_PRODUCER_ACKS: str = "all"
    KAFKA_CONSUMER_AUTO_OFFSET_RESET: str = "earliest"


    KAFKA_WORKER_MAX_CONCURRENT: int = 6


    RAZORPAY_KEY_ID: str = ""
    RAZORPAY_KEY_SECRET: str = ""
    RAZORPAY_WEBHOOK_SECRET: str = ""

    # Premium plan shown as the upsell at the start of a new conversation.
    PREMIUM_PLAN_AMOUNT_RUPEES: int = 499
    PREMIUM_PLAN_DAYS: int = 21


    DAILY_CHECKIN_ENABLED: bool = True
    DAILY_CHECKIN_HOUR_UTC: int = 9
    DAILY_CHECKIN_MIN_GAP_HOURS: int = 20


    DEFAULT_PLAN_CATEGORY: str = "weight_loss"

    # How long (seconds) a user's onboarding session may sit idle before
    # a fresh "/premium" or payment restarts it from question 1 instead
    # of resuming a stale, possibly-abandoned session.
    ONBOARDING_SESSION_TIMEOUT_SECONDS: int = 60 * 60 * 24


    PLAN_GENERATION_MAX_OUTPUT_TOKENS: int = 8000


    SYMPTOM_INTAKE_MAX_QUESTIONS: int = 4


    SYMPTOM_INTAKE_SESSION_TIMEOUT_SECONDS: int = 60 * 60


    SESSION_GAP_SECONDS: int = 6 * 60 * 60


    PREMIUM_REOFFER_MIN_GAP_SECONDS: int = 24 * 60 * 60

    # Logging
    LOG_LEVEL: str = "INFO"


@lru_cache()
def get_settings() -> Settings:
    """Get application settings (cached)."""
    return Settings()