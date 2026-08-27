from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # WhatsApp Configuration
    PHONE_NUMBER_ID: str
    WHATSAPP_TOKEN: str
    VERIFY_TOKEN: str

    # LLM Configuration (Gemini)
    GEMINI_API_KEY: str

    # OpenAI (optional, for future use)
    OPENAI_API_KEY: str = ""

    MAX_HISTORY_TURNS: int = 5
    GEMINI_MAX_CONCURRENT_REQUESTS: int = 8
    GEMINI_MAX_RETRIES: int = 5
    GEMINI_RETRY_BASE_DELAY_SECONDS: float = 1.0
    GEMINI_RETRY_MAX_DELAY_SECONDS: float = 12.0

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
    KAFKA_BOOTSTRAP_SERVERS: str = "kafka:9092"

    # Inbound topic: webhook receivers publish raw WhatsApp messages here,
    KAFKA_INBOUND_TOPIC: str = "whatsapp.inbound"
    KAFKA_OUTBOUND_TOPIC: str = "whatsapp.outbound"
    KAFKA_NUM_PARTITIONS: int = 6
    KAFKA_INBOUND_GROUP_ID: str = "whatsapp-bot-inbound-workers"
    KAFKA_OUTBOUND_GROUP_ID: str = "whatsapp-bot-outbound-workers"

    KAFKA_PRODUCER_ACKS: str = "all"
    KAFKA_CONSUMER_AUTO_OFFSET_RESET: str = "earliest"

    KAFKA_WORKER_MAX_CONCURRENT: int = 6

    # --- Razorpay PG ---
    RAZORPAY_KEY_ID: str = ""
    RAZORPAY_KEY_SECRET: str = ""
    RAZORPAY_WEBHOOK_SECRET: str = ""



    # --- PhonePe PG (Standard Checkout v2) ---
    # PHONEPE_CLIENT_ID: str = ""
    # PHONEPE_CLIENT_SECRET: str = ""
    # PHONEPE_CLIENT_VERSION: str = "1"
    # PHONEPE_ENV: str = "production"
    # PHONEPE_WEBHOOK_USERNAME: str = ""
    # PHONEPE_WEBHOOK_PASSWORD: str = ""
    # PHONEPE_REDIRECT_URL: str = ""
    # PHONEPE_LINK_EXPIRE_AFTER_SECONDS: int = 3600

    WHATSAPP_NUMBER: str = ""  # The dialable WhatsApp Business number in international format, no symbols (e.g. "919876543210")

    # Premium plan shown as the upsell at the start of a new conversation.
    PREMIUM_PLAN_AMOUNT_RUPEES: int = 499
    PREMIUM_PLAN_DAYS: int = 21

    DAILY_CHECKIN_ENABLED: bool = True
    DAILY_CHECKIN_HOUR_UTC: int = 9
    DAILY_CHECKIN_MIN_GAP_HOURS: int = 20

    # WhatsApp only allows free-form text within 24h of the user's last
    # inbound message; outside that window a plain send_text_message call
    # is accepted by the API ("sent") but never actually delivered to the
    # phone. This must be the name of a Meta-approved template (with one
    # text parameter for the day number) used to re-open the session when
    # a premium user's check-in is due but they haven't messaged recently.
    # Leave empty to keep the old (broken-outside-24h) behavior.
    DAILY_CHECKIN_REENGAGEMENT_TEMPLATE: str = ""
    # Safety margin under the real 24h limit, so a check-in queued right
    # at the edge doesn't get sent as free text a few minutes before the
    # window actually closes.
    WHATSAPP_SESSION_WINDOW_HOURS: float = 23.0

    DEFAULT_PLAN_CATEGORY: str = "weight_loss"
    PLAN_GENERATION_MAX_OUTPUT_TOKENS: int = 14000

    SYMPTOM_INTAKE_MAX_QUESTIONS: int = 4
    SYMPTOM_INTAKE_SESSION_TIMEOUT_SECONDS: int = 60 * 60
    PREMIUM_REOFFER_MIN_GAP_SECONDS: int = 24 * 60 * 60

    # How many health questions a non-premium user gets answered for free
    # before every further question gets the payment link + app features
    # instead of an actual answer. Resets to 0 whenever the user buys/
    # renews Premium (see ConversationMemory.activate_subscription).
    FREE_QUESTION_LIMIT: int = 5

    # Logging
    LOG_LEVEL: str = "INFO"


@lru_cache()
def get_settings() -> Settings:
    """Get application settings (cached)."""
    return Settings()