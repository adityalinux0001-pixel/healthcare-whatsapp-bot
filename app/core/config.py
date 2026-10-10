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
    DAILY_CHECKIN_HOUR_UTC: int = 2  # 3:00 UTC = 8:30 AM IST
    DAILY_CHECKIN_MIN_GAP_HOURS: int = 8  # min hours between two plan messages (Day 1 is sent right after payment)

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
    PLAN_GENERATION_MAX_OUTPUT_TOKENS: int = 16000  # per 7-day chunk (plan is generated in chunks)

    SYMPTOM_INTAKE_MAX_QUESTIONS: int = 4
    SYMPTOM_INTAKE_SESSION_TIMEOUT_SECONDS: int = 60 * 60
    PREMIUM_REOFFER_MIN_GAP_SECONDS: int = 24 * 60 * 60

    # How many health questions a non-premium user gets answered for free
    # before every further question gets the payment link + app features
    # instead of an actual answer. Resets to 0 whenever the user buys/
    # renews Premium (see ConversationMemory.activate_subscription).
    # Free questions before payment. 0 = no free questions: the payment offer
    # is sent right after onboarding and every question is gated until paid.
    FREE_QUESTION_LIMIT: int = 0

    # Per-user daily chat limit (Meta bills every outbound service message).
    # Applies to PAID (premium) users only — it starts counting after payment.
    # Before payment the FREE_QUESTION_LIMIT (5 questions total) applies instead.
    # Each paid user may send DAILY_MESSAGE_LIMIT chat messages per day; after the
    # last allowed one they get a single reminder, further messages that day
    # are ignored silently (no reply = no cost), and the counter resets at
    # local midnight. 0 = unlimited.
    # Can be changed WITHOUT redeploy / restart, e.g. to test with 5:
    #   docker compose exec redis redis-cli set config:daily_message_limit 5
    #   docker compose exec redis redis-cli del config:daily_message_limit   (back to .env value)
    DAILY_MESSAGE_LIMIT: int = 25
    # Day boundary for the reset. 330 = India (IST, UTC+5:30).
    DAILY_LIMIT_UTC_OFFSET_MINUTES: int = 330
    DAILY_LIMIT_REACHED_MESSAGE: str = (
        "⚠️ You've reached today's limit of {limit} messages.\n"
        "Your limit resets tomorrow 🌅 — you can chat with me again then. Thank you for understanding 🙏"
    )

    # Daily task confirmation (WhatsApp reply buttons).
    # Every daily plan message is followed by a "Done / Not done" button
    # message. The user's tap re-opens the 24h service window AND is the
    # gate for tomorrow's plan: no confirmation -> next day is NOT sent.
    TASK_CONFIRM_NOTE: str = (
        "⚠️ Note: please confirm whether you completed today's task or not, "
        "otherwise tomorrow's task will not be sent."
    )
    TASK_CONFIRM_DONE_LABEL: str = "✅ Task done"      # max 20 chars (Meta limit)
    TASK_CONFIRM_NOT_DONE_LABEL: str = "❌ Not done"   # max 20 chars (Meta limit)
    TASK_CONFIRM_DONE_ACK: str = "Great job! 🎉 Tomorrow's task will arrive at your chosen time."
    TASK_CONFIRM_NOT_DONE_ACK: str = "No worries 🙂 Try again — tomorrow's task will arrive at your chosen time."

    # Logging
    LOG_LEVEL: str = "INFO"


@lru_cache()
def get_settings() -> Settings:
    """Get application settings (cached)."""
    return Settings()