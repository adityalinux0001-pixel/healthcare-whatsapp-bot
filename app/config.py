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

    max_history_turns: int = 5

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

    # Queue backend selection. Supported values: rq, celery, huey, kafka.
    queue_backend: str = "rq"
    queue_name: str = "default"

    # --- Kafka (optional queue_backend="kafka") ---
    # Bootstrap servers, comma-separated (e.g. "kafka:9092" in Docker Compose,
    # or a cluster of "broker1:9092,broker2:9092" in production).
    kafka_bootstrap_servers: str = "kafka:9092"

    # Inbound topic: webhook receivers publish raw WhatsApp messages here,
    # keyed by phone number so all messages from one user land on the same
    # partition (preserves per-user ordering; see kafka_client.py).
    kafka_inbound_topic: str = "whatsapp.inbound"

    # Outbound topic: anything that should be sent back to WhatsApp is
    # published here instead of calling the Cloud API inline. A dedicated
    # outbound consumer (outbound_worker.py) drains this topic and does the
    # actual HTTP POSTs to WhatsApp, keyed by phone number for the same
    # per-user-ordering reason as inbound.
    kafka_outbound_topic: str = "whatsapp.outbound"

    # Number of partitions to create for each topic if auto-creation is used
    # or if you provision them yourself with kafka-topics.sh. More
    # partitions = more parallel workers can consume concurrently, but
    # ordering is only guaranteed within a partition (i.e. per phone
    # number, given the partitioning key below), not across the topic.
    kafka_num_partitions: int = 6

    # Consumer group IDs — all worker processes with the same group id
    # share the topic's partitions (each partition goes to exactly one
    # consumer in the group at a time), giving you horizontal scaling by
    # just starting more worker processes/containers.
    kafka_inbound_group_id: str = "whatsapp-bot-inbound-workers"
    kafka_outbound_group_id: str = "whatsapp-bot-outbound-workers"

    # librdkafka producer/consumer tuning. acks=all is the safe default
    # (wait for all in-sync replicas) — drop to "1" only if you've decided
    # you can tolerate occasionally losing a message on broker failover.
    kafka_producer_acks: str = "all"
    kafka_consumer_auto_offset_reset: str = "earliest"

    # How many inbound messages a single worker process will handle
    # concurrently (as asyncio tasks) instead of strictly one-at-a-time.
    # This is what actually fixes "second/third user has to wait for the
    # first user's reply to finish" — before this, a single worker
    # process processed messages fully serially even though the topic
    # has multiple partitions, because handle_message() blocked the only
    # consumer loop until it returned.
    #
    # NOTE: this USED to need to stay <= kafka_num_partitions, back when
    # run_consumer_loop_async gated concurrency per-partition (a single
    # partition could only have one message in flight). That gate is now
    # per phone-number-key instead (see run_consumer_loop_async's
    # docstring in app/kafka_client.py) — per-user ordering is preserved
    # without tying this to the partition count, so it's fine to set
    # this higher than kafka_num_partitions if you have enough concurrent
    # users to benefit and your downstream (Gemini rate limits, DB pool
    # size) can handle it.
    kafka_worker_max_concurrent: int = 6

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

    # Daily health check-in (premium feature) — every day of the 21-day
    # premium plan, a background job SENDS the pre-generated message for
    # that day (see PREMIUM PLAN PREGENERATION below — no LLM call happens
    # here anymore), then a same-day follow-up question is asked and the
    # conversation continues naturally from the user's reply (handled by
    # the normal webhook flow in app/main.py). hour is in 24h format, UTC
    # by default — adjust daily_checkin_hour/timezone via .env to match
    # your users.
    daily_checkin_enabled: bool = True
    daily_checkin_hour_utc: int = 9
    daily_checkin_min_gap_hours: int = 20

    # ------------------------------------------------------------------
    # PREMIUM PLAN PREGENERATION (onboarding -> one LLM call -> 21 rows)
    # ------------------------------------------------------------------
    # Default plan category. This is the single knob that applies when a
    # user doesn't pick one explicitly — every current user gets
    # "weight_loss". Additional categories (e.g. "yoga", "bulking") are
    # added later purely as new prompt templates in app/llm.py's
    # PLAN_CATEGORY_PROMPTS dict; nothing else in the pipeline needs to
    # change to support them.
    default_plan_category: str = "weight_loss"

    # How long (seconds) a user's onboarding session may sit idle before
    # a fresh "/premium" or payment restarts it from question 1 instead
    # of resuming a stale, possibly-abandoned session.
    onboarding_session_timeout_seconds: int = 60 * 60 * 24

    # Safety ceiling for the single "generate all 21 days" Gemini call —
    # this one call matters more than a normal chat turn (if it fails,
    # the whole premium plan fails to materialize), so it gets its own
    # generous output budget separate from other calls.
    plan_generation_max_output_tokens: int = 8000

    # Symptom intake — hard cap on how many short one-at-a-time questions
    # (see SYMPTOM INTAKE MODE in app/llm.py's SYSTEM_PROMPT) the bot may
    # ask before it is FORCED (in code, not left to the LLM's judgment) to
    # stop asking and give its actual answer/guidance instead. This exists
    # because relying on the model alone to decide "I have enough now" was
    # unreliable — it sometimes kept asking indefinitely, or repeated a
    # question it had already asked.
    symptom_intake_max_questions: int = 4

    # If a user goes quiet for longer than this after the last intake
    # question, the NEXT message is treated as a fresh conversation (the
    # question counter resets) rather than continuing to count against an
    # abandoned symptom discussion from hours/days ago.
    symptom_intake_session_timeout_seconds: int = 60 * 60

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