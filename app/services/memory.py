import os
import logging
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import List, Dict, Optional

from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row

logger = logging.getLogger(__name__)


class ConversationMemory:
    """
    Enhanced conversation memory with support for:
    - Text and audio message storage
    - Message retrieval with last 5 context
    - Timestamp tracking
    - Audio file management

    STEP 1 OF THE MULTI-WORKER MIGRATION: this used to be a SQLite file.
    SQLite's single writer lock is fine for one process, but it becomes
    the bottleneck the moment you run more than one worker (step 4) —
    concurrent writers just queue up behind the file lock. Postgres
    handles concurrent writers natively and shares state across worker
    processes over the network instead of a local file.

    The public method signatures are unchanged from the SQLite version on
    purpose: every caller in app/main.py invokes these via
    `asyncio.to_thread(memory.some_method, ...)`, and psycopg's sync
    driver is blocking just like sqlite3 was — so that calling convention
    stays correct without touching main.py at all.
    """

    # Project root = the directory that contains the "app" package.
    # Class-level so _resolve_path (a classmethod) can use it before/
    # without an instance existing.
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def __init__(self, database_url: Optional[str] = None,
                 audio_dir: str = "audio_storage",
                 pool_min_size: int = 1,
                 pool_max_size: int = 5):
        from app.core.config import get_settings
        settings = get_settings()

        self.database_url = database_url or settings.DATABASE_URL


        self.audio_dir = self._resolve_path(audio_dir)
        os.makedirs(self.audio_dir, exist_ok=True)

        logger.info(f"ConversationMemory connecting to Postgres: {self._safe_url()}")


        self.pool = ConnectionPool(
            conninfo=self.database_url,
            min_size=pool_min_size,
            max_size=pool_max_size,
            kwargs={"autocommit": False},
        )
        self.init_db()

    def _safe_url(self) -> str:
        """Database URL with the password redacted, for logging."""
        try:
            if "@" in self.database_url and "://" in self.database_url:
                scheme, rest = self.database_url.split("://", 1)
                creds, host = rest.split("@", 1)
                user = creds.split(":", 1)[0]
                return f"{scheme}://{user}:***@{host}"
        except Exception:
            pass
        return "***"

    @classmethod
    def _resolve_path(cls, path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.join(cls._PROJECT_ROOT, path)

    @contextmanager
    def _get_conn(self):
        """Borrow a connection from the pool. Commits on success, rolls
        back and re-raises on error — mirrors the old SQLite context
        manager's behavior so every method body below stays the same
        shape as before."""
        with self.pool.connection() as conn:
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def init_db(self):
        try:
            self._init_db()
        except Exception:
            logger.error(
                f"❌ Failed to initialize/connect to Postgres at {self._safe_url()} — "
                f"check DATABASE_URL and that the database is reachable.",
                exc_info=True,
            )
            raise

    def _init_db(self):
        with self._get_conn() as conn:
            cur = conn.cursor()

            cur.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    phone_number TEXT PRIMARY KEY,
                    name TEXT,
                    summary TEXT DEFAULT '',
                    last_message_at TIMESTAMPTZ,
                    created_at TIMESTAMPTZ DEFAULT now()
                )
            ''')


            cur.execute('''
                ALTER TABLE users
                ADD COLUMN IF NOT EXISTS category TEXT NOT NULL DEFAULT 'weight_loss'
            ''')


            cur.execute('''
                ALTER TABLE users
                ADD COLUMN IF NOT EXISTS preferred_checkin_hour_utc INTEGER
            ''')

            cur.execute('''
                CREATE TABLE IF NOT EXISTS chat_history (
                    id BIGSERIAL PRIMARY KEY,
                    phone_number TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    message_type TEXT DEFAULT 'text',
                    audio_file_path TEXT,
                    audio_transcription TEXT,
                    timestamp TIMESTAMPTZ NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT now(),
                    FOREIGN KEY (phone_number) REFERENCES users(phone_number)
                )
            ''')

            cur.execute('''
                CREATE INDEX IF NOT EXISTS idx_phone_timestamp
                ON chat_history(phone_number, timestamp DESC)
            ''')



            cur.execute('''
                CREATE TABLE IF NOT EXISTS followup_suggestions (
                    id BIGSERIAL PRIMARY KEY,
                    phone_number TEXT NOT NULL,
                    suggestion TEXT NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT now()
                )
            ''')
            cur.execute('''
                CREATE INDEX IF NOT EXISTS idx_followup_phone
                ON followup_suggestions(phone_number, created_at DESC)
            ''')

            cur.execute('''
                CREATE TABLE IF NOT EXISTS payment_links (
                    payment_link_id TEXT PRIMARY KEY,
                    phone_number TEXT NOT NULL,
                    amount_paise INTEGER NOT NULL,
                    status TEXT NOT NULL DEFAULT 'created',
                    razorpay_payment_id TEXT,
                    short_url TEXT,
                    created_at TIMESTAMPTZ DEFAULT now(),
                    paid_at TIMESTAMPTZ
                )
            ''')

            cur.execute('''
                ALTER TABLE payment_links ADD COLUMN IF NOT EXISTS short_url TEXT
            ''')
            cur.execute('''
                CREATE INDEX IF NOT EXISTS idx_payment_links_phone
                ON payment_links(phone_number, created_at DESC)
            ''')

            cur.execute('''
                CREATE TABLE IF NOT EXISTS subscriptions (
                    phone_number TEXT PRIMARY KEY,
                    plan_name TEXT NOT NULL DEFAULT 'premium_21day',
                    started_at TIMESTAMPTZ NOT NULL,
                    expires_at TIMESTAMPTZ NOT NULL,
                    payment_link_id TEXT,
                    expiry_notified BOOLEAN NOT NULL DEFAULT FALSE,
                    updated_at TIMESTAMPTZ DEFAULT now()
                )
            ''')


            cur.execute('''
                ALTER TABLE subscriptions
                ADD COLUMN IF NOT EXISTS expiry_notified BOOLEAN NOT NULL DEFAULT FALSE
            ''')


            cur.execute('''
                CREATE TABLE IF NOT EXISTS daily_checkins (
                    id BIGSERIAL PRIMARY KEY,
                    phone_number TEXT NOT NULL,
                    day_number INTEGER NOT NULL,
                    message TEXT NOT NULL,
                    sent_at TIMESTAMPTZ DEFAULT now()
                )
            ''')
            cur.execute('''
                CREATE INDEX IF NOT EXISTS idx_daily_checkins_phone
                ON daily_checkins(phone_number, sent_at DESC)
            ''')


            cur.execute('''
                CREATE TABLE IF NOT EXISTS symptom_sessions (
                    phone_number TEXT PRIMARY KEY,
                    question_count INTEGER NOT NULL DEFAULT 0,
                    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
            ''')


            cur.execute('''
                CREATE TABLE IF NOT EXISTS premium_plans (
                    id BIGSERIAL PRIMARY KEY,
                    phone_number TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT 'weight_loss',
                    day_number INTEGER NOT NULL,
                    message_text TEXT NOT NULL,
                    followup_question TEXT,
                    followup_answer TEXT,
                    followup_answered_at TIMESTAMPTZ,
                    created_at TIMESTAMPTZ DEFAULT now(),
                    sent_at TIMESTAMPTZ,
                    -- awaiting_followup marks the window between "today's
                    -- message was just sent" and "the user has answered (or
                    -- the next day's message supersedes it)" — the webhook
                    -- handler in app/main.py checks this to decide whether
                    -- an incoming text should be captured as this day's
                    -- follow-up answer instead of going through the normal
                    -- Q&A path.
                    awaiting_followup BOOLEAN NOT NULL DEFAULT FALSE,
                    UNIQUE (phone_number, day_number)
                )
            ''')
            cur.execute('''
                CREATE INDEX IF NOT EXISTS idx_premium_plans_phone_day
                ON premium_plans(phone_number, day_number)
            ''')
            cur.execute('''
                CREATE INDEX IF NOT EXISTS idx_premium_plans_awaiting
                ON premium_plans(phone_number, awaiting_followup)
                WHERE awaiting_followup = TRUE
            ''')


            cur.execute('''
                CREATE TABLE IF NOT EXISTS onboarding_sessions (
                    phone_number TEXT PRIMARY KEY,
                    category TEXT NOT NULL DEFAULT 'weight_loss',
                    question_index INTEGER NOT NULL DEFAULT 0,
                    answers JSONB NOT NULL DEFAULT '{}'::jsonb,
                    is_complete BOOLEAN NOT NULL DEFAULT FALSE,
                    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
            ''')


            cur.execute('''
                CREATE TABLE IF NOT EXISTS dead_letter_messages (
                    id BIGSERIAL PRIMARY KEY,
                    phone_number TEXT NOT NULL,
                    payload JSONB NOT NULL,
                    failure_reason TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 1,
                    status TEXT NOT NULL DEFAULT 'pending',
                    first_failed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    last_failed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    resolved_at TIMESTAMPTZ
                )
            ''')
            cur.execute('''
                CREATE INDEX IF NOT EXISTS idx_dead_letter_status
                ON dead_letter_messages(status, first_failed_at)
            ''')
            cur.execute('''
                CREATE INDEX IF NOT EXISTS idx_dead_letter_phone
                ON dead_letter_messages(phone_number, first_failed_at DESC)
            ''')

            conn.commit()

    # ------------------------------------------------------------------
    # Users
    # ------------------------------------------------------------------

    def get_customer(self, phone_number: str) -> Dict:
        with self._get_conn() as conn:
            cur = conn.cursor(row_factory=dict_row)
            cur.execute(
                "SELECT name, summary, last_message_at FROM users WHERE phone_number = %s",
                (phone_number,),
            )
            row = cur.fetchone()
            if row:
                return dict(row)
            return {"name": "Customer", "summary": "", "last_message_at": None}

    def save_message(
        self,
        phone_number: str,
        role: str,
        content: str,
        message_type: str = "text",
        audio_file_path: Optional[str] = None,
        audio_transcription: Optional[str] = None,
    ) -> int:
        now = datetime.utcnow()
        try:
            with self._get_conn() as conn:
                cur = conn.cursor()

                cur.execute('''
                    INSERT INTO chat_history
                    (phone_number, role, content, message_type, audio_file_path, audio_transcription, timestamp)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                ''', (phone_number, role, content, message_type, audio_file_path, audio_transcription, now))
                message_id = cur.fetchone()[0]

                cur.execute('''
                    INSERT INTO users (phone_number, last_message_at)
                    VALUES (%s, %s)
                    ON CONFLICT (phone_number) DO UPDATE SET last_message_at = EXCLUDED.last_message_at
                ''', (phone_number, now))

                conn.commit()
                logger.debug(
                    f"Saved message id={message_id} phone={phone_number} role={role} type={message_type}"
                )
                return message_id
        except Exception:
            logger.error(
                f"❌ Failed to save message for {phone_number} (role={role}, type={message_type}) "
                f"to Postgres",
                exc_info=True,
            )
            raise

    def get_last_messages(self, phone_number: str, limit: int = 5) -> List[Dict]:
        with self._get_conn() as conn:
            cur = conn.cursor(row_factory=dict_row)
            cur.execute('''
                SELECT id, role, content, message_type, audio_transcription, timestamp
                FROM chat_history
                WHERE phone_number = %s
                ORDER BY timestamp DESC
                LIMIT %s
            ''', (phone_number, limit))
            rows = cur.fetchall()
            return [dict(row) for row in reversed(rows)]

    def get_conversation_context(self, phone_number: str, limit: int = 5) -> str:
        messages = self.get_last_messages(phone_number, limit)
        if not messages:
            return ""

        context_lines = ["[Recent Conversation Context]"]
        for msg in messages:
            role_display = "User" if msg['role'] == "user" else "Assistant"
            msg_type = f" ({msg['message_type'].upper()})" if msg['message_type'] != "text" else ""
            text_content = msg.get('audio_transcription') or msg['content']
            ts = msg.get('timestamp')
            timestamp = ts.strftime("%Y-%m-%d %H:%M:%S") if hasattr(ts, "strftime") else str(ts)[:19]
            context_lines.append(f"{timestamp} - {role_display}{msg_type}: {text_content}")

        return "\n".join(context_lines)

    def update_summary(self, phone_number: str, new_summary: str) -> None:
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute('''
                INSERT INTO users (phone_number, summary)
                VALUES (%s, %s)
                ON CONFLICT (phone_number) DO UPDATE SET summary = EXCLUDED.summary
            ''', (phone_number, new_summary))
            conn.commit()

    def set_user_name(self, phone_number: str, name: str) -> None:
        if not name:
            return
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute('''
                INSERT INTO users (phone_number, name)
                VALUES (%s, %s)
                ON CONFLICT (phone_number) DO UPDATE SET name = EXCLUDED.name
            ''', (phone_number, name))
            conn.commit()

    # ------------------------------------------------------------------
    # Audio files (unchanged — still local disk)
    # ------------------------------------------------------------------

    def save_audio_file(self, phone_number: str, audio_bytes: bytes) -> str:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        phone_safe = phone_number.replace("+", "").replace("-", "")
        filename = f"{phone_safe}_{timestamp}.ogg"
        filepath = os.path.join(self.audio_dir, filename)
        with open(filepath, 'wb') as f:
            f.write(audio_bytes)
        return filepath

    def get_audio_file(self, filepath: str) -> Optional[bytes]:
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                return f.read()
        return None

    def delete_old_audio_files(self, phone_number: str, keep_count: int = 10) -> None:
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute('''
                SELECT audio_file_path FROM chat_history
                WHERE phone_number = %s AND audio_file_path IS NOT NULL
                ORDER BY timestamp DESC
                OFFSET %s
            ''', (phone_number, keep_count))
            old_files = cur.fetchall()
            for (filepath,) in old_files:
                if filepath and os.path.exists(filepath):
                    os.remove(filepath)

    def get_message_count(self, phone_number: str) -> int:
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT COUNT(*) FROM chat_history WHERE phone_number = %s",
                (phone_number,),
            )
            return cur.fetchone()[0]

    # ------------------------------------------------------------------
    # Follow-up suggestions
    # ------------------------------------------------------------------

    def save_followup_suggestion(self, phone_number: str, suggestion: str) -> None:
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO followup_suggestions (phone_number, suggestion) VALUES (%s, %s)",
                (phone_number, suggestion),
            )
            conn.commit()

    def get_recent_followups(self, phone_number: str, limit: int = 5) -> List[str]:
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute('''
                SELECT suggestion FROM followup_suggestions
                WHERE phone_number = %s
                ORDER BY created_at DESC
                LIMIT %s
            ''', (phone_number, limit))
            return [row[0] for row in cur.fetchall()]

    def get_user_stats(self, phone_number: str) -> Dict:
        with self._get_conn() as conn:
            cur = conn.cursor(row_factory=dict_row)
            cur.execute('''
                SELECT
                    COUNT(*) as total_messages,
                    SUM(CASE WHEN role='user' THEN 1 ELSE 0 END) as user_messages,
                    SUM(CASE WHEN role='assistant' THEN 1 ELSE 0 END) as assistant_messages,
                    SUM(CASE WHEN message_type='audio' THEN 1 ELSE 0 END) as audio_messages
                FROM chat_history
                WHERE phone_number = %s
            ''', (phone_number,))
            row = cur.fetchone()
            return {
                'total_messages': row['total_messages'] or 0,
                'user_messages': row['user_messages'] or 0,
                'assistant_messages': row['assistant_messages'] or 0,
                'audio_messages': row['audio_messages'] or 0,
            }

    # ------------------------------------------------------------------
    # Session detection
    # ------------------------------------------------------------------

    def get_seconds_since_last_message(self, phone_number: str) -> Optional[float]:
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT last_message_at FROM users WHERE phone_number = %s",
                (phone_number,),
            )
            row = cur.fetchone()
            if not row or not row[0]:
                return None
            last = row[0]
            if last.tzinfo is not None:
                last = last.replace(tzinfo=None)
            return (datetime.utcnow() - last).total_seconds()

    # ------------------------------------------------------------------
    # Razorpay payments
    # ------------------------------------------------------------------

    def save_payment_link(
        self, payment_link_id: str, phone_number: str, amount_paise: int,
        short_url: Optional[str] = None,
    ) -> None:
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute('''
                INSERT INTO payment_links (payment_link_id, phone_number, amount_paise, status, short_url)
                VALUES (%s, %s, %s, 'created', %s)
                ON CONFLICT (payment_link_id) DO NOTHING
            ''', (payment_link_id, phone_number, amount_paise, short_url))
            conn.commit()

    def get_payment_link(self, payment_link_id: str) -> Optional[Dict]:
        with self._get_conn() as conn:
            cur = conn.cursor(row_factory=dict_row)
            cur.execute(
                "SELECT * FROM payment_links WHERE payment_link_id = %s",
                (payment_link_id,),
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def get_latest_payment_link_for_user(self, phone_number: str) -> Optional[Dict]:
        """
        Most recent payment link created for this user, regardless of
        status. Used to throttle repeat premium offers — if we already
        sent one recently and the user hasn't paid yet, we don't want to
        blast a fresh link on literally every message they send.
        """
        with self._get_conn() as conn:
            cur = conn.cursor(row_factory=dict_row)
            cur.execute('''
                SELECT * FROM payment_links
                WHERE phone_number = %s
                ORDER BY created_at DESC
                LIMIT 1
            ''', (phone_number,))
            row = cur.fetchone()
            return dict(row) if row else None

    def mark_payment_link_paid(self, payment_link_id: str, razorpay_payment_id: str) -> Optional[str]:
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT phone_number, status FROM payment_links WHERE payment_link_id = %s",
                (payment_link_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            phone_number, status = row
            if status == "paid":
                return phone_number
            cur.execute('''
                UPDATE payment_links
                SET status = 'paid', razorpay_payment_id = %s, paid_at = %s
                WHERE payment_link_id = %s
            ''', (razorpay_payment_id, datetime.utcnow(), payment_link_id))
            conn.commit()
            return phone_number

    def activate_subscription(self, phone_number: str, days: int, payment_link_id: str,
                               plan_name: str = "premium_21day") -> str:
        now = datetime.utcnow()
        expires_at = now + timedelta(days=days)
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute('''
                INSERT INTO subscriptions (phone_number, plan_name, started_at, expires_at, payment_link_id, expiry_notified)
                VALUES (%s, %s, %s, %s, %s, FALSE)
                ON CONFLICT (phone_number) DO UPDATE SET
                    plan_name = EXCLUDED.plan_name,
                    started_at = EXCLUDED.started_at,
                    expires_at = EXCLUDED.expires_at,
                    payment_link_id = EXCLUDED.payment_link_id,
                    expiry_notified = FALSE,
                    updated_at = now()
            ''', (phone_number, plan_name, now, expires_at, payment_link_id))
            conn.commit()
        return expires_at.isoformat()

    def get_subscription(self, phone_number: str) -> Optional[Dict]:
        with self._get_conn() as conn:
            cur = conn.cursor(row_factory=dict_row)
            cur.execute(
                "SELECT * FROM subscriptions WHERE phone_number = %s",
                (phone_number,),
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def is_premium_active(self, phone_number: str) -> bool:
        sub = self.get_subscription(phone_number)
        if not sub:
            return False
        expires_at = sub["expires_at"]
        if expires_at is None:
            return False
        if expires_at.tzinfo is not None:
            expires_at = expires_at.replace(tzinfo=None)
        return datetime.utcnow() < expires_at

    def mark_subscription_expiry_notified(self, phone_number: str) -> None:
        """
        Marks that the user has already been sent the one-time "your plan
        has expired" message for their CURRENT subscription row, so
        _maybe_send_premium_offer() sends that specific message exactly
        once per expiry instead of on every message after expiry.
        Reset back to FALSE automatically the next time they activate a
        new subscription (see activate_subscription), so the next expiry
        gets its own one-time notification too.
        """
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute('''
                UPDATE subscriptions
                SET expiry_notified = TRUE, updated_at = now()
                WHERE phone_number = %s
            ''', (phone_number,))
            conn.commit()

    # ------------------------------------------------------------------
    # Daily premium check-ins (21-day plan)
    # ------------------------------------------------------------------

    def get_checkins_sent_count(self, phone_number: str, since: datetime) -> int:
        """How many daily check-ins have already been sent for this user
        since `since` (normally the subscription's started_at) — used to
        compute which day number of the plan today is."""
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute('''
                SELECT COUNT(*) FROM daily_checkins
                WHERE phone_number = %s AND sent_at >= %s
            ''', (phone_number, since))
            return cur.fetchone()[0]

    def get_last_checkin_sent_at(self, phone_number: str) -> Optional[datetime]:
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute('''
                SELECT sent_at FROM daily_checkins
                WHERE phone_number = %s
                ORDER BY sent_at DESC
                LIMIT 1
            ''', (phone_number,))
            row = cur.fetchone()
            return row[0] if row else None

    def get_recent_checkin_messages(self, phone_number: str, limit: int = 5) -> List[str]:
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute('''
                SELECT message FROM daily_checkins
                WHERE phone_number = %s
                ORDER BY sent_at DESC
                LIMIT %s
            ''', (phone_number, limit))
            return [row[0] for row in cur.fetchall()]

    def save_daily_checkin(self, phone_number: str, day_number: int, message: str) -> None:
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute('''
                INSERT INTO daily_checkins (phone_number, day_number, message)
                VALUES (%s, %s, %s)
            ''', (phone_number, day_number, message))
            conn.commit()

    def get_active_premium_users(self) -> List[Dict]:
        """All users whose 21-day (configurable) premium subscription is
        currently active — the candidate list the scheduled daily
        check-in job iterates over once a day. Includes
        preferred_checkin_hour_utc (NULL if the user was never asked /
        hasn't answered yet) so the scheduler can send each user's
        check-in at THEIR chosen hour instead of one fixed hour for
        everyone."""
        with self._get_conn() as conn:
            cur = conn.cursor(row_factory=dict_row)
            cur.execute('''
                SELECT s.phone_number, s.started_at, s.expires_at,
                       u.preferred_checkin_hour_utc
                FROM subscriptions s
                LEFT JOIN users u ON u.phone_number = s.phone_number
                WHERE s.expires_at > now()
            ''')
            return [dict(row) for row in cur.fetchall()]

    def set_preferred_checkin_hour(self, phone_number: str, hour_utc: int) -> None:
        """Store the user's chosen daily check-in hour (0-23, UTC),
        collected once during onboarding. Upserts into `users` so it
        works even if save_premium_plan's INSERT ... ON CONFLICT for this
        phone_number hasn't run yet."""
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute('''
                INSERT INTO users (phone_number, preferred_checkin_hour_utc)
                VALUES (%s, %s)
                ON CONFLICT (phone_number) DO UPDATE SET
                    preferred_checkin_hour_utc = EXCLUDED.preferred_checkin_hour_utc
            ''', (phone_number, hour_utc))
            conn.commit()

    def get_preferred_checkin_hour(self, phone_number: str) -> Optional[int]:
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT preferred_checkin_hour_utc FROM users WHERE phone_number = %s",
                (phone_number,),
            )
            row = cur.fetchone()
            return row[0] if row and row[0] is not None else None

    # ------------------------------------------------------------------
    # Premium plan pregeneration (21 rows written once, sent one/day)
    # ------------------------------------------------------------------

    def save_premium_plan(self, phone_number: str, category: str, days: List[Dict]) -> None:
        """
        Persist the ENTIRE pregenerated plan in one transaction: `days` is
        a list of {"message": str, "followup_question": str} in day order
        (index 0 = day 1), as returned by
        app.llm.generate_premium_plan(). Called exactly once, right after
        onboarding finishes — see app/onboarding.py.

        Replaces any existing plan rows for this user first (safe to call
        again if, say, an operator wants to regenerate a user's plan from
        scratch), so this is idempotent per call.
        """
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM premium_plans WHERE phone_number = %s", (phone_number,))
            for day_number, day in enumerate(days, start=1):
                cur.execute('''
                    INSERT INTO premium_plans
                        (phone_number, category, day_number, message_text, followup_question)
                    VALUES (%s, %s, %s, %s, %s)
                ''', (phone_number, category, day_number, day["message"], day["followup_question"]))
            cur.execute('''
                INSERT INTO users (phone_number, category)
                VALUES (%s, %s)
                ON CONFLICT (phone_number) DO UPDATE SET category = EXCLUDED.category
            ''', (phone_number, category))
            conn.commit()
        logger.info(f"✅ Saved {len(days)}-day '{category}' premium plan for {phone_number}")

    def get_premium_plan_day(self, phone_number: str, day_number: int) -> Optional[Dict]:
        """Fetch the pre-written row for a specific day — this is the ONLY
        thing the daily scheduler does to get content; no LLM call."""
        with self._get_conn() as conn:
            cur = conn.cursor(row_factory=dict_row)
            cur.execute('''
                SELECT * FROM premium_plans
                WHERE phone_number = %s AND day_number = %s
            ''', (phone_number, day_number))
            row = cur.fetchone()
            return dict(row) if row else None

    def get_next_unsent_plan_day(self, phone_number: str) -> Optional[Dict]:
        """The lowest-numbered day for this user that hasn't been sent yet
        (sent_at IS NULL) — this IS the "compute current day number" +
        "fetch row" step from the architecture diagram, combined: instead
        of separately counting elapsed days, we just take whichever
        pregenerated row is next in line. Returns None once all days (or
        no plan at all) exist."""
        with self._get_conn() as conn:
            cur = conn.cursor(row_factory=dict_row)
            cur.execute('''
                SELECT * FROM premium_plans
                WHERE phone_number = %s AND sent_at IS NULL
                ORDER BY day_number ASC
                LIMIT 1
            ''', (phone_number,))
            row = cur.fetchone()
            return dict(row) if row else None

    def mark_plan_day_sent(self, phone_number: str, day_number: int) -> None:
        """Mark today's row as sent and open the same-day follow-up
        window (awaiting_followup=TRUE) so the next incoming text message
        from this user is captured as the answer to THIS day's
        followup_question instead of going through normal Q&A."""
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute('''
                UPDATE premium_plans
                SET sent_at = now(), awaiting_followup = TRUE
                WHERE phone_number = %s AND day_number = %s
            ''', (phone_number, day_number))
            conn.commit()

    def get_awaiting_followup_day(self, phone_number: str) -> Optional[Dict]:
        """The plan day (if any) currently waiting on a same-day follow-up
        answer from this user. Checked by the webhook handler on every
        incoming text so a reply right after a check-in gets logged
        against the right day instead of just vanishing into normal chat."""
        with self._get_conn() as conn:
            cur = conn.cursor(row_factory=dict_row)
            cur.execute('''
                SELECT * FROM premium_plans
                WHERE phone_number = %s AND awaiting_followup = TRUE
                ORDER BY day_number DESC
                LIMIT 1
            ''', (phone_number,))
            row = cur.fetchone()
            return dict(row) if row else None

    def save_followup_answer(self, phone_number: str, day_number: int, answer: str) -> None:
        """Log the user's same-day follow-up answer and close the
        follow-up window. IMPORTANT: this only writes to this one row —
        by design (see architecture) it never touches message_text for
        any other day. Every other day's content stays exactly as
        pregenerated."""
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute('''
                UPDATE premium_plans
                SET followup_answer = %s, followup_answered_at = now(), awaiting_followup = FALSE
                WHERE phone_number = %s AND day_number = %s
            ''', (answer, phone_number, day_number))
            conn.commit()

    def has_premium_plan(self, phone_number: str) -> bool:
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT 1 FROM premium_plans WHERE phone_number = %s LIMIT 1",
                (phone_number,),
            )
            return cur.fetchone() is not None

    def get_user_category(self, phone_number: str) -> Optional[str]:
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT category FROM users WHERE phone_number = %s", (phone_number,))
            row = cur.fetchone()
            return row[0] if row else None

    # ------------------------------------------------------------------
    # Onboarding session (subscribe -> category -> questions -> plan gen)
    # ------------------------------------------------------------------

    def get_onboarding_session(self, phone_number: str) -> Optional[Dict]:
        with self._get_conn() as conn:
            cur = conn.cursor(row_factory=dict_row)
            cur.execute(
                "SELECT * FROM onboarding_sessions WHERE phone_number = %s",
                (phone_number,),
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def start_onboarding_session(self, phone_number: str, category: str) -> None:
        """Start (or restart from question 1) an onboarding session for
        this user with the given category. Called right after a
        subscription is activated (payment webhook) — see
        app/onboarding.py."""
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute('''
                INSERT INTO onboarding_sessions (phone_number, category, question_index, answers, is_complete, started_at, updated_at)
                VALUES (%s, %s, 0, '{}'::jsonb, FALSE, now(), now())
                ON CONFLICT (phone_number) DO UPDATE SET
                    category = EXCLUDED.category,
                    question_index = 0,
                    answers = '{}'::jsonb,
                    is_complete = FALSE,
                    started_at = now(),
                    updated_at = now()
            ''', (phone_number, category))
            conn.commit()

    def save_onboarding_answer(self, phone_number: str, question_key: str, answer: str) -> int:
        """Store the answer to the current question under `question_key`,
        advance question_index by 1, and return the NEW question_index
        (i.e. which question to ask next).

        Both %s placeholders inside jsonb_build_object are explicitly
        cast to ::text — without this, Postgres can't infer a type for
        the first parameter ($1, the key) since jsonb_build_object is
        variadic/polymorphic, and raises "could not determine data type
        of parameter $1" (seen in production logs) even though the value
        being passed is a perfectly normal string.
        """
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute('''
                UPDATE onboarding_sessions
                SET answers = answers || jsonb_build_object(%s::text, %s::text),
                    question_index = question_index + 1,
                    updated_at = now()
                WHERE phone_number = %s
                RETURNING question_index
            ''', (question_key, answer, phone_number))
            row = cur.fetchone()
            conn.commit()
            return row[0] if row else 0

    def mark_onboarding_complete(self, phone_number: str) -> Dict:
        """Mark the session complete and return the full accumulated
        answers dict, ready to hand to app.llm.generate_premium_plan()."""
        with self._get_conn() as conn:
            cur = conn.cursor(row_factory=dict_row)
            cur.execute('''
                UPDATE onboarding_sessions
                SET is_complete = TRUE, updated_at = now()
                WHERE phone_number = %s
                RETURNING answers, category
            ''', (phone_number,))
            row = cur.fetchone()
            conn.commit()
            return dict(row) if row else {"answers": {}, "category": "weight_loss"}

    def is_onboarding_in_progress(self, phone_number: str) -> bool:
        session = self.get_onboarding_session(phone_number)
        return bool(session and not session["is_complete"])

    # ------------------------------------------------------------------
    # Symptom intake session (deterministic question-count cap)
    # ------------------------------------------------------------------

    def get_symptom_question_count(self, phone_number: str) -> int:
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT question_count FROM symptom_sessions WHERE phone_number = %s",
                (phone_number,),
            )
            row = cur.fetchone()
            return row[0] if row else 0

    def increment_symptom_question_count(self, phone_number: str) -> int:
        """Increment (creating the row if needed) and return the new
        count. Called once every time the bot sends an intake-style
        question, so the count always reflects reality regardless of what
        the LLM itself thinks it has asked."""
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute('''
                INSERT INTO symptom_sessions (phone_number, question_count, started_at, updated_at)
                VALUES (%s, 1, now(), now())
                ON CONFLICT (phone_number) DO UPDATE SET
                    question_count = symptom_sessions.question_count + 1,
                    updated_at = now()
                RETURNING question_count
            ''', (phone_number,))
            new_count = cur.fetchone()[0]
            conn.commit()
            return new_count

    def reset_symptom_session(self, phone_number: str) -> None:
        """Reset the intake question counter back to 0 — called once the
        bot has actually given its answer/guidance (intake finished), so
        the NEXT new complaint starts counting from zero again."""
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute('''
                INSERT INTO symptom_sessions (phone_number, question_count, started_at, updated_at)
                VALUES (%s, 0, now(), now())
                ON CONFLICT (phone_number) DO UPDATE SET
                    question_count = 0,
                    started_at = now(),
                    updated_at = now()
            ''', (phone_number,))
            conn.commit()

    def get_symptom_session_age_seconds(self, phone_number: str) -> Optional[float]:
        """How long ago the current intake session started — used to
        auto-expire a stale session (e.g. user went quiet for hours then
        came back with something unrelated) so the counter doesn't
        artificially cap an unrelated, brand-new conversation."""
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT started_at FROM symptom_sessions WHERE phone_number = %s",
                (phone_number,),
            )
            row = cur.fetchone()
            if not row:
                return None
            started_at = row[0]
            if started_at.tzinfo is not None:
                started_at = started_at.replace(tzinfo=None)
            return (datetime.utcnow() - started_at).total_seconds()



    def save_dead_letter(
        self, phone_number: str, payload: dict, failure_reason: str,
    ) -> int:
        """Persist a failed enqueue. If an identical *pending* payload for
        this phone number already exists (e.g. two rapid retries hit the
        same underlying outage), bump its attempt count instead of
        creating a duplicate row."""
        import json as _json

        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute('''
                SELECT id FROM dead_letter_messages
                WHERE phone_number = %s
                  AND status = 'pending'
                  AND payload = %s::jsonb
                ORDER BY first_failed_at DESC
                LIMIT 1
            ''', (phone_number, _json.dumps(payload)))
            existing = cur.fetchone()

            if existing:
                dead_letter_id = existing[0]
                cur.execute('''
                    UPDATE dead_letter_messages
                    SET attempts = attempts + 1,
                        failure_reason = %s,
                        last_failed_at = now()
                    WHERE id = %s
                ''', (failure_reason, dead_letter_id))
            else:
                cur.execute('''
                    INSERT INTO dead_letter_messages
                        (phone_number, payload, failure_reason)
                    VALUES (%s, %s::jsonb, %s)
                    RETURNING id
                ''', (phone_number, _json.dumps(payload), failure_reason))
                dead_letter_id = cur.fetchone()[0]

            conn.commit()
            return dead_letter_id

    def get_pending_dead_letters(self, limit: int = 100) -> List[Dict]:
        """Oldest-first batch of not-yet-recovered failed messages —
        used by the replay worker / admin endpoint."""
        with self._get_conn() as conn:
            cur = conn.cursor(row_factory=dict_row)
            cur.execute('''
                SELECT * FROM dead_letter_messages
                WHERE status = 'pending'
                ORDER BY first_failed_at ASC
                LIMIT %s
            ''', (limit,))
            return [dict(row) for row in cur.fetchall()]

    def mark_dead_letter_resolved(self, dead_letter_id: int) -> None:
        """Call after successfully replaying a dead-lettered message."""
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute('''
                UPDATE dead_letter_messages
                SET status = 'resolved', resolved_at = now()
                WHERE id = %s
            ''', (dead_letter_id,))
            conn.commit()

    def mark_dead_letter_failed_permanently(self, dead_letter_id: int) -> None:
        """Call when a replay attempt itself fails again past a retry
        ceiling — keeps it out of get_pending_dead_letters() so the
        replay worker doesn't spin forever, while still keeping the row
        for manual/operator inspection."""
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute('''
                UPDATE dead_letter_messages
                SET status = 'failed_permanently'
                WHERE id = %s
            ''', (dead_letter_id,))
            conn.commit()

    def count_pending_dead_letters(self) -> int:
        """Cheap count for a health/metrics endpoint — alert if this
        stays above zero for more than a few minutes."""
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT COUNT(*) FROM dead_letter_messages WHERE status = 'pending'"
            )
            return cur.fetchone()[0]