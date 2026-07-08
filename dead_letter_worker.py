"""
Dead-letter replay worker.

Run this as its own long-lived process (own container/service, same
pattern as worker.py and outbound_worker.py) alongside the app.

WHY THIS EXISTS
----------------
app/queueing.py's enqueue_incoming() retries transient publish failures
inline, then — if the queue backend is still unavailable after those
retries — persists the raw message to the `dead_letter_messages` table in
Postgres instead of losing it (see app/memory.py's save_dead_letter()).

That's enough to guarantee no message is silently dropped, but a message
sitting in Postgres forever still isn't "delivered" — the user is still
waiting on a reply. This worker is the other half: it periodically wakes
up, pulls pending dead-lettered rows, and re-attempts the exact same
enqueue_incoming() call that failed the first time. If the underlying
outage (Kafka producer not ready, broker down, network blip) has cleared,
the message flows through normally and gets marked resolved. If it's
still failing, it's left pending and gets picked up again next cycle.

This gives the pipeline a genuine at-least-once guarantee end-to-end:
webhook -> enqueue (retried) -> dead-letter (durable) -> replay (retried)
-> normal processing, with the only unrecoverable case being Postgres
itself being down (already logged at CRITICAL where it happens).

USAGE
-----
    python dead_letter_worker.py

Configure poll interval / batch size via env vars (see Settings below) or
just edit the constants — this is intentionally a small, simple loop
rather than another queue backend, since dead-letter volume should be
near-zero in steady state.
"""

from __future__ import annotations

import logging
import sys
import time

from app.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("dead_letter_worker")

# How often to check for pending dead-lettered messages.
POLL_INTERVAL_SECONDS = 30

# How many pending rows to attempt per cycle. Kept small on purpose —
# this worker is a safety net for rare failures, not a bulk pipeline; if
# this number is regularly maxed out, that's a signal the real queue
# backend is unhealthy and needs attention, not a reason to raise the
# batch size.
BATCH_SIZE = 50

# After this many total attempts (see dead_letter_messages.attempts,
# incremented both on the original enqueue_incoming failures AND on
# replay failures below), stop auto-retrying and mark the row
# failed_permanently so an operator investigates instead of it silently
# retrying forever against a message that can never succeed (e.g.
# malformed payload).
MAX_TOTAL_ATTEMPTS = 10


def _replay_one(memory, row: dict) -> None:
    from app.queueing import enqueue_incoming

    dead_letter_id = row["id"]
    phone_number = row["phone_number"]
    payload = row["payload"]
    attempts = row["attempts"]

    if attempts >= MAX_TOTAL_ATTEMPTS:
        logger.error(
            f"❌ Dead-letter id={dead_letter_id} phone={phone_number} exceeded "
            f"{MAX_TOTAL_ATTEMPTS} total attempts — marking failed_permanently "
            "for manual review."
        )
        memory.mark_dead_letter_failed_permanently(dead_letter_id)
        return

    try:
        # enqueue_incoming has its own retry loop internally too, so this
        # single call already gets a few attempts against the queue
        # backend before falling through to re-dead-lettering (which
        # save_dead_letter() correctly treats as "same pending row, bump
        # attempts" rather than creating a duplicate).
        job_ref = enqueue_incoming(payload)
        if job_ref.startswith("dead_letter:"):
            # enqueue_incoming re-dead-lettered it again (still failing).
            # save_dead_letter() already bumped attempts/last_failed_at
            # for us — nothing else to do this cycle.
            logger.warning(
                f"⏳ Dead-letter id={dead_letter_id} phone={phone_number} still "
                "failing — left pending for next cycle."
            )
            return

        logger.info(
            f"✅ Dead-letter id={dead_letter_id} phone={phone_number} replayed "
            f"successfully (job_ref={job_ref}) — marking resolved."
        )
        memory.mark_dead_letter_resolved(dead_letter_id)
    except Exception:
        # Only reachable if enqueue_incoming's own dead-letter fallback
        # also failed (i.e. Postgres itself is unreachable) — nothing
        # more this worker can do this cycle.
        logger.error(
            f"❌ Replay of dead-letter id={dead_letter_id} phone={phone_number} "
            "raised unexpectedly — will retry next cycle.",
            exc_info=True,
        )


def main() -> None:
    from app.main import memory  # reuses the app's Postgres connection pool

    settings = get_settings()
    logger.info(
        "🟢 Dead-letter replay worker started | poll_interval=%ss batch_size=%s "
        "queue_backend=%s",
        POLL_INTERVAL_SECONDS,
        BATCH_SIZE,
        settings.queue_backend,
    )

    while True:
        try:
            pending = memory.get_pending_dead_letters(limit=BATCH_SIZE)
            if pending:
                logger.info(f"🔁 Replaying {len(pending)} pending dead-lettered message(s)")
                for row in pending:
                    _replay_one(memory, row)
            else:
                logger.debug("No pending dead-lettered messages.")
        except KeyboardInterrupt:
            logger.info("Dead-letter worker interrupted — shutting down.")
            return
        except Exception:
            logger.error("Unexpected error in dead-letter worker loop.", exc_info=True)

        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()