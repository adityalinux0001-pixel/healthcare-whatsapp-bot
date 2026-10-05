"""
Single place that delivers one premium plan day to a user.

Used by all three send paths (so they can never drift apart again):
  - app/services/daily_checkin.py      (scheduled daily job)
  - app/services/onboarding.py         (Day 1, right after payment)
  - app/api/main.py                    (_flush_pending_checkin_day)

Message-count optimisation (Meta bills per outbound service message):
  OLD: plan text + follow-up question text            = 2 messages/day
  NEW: plan text + [follow-up + note + Done/Not-done] = 2 messages/day
       and just ONE message when everything fits in WhatsApp's 1024-char
       button-body limit.

The button message does two jobs:
  1. It collects the user's "task done / not done" confirmation.
  2. The tap is an inbound message, so it re-opens the 24h service window
     — which is exactly what lets us send tomorrow's plan as free-form
     (see the gate in daily_checkin._send_checkin_for_user: tomorrow is
     only sent once today is confirmed).
"""

import asyncio
import logging

from app.core.config import get_settings
from app.services.whatsapp import send_text_message, send_reply_buttons

logger = logging.getLogger(__name__)
settings = get_settings()

TASK_DONE_PREFIX = "task_done:"
TASK_NOT_DONE_PREFIX = "task_not_done:"

_BUTTON_BODY_LIMIT = 1024


def build_confirm_body(followup_question: str | None) -> str:
    parts = []
    if followup_question:
        parts.append(followup_question.strip())
    parts.append(settings.TASK_CONFIRM_NOTE)
    return "\n\n".join(parts)


def _buttons_for(day_number: int) -> list[tuple[str, str]]:
    return [
        (f"{TASK_DONE_PREFIX}{day_number}", settings.TASK_CONFIRM_DONE_LABEL),
        (f"{TASK_NOT_DONE_PREFIX}{day_number}", settings.TASK_CONFIRM_NOT_DONE_LABEL),
    ]


async def send_plan_day(
    memory,
    phone_number: str,
    plan_day: dict,
    prefix: str = "",
) -> bool:
    """
    Send one plan day + its Done/Not-done confirmation buttons, mark the
    day as sent and save everything to chat history. The caller must
    already have claimed the day (memory.claim_plan_day_for_send).

    prefix: optional text glued in front of the day message (used by
    Day 1 so "your plan is ready" doesn't cost its own message).

    Returns True if the plan content itself was delivered.
    """
    day_number = plan_day["day_number"]
    header = f"*Day {day_number} of {settings.PREMIUM_PLAN_DAYS}* 🗓️\n\n"
    message = f"{prefix}{header}{plan_day['message_text']}"
    confirm_body = build_confirm_body(plan_day.get("followup_question"))

    combined = f"{message}\n\n{confirm_body}"
    single_message = len(combined) <= _BUTTON_BODY_LIMIT

    try:
        if single_message:
            await send_reply_buttons(phone_number, combined, _buttons_for(day_number))
        else:
            await send_text_message(phone_number, message)
    except Exception as e:
        logger.error(f"❌ Failed to send day {day_number} to {phone_number}: {e}", exc_info=True)
        return False

    await asyncio.to_thread(memory.mark_plan_day_sent, phone_number, day_number)
    await asyncio.to_thread(
        memory.save_message, phone_number, "assistant",
        combined if single_message else message, message_type="text",
    )
    logger.info(
        f"✅ Sent day {day_number}/{settings.PREMIUM_PLAN_DAYS} to {phone_number} "
        f"({'1 message' if single_message else '2 messages'})"
    )

    if single_message:
        return True

    # Plan text was too long to share a message with the buttons.
    try:
        await send_reply_buttons(phone_number, confirm_body[:_BUTTON_BODY_LIMIT], _buttons_for(day_number))
        await asyncio.to_thread(
            memory.save_message, phone_number, "assistant", confirm_body, message_type="text"
        )
    except Exception as e:
        logger.error(
            f"❌ Buttons failed for day {day_number} / {phone_number}, falling back to plain text: {e}",
            exc_info=True,
        )
        try:
            await send_text_message(phone_number, confirm_body)
            await asyncio.to_thread(
                memory.save_message, phone_number, "assistant", confirm_body, message_type="text"
            )
        except Exception as e2:
            logger.error(f"❌ Fallback confirm text also failed for {phone_number}: {e2}", exc_info=True)
    return True
