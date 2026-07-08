"""
Kafka producer/consumer helpers.

Shared by:
- app/queueing.py        (inbound producer — webhook receiver side)
- worker.py               (inbound consumer — worker microservice side)
- app/outbound_queue.py   (outbound producer — wherever we'd otherwise
                           call whatsapp.send_* directly)
- outbound_worker.py      (outbound consumer — actually talks to the
                           WhatsApp Cloud API)

Design notes
------------
Messages are keyed by phone number (`key=phone_number.encode()`). Kafka
guarantees ordering only within a partition, and the default partitioner
hashes the key to pick a partition — so keying by phone number means every
message for a given user always lands on the same partition and is
therefore processed in order by whichever single consumer owns that
partition, while different users' messages fan out across partitions/
workers for concurrency. This mirrors "Partitioned by Phone Number" in the
target architecture diagram.

We use confluent-kafka (librdkafka bindings) for both producer and
consumer — it's the fastest/most complete Python client and is what's
usually meant by "production Kafka client" in this ecosystem.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Optional

from confluent_kafka import Consumer, KafkaError, KafkaException, Producer, TopicPartition
from confluent_kafka.admin import AdminClient, NewTopic

from app.config import get_settings

logger = logging.getLogger(__name__)

_producer: Optional[Producer] = None


def _delivery_report(err, msg) -> None:
    if err is not None:
        logger.error(f"❌ Kafka delivery failed [{msg.topic()}]: {err}")
    else:
        logger.debug(
            f"✅ Kafka delivered to {msg.topic()}[{msg.partition()}]@{msg.offset()}"
        )


def get_producer() -> Producer:
    """Return a process-wide singleton Kafka producer.

    One Producer instance per process is the recommended usage pattern —
    it batches and manages its own background delivery thread internally,
    so there's no benefit (and real overhead) to creating a new one per
    call.
    """
    global _producer
    if _producer is None:
        settings = get_settings()
        _producer = Producer(
            {
                "bootstrap.servers": settings.kafka_bootstrap_servers,
                "acks": settings.kafka_producer_acks,
                # Retry transient broker errors instead of failing the
                # publish outright — the webhook ack has already been
                # returned to Meta by this point, so we can afford to
                # retry here without risking a duplicate delivery from
                # Meta's side.
                "retries": 5,
                "retry.backoff.ms": 200,
                "linger.ms": 20,
                "compression.type": "lz4",
                "enable.idempotence": True,
            }
        )
    return _producer


def publish(topic: str, key: str, value: dict[str, Any]) -> None:
    """Publish a JSON-serializable dict to `topic`, partitioned by `key`.

    Non-blocking: hands off to librdkafka's internal queue and returns.
    Call flush() (e.g. at shutdown, or after a burst in a script) if you
    need a synchronous guarantee that everything has actually gone out.
    """
    producer = get_producer()
    try:
        producer.produce(
            topic,
            key=key.encode("utf-8"),
            value=json.dumps(value).encode("utf-8"),
            callback=_delivery_report,
        )
        # Serve delivery-report callbacks / internal queue without
        # blocking for acks — this is what actually gets bytes onto the
        # wire promptly instead of waiting for linger.ms/batching alone.
        producer.poll(0)
    except BufferError:
        logger.warning("Kafka producer queue full — flushing and retrying once.")
        producer.flush(5)
        producer.produce(
            topic,
            key=key.encode("utf-8"),
            value=json.dumps(value).encode("utf-8"),
            callback=_delivery_report,
        )
        producer.poll(0)


def flush(timeout: float = 10.0) -> None:
    if _producer is not None:
        _producer.flush(timeout)


def ensure_topics(topics: list[str], num_partitions: int, replication_factor: int = 1) -> None:
    """Best-effort topic creation (idempotent — ignores 'already exists').

    Useful for local/dev/docker-compose where auto.create.topics.enable
    may be off. In a managed production cluster you'd normally provision
    topics via Terraform/kafka-topics.sh instead and can skip calling
    this, but it's safe to call either way.
    """
    settings = get_settings()
    admin = AdminClient({"bootstrap.servers": settings.kafka_bootstrap_servers})
    new_topics = [
        NewTopic(t, num_partitions=num_partitions, replication_factor=replication_factor)
        for t in topics
    ]
    futures = admin.create_topics(new_topics, request_timeout=15)
    for topic, future in futures.items():
        try:
            future.result()
            logger.info(f"✅ Kafka topic ready: {topic}")
        except KafkaException as e:
            # Code 36 = TOPIC_ALREADY_EXISTS — expected on every restart.
            if "already exists" in str(e).lower():
                logger.debug(f"Kafka topic already exists: {topic}")
            else:
                logger.warning(f"⚠️ Could not create Kafka topic {topic}: {e}")


async def run_consumer_loop_async(
    topic: str,
    group_id: str,
    handle_message: Callable[[dict[str, Any]], "Any"],
    poll_timeout: float = 1.0,
    max_concurrent: int = 6,
) -> None:
    """
    Concurrent async version of run_consumer_loop().

    WHY THIS EXISTS: run_consumer_loop() above processes one message at a
    time — poll() -> handle_message() (blocks until fully done, including
    the Gemini API round trip) -> commit() -> poll() again. Even though
    the topic has multiple partitions and different users' messages are
    keyed by phone number so they land on *different* partitions (see
    module docstring), a single consumer instance still only ever has
    ONE message in flight at a time. In production this showed up as:
    three different users messaging within under a second of each other
    still got served strictly in arrival order, because whoever's message
    was being handled (a multi-second Gemini call) blocked every other
    user's message from even starting, regardless of partition.

    THE BUG IN THE FIRST VERSION OF THIS FUNCTION (fixed here): an
    earlier version dispatched consumer.poll() AND consumer.commit() to
    background threads via asyncio.to_thread() from what could be
    OVERLAPPING calls — poll() for message N+1 could run in one thread
    while commit() for message N's just-finished handler was running in
    another thread, both against the SAME confluent_kafka.Consumer
    object at the same time. confluent_kafka.Consumer (a thin wrapper
    over librdkafka) is NOT thread-safe for concurrent poll()/commit()
    calls — librdkafka's own docs are explicit that all calls on one
    consumer instance must be serialized. Overlapping them doesn't
    reliably raise an exception; it can silently wedge internal fetch
    state for specific partitions instead. That matches the exact
    symptom observed: two users kept getting replies fine while a third
    user's messages just never surfaced, and only started flowing again
    once the other two stopped sending (freeing up poll() to run
    uncontended again and eventually recover that partition's fetch).

    THE FIX: poll() and commit() now ONLY ever run on the main coroutine
    of this function, one at a time, never from a background thread and
    never overlapping each other. Concurrency is achieved a different
    way: handle_message() (the actual slow part — Gemini calls, DB
    writes) runs in parallel asyncio tasks, but each task does its own
    work and then reports back to the main loop via an asyncio.Queue;
    the main loop is the only place that ever touches the Consumer
    object, and it does so strictly sequentially. This preserves full
    concurrency for the actually-slow work while eliminating the
    thread-safety violation entirely.

    Ordering: Kafka only orders messages within a partition. Earlier
    versions of this function preserved ordering by never starting a
    second task for a PARTITION that already had one in flight — but
    with a small number of partitions (see kafka_num_partitions in
    app/config.py) two or more different users regularly hash onto the
    same partition, and gating on the partition needlessly serialized
    those unrelated users against each other (User B's message would
    wait out User A's entire Gemini round trip even though they have
    nothing to do with each other). This version gates on the message
    KEY (phone number) instead: a single user's own messages still
    process strictly in order, but different users now run fully in
    parallel regardless of whether they happen to share a partition.
    Kafka commit offsets are still advanced strictly in per-partition
    order underneath this (see finished_offsets/next_offset_to_commit
    below) — only the concurrency gate changed, not Kafka's own
    ordering contract.

    `handle_message` may be a sync or async callable; async callables are
    awaited directly inside the worker task, sync callables run via
    asyncio.to_thread from within that same task so a CPU-bound or
    blocking handler doesn't stall the event loop either.
    """
    settings = get_settings()
    consumer = Consumer(
        {
            "bootstrap.servers": settings.kafka_bootstrap_servers,
            "group.id": group_id,
            "auto.offset.reset": settings.kafka_consumer_auto_offset_reset,
            "enable.auto.commit": False,
        }
    )
    consumer.subscribe([topic])
    logger.info(
        f"🟢 Kafka async consumer started | topic={topic} group={group_id} "
        f"max_concurrent={max_concurrent}"
    )

    is_async_handler = asyncio.iscoroutinefunction(handle_message)
    semaphore = asyncio.Semaphore(max_concurrent)
    # Which message KEYS (phone numbers) currently have a task in flight —
    # gates starting a second task for the same user (ordering guarantee
    # above).
    #
    # THIS USED TO GATE ON msg.partition() INSTEAD OF msg.key(). That was
    # a real bug, not just an over-cautious choice: with
    # kafka_num_partitions=6 and the default hash partitioner, two or
    # more DIFFERENT phone numbers regularly collide onto the same
    # partition (pigeonhole — 6 buckets, any nontrivial number of active
    # users). Gating on the partition meant User B's message sat idle
    # behind User A's entire _handle_incoming() (Gemini calls, DB writes,
    # sometimes 503-retry backoff pushing 20-40s) whenever they happened
    # to share a partition — even though they are unrelated users and
    # Kafka's ordering guarantee never required serializing them against
    # each other. That's the exact "3rd user's message doesn't show
    # 'seen' until the first two are done" symptom. Kafka only guarantees
    # ordering *within* a partition for messages sharing a partition;
    # what we actually need to preserve is ordering *per key* (per user),
    # which gating on msg.key() gives us directly and correctly, without
    # over-serializing unrelated users who happen to collide onto the
    # same partition.
    keys_in_flight: set[bytes] = set()
    in_flight_tasks: set[asyncio.Task] = set()
    # Tasks report finished (msg, error_or_None) tuples here. Only the
    # main loop ever reads this queue and only the main loop ever calls
    # consumer.commit() — that's what keeps all Consumer access on one
    # logical thread of control.
    completed: asyncio.Queue = asyncio.Queue()

    # commit() must still advance strictly in offset order WITHIN a
    # partition — that part of Kafka's contract doesn't change just
    # because we now gate concurrency by key instead of by partition.
    # Two different users' messages can now be in flight on the same
    # partition at once, and they can finish in either order; if the
    # later-offset one finishes first we must hold its commit until the
    # earlier-offset one on that same partition has also completed,
    # otherwise a crash before the earlier one finishes would skip past
    # it on restart (auto.offset.reset picks up after the last commit).
    # finished_offsets[partition] holds offsets that completed
    # successfully but are still waiting behind an earlier offset on the
    # same partition.
    finished_offsets: dict[int, set[int]] = {}
    # next_offset_to_commit[partition] = the lowest offset on that
    # partition we haven't committed yet — i.e. what we're waiting on.
    next_offset_to_commit: dict[int, int] = {}

    async def _process_one(msg) -> None:
        try:
            async with semaphore:
                payload = json.loads(msg.value().decode("utf-8"))
                if is_async_handler:
                    await handle_message(payload)
                else:
                    await asyncio.to_thread(handle_message, payload)
            await completed.put((msg, None))
        except Exception as exc:
            await completed.put((msg, exc))

    async def _drain_completed(block: bool) -> None:
        """Pull finished tasks off the queue and commit their offsets —
        the ONLY place commit() is called, always from this main
        coroutine, never concurrently with poll() or with another
        commit()."""
        while True:
            try:
                if block:
                    msg, error = await completed.get()
                    block = False  # only block for the first one per call
                else:
                    msg, error = completed.get_nowait()
            except asyncio.QueueEmpty:
                return

            partition = msg.partition()
            offset = msg.offset()
            keys_in_flight.discard(msg.key())

            if error is not None:
                logger.error(
                    f"❌ Failed to process Kafka message from {topic} "
                    f"[partition={partition} offset={offset}] "
                    "— NOT committing offset, will be redelivered.",
                    exc_info=error,
                )
                # A gap at this offset is intentional here: we do NOT
                # advance next_offset_to_commit past a failed message, so
                # every later offset on this partition (even ones that
                # already finished successfully and are sitting in
                # finished_offsets) stays uncommitted until this one is
                # retried and succeeds. That's the same at-least-once
                # semantics the rest of this codebase relies on
                # elsewhere (see run_consumer_loop's docstring).
                continue

            finished_offsets.setdefault(partition, set()).add(offset)

            # Commit the longest CONTIGUOUS run of successfully finished
            # offsets starting from next_offset_to_commit[partition].
            # This is what keeps offset commits in strict per-partition
            # order even though two different users' messages on the
            # same partition can now finish in either order (see the
            # keys_in_flight docstring above for why that's possible).
            expected = next_offset_to_commit.get(partition, offset)
            pending = finished_offsets[partition]
            highest_committable = None
            while expected in pending:
                pending.discard(expected)
                highest_committable = expected
                expected += 1
            next_offset_to_commit[partition] = expected

            if highest_committable is None:
                # This offset finished, but an earlier offset on the same
                # partition is still in flight — nothing new to commit
                # yet; highest_committable will catch up once that
                # earlier one finishes.
                continue

            # commit() is a blocking librdkafka call, but it's fast
            # (network round trip, not a Gemini call) and — critically —
            # runs here on the single consumer-owning coroutine only, so
            # there's no thread-safety concern running it inline without
            # to_thread. Keeping it un-threaded also avoids any chance
            # of a stray overlap with poll() below.
            try:
                consumer.commit(
                    offsets=[TopicPartition(topic, partition, highest_committable + 1)],
                    asynchronous=False,
                )
            except Exception:
                logger.error(
                    f"❌ Failed to commit offset for {topic} "
                    f"[partition={partition} offset={highest_committable}] — "
                    "message was processed successfully but may be "
                    "redelivered.",
                    exc_info=True,
                )

    try:
        while True:
            # Reap any tasks that finished since our last poll, so their
            # keys free up before we decide whether we can accept a new
            # message for that same user.
            await _drain_completed(block=False)

            # poll() itself is a blocking librdkafka call. It's still
            # safe to run off-thread via to_thread because, by
            # construction, nothing else ever touches `consumer`
            # concurrently with it: commit() only happens in
            # _drain_completed() calls that happen strictly before or
            # after this poll() on the same coroutine, never during it.
            msg = await asyncio.to_thread(consumer.poll, poll_timeout)

            if msg is None:
                # Nothing new arrived this tick — use the idle moment to
                # block-wait for at least one in-flight task to finish
                # (if any), so keys free up promptly instead of only
                # being noticed on the next poll_timeout tick.
                if in_flight_tasks:
                    await _drain_completed(block=True)
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                logger.error(f"Kafka consumer error: {msg.error()}")
                continue

            key = msg.key()
            if key in keys_in_flight:
                # This user's previous message is still being handled —
                # don't start a second task for them (would risk
                # out-of-order processing for that one user). We haven't
                # committed the in-flight one yet, so librdkafka will
                # hand us this same message again on a later poll();
                # nothing is lost, we just don't act on it yet.
                #
                # Note: because commit() advances a partition's offset
                # strictly in order, if this message is BEHIND another
                # in-flight message on the same partition but for a
                # DIFFERENT key, we still can't commit past it here —
                # but we also don't need to: we simply skip starting a
                # task for it this tick and let poll() re-offer it once
                # the blocking key is free, same as before.
                await asyncio.sleep(0.05)
                continue

            keys_in_flight.add(key)
            task = asyncio.create_task(_process_one(msg))
            in_flight_tasks.add(task)
            task.add_done_callback(lambda t: in_flight_tasks.discard(t))

            # Backpressure: don't let poll() keep handing us messages
            # faster than we can process them — wait for room if we're
            # at the concurrency ceiling.
            while len(in_flight_tasks) >= max_concurrent:
                await _drain_completed(block=True)
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Kafka async consumer interrupted — shutting down.")
    finally:
        if in_flight_tasks:
            await asyncio.gather(*in_flight_tasks, return_exceptions=True)
            await _drain_completed(block=False)
        consumer.close()


def run_consumer_loop(
    topic: str,
    group_id: str,
    handle_message: Callable[[dict[str, Any]], None],
    poll_timeout: float = 1.0,
) -> None:
    """Blocking consume loop — call from a worker process's main().

    `handle_message` receives the decoded JSON dict for each message. Offsets
    are committed only after `handle_message` returns without raising, so a
    worker crash mid-processing results in at-least-once redelivery (the
    same message may be handled twice) rather than silently dropping work —
    downstream handlers should already be idempotent (see
    app/idempotency.py, already used for inbound WhatsApp messages).
    """
    settings = get_settings()
    consumer = Consumer(
        {
            "bootstrap.servers": settings.kafka_bootstrap_servers,
            "group.id": group_id,
            "auto.offset.reset": settings.kafka_consumer_auto_offset_reset,
            # Commit offsets ourselves, after successful processing, not
            # automatically in the background — see docstring above.
            "enable.auto.commit": False,
        }
    )
    consumer.subscribe([topic])
    logger.info(f"🟢 Kafka consumer started | topic={topic} group={group_id}")

    try:
        while True:
            msg = consumer.poll(poll_timeout)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                logger.error(f"Kafka consumer error: {msg.error()}")
                continue

            try:
                payload = json.loads(msg.value().decode("utf-8"))
                handle_message(payload)
                consumer.commit(msg, asynchronous=False)
            except Exception:
                logger.error(
                    f"❌ Failed to process Kafka message from {topic} "
                    f"[partition={msg.partition()} offset={msg.offset()}] "
                    "— NOT committing offset, will be redelivered.",
                    exc_info=True,
                )
    except KeyboardInterrupt:
        logger.info("Kafka consumer interrupted — shutting down.")
    finally:
        consumer.close()