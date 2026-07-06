# vector_utils.py
import asyncio
import hashlib
import json
import logging
import re
import uuid
from functools import lru_cache
from openai import AsyncOpenAI
from pinecone import Pinecone, ServerlessSpec

from app.config import get_settings

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"   # 1536 dims
EMBEDDING_DIMENSION = 1536

# ---------------------------------------------------------------------------
# Chit-chat / greeting short-circuit
# ---------------------------------------------------------------------------
# Every incoming message currently pays for an OpenAI embedding round-trip
# + a Pinecone query round-trip before the bot can even start composing a
# reply — including for messages like "hi", "thanks", "ok" that will never
# retrieve anything useful from the knowledge base anyway. Skipping RAG for
# these removes two full network round-trips from the critical path for a
# large fraction of real WhatsApp traffic (greetings, acks, small talk).
#
# This is intentionally a cheap, local, regex-based heuristic — NOT an LLM
# call — so the check itself costs ~0ms. It only needs to catch the common,
# obvious cases; anything even slightly ambiguous falls through to normal
# RAG retrieval, so we never risk suppressing context for a real question.
_CHITCHAT_PATTERNS = re.compile(
    r"^\s*("
    r"hi+|hello+|hey+|yo|sup|hola|namaste|namaskar"
    r"|good\s*(morning|afternoon|evening|night)"
    r"|(?:ok(?:ay)?|okk+|alright|fine|cool|great|nice|got it|understood|sounds good)"
    r"|thanks?(?:\s*you)?|thank\s*you|thx|ty|tq"
    r"|bye+|goodbye|see\s*ya|take care"
    r"|yes|yep|yup|no|nope|nah"
    r"|(?:😀|😃|😄|😁|🙂|👍|👌|🙏|❤️|😊)+"
    r")[\s!.,?]*$",
    re.IGNORECASE,
)


def should_use_rag(user_message: str) -> bool:
    """
    Cheap local heuristic (no API call): returns False for obvious
    greetings/acks/small-talk where knowledge-base retrieval would never
    add anything, True otherwise (including anything ambiguous — the
    default is to retrieve, this only opts OUT of the clear-cut cases).
    """
    settings = get_settings()
    if not settings.rag_skip_chitchat:
        return True

    text = (user_message or "").strip()
    if not text:
        return False
    # Very short messages with no letters (e.g. just punctuation/emoji)
    if len(text) <= 2:
        return False

    return not bool(_CHITCHAT_PATTERNS.match(text))


# ---------------------------------------------------------------------------
# Embedding cache (Redis)
# ---------------------------------------------------------------------------
# Repeated/near-identical questions are extremely common in FAQ-style
# WhatsApp bots ("pricing?", "pricing", "what's the pricing"). Caching the
# embedding for a normalized query text means the second+ time a user (or
# a different user) asks essentially the same thing, we skip the OpenAI
# embedding round-trip entirely and go straight to the Pinecone query.
def _normalize_query_for_cache(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip().lower())


def _embedding_cache_key(query: str) -> str:
    normalized = _normalize_query_for_cache(query)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"wa:embcache:{EMBEDDING_MODEL}:{digest}"


# Pinecone client (singleton)

@lru_cache(maxsize=1)
def get_pinecone_index():
    """
    Returns a Pinecone Index object.
    Creates the index if it doesn't exist yet (serverless, AWS us-east-1).
    Cached so we only initialise once per process.
    """
    settings = get_settings()
    pc = Pinecone(api_key=settings.pinecone_api_key)

    existing = [idx.name for idx in pc.list_indexes()]
    if settings.pinecone_index_name not in existing:
        logger.info(f"Creating Pinecone index '{settings.pinecone_index_name}' ...")
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=settings.pinecone_dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        logger.info("Index created!")
    else:
        logger.info(f"Pinecone index '{settings.pinecone_index_name}' already exists!")

    return pc.Index(settings.pinecone_index_name)


# Embedding

@lru_cache(maxsize=1)
def _get_openai_client() -> AsyncOpenAI:
    # Was constructed fresh on every embed call — cache it so the
    # underlying HTTP connection pool is reused across requests instead of
    # paying a new-connection cost on every single message.
    settings = get_settings()
    return AsyncOpenAI(api_key=settings.openai_api_key)


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of strings using text-embedding-3-small.
    Returns a list of float vectors, one per input text.
    """
    client = _get_openai_client()

    response = await client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    vectors = [item.embedding for item in response.data]
    logger.info(f"Embedded {len(texts)} chunk(s) using {EMBEDDING_MODEL}")
    return vectors


async def embed_query(query: str) -> list[float]:
    """
    Embed a single query string, using a Redis cache keyed on the
    normalized query text so repeated/near-identical questions skip the
    OpenAI embedding round-trip entirely. Falls back to a live embed call
    on any cache error (Redis down, etc.) rather than failing the request.
    """
    settings = get_settings()
    if not settings.embedding_cache_enabled:
        vectors = await embed_texts([query])
        return vectors[0]

    from app.redis_client import get_redis

    key = _embedding_cache_key(query)
    try:
        r = get_redis()
        cached = await r.get(key)
        if cached:
            logger.info(f"Embedding cache HIT for query: '{query[:60]}'")
            return json.loads(cached)
    except Exception as e:
        logger.warning(f"Embedding cache read failed, embedding live: {e}")

    vectors = await embed_texts([query])
    vector = vectors[0]

    try:
        r = get_redis()
        await r.set(key, json.dumps(vector), ex=settings.embedding_cache_ttl_seconds)
    except Exception as e:
        logger.warning(f"Embedding cache write failed (non-fatal): {e}")

    return vector


# Insertion
async def upsert_chunks(chunks: list[str], source: str) -> dict:
    """
    Embed a list of text chunks and upsert them into Pinecone under `source`.
    """
    if not chunks:
        raise ValueError("No chunks to ingest.")

    vectors = await embed_texts(chunks)

    index = get_pinecone_index()
    records = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        record_id = f"{source}__chunk_{i}__{uuid.uuid4().hex[:8]}"
        records.append({
            "id": record_id,
            "values": vector,
            "metadata": {
                "text": chunk,
                "source": source,
                "chunk_id": i,
                "chunk_total": len(chunks),
            },
        })

    # Upsert in batches of 100
    # index.upsert() is a blocking (sync) network call from the Pinecone
    # SDK — running it directly inside this async function would stall the
    # whole event loop (and every concurrent WhatsApp user) for its
    # duration, so it's offloaded to a worker thread.
    batch_size = 100
    for i in range(0, len(records), batch_size):
        batch = records[i: i + batch_size]
        await asyncio.to_thread(index.upsert, vectors=batch)
        logger.info(f"Upserted batch {i // batch_size + 1}: {len(batch)} vectors")

    logger.info(f"Ingested '{source}': {len(chunks)} chunks into Pinecone")
    return {"chunks_ingested": len(chunks), "source": source}


# Deletion
async def delete_source(source: str) -> dict:
    """
    Delete all vectors for a given source from Pinecone.
    Uses metadata filter — requires Pinecone index to have metadata filtering enabled.
    """
    index = get_pinecone_index()
    await asyncio.to_thread(index.delete, filter={"source": {"$eq": source}})
    logger.info(f"Deleted all vectors for source='{source}'")
    return {"deleted_source": source}


# Retrieval
async def retrieve_context(query: str, top_k: int | None = None) -> list[dict]:
    """
    Embed the query, search Pinecone, return top-k matching chunks.

    Skips embedding + Pinecone entirely for obvious chit-chat/greetings
    (see should_use_rag()) — pure latency win, returns [] immediately
    instead of paying for two network round-trips that would never
    surface anything useful anyway.
    """
    settings = get_settings()
    k = top_k or settings.rag_top_k

    if not should_use_rag(query):
        logger.info(f"Skipping RAG (chit-chat heuristic) for query: '{query[:60]}'")
        return []

    query_vector = await embed_query(query)
    index = get_pinecone_index()

    # index.query() is a blocking (sync) network call — this runs on every
    # single chat message, so leaving it un-offloaded meant every incoming
    # WhatsApp message stalled the entire event loop (all other users too)
    # for the round-trip time to Pinecone.
    results = await asyncio.to_thread(
        index.query,
        vector=query_vector,
        top_k=k,
        include_metadata=True,
    )

    chunks = []
    for match in results.matches:
        chunks.append({
            "text": match.metadata.get("text", ""),
            "source": match.metadata.get("source", "unknown"),
            "chunk_id": match.metadata.get("chunk_id", ""),
            "score": round(match.score, 4),
        })

    logger.info(f"Retrieved {len(chunks)} chunks for query: '{query[:60]}'")
    for c in chunks:
        logger.debug(f"  score={c['score']} source={c['source']} text={c['text'][:60]}")

    return chunks


# Misc helpers
def build_context_block(chunks: list[dict]) -> str:
    """Format retrieved chunks into a clean context string."""
    if not chunks:
        return ""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[{i}] (source: {chunk['source']})\n{chunk['text']}")
    return "\n\n".join(parts)


async def get_index_stats() -> dict:
    """Return Pinecone index stats — total vector count, dimension, namespaces."""
    settings = get_settings()
    index = get_pinecone_index()
    stats = await asyncio.to_thread(index.describe_index_stats)
    return {
        "index": settings.pinecone_index_name,
        "total_vectors": stats.total_vector_count,
        "dimension": stats.dimension,
        "namespaces": dict(stats.namespaces) if stats.namespaces else {},
    }