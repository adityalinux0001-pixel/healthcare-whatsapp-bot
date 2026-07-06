import logging
import re
import tiktoken
import pypdf
import io
from app.vector_utils import upsert_chunks, delete_source as _delete_source

logger = logging.getLogger(__name__)

# Chunking config
CHUNK_TOKENS    = 300
OVERLAP_TOKENS  = 50
TOKENIZER       = "cl100k_base"   # same tokenizer as of text-embedding-3-small


# Text cleaning
def clean_text(text: str) -> str:
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)       # collapse excess blank lines
    text = re.sub(r"[ \t]+", " ", text)           # collapse spaces/tabs
    return text.strip()


# Chunking
def chunk_text(text: str, chunk_tokens: int = CHUNK_TOKENS, overlap: int = OVERLAP_TOKENS) -> list[str]:
    enc = tiktoken.get_encoding(TOKENIZER)
    tokens = enc.encode(text)

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_tokens, len(tokens))
        chunk_tokens_slice = tokens[start:end]
        piece = enc.decode(chunk_tokens_slice)
        chunks.append(piece)
        if end == len(tokens):
            break
        start += chunk_tokens - overlap   # slide window with overlap

    logger.info(f"Chunked into {len(chunks)} pieces ({len(tokens)} total tokens)")
    return chunks


# File readers
def read_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF's text layer (no OCR)."""
    try:
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)
    except ImportError:
        raise RuntimeError(
            "pypdf not installed. Run: pip install pypdf  to enable PDF ingestion."
        )


def read_text_file(file_bytes: bytes) -> str:
    """Decode a .txt or .md file. Markdown is ingested as plain text —
    formatting characters are left as-is; clean_text() handles whitespace."""
    return file_bytes.decode("utf-8", errors="ignore")

def read_docx(file_bytes: bytes) -> str:
    """Extract text from a .docx file."""
    try:
        import io
        from docx import Document

        doc = Document(io.BytesIO(file_bytes))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)

    except ImportError:
        raise RuntimeError(
            "python-docx not installed. Run: pip install python-docx to enable DOCX ingestion."
        )

SUPPORTED_EXTENSIONS = (".txt", ".md", ".pdf", ".docx")


def read_file(filename: str, file_bytes: bytes) -> str:
    """Dispatch to the right reader based on file extension."""
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return read_pdf(file_bytes)
    if lower.endswith(".docx"):
        return read_docx(file_bytes)
    if lower.endswith(".txt") or lower.endswith(".md"):
        return read_text_file(file_bytes)
    raise ValueError(
        f"Unsupported file type for '{filename}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
    )


# ─── Core ingest function ─────────────────────────────────────────────────────

async def ingest_text(
    text: str,
    source: str = "manual",
    chunk_tokens: int = CHUNK_TOKENS,
    overlap: int = OVERLAP_TOKENS,
) -> dict:
    """
    Full pipeline: clean → chunk → embed → upsert to Pinecone.

    Args:
        text:         Raw document text.
        source:       Label stored in metadata (e.g. filename, URL, topic name).
        chunk_tokens: Tokens per chunk.
        overlap:      Overlap tokens between chunks.

    Returns:
        {"chunks_ingested": int, "source": str}
    """
    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("Document is empty after cleaning.")

    chunks = chunk_text(cleaned, chunk_tokens, overlap)
    return await upsert_chunks(chunks, source)


async def delete_source(source: str) -> dict:
    """Delete all vectors for a given source from Pinecone."""
    return await _delete_source(source)
