"""
Audio Handler Module
Handles audio transcription, processing, and model selection
Supports both OpenAI's Whisper (audio-to-text) and text-to-speech models
"""

import logging
import httpx
import asyncio
import subprocess
import tempfile
import os
from typing import Optional, Tuple
from app.core.config import get_settings

logger = logging.getLogger("audio_handler")
settings = get_settings()

# Reuse one pooled/keep-alive client instead of opening a new TCP+TLS
# connection to OpenAI on every single voice message.
_http_client: httpx.AsyncClient | None = None


def _client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
        )
    return _http_client

# Audio Model Configuration
AUDIO_MODELS = {
    # Transcription Models (Speech-to-Text)
    "transcription": {
        "openai_whisper_large_v3": {
            "provider": "openai",
            "type": "speech-to-text",
            "model_id": "whisper-1",
            "description": "OpenAI Whisper Large V3 - Most accurate, supports 98 languages",
            "languages": "98 languages",
            "accuracy": "High",
        },
        "openai_whisper_small": {
            "provider": "openai",
            "type": "speech-to-text",
            "model_id": "whisper-1",
            "description": "OpenAI Whisper Small - Faster, good accuracy",
            "languages": "98 languages",
            "accuracy": "Good",
        },
    },
    # Text-to-Speech Models
    "tts": {
        "openai_tts_1": {
            "provider": "openai",
            "type": "text-to-speech",
            "model_id": "tts-1",
            "description": "OpenAI TTS-1 - Fast, real-time capable",
            "voices": ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
            "latency": "Low",
        },
        "openai_tts_1_hd": {
            "provider": "openai",
            "type": "text-to-speech",
            "model_id": "tts-1-hd",
            "description": "OpenAI TTS-1 HD - Higher quality audio",
            "voices": ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
            "latency": "Medium",
        },
        "google_cloud_tts": {
            "provider": "google_cloud",
            "type": "text-to-speech",
            "model_id": "neural2",
            "description": "Google Cloud Text-to-Speech - Natural sounding",
            "voices": "200+ voices across 40+ languages",
            "latency": "Medium",
        },
        "aws_polly": {
            "provider": "aws",
            "type": "text-to-speech",
            "model_id": "neural",
            "description": "AWS Polly Neural - High quality synthesis",
            "voices": "150+ voices",
            "latency": "Medium",
        },
    },
}

# Text Model Configuration
TEXT_MODELS = {
    "openai_gpt4": {
        "provider": "openai",
        "model_id": "gpt-4-turbo",
        "description": "OpenAI GPT-4 Turbo - Most capable, best for complex reasoning",
        "context_window": "128k tokens",
        "training_data": "April 2024",
        "use_case": "Complex analysis, multi-step reasoning",
    },
    "openai_gpt4_mini": {
        "provider": "openai",
        "model_id": "gpt-4-mini",
        "description": "OpenAI GPT-4 Mini - Fast, cost-effective",
        "context_window": "128k tokens",
        "training_data": "April 2024",
        "use_case": "Quick responses, standard queries",
    },
    "openai_gpt35_turbo": {
        "provider": "openai",
        "model_id": "gpt-3.5-turbo",
        "description": "OpenAI GPT-3.5 Turbo - Very fast, cost-effective",
        "context_window": "16k tokens",
        "training_data": "April 2024",
        "use_case": "Quick responses, simple tasks",
    },
    "anthropic_claude_3_opus": {
        "provider": "anthropic",
        "model_id": "claude-3-opus-20240229",
        "description": "Claude 3 Opus - Most capable Claude model",
        "context_window": "200k tokens",
        "training_data": "August 2024",
        "use_case": "Complex analysis, long context",
    },
    "anthropic_claude_3_sonnet": {
        "provider": "anthropic",
        "model_id": "claude-3-sonnet-20240229",
        "description": "Claude 3 Sonnet - Balanced capability and speed",
        "context_window": "200k tokens",
        "training_data": "August 2024",
        "use_case": "General purpose, balanced performance",
    },
    "anthropic_claude_3_haiku": {
        "provider": "anthropic",
        "model_id": "claude-3-haiku-20240307",
        "description": "Claude 3 Haiku - Fast, cost-effective",
        "context_window": "200k tokens",
        "training_data": "August 2024",
        "use_case": "Quick responses, simple tasks",
    },
    "google_gemini_pro": {
        "provider": "google",
        "model_id": "gemini-pro",
        "description": "Google Gemini Pro - Multimodal capabilities",
        "context_window": "30k tokens",
        "training_data": "April 2024",
        "use_case": "Multimodal tasks, general purpose",
    },
    "mistral_large": {
        "provider": "mistral",
        "model_id": "mistral-large",
        "description": "Mistral Large - High performance open model",
        "context_window": "32k tokens",
        "training_data": "September 2024",
        "use_case": "Complex reasoning, long context",
    },
    "llama2_70b": {
        "provider": "meta",
        "model_id": "llama-2-70b",
        "description": "LLaMA 2 70B - Open source, capable model",
        "context_window": "4k tokens",
        "training_data": "July 2023",
        "use_case": "General purpose, open source",
    },
}


async def get_audio_duration_seconds(audio_bytes: bytes, audio_format: str = "ogg") -> Optional[float]:
    """
    Get the duration of an audio clip in seconds using ffprobe (already
    present on the system alongside ffmpeg — no new dependency needed).
    Writes the bytes to a temp file since ffprobe needs a file path/stdin
    with a known container, then reads back the duration from its JSON
    output. Runs the subprocess off the event loop via asyncio.

    Returns None if duration can't be determined (caller should decide
    how to handle that — e.g. fail open and let the message through,
    or fail closed and reject it).
    """
    tmp_path = None
    ffprobe_bin = os.environ.get("FFPROBE_PATH", "ffprobe")
    try:
        with tempfile.NamedTemporaryFile(suffix=f".{audio_format}", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name


        result = await asyncio.to_thread(
            subprocess.run,
            [
                ffprobe_bin, "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                tmp_path,
            ],
            capture_output=True,
            timeout=15,
        )

        if result.returncode != 0:
            logger.error(f"ffprobe failed (rc={result.returncode}): {result.stderr.decode(errors='ignore')}")
            return None

        duration_str = result.stdout.decode(errors="ignore").strip()
        return float(duration_str) if duration_str else None
    except FileNotFoundError:
        logger.error(
            f"ffprobe executable not found (tried '{ffprobe_bin}'). "
            f"Set FFPROBE_PATH in .env to the full path, e.g. "
            f"FFPROBE_PATH=C:\\ffmpeg\\bin\\ffprobe.exe"
        )
        return None
    except Exception as e:
        logger.error(f"Failed to get audio duration: {type(e).__name__}: {e}", exc_info=True)
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


async def transcribe_audio(audio_bytes: bytes, audio_format: str = "ogg") -> Optional[dict]:
    """
    Transcribe audio using OpenAI Whisper API.

    Uses response_format="verbose_json" so Whisper also returns the
    language it actually detected (e.g. "english", "hindi", "gujarati").
    Short/accented clips confuse Whisper's auto-detect into picking the
    wrong Indic script (e.g. Gujarati letters for English speech) — we
    can't fully prevent that, but by trusting Whisper's OWN reported
    language instead of re-guessing from the (possibly garbled) output
    text, we avoid compounding the error with a second wrong guess.

    Returns:
        dict with "text" and "language", or None if failed.
    """
    try:
        files = {'file': (f'audio.{audio_format}', audio_bytes)}
        data = {
            'model': 'whisper-1',
            'response_format': 'verbose_json',
        }
        headers = {'Authorization': f'Bearer {settings.OPENAI_API_KEY}'}

        response = await _client().post(
            "https://api.openai.com/v1/audio/transcriptions",
            files=files,
            data=data,
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()
        return {
            "text": result.get("text", ""),
            "language": result.get("language", ""),  # e.g. "english", "hindi", "gujarati"
        }
    except Exception as e:
        logger.error(f"Audio transcription error: {e}")
        return None


def get_available_models(model_type: str = "all") -> dict:
    """
    Get list of available models.
    
    Args:
        model_type: "audio", "text", or "all"
        
    Returns:
        Dictionary of available models
    """
    if model_type == "audio":
        return AUDIO_MODELS
    elif model_type == "text":
        return TEXT_MODELS
    else:
        return {
            "audio": AUDIO_MODELS,
            "text": TEXT_MODELS
        }


def get_model_info(model_type: str, model_key: str) -> Optional[dict]:
    """
    Get detailed information about a specific model.
    
    Args:
        model_type: "audio" or "text"
        model_key: Model identifier
        
    Returns:
        Model information dictionary
    """
    if model_type == "audio":
        for category, models in AUDIO_MODELS.items():
            if model_key in models:
                return models[model_key]
    elif model_type == "text":
        if model_key in TEXT_MODELS:
            return TEXT_MODELS[model_key]
    return None


def format_models_for_display() -> str:
    """Format models information as readable text."""
    output = []
    
    output.append("=" * 80)
    output.append("AVAILABLE AUDIO MODELS")
    output.append("=" * 80)
    
    output.append("\n🎙️ Speech-to-Text (Transcription) Models:")
    for model_key, model_info in AUDIO_MODELS.get("transcription", {}).items():
        output.append(f"\n  • {model_key}")
        output.append(f"    Provider: {model_info['provider']}")
        output.append(f"    Description: {model_info['description']}")
        output.append(f"    Languages: {model_info.get('languages', 'N/A')}")
        output.append(f"    Accuracy: {model_info.get('accuracy', 'N/A')}")
    
    output.append("\n\n🔊 Text-to-Speech Models:")
    for model_key, model_info in AUDIO_MODELS.get("tts", {}).items():
        output.append(f"\n  • {model_key}")
        output.append(f"    Provider: {model_info['provider']}")
        output.append(f"    Description: {model_info['description']}")
        output.append(f"    Voices: {model_info.get('voices', 'N/A')}")
        output.append(f"    Latency: {model_info.get('latency', 'N/A')}")
    
    output.append("\n\n" + "=" * 80)
    output.append("AVAILABLE TEXT MODELS")
    output.append("=" * 80 + "\n")
    
    for model_key, model_info in TEXT_MODELS.items():
        output.append(f"• {model_key}")
        output.append(f"  Provider: {model_info['provider']}")
        output.append(f"  Model ID: {model_info['model_id']}")
        output.append(f"  Description: {model_info['description']}")
        output.append(f"  Context Window: {model_info.get('context_window', 'N/A')}")
        output.append(f"  Training Data: {model_info.get('training_data', 'N/A')}")
        output.append(f"  Use Case: {model_info.get('use_case', 'N/A')}\n")
    
    return "\n".join(output)