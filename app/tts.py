"""
Gemini Flash Text-to-Speech integration for the Language Tutor Agent.

Uses Gemini 2.5 Flash TTS via the google-genai SDK. One model serves all
three languages — language is controlled by the text content itself.

Gemini TTS returns raw PCM audio (audio/L16;codec=pcm;rate=24000), so we
wrap it in a proper WAV container before saving.

Requirements:
    - GEMINI_API_KEY env var set to a Gemini API key
"""

import base64
import os
import re
import struct
import time
import uuid
from pathlib import Path

from google import genai
from google.genai import types

from .exceptions import TTSError
from .logging_config import get_logger

logger = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
AUDIO_DIR = BASE_DIR / "audio"

# Ensure audio directory exists
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# Gemini TTS model (preview — language coverage may shift)
TTS_MODEL = "gemini-2.5-flash-preview-tts"

# Use a single consistent feminine voice across all languages.
_TTS_VOICE_NAME = "Erinome"

# Retry config for TTS API calls
_MAX_RETRIES = 3
_RETRY_BACKOFF = 1.5  # seconds between retries, doubled each attempt


def _strip_markdown(text: str) -> str:
    """Remove markdown formatting so Gemini TTS reads clean text.

    Strips: **bold**, *italic*, `code`, # headers, --- hrules, > blockquotes,
    markdown links, and HTML tags. Preserves line breaks and plain text.
    """
    # Strip code blocks (``` ... ```)
    text = re.sub(r'```[\s\S]*?```', '', text)
    # Strip inline code
    text = re.sub(r'`([^`]+)`', r'\1', text)
    # Strip **bold** and __bold__
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    # Strip *italic* and _italic_
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)
    # Strip markdown links [text](url) -> text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Strip headers (# ...)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Strip horizontal rules
    text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
    # Strip blockquotes
    text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)
    # Strip HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Strip ~~strikethrough~~
    text = re.sub(r'~~([^~]+)~~', r'\1', text)
    # Clean up: collapse multiple newlines, trim
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _build_tts_text(text: str, language: str, speed: str) -> str:
    """Return stripped plain text for Gemini TTS.

    Gemini TTS models only accept plain text to speak — instructions, preamble
    text, or structural hints cause the model to try to generate text, which
    triggers a 400 INVALID_ARGUMENT error or (worse) produces a spoofed short
    clip. Speed control is handled client-side via Audio.playbackRate.

    ponytail: Gemini preview TTS may still occasionally misinterpret CJK text
    as instructions on some voices. Upgrade path: switch to a dedicated TTS
    service (Azure Speech, ElevenLabs) if this becomes frequent in production.
    """
    text = _strip_markdown(text)

    # Strip trailing prompting-language patterns that confuse TTS models.
    # "Speak in X" and "say X" lines are common LLM fillers.
    text = re.sub(r'(?i)^speak (in |mostly in )?\w+\.?\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'(?i)^say\s+"[^"]*"\.?\s*', '', text, flags=re.MULTILINE)

    # Strip known stage-direction tokens inside parentheses — keeps real
    # parenthetical content like IELTS band descriptors intact (Issue #39 review).
    _SD = (
        'smiles?|chuckles?|laughs?|sighs?|nods?|pauses?|be kind|be gentle|'
        'warmly|gently|softly|happily|kindly|patiently|encouragingly|'
        'thoughtfully|seriously|cheerfully|calmly|slowly|carefully|'
        'briefly|simply|clearly|quietly|firmly|politely|respectfully|'
        'apologetically|sympathetically|enthusiastically|playfully|'
        'grinning|smiling|frowning|winking|nodding|shaking head|'
        'with a smile|with a laugh|with a nod|with a sigh|with a chuckle|'
        'lightheartedly|jokingly|teasingly|soothingly|reassuringly|'
        'excitedly|curiously|confidently|honestly|candidly|frankly'
    )
    # Match one or more stage-direction tokens separated by spaces inside parens,
    # e.g. (smiles), (chuckles), (smiles warmly), (shaking head quietly)
    text = re.sub(rf'\(\s*(?i:{_SD})(?:\s+(?i:{_SD}))*\s*\)', '', text)

    # Strip bracket nicknames like [Student's Name] or [Tutor's Name]
    text = re.sub(r'\[[^\]]*\]', '', text)

    # Truncate very long text: Gemini TTS has practical length limits;
    # 3000 chars is generous for a single tutor response.
    max_len = 3000
    if len(text) > max_len:
        # ponytail: naive truncation — break at the last sentence boundary
        truncated = text[:max_len]
        last_sentence = max(truncated.rfind('.'), truncated.rfind('!'), truncated.rfind('?'), truncated.rfind('\n'))
        if last_sentence > max_len // 2:
            text = truncated[:last_sentence + 1]
        else:
            text = truncated

    # Clean up extra whitespace from all the stripping above
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


import subprocess

def _pcm_to_wav(pcm_data: bytes, sample_rate: int = 24000, num_channels: int = 1, bits_per_sample: int = 16) -> bytes:
    """Wrap raw PCM audio data in a WAV container header (fallback).
    Retained as a fallback when ffmpeg is not available.
    """
    byte_rate = sample_rate * num_channels * (bits_per_sample // 8)
    block_align = num_channels * (bits_per_sample // 8)
    data_size = len(pcm_data)

    wav = bytearray()
    wav += b"RIFF"
    wav += struct.pack("<I", 36 + data_size)
    wav += b"WAVE"
    wav += b"fmt "
    wav += struct.pack("<I", 16)
    wav += struct.pack("<H", 1)
    wav += struct.pack("<H", num_channels)
    wav += struct.pack("<I", sample_rate)
    wav += struct.pack("<I", byte_rate)
    wav += struct.pack("<H", block_align)
    wav += struct.pack("<H", bits_per_sample)
    wav += b"data"
    wav += struct.pack("<I", data_size)
    wav += pcm_data
    return bytes(wav)


def _pcm_to_mp3(pcm_data: bytes, sample_rate: int = 24000, num_channels: int = 1, bits_per_sample: int = 16) -> bytes | None:
    """Convert raw PCM audio to MP3 using ffmpeg.

    Gemini TTS returns raw PCM audio (audio/L16;codec=pcm;rate=24000).
    Rather than wrapping in WAV (which is ~10x larger), we convert directly
    to MP3 for efficient transport over bandwidth-limited hosting (Railway
    starter plan, Vercel edge, etc.).

    Falls back to WAV if ffmpeg is not available.
    """
    try:
        pcm_size = len(pcm_data)
        byte_rate = sample_rate * num_channels * (bits_per_sample // 8)
        block_align = num_channels * (bits_per_sample // 8)

        wav = bytearray()
        wav += b"RIFF"
        wav += struct.pack("<I", 36 + pcm_size)
        wav += b"WAVE"
        wav += b"fmt "
        wav += struct.pack("<I", 16)
        wav += struct.pack("<H", 1)
        wav += struct.pack("<H", num_channels)
        wav += struct.pack("<I", sample_rate)
        wav += struct.pack("<I", byte_rate)
        wav += struct.pack("<H", block_align)
        wav += struct.pack("<H", bits_per_sample)
        wav += b"data"
        wav += struct.pack("<I", pcm_size)
        wav += pcm_data

        result = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "error",
                "-f", "wav",
                "-i", "pipe:0",
                "-f", "mp3",
                "-b:a", "32k",
                "-ar", str(sample_rate),
                "-ac", str(num_channels),
                "pipe:1",
            ],
            input=bytes(wav),
            capture_output=True,
            timeout=30,
        )
        if result.returncode == 0 and len(result.stdout) > 0:
            return result.stdout
        logger.warning(
            "ffmpeg MP3 conversion failed (rc=%d, stderr=%s) — falling back to WAV",
            result.returncode,
            result.stderr.decode(errors="replace")[:200],
        )
    except FileNotFoundError:
        logger.warning("ffmpeg not found — falling back to WAV")
    except subprocess.TimeoutExpired:
        logger.warning("ffmpeg timed out — falling back to WAV")
    except Exception as exc:
        logger.warning("ffmpeg error: %s — falling back to WAV", exc)

    return None


def _get_client() -> genai.Client | None:
    """Create a Gemini API client using GEMINI_API_KEY."""
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        logger.warning("GEMINI_API_KEY not set — skipping speech synthesis")
        return None
    return genai.Client(api_key=api_key)


def _user_dir(user_id: str) -> str:
    """Return a deterministic opaque directory name for a user (Issue #29).
    
    Uses SHA256 hash so the folder name doesn't contain the user's email or
    any PII. Same user_id always maps to the same directory.
    """
    import hashlib
    return hashlib.sha256(user_id.encode()).hexdigest()[:16]


def synthesize_speech(text: str, language: str, speed: str = "normal", user_id: str = "", session_id: str = "") -> str | None:
    """Synthesize speech from text using Gemini Flash TTS and return a relative URL to the audio file.

    Args:
        text: The text to convert to speech (the model handles varying lengths).
        language: Language code — 'en', 'ko', or 'ja'.
        speed: Playback speed — 'normal' (natural pace) or 'slow' (slower, with pauses).
        user_id: Authenticated user ID for per-user audio directories.
        session_id: Session ID for per-session audio organization.

    Returns:
        Relative URL path like 'user_id/session_id/abc123.wav', or None if TTS is not configured
        or the text is empty.
    """
    if not text or not text.strip():
        return None

    client = _get_client()
    if client is None:
        return None

    tts_text = _build_tts_text(text, language, speed)

    if not tts_text or not tts_text.strip():
        return None

    last_error = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            speech_config = types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=_TTS_VOICE_NAME),
                ),
            )
            response = client.models.generate_content(
                model=TTS_MODEL,
                contents=tts_text,
                config=types.GenerateContentConfig(
                    response_modalities=["Audio"],
                    speech_config=speech_config,
                ),
            )

            # Extract audio from the response
            if not response.candidates:
                logger.warning("TTS: No response candidates returned (attempt %d/%d)", attempt, _MAX_RETRIES)
                last_error = "no candidates"
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_BACKOFF * (2 ** (attempt - 1)))
                continue

            parts = response.candidates[0].content.parts if response.candidates[0].content else []
            if not parts:
                logger.warning("TTS: No content parts in response (attempt %d/%d)", attempt, _MAX_RETRIES)
                last_error = "no content parts"
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_BACKOFF * (2 ** (attempt - 1)))
                continue

            # Find the first audio/inline_data part
            audio_blob = None
            for part in parts:
                if hasattr(part, "inline_data") and part.inline_data:
                    audio_blob = part.inline_data
                    break

            if audio_blob is None:
                logger.warning("TTS: No audio data in response parts (attempt %d/%d)", attempt, _MAX_RETRIES)
                last_error = "no audio data"
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_BACKOFF * (2 ** (attempt - 1)))
                continue

            audio_bytes = audio_blob.data
            if isinstance(audio_bytes, str):
                audio_bytes = base64.b64decode(audio_bytes)

            mime_type = getattr(audio_blob, "mime_type", "")

            # Gemini TTS returns raw PCM — wrap in WAV container for browser playback
            if "pcm" in mime_type.lower() or "L16" in mime_type:
                # Extract sample rate from mime_type if present (e.g. "audio/L16;codec=pcm;rate=24000")
                sample_rate = 24000
                if "rate=" in mime_type:
                    try:
                        sample_rate = int(mime_type.split("rate=")[-1].split(";")[0])
                    except ValueError:
                        pass
                # Convert to MP3 for efficient transport (~10x smaller than WAV)
                mp3_bytes = _pcm_to_mp3(audio_bytes, sample_rate=sample_rate)
                if mp3_bytes is not None:
                    audio_bytes = mp3_bytes
                    filename = f"{uuid.uuid4().hex[:12]}.mp3"
                else:
                    # Fallback to WAV if ffmpeg unavailable
                    audio_bytes = _pcm_to_wav(audio_bytes, sample_rate=sample_rate)
                    filename = f"{uuid.uuid4().hex[:12]}.wav"
            elif "mpeg" in mime_type.lower() or "mp3" in mime_type.lower():
                filename = f"{uuid.uuid4().hex[:12]}.mp3"
            else:
                filename = f"{uuid.uuid4().hex[:12]}.wav"

            # Save per-user/session directory (Issue #8)
            # Sanitize user_id so email addresses don't leak into folder names (Issue #29)
            save_dir = AUDIO_DIR
            if user_id and session_id:
                user_dir = _user_dir(user_id)
                save_dir = AUDIO_DIR / user_dir / session_id
            save_dir.mkdir(parents=True, exist_ok=True)
            filepath = save_dir / filename
            with open(filepath, "wb") as f:
                f.write(audio_bytes)

            # Return relative path — the frontend constructs the full URL
            if user_id and session_id:
                user_dir = _user_dir(user_id)
                return f"{user_dir}/{session_id}/{filename}"
            return filename

        except Exception as exc:
            logger.warning("TTS: Gemini speech synthesis failed (attempt %d/%d): %s", attempt, _MAX_RETRIES, exc)
            last_error = str(exc)
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_BACKOFF * (2 ** (attempt - 1)))

    logger.warning("TTS: All %d attempts failed. Last error: %s", _MAX_RETRIES, last_error)
    # Raise TTSError so the route can map it to a 502 response
    raise TTSError(f"Audio generation failed after {_MAX_RETRIES} attempts: {last_error}")
