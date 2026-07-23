"""
Gemini Flash Text-to-Speech integration for the Language Tutor Agent.

Uses Gemini TTS via the google-genai SDK. One model serves all three
languages — language is controlled by the text content itself.

Gemini TTS returns raw PCM audio (audio/L16;codec=pcm;rate=24000), so we
convert it to MP3 via ffmpeg and cache the result on disk.

Audio files are cached in RAILWAY_VOLUME_PATH/audio/ (or data/audio/ locally)
keyed by SHA-256 hash of the cleaned TTS text. This saves on Gemini API costs
when the same text is requested again (e.g. replaying previous responses).

Requirements:
    - GEMINI_API_KEY env var set to a Gemini API key
    - ffmpeg installed (for PCM → MP3 conversion)
"""

import base64
import hashlib
import os
import re
import struct
import subprocess
import time
from pathlib import Path

from google import genai
from google.genai import types

from .exceptions import TTSError
from .logging_config import get_logger

logger = get_logger(__name__)

# Gemini TTS model — 2.5 Flash preview supports streaming (cheaper than 3.1)
TTS_MODEL = "gemini-2.5-flash-preview-tts"

# Use a single consistent feminine voice across all languages.
_TTS_VOICE_NAME = "Erinome"

# Retry config for TTS API calls
_MAX_RETRIES = 3
_RETRY_BACKOFF = 1.5  # seconds between retries, doubled each attempt

# MP3 bitrate for speech — 48 kbps is a good balance for voice quality vs size
_MP3_BITRATE = "48k"

# Audio cache directory — same volume as the database
_VOLUME_PATH = os.getenv("RAILWAY_VOLUME_PATH", "")
if _VOLUME_PATH:
    AUDIO_CACHE_DIR = Path(_VOLUME_PATH) / "audio"
else:
    AUDIO_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "audio"
AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)


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


def _get_cache_path(tts_text: str) -> Path:
    """Return the cache file path for a given TTS text.

    Uses SHA-256 hash (first 16 hex chars) as the filename to avoid
    filesystem issues with long or special-character text.
    """
    text_hash = hashlib.sha256(tts_text.encode("utf-8")).hexdigest()[:16]
    return AUDIO_CACHE_DIR / f"{text_hash}.mp3"


def _pcm_to_mp3(pcm_data: bytes, sample_rate: int = 24000) -> bytes:
    """Convert raw PCM audio to MP3 using ffmpeg.

    Args:
        pcm_data: Raw PCM audio bytes (16-bit, mono).
        sample_rate: Sample rate in Hz (default 24000).

    Returns:
        MP3-encoded audio bytes.

    Raises:
        TTSError: If ffmpeg is not found or conversion fails.
    """
    try:
        proc = subprocess.run(
            [
                "ffmpeg",
                "-y",                          # overwrite output
                "-f", "s16le",                 # input format: signed 16-bit little-endian
                "-ar", str(sample_rate),       # input sample rate
                "-ac", "1",                    # input channels: mono
                "-i", "pipe:0",                # read from stdin
                "-codec:a", "libmp3lame",      # MP3 encoder
                "-b:a", _MP3_BITRATE,          # bitrate
                "-f", "mp3",                   # output format
                "pipe:1",                      # write to stdout
            ],
            input=pcm_data,
            capture_output=True,
            timeout=30,
        )
    except FileNotFoundError:
        raise TTSError("ffmpeg not found — install ffmpeg to use MP3 TTS")
    except subprocess.TimeoutExpired:
        raise TTSError("ffmpeg MP3 conversion timed out after 30 seconds")

    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace")[:500]
        raise TTSError(f"ffmpeg MP3 conversion failed (code {proc.returncode}): {stderr}")

    return proc.stdout


def _pcm_to_wav(pcm_data: bytes, sample_rate: int = 24000, num_channels: int = 1, bits_per_sample: int = 16) -> bytes:
    """Wrap raw PCM audio data in a WAV container header.

    Pure Python — no ffmpeg dependency needed (Issue #43).
    Kept as a fallback if ffmpeg is unavailable.
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


def _get_client() -> genai.Client | None:
    """Create a Gemini API client using GEMINI_API_KEY."""
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        logger.warning("GEMINI_API_KEY not set — skipping speech synthesis")
        return None
    return genai.Client(api_key=api_key)


def synthesize_speech(text: str, language: str, speed: str = "normal") -> tuple[bytes, str] | None:
    """Synthesize speech from text using Gemini Flash TTS.

    Returns (audio_bytes, mime_type) tuple. Audio is MP3-encoded and cached
    on disk for future requests. The caller (FastAPI route) streams the bytes
    directly to the frontend.

    Args:
        text: The text to convert to speech (the model handles varying lengths).
        language: Language code — 'en', 'ko', or 'ja'.
        speed: Playback speed — 'normal' (natural pace) or 'slow' (slower, with pauses).

    Returns:
        Tuple of (audio_bytes, mime_type) like (b'...', 'audio/mpeg'),
        or None if TTS is not configured or the text is empty.
    """
    if not text or not text.strip():
        return None

    client = _get_client()
    if client is None:
        return None

    tts_text = _build_tts_text(text, language, speed)

    if not tts_text or not tts_text.strip():
        return None

    # Check cache first — saves Gemini API calls and speeds up replay
    cache_path = _get_cache_path(tts_text)
    if cache_path.exists():
        logger.info("TTS cache hit: %s", cache_path.name)
        audio_bytes = cache_path.read_bytes()
        return (audio_bytes, "audio/mpeg")

    logger.info("TTS cache miss: %s — calling Gemini API", cache_path.name)

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

            # Gemini TTS returns raw PCM — convert to MP3 and cache
            if "pcm" in mime_type.lower() or "L16" in mime_type:
                # Extract sample rate from mime_type if present (e.g. "audio/L16;codec=pcm;rate=24000")
                sample_rate = 24000
                if "rate=" in mime_type:
                    try:
                        sample_rate = int(mime_type.split("rate=")[-1].split(";")[0])
                    except ValueError:
                        pass

                # Convert PCM to MP3 via ffmpeg
                try:
                    mp3_bytes = _pcm_to_mp3(audio_bytes, sample_rate=sample_rate)
                except TTSError:
                    # Fallback to WAV if ffmpeg is unavailable
                    logger.warning("ffmpeg MP3 conversion failed, falling back to WAV")
                    wav_bytes = _pcm_to_wav(audio_bytes, sample_rate=sample_rate)
                    return (wav_bytes, "audio/wav")

                # Cache the MP3 on disk
                try:
                    cache_path.write_bytes(mp3_bytes)
                    logger.info("TTS cached to: %s (%d bytes)", cache_path.name, len(mp3_bytes))
                except OSError as exc:
                    logger.warning("TTS: Failed to write cache file %s: %s", cache_path.name, exc)
                    # Non-fatal — still return the audio

                return (mp3_bytes, "audio/mpeg")

            elif "mpeg" in mime_type.lower() or "mp3" in mime_type.lower():
                # Gemini returned MP3 directly — cache and return as-is
                try:
                    cache_path.write_bytes(audio_bytes)
                    logger.info("TTS cached to: %s (%d bytes)", cache_path.name, len(audio_bytes))
                except OSError as exc:
                    logger.warning("TTS: Failed to write cache file %s: %s", cache_path.name, exc)
                return (audio_bytes, "audio/mpeg")

            else:
                # Unknown format — try MP3 conversion, fallback to WAV
                try:
                    mp3_bytes = _pcm_to_mp3(audio_bytes)
                    try:
                        cache_path.write_bytes(mp3_bytes)
                    except OSError:
                        pass
                    return (mp3_bytes, "audio/mpeg")
                except TTSError:
                    wav_bytes = _pcm_to_wav(audio_bytes)
                    return (wav_bytes, "audio/wav")

        except Exception as exc:
            logger.warning("TTS: Gemini speech synthesis failed (attempt %d/%d): %s", attempt, _MAX_RETRIES, exc)
            last_error = str(exc)
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_BACKOFF * (2 ** (attempt - 1)))

    logger.warning("TTS: All %d attempts failed. Last error: %s", _MAX_RETRIES, last_error)
    # Raise TTSError so the route can map it to a 502 response
    raise TTSError(f"Audio generation failed after {_MAX_RETRIES} attempts: {last_error}")