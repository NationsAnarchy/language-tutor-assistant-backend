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
from contextlib import contextmanager
import hashlib
import os
import re
import struct
import subprocess
import threading
import time
import uuid
from pathlib import Path

from google import genai
from google.genai import types

from .config import data_dir
from .exceptions import TTSError
from .logging_config import get_logger
from .text_utils import strip_markdown, strip_stage_directions

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

# Long tutor replies can take longer than the client and proxy timeouts to
# synthesize. Keep audio as a concise companion to the complete written reply.
_MAX_SPOKEN_CHARACTERS = 600
_FULL_RESPONSE_NOTICES = {
    "en": "For the full answer, please read the message above.",
    "ko": "전체 답변은 위의 메시지에서 확인해 주세요.",
    "ja": "詳しい回答は、上のメッセージを読んでください。",
}

# Audio cache directory — same volume as the database
AUDIO_CACHE_DIR = data_dir() / "audio"
AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_CACHE_MAX_BYTES = int(os.getenv("AUDIO_CACHE_MAX_BYTES", str(800 * 1024 * 1024)))

_cache_locks: dict[str, tuple[threading.Lock, int]] = {}
_cache_locks_guard = threading.Lock()


@contextmanager
def _acquire_cache_lock(cache_path: Path):
    """Acquire a per-key lock with refcounting, pruning when no longer in use."""
    key = cache_path.name
    with _cache_locks_guard:
        if key not in _cache_locks:
            _cache_locks[key] = (threading.Lock(), 0)
        lock, refcount = _cache_locks[key]
        _cache_locks[key] = (lock, refcount + 1)

    lock.acquire()
    try:
        yield
    finally:
        lock.release()
        with _cache_locks_guard:
            if key in _cache_locks:
                lock, refcount = _cache_locks[key]
                if refcount <= 1:
                    del _cache_locks[key]
                else:
                    _cache_locks[key] = (lock, refcount - 1)


def evict_cache_if_needed(max_bytes: int | None = None) -> int:
    """Evict oldest cached audio files when cache directory exceeds max_bytes.

    Returns the number of evicted files.
    """
    if max_bytes is None:
        max_bytes = AUDIO_CACHE_MAX_BYTES

    try:
        files = [f for f in AUDIO_CACHE_DIR.iterdir() if f.is_file() and f.suffix in {".mp3", ".wav"}]
        if not files:
            return 0

        file_stats: list[tuple[Path, int, float]] = []
        total_size = 0
        for f in files:
            try:
                st = f.stat()
                file_stats.append((f, st.st_size, st.st_mtime))
                total_size += st.st_size
            except OSError:
                continue

        if total_size <= max_bytes:
            return 0

        # Target 80% of max_bytes to avoid oscillating eviction
        target_size = int(max_bytes * 0.8)
        # Sort oldest first by modification time
        file_stats.sort(key=lambda item: item[2])

        evicted_count = 0
        for f, sz, _ in file_stats:
            if total_size <= target_size:
                break
            try:
                f.unlink(missing_ok=True)
                total_size -= sz
                evicted_count += 1
                logger.info("Evicted audio cache file: %s (freed %d bytes)", f.name, sz)
            except OSError as exc:
                logger.warning("Failed to evict %s: %s", f.name, exc)

        return evicted_count
    except Exception as exc:
        logger.warning("Error during audio cache eviction: %s", exc)
        return 0


def _atomic_write_cache(cache_path: Path, data: bytes) -> None:
    """Atomically write data to cache_path and trigger eviction if needed."""
    temp_path = cache_path.with_name(f"{cache_path.name}.tmp.{uuid.uuid4().hex[:8]}")
    try:
        temp_path.write_bytes(data)
        temp_path.replace(cache_path)
        logger.info("TTS cached to: %s (%d bytes)", cache_path.name, len(data))
    except OSError as exc:
        logger.warning("TTS: Failed to write cache file %s: %s", cache_path.name, exc)
    finally:
        temp_path.unlink(missing_ok=True)
    evict_cache_if_needed()


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
    text = strip_markdown(text)

    # Strip trailing prompting-language patterns that confuse TTS models.
    # "Speak in X" and "say X" lines are common LLM fillers.
    text = re.sub(r'(?i)^speak (in |mostly in )?\w+\.?\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'(?i)^say\s+"[^"]*"\.?\s*', '', text, flags=re.MULTILINE)

    # Strip known stage-direction tokens inside parentheses — keeps real
    # parenthetical content like IELTS band descriptors intact (Issue #39 review).
    text = strip_stage_directions(text)

    # Strip bracket nicknames like [Student's Name] or [Tutor's Name]
    text = re.sub(r'\[[^\]]*\]', '', text)

    # Clean up extra whitespace from all the stripping above
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()

    if len(text) <= _MAX_SPOKEN_CHARACTERS:
        return text

    # Reserve room for an explanation that the written response contains the
    # complete answer. Prefer a natural sentence ending; if the reply has a
    # very long sentence, fall back to the nearest word boundary instead.
    notice = _FULL_RESPONSE_NOTICES.get(language, _FULL_RESPONSE_NOTICES["en"])
    content_limit = _MAX_SPOKEN_CHARACTERS - len(notice) - 2
    candidate = text[:content_limit]
    sentence_ends = list(re.finditer(r'[.!?。！？](?:\s|$)', candidate))
    if sentence_ends and sentence_ends[-1].end() > content_limit // 2:
        excerpt = candidate[:sentence_ends[-1].end()].strip()
    else:
        word_end = max(candidate.rfind(' '), candidate.rfind('\n'))
        excerpt = candidate[:word_end if word_end > content_limit // 2 else content_limit].rstrip(' ,;:')

    return f"{excerpt}\n\n{notice}"


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


def _cache_lock(cache_path: Path) -> threading.Lock:
    """Return the process-local lock for a cache key (backwards-compatibility helper)."""
    with _cache_locks_guard:
        if cache_path.name not in _cache_locks:
            _cache_locks[cache_path.name] = (threading.Lock(), 0)
        return _cache_locks[cache_path.name][0]


def synthesize_speech(text: str, language: str, speed: str = "normal") -> tuple[bytes, str] | None:
    """Synthesize speech once per local cache key and return the audio bytes."""
    tts_text = _build_tts_text(text, language, speed)
    if not tts_text:
        return None

    cache_path = _get_cache_path(tts_text)
    with _acquire_cache_lock(cache_path):
        if cache_path.exists():
            logger.info("TTS cache hit: %s", cache_path.name)
            try:
                os.utime(cache_path, None)
            except OSError:
                pass
            return (cache_path.read_bytes(), "audio/mpeg")
        return _synthesize_speech_uncached(tts_text, cache_path)


def _synthesize_speech_uncached(tts_text: str, cache_path: Path) -> tuple[bytes, str] | None:
    """Synthesize speech from text using Gemini Flash TTS.

    Returns (audio_bytes, mime_type) tuple. Audio is MP3-encoded and cached
    on disk for future requests. The caller (FastAPI route) streams the bytes
    directly to the frontend.

    Args:
        tts_text: Cleaned text to convert to speech.
        cache_path: MP3 cache path derived from ``tts_text``.

    Returns:
        Tuple of (audio_bytes, mime_type) like (b'...', 'audio/mpeg'),
        or None if TTS is not configured or the text is empty.
    """
    client = _get_client()
    if client is None:
        return None

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

                # Cache the MP3 on disk atomically
                _atomic_write_cache(cache_path, mp3_bytes)
                return (mp3_bytes, "audio/mpeg")

            elif "mpeg" in mime_type.lower() or "mp3" in mime_type.lower():
                # Gemini returned MP3 directly — cache and return as-is
                _atomic_write_cache(cache_path, audio_bytes)
                return (audio_bytes, "audio/mpeg")

            else:
                # Unknown format — try MP3 conversion, fallback to WAV
                try:
                    mp3_bytes = _pcm_to_mp3(audio_bytes)
                    _atomic_write_cache(cache_path, mp3_bytes)
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
