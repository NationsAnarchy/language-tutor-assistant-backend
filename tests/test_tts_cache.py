"""Tests for safe cached-audio delivery and local cache de-duplication."""

import threading
from concurrent.futures import ThreadPoolExecutor

from app import tts


def test_synthesize_speech_deduplicates_concurrent_cache_misses(tmp_path, monkeypatch):
    """Equivalent simultaneous requests perform a single uncached synthesis."""
    monkeypatch.setattr(tts, "AUDIO_CACHE_DIR", tmp_path)
    monkeypatch.setattr(tts, "_cache_locks", {})
    started = threading.Event()
    release = threading.Event()
    calls = 0
    calls_lock = threading.Lock()

    def fake_uncached(tts_text, cache_path):
        nonlocal calls
        with calls_lock:
            calls += 1
        started.set()
        release.wait(timeout=2)
        cache_path.write_bytes(b"mp3-data")
        return b"mp3-data", "audio/mpeg"

    monkeypatch.setattr(tts, "_synthesize_speech_uncached", fake_uncached)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(tts.synthesize_speech, "Hello", "en")
        assert started.wait(timeout=1)
        second = executor.submit(tts.synthesize_speech, "Hello", "en")
        release.set()
        assert first.result(timeout=2) == (b"mp3-data", "audio/mpeg")
        assert second.result(timeout=2) == (b"mp3-data", "audio/mpeg")

    assert calls == 1