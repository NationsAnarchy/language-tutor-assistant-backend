"""Tests for safe cached-audio delivery and local cache de-duplication."""

import os
import threading
import time
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
    # Verify locks are pruned once operations complete
    assert len(tts._cache_locks) == 0


def test_evict_cache_if_needed_removes_oldest_files(tmp_path, monkeypatch):
    """LRU/mtime eviction trims directory to target size."""
    monkeypatch.setattr(tts, "AUDIO_CACHE_DIR", tmp_path)

    # Create 3 files with distinct mtimes
    f1 = tmp_path / "old.mp3"
    f2 = tmp_path / "middle.mp3"
    f3 = tmp_path / "new.mp3"

    f1.write_bytes(b"A" * 1000)
    f2.write_bytes(b"B" * 1000)
    f3.write_bytes(b"C" * 1000)

    now = time.time()
    os.utime(f1, (now - 300, now - 300))
    os.utime(f2, (now - 200, now - 200))
    os.utime(f3, (now - 100, now - 100))

    # Total size is 3000 bytes. Cap at 2500 bytes -> evicts oldest
    evicted = tts.evict_cache_if_needed(max_bytes=2500)
    assert evicted >= 1
    assert not f1.exists()  # Oldest is evicted
    assert f3.exists()      # Newest remains


def test_get_cached_audio_requires_auth(tmp_path, monkeypatch):
    """GET /audio/{hash}.mp3 returns 401 when unauthenticated."""
    from fastapi.testclient import TestClient
    from app import main

    monkeypatch.setattr(tts, "AUDIO_CACHE_DIR", tmp_path)
    client = TestClient(main.app)
    response = client.get("/audio/0123456789abcdef.mp3")
    assert response.status_code == 401


def test_get_cached_audio_authenticated_and_headers(tmp_path, monkeypatch):
    """GET /audio/{hash}.mp3 returns 200 with private, no-cache when authenticated."""
    from fastapi.testclient import TestClient
    from app import main

    monkeypatch.setattr(tts, "AUDIO_CACHE_DIR", tmp_path)
    monkeypatch.setattr("app.main.verify_token", lambda tok: {"sub": "user-1"})

    audio_hash = "abcdef0123456789"
    cache_file = tmp_path / f"{audio_hash}.mp3"
    cache_file.write_bytes(b"test-audio-content")

    client = TestClient(main.app)
    response = client.get(
        f"/audio/{audio_hash}.mp3",
        headers={"Authorization": "Bearer valid-token"},
    )
    assert response.status_code == 200
    assert response.content == b"test-audio-content"
    assert response.headers["content-type"] == "audio/mpeg"
    assert "private" in response.headers["cache-control"]
    assert "no-cache" in response.headers["cache-control"]