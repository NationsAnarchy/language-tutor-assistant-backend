"""Tests for concise, learner-friendly TTS input."""

from app.tts import _MAX_SPOKEN_CHARACTERS, _build_tts_text


def test_tts_keeps_short_responses_intact():
    text = "Great work. Your answer is correct."
    assert _build_tts_text(text, "en", "normal") == text


def test_tts_shortens_long_responses_at_a_sentence_boundary_with_notice():
    text = "First complete sentence. " + ("Useful detail. " * 100)

    result = _build_tts_text(text, "en", "normal")

    assert len(result) <= _MAX_SPOKEN_CHARACTERS
    assert result.endswith("For the full answer, please read the message above.")
    assert result.split("\n\n", 1)[0].endswith(".")


def test_tts_uses_a_localized_notice_for_long_responses():
    result = _build_tts_text("설명입니다. " * 200, "ko", "normal")

    assert result.endswith("전체 답변은 위의 메시지에서 확인해 주세요.")
