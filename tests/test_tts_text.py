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


def test_tts_handles_quotes_and_avoids_cutting_at_numbered_markers():
    # Similar to the user's software development vocabulary message
    sample = (
        "Here is a curated list of common software development vocabulary:\n\n"
        "Key Nouns\n"
        "1. Repository (or 'Repo')\n"
        "Meaning: A central digital storage place where code and files are kept.\n"
        "Example: \"Make sure to clone the repository before you start working on the project.\"\n\n"
        "2. Framework\n"
        "Meaning: A pre-built structure that developers use as a foundation to build software.\n"
        "Example: \"React is a widely used front-end framework for building user interfaces.\"\n\n"
        "3. Bug or Defect\n"
        "Meaning: An error, flaw, or unexpected behavior in code.\n"
        "Example: \"The QA team found a minor bug in the payment system.\""
    )

    result = _build_tts_text(sample, "en", "normal")

    assert len(result) <= _MAX_SPOKEN_CHARACTERS
    assert result.endswith("For the full answer, please read the message above.")
    # Ensure it doesn't leave a dangling "2." or "2" right before the notice
    assert "2.\n\nFor the full answer" not in result
    assert "2\n\nFor the full answer" not in result
    # Ensure it ends on a complete sentence with closing quote or punctuation
    first_part = result.rsplit("\n\nFor the full answer", 1)[0]
    assert first_part.endswith('"') or first_part.endswith('.')
