"""
Shared text utilities for the Language Tutor Agent.

Provides helpers for:
- Normalizing Gemini's list-of-content-parts into plain strings
- Stripping markdown formatting
- Stripping stage directions for TTS and session matching
"""

import re
from typing import Any

# Stage-direction tokens used in TTS cleanup and session content matching.
_SD_TAGS = (
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


def extract_text(content: object) -> str:
    """Normalize Gemini's list-of-content-parts into a plain string.

    Gemini returns content as [{'type': 'text', 'text': '...'}] but the
    rest of the codebase expects plain strings. This helper handles both
    formats transparently.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(part.get("text", ""))
            elif hasattr(part, "text"):
                parts.append(part.text)
            else:
                parts.append(str(part))
        return "".join(parts)
    return str(content)


def strip_markdown(text: str) -> str:
    """Remove markdown formatting so the result is clean plain text.

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


def strip_stage_directions(text: str) -> str:
    """Strip known stage-direction tokens inside parentheses.

    E.g. "(smiles)", "(chuckles warmly)", "(shaking head quietly)" → removed.
    Keeps real parenthetical content like IELTS band descriptors intact.
    """
    text = re.sub(rf'\(\s*(?i:{_SD_TAGS})(?:\s+(?i:{_SD_TAGS}))*\s*\)', '', text)
    return text


def strip_for_matching(text: str) -> str:
    """Strip markdown and normalize whitespace for content matching.

    Used by sessions.py to match assistant messages when setting audio_hash.
    Composes strip_markdown + strip_stage_directions + bracket cleanup + normalization.
    """
    text = strip_markdown(text)
    text = strip_stage_directions(text)
    # Strip bracket nicknames like [Student's Name] or [Tutor's Name]
    text = re.sub(r'\[[^\]]*\]', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()