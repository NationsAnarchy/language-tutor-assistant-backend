"""Internal tool execution and prompt-context helpers for agent nodes."""

from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from .logging_config import get_logger
from .text_utils import extract_text

logger = get_logger(__name__)

CANONICAL_MISTAKE_TYPES = frozenset({"grammar", "vocabulary", "pronunciation", "spelling"})

# The mistake-review UI intentionally has four broad categories. Normalize the
# common, more specific labels an LLM may produce into that stable contract.
MISTAKE_TYPE_ALIASES = {
    "grammar": "grammar",
    "grammatical": "grammar",
    "syntax": "grammar",
    "sentence structure": "grammar",
    "word order": "grammar",
    "tense": "grammar",
    "verb tense": "grammar",
    "verb form": "grammar",
    "conjugation": "grammar",
    "article": "grammar",
    "preposition": "grammar",
    "particle": "grammar",
    "agreement": "grammar",
    "vocabulary": "vocabulary",
    "vocab": "vocabulary",
    "lexical": "vocabulary",
    "word choice": "vocabulary",
    "word usage": "vocabulary",
    "word meaning": "vocabulary",
    "collocation": "vocabulary",
    "idiom": "vocabulary",
    "pronunciation": "pronunciation",
    "pronounciation": "pronunciation",  # Common misspelling.
    "phonetic": "pronunciation",
    "phonetics": "pronunciation",
    "accent": "pronunciation",
    "intonation": "pronunciation",
    "stress": "pronunciation",
    "tone": "pronunciation",
    "spelling": "spelling",
    "typo": "spelling",
    "orthography": "spelling",
    "orthographic": "spelling",
    "punctuation": "spelling",
}


def normalize_mistake_type(value: Any) -> str | None:
    """Return a canonical mistake type, or ``None`` when it cannot be classified."""
    if not isinstance(value, str):
        return None
    normalized = " ".join(value.strip().casefold().replace("_", " ").replace("-", " ").split())
    return MISTAKE_TYPE_ALIASES.get(normalized)


def user_facing_messages(messages: list[Any]) -> list[Any]:
    """Remove internal function-calling messages from persisted conversation."""
    return [
        message for message in messages
        if not isinstance(message, ToolMessage)
        and not (isinstance(message, AIMessage) and getattr(message, "tool_calls", None))
    ]


def messages_for_response(system_prompt: str, messages: list[Any]) -> list[Any]:
    """Build valid conversation history while preserving private tool results."""
    tool_results = [extract_text(message.content) for message in messages if isinstance(message, ToolMessage)]
    prompt_messages: list[Any] = [SystemMessage(content=system_prompt)]
    if tool_results:
        prompt_messages.append(SystemMessage(content=(
            "Internal tool results (use these; do not mention tools):\n"
            + "\n\n".join(tool_results)
        )))
    return prompt_messages + user_facing_messages(messages)


def execute_tool_calls(response: Any, state: dict[str, Any], tools_by_name: dict[str, Any]) -> list[ToolMessage]:
    """Execute function calls and apply their turn-local state changes."""
    results: list[ToolMessage] = []
    for tool_call in response.tool_calls:
        tool = tools_by_name.get(tool_call["name"])
        if tool is None:
            continue
        try:
            args = dict(tool_call["args"])
            if tool_call["name"] == "log_mistake":
                mistake_type = normalize_mistake_type(args.get("mistake_type"))
                detail = args.get("detail")
                if mistake_type is None or not isinstance(detail, str) or not detail.strip():
                    logger.warning(
                        "Skipping log_mistake call with invalid category or empty detail",
                        extra={"mistake_type": str(args.get("mistake_type", ""))[:100]},
                    )
                    results.append(ToolMessage(
                        content=(
                            "Mistake was not recorded: use one of grammar, vocabulary, "
                            "pronunciation, or spelling and provide a non-empty detail."
                        ),
                        tool_call_id=tool_call["id"],
                    ))
                    continue
                args["mistake_type"] = mistake_type
                args["detail"] = detail.strip()
            # The model should not have to reconstruct private session history.
            # Supply only a compact recent-mistake summary to mistake review.
            if tool_call["name"] == "generate_exercise" and args.get("skill") == "mistake_review":
                recent_mistakes = state.get("mistake_log", [])[-5:]
                args["recent_mistakes"] = "; ".join(
                    f"[{mistake.get('type', 'unknown')}] {mistake.get('detail', '')}"
                    for mistake in recent_mistakes
                )
            result = tool.invoke(args)
            results.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
            if tool_call["name"] == "generate_exercise":
                state["last_exercise"] = {
                    "active": True,
                    "language": state["language"],
                    "level": state["level"],
                    "context": str(result),
                }
            elif tool_call["name"] == "log_mistake":
                state.setdefault("mistake_log", []).append({
                    "type": args["mistake_type"],
                    "detail": args["detail"],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
        except Exception as exc:
            logger.warning("Tool '%s' failed: %s", tool_call["name"], exc)
            results.append(ToolMessage(content=f"Tool error: {exc}", tool_call_id=tool_call["id"]))
    return results
