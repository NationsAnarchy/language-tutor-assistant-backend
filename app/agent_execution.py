"""Internal tool execution and prompt-context helpers for agent nodes."""

from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from .logging_config import get_logger
from .text_utils import extract_text

logger = get_logger(__name__)


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
            result = tool.invoke(tool_call["args"])
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
                    "type": tool_call["args"].get("mistake_type", "unknown"),
                    "detail": tool_call["args"].get("detail", ""),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
        except Exception as exc:
            logger.warning("Tool '%s' failed: %s", tool_call["name"], exc)
            results.append(ToolMessage(content=f"Tool error: {exc}", tool_call_id=tool_call["id"]))
    return results
