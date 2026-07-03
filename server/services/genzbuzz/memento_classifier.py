"""LLM classifier for high-signal memento notifications."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from ...config import get_settings
from ...logging_config import logger
from ...openrouter_client import OpenRouterError, request_chat_completion


_TOOL_NAME = "mark_memento_importance"
_TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": _TOOL_NAME,
        "description": (
            "Decide whether a memento update should be proactively surfaced, "
            "and classify it for notification routing."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "important": {
                    "type": "boolean",
                    "description": (
                        "Set true only when this memento is meaningfully high-signal for the recipient."
                    ),
                },
                "category": {
                    "type": "string",
                    "enum": ["milestone", "tragedy", "favorite_things", "other"],
                    "description": "Primary memento significance category.",
                },
                "summary": {
                    "type": "string",
                    "description": (
                        "Brief 1-2 sentence recipient-facing summary describing what happened and why it matters."
                    ),
                },
            },
            "required": ["important", "category"],
            "additionalProperties": False,
        },
    },
}

_SYSTEM_PROMPT = (
    "You classify memento updates for proactive notifications. Mark important=true only when the "
    "content is emotionally meaningful or high-signal for the recipient. Favor these categories: "
    "milestone (major life progress/achievement), tragedy (loss/crisis/grief), favorite_things "
    "(deep personal favorites/preferences that strengthen relational memory). Use other for low-signal "
    "or routine updates. For important updates, include a concise summary."
)


@dataclass(frozen=True)
class MementoCandidate:
    queue_id: str
    note_id: str
    recipient_email: str
    sender_name: str
    sender_id: int
    queued_at: Optional[datetime]
    note_excerpt: str


def _format_candidate(candidate: MementoCandidate) -> str:
    queued_at = candidate.queued_at.isoformat() if candidate.queued_at else "unknown"
    return (
        f"Queue ID: {candidate.queue_id}\n"
        f"Note ID: {candidate.note_id}\n"
        f"Recipient Email: {candidate.recipient_email}\n"
        f"Sender Name: {candidate.sender_name or 'Unknown'}\n"
        f"Sender ID: {candidate.sender_id}\n"
        f"Queued At: {queued_at}\n\n"
        f"Memento Content:\n{candidate.note_excerpt or '(empty)'}"
    )


async def classify_memento_importance(candidate: MementoCandidate) -> Optional[Dict[str, str]]:
    settings = get_settings()
    api_key = settings.openrouter_api_key
    model = settings.email_classifier_model

    if not api_key:
        logger.warning("Skipping memento classification; OpenRouter API key missing")
        return None

    messages = [{"role": "user", "content": _format_candidate(candidate)}]

    try:
        response = await request_chat_completion(
            model=model,
            messages=messages,
            system=_SYSTEM_PROMPT,
            api_key=api_key,
            tools=[_TOOL_SCHEMA],
        )
    except OpenRouterError as exc:
        logger.error(
            "Memento classification failed",
            extra={"queue_id": candidate.queue_id, "error": str(exc)},
        )
        return None
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception(
            "Unexpected memento classification error",
            extra={"queue_id": candidate.queue_id, "error": str(exc)},
        )
        return None

    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    tool_calls = message.get("tool_calls") or []

    for tool_call in tool_calls:
        function_block = tool_call.get("function") or {}
        if function_block.get("name") != _TOOL_NAME:
            continue

        arguments = _coerce_arguments(function_block.get("arguments"))
        if arguments is None:
            return None

        important = bool(arguments.get("important"))
        category = str(arguments.get("category") or "other").strip().lower()
        if category not in {"milestone", "tragedy", "favorite_things", "other"}:
            category = "other"

        if not important:
            return None

        summary = str(arguments.get("summary") or "").strip()
        if not summary:
            return None

        return {
            "category": category,
            "summary": summary,
        }

    return None


def _coerce_arguments(raw: Any) -> Optional[Dict[str, Any]]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        if not raw.strip():
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return parsed
        return None
    return None


__all__ = ["MementoCandidate", "classify_memento_importance"]
