"""GenZbuzz execution tools for confirmation and send policy enforcement."""

from __future__ import annotations

import base64
from datetime import datetime, timezone
from functools import partial
import hashlib
import re
import time
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from server.services.execution import get_execution_agent_logs
from server.services.genzbuzz import get_genzbuzz_bridge_service, get_genzbuzz_policy_service


_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "genzbuzz_create_send_confirmation",
            "description": "Create a confirmation request before final send.",
            "parameters": {
                "type": "object",
                "properties": {
                    "channel": {
                        "type": "string",
                        "description": "Message channel, usually imessage.",
                    },
                    "message_kind": {
                        "type": "string",
                        "description": "Flow type: bonding, waiting, invite, or fallback.",
                    },
                    "message_preview": {
                        "type": "string",
                        "description": "Short preview of the outbound message.",
                    },
                    "prompt_text": {
                        "type": "string",
                        "description": "Custom confirmation prompt wording.",
                    },
                    "ttl_minutes": {
                        "type": "integer",
                        "description": "How long confirmation stays valid.",
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Optional metadata for audit/logging.",
                    },
                },
                "required": ["channel", "message_kind", "message_preview"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "genzbuzz_resolve_send_confirmation",
            "description": "Resolve user reply to a pending send confirmation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "confirmation_id": {
                        "type": "string",
                        "description": "Confirmation identifier returned by creation tool.",
                    },
                    "user_reply": {
                        "type": "string",
                        "description": "User response text, such as yes/cancel/edit.",
                    },
                },
                "required": ["confirmation_id", "user_reply"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "genzbuzz_evaluate_send_window",
            "description": "Evaluate daytime-only send policy window.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "genzbuzz_record_delivery_attempt",
            "description": "Record a delivery attempt and return escalation instructions after repeated failures.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "Stable identifier for the outbound task.",
                    },
                    "success": {
                        "type": "boolean",
                        "description": "Whether this attempt succeeded.",
                    },
                    "error_message": {
                        "type": "string",
                        "description": "Failure reason when success is false.",
                    },
                    "terminal_failure": {
                        "type": "boolean",
                        "description": "Set true when failure is non-retriable and should stop immediately.",
                    },
                },
                "required": ["task_id", "success"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "genzbuzz_process_due_retries",
            "description": "Process due retry jobs for failed waiting/bonding/spontaneous sends.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum retry jobs to process in this run.",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "genzbuzz_get_waiting_prompt_context",
            "description": "Fetch waiting prompt context with can_accept_reply and current prompt text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "integer",
                        "description": "GenZbuzz user id.",
                    },
                },
                "required": ["user_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "genzbuzz_get_active_waiting_cycle",
            "description": "Fetch active waiting cycle details for a user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "integer",
                        "description": "GenZbuzz user id.",
                    },
                },
                "required": ["user_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "genzbuzz_start_new_friend_onboarding",
            "description": "Start Messenger new-friend onboarding using existing legacy onboarding flow.",
            "parameters": {
                "type": "object",
                "properties": {
                    "psid": {
                        "type": "string",
                        "description": "Messenger PSID for the user session.",
                    },
                    "user_id": {
                        "type": "integer",
                        "description": "GenZbuzz user id.",
                    },
                },
                "required": ["psid", "user_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "genzbuzz_submit_waiting_text_reply",
            "description": "Submit a waiting-mode text reply through WordPress AJAX contract with daytime policy gating.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "integer",
                        "description": "GenZbuzz user id.",
                    },
                    "text_answer": {
                        "type": "string",
                        "description": "Waiting prompt text reply.",
                    },
                    "voice_note_base64": {
                        "type": "string",
                        "description": "Optional base64 audio bytes for voice note reply.",
                    },
                    "voice_note_filename": {
                        "type": "string",
                        "description": "Optional filename for voice note upload.",
                    },
                    "voice_note_mime": {
                        "type": "string",
                        "description": "Optional MIME type for voice note. Must be audio/*.",
                    },
                    "photo_base64": {
                        "type": "string",
                        "description": "Optional base64 bytes for photo attachment.",
                    },
                    "photo_filename": {
                        "type": "string",
                        "description": "Optional filename for photo attachment.",
                    },
                    "photo_mime": {
                        "type": "string",
                        "description": "Optional MIME type for photo attachment. Must be image/* (no video).",
                    },
                    "task_id": {
                        "type": "string",
                        "description": "Stable delivery task id for retry tracking.",
                    },
                    "confirmation_id": {
                        "type": "string",
                        "description": "Confirmed send approval id for this message.",
                    },
                },
                "required": ["user_id", "confirmation_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "genzbuzz_submit_bonding_text_reply",
            "description": "Submit a bonding text reply through WordPress AJAX contract with daytime policy gating.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "integer",
                        "description": "GenZbuzz user id.",
                    },
                    "asker_id": {
                        "type": "integer",
                        "description": "Asker/friend user id.",
                    },
                    "text_answer": {
                        "type": "string",
                        "description": "Bonding text reply.",
                    },
                    "voice_note_base64": {
                        "type": "string",
                        "description": "Optional base64 audio bytes for voice note reply.",
                    },
                    "voice_note_filename": {
                        "type": "string",
                        "description": "Optional filename for voice note upload.",
                    },
                    "voice_note_mime": {
                        "type": "string",
                        "description": "Optional MIME type for voice note. Must be audio/*.",
                    },
                    "photo_base64": {
                        "type": "string",
                        "description": "Optional base64 bytes for photo attachment.",
                    },
                    "photo_filename": {
                        "type": "string",
                        "description": "Optional filename for photo attachment.",
                    },
                    "photo_mime": {
                        "type": "string",
                        "description": "Optional MIME type for photo attachment. Must be image/* (no video).",
                    },
                    "task_id": {
                        "type": "string",
                        "description": "Stable delivery task id for retry tracking.",
                    },
                    "confirmation_id": {
                        "type": "string",
                        "description": "Confirmed send approval id for this message.",
                    },
                },
                "required": ["user_id", "asker_id", "confirmation_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "genzbuzz_create_draft",
            "description": "Create a persistent GenZbuzz draft for waiting, bonding, or spontaneous flows.",
            "parameters": {
                "type": "object",
                "properties": {
                    "draft_kind": {
                        "type": "string",
                        "enum": ["waiting", "bonding", "spontaneous"],
                        "description": "Draft flow kind.",
                    },
                    "channel": {
                        "type": "string",
                        "description": "Message channel, usually imessage.",
                    },
                    "title": {
                        "type": "string",
                        "description": "Optional short title for operator review.",
                    },
                    "payload": {
                        "type": "object",
                        "description": "Kind-specific payload used later for execution.",
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Optional metadata for audit and traceability.",
                    },
                },
                "required": ["draft_kind", "payload"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "genzbuzz_get_draft",
            "description": "Fetch a persisted draft by id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "draft_id": {
                        "type": "string",
                        "description": "Draft id returned by genzbuzz_create_draft.",
                    },
                },
                "required": ["draft_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "genzbuzz_update_draft",
            "description": "Update a persisted draft payload before confirmation/send.",
            "parameters": {
                "type": "object",
                "properties": {
                    "draft_id": {
                        "type": "string",
                        "description": "Draft id to update.",
                    },
                    "payload": {
                        "type": "object",
                        "description": "Replacement payload for the draft.",
                    },
                    "title": {
                        "type": "string",
                        "description": "Updated title for operator review.",
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Replacement metadata object.",
                    },
                    "expected_version": {
                        "type": "integer",
                        "description": "Optional optimistic-concurrency version check.",
                    },
                },
                "required": ["draft_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "genzbuzz_execute_draft",
            "description": "Execute a previously created draft after confirmation checks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "draft_id": {
                        "type": "string",
                        "description": "Draft id to execute.",
                    },
                    "confirmation_id": {
                        "type": "string",
                        "description": "Confirmed send approval id matching this exact draft preview.",
                    },
                    "task_id": {
                        "type": "string",
                        "description": "Optional stable task id for retries and traceability.",
                    },
                },
                "required": ["draft_id", "confirmation_id"],
                "additionalProperties": False,
            },
        },
    },
]

_LOG_STORE = get_execution_agent_logs()
_POLICY = get_genzbuzz_policy_service()
_BRIDGE = get_genzbuzz_bridge_service()

_ACTION_WAITING_REPLY = "waiting_reply"
_ACTION_BONDING_REPLY = "bonding_reply"
_ACTION_SPONTANEOUS_MEMENTO = "spontaneous_memento"


def _fingerprint_text(value: str) -> str:
    return hashlib.sha1((value or "").strip().encode("utf-8")).hexdigest()


def _coerce_status_code(send_result: Dict[str, Any]) -> int:
    raw = send_result.get("http_status") or send_result.get("status_code") or send_result.get("response_code")
    try:
        return int(str(raw).strip())
    except Exception:
        return 0


def _extract_retry_after_seconds(send_result: Dict[str, Any]) -> int:
    raw = str(send_result.get("retry_after") or "").strip()
    if not raw:
        return 0

    # Retry-After can be a whole-second number or an HTTP date.
    try:
        value = int(raw)
        if value > 0:
            return value
    except Exception:
        pass

    match = re.search(r"(\d+)", raw)
    if match:
        try:
            value = int(match.group(1))
            if value > 0:
                return value
        except Exception:
            return 0

    return 0


def _classify_retry_profile(*, send_result: Dict[str, Any], latency_ms: int) -> Dict[str, Any]:
    status_code = _coerce_status_code(send_result)
    error_text = str(send_result.get("error") or send_result.get("message") or "").strip().lower()
    retry_after_seconds = _extract_retry_after_seconds(send_result)

    terminal_failure = False
    if status_code in {400, 401, 403, 404, 405, 410, 422}:
        terminal_failure = True
    if status_code in {408, 409, 423, 425, 429}:
        terminal_failure = False

    transient_markers = (
        "timeout",
        "timed out",
        "temporarily",
        "try again",
        "connection reset",
        "connection aborted",
        "upstream",
        "service unavailable",
        "rate limit",
    )
    hard_markers = (
        "invalid",
        "malformed",
        "forbidden",
        "unauthorized",
        "auth",
        "scope",
        "token",
        "outside of allowed window",
        "retry_invalid_",
    )

    if any(marker in error_text for marker in transient_markers):
        terminal_failure = False
    elif any(marker in error_text for marker in hard_markers):
        terminal_failure = True

    if status_code >= 500:
        terminal_failure = False

    service_health = "healthy"
    if status_code == 429 or retry_after_seconds > 0:
        service_health = "degraded"
    if latency_ms >= 8000:
        service_health = "flapping"
    elif latency_ms >= 3000 and service_health == "healthy":
        service_health = "degraded"

    if not terminal_failure and retry_after_seconds == 0 and status_code == 429:
        retry_after_seconds = 60

    retry_hint: Dict[str, Any] = {
        "latency_ms": max(0, int(latency_ms)),
        "service_health": service_health,
    }
    if retry_after_seconds > 0:
        retry_hint["retry_after_seconds"] = retry_after_seconds

    return {
        "terminal_failure": terminal_failure,
        "status_code": status_code,
        "retry_hint": retry_hint,
        "error_text": error_text,
    }


def _decode_base64_payload(raw_value: Optional[str]) -> Optional[bytes]:
    if not raw_value:
        return None
    candidate = raw_value.strip()
    if "," in candidate and candidate.lower().startswith("data:"):
        candidate = candidate.split(",", 1)[1].strip()
    try:
        return base64.b64decode(candidate, validate=True)
    except Exception:
        return None


def _validate_media_inputs(
    *,
    text_answer: Optional[str],
    voice_note_base64: Optional[str],
    voice_note_mime: Optional[str],
    photo_base64: Optional[str],
    photo_mime: Optional[str],
) -> Dict[str, Any]:
    has_text = bool((text_answer or "").strip())
    has_voice = bool((voice_note_base64 or "").strip())

    if has_text and has_voice:
        return {"ok": False, "error": "Provide either text or voice note, not both."}
    if not has_text and not has_voice:
        return {"ok": False, "error": "Either text or voice note is required."}

    if has_voice:
        mime = (voice_note_mime or "audio/m4a").strip().lower()
        if not mime.startswith("audio/"):
            return {"ok": False, "error": "Voice note MIME must be audio/*."}

    if (photo_base64 or "").strip():
        mime = (photo_mime or "image/jpeg").strip().lower()
        if mime.startswith("video/"):
            return {"ok": False, "error": "Video attachments are not allowed. Photo must be image/*."}
        if not mime.startswith("image/"):
            return {"ok": False, "error": "Photo MIME must be image/*."}

    return {"ok": True}


def _normalize_draft_kind(raw_kind: str) -> str:
    kind = (raw_kind or "").strip().lower()
    if kind in {"waiting", "waiting_prompt"}:
        return "waiting"
    if kind in {"bonding", "bonding_prompt", "bonding_cycle"}:
        return "bonding"
    if kind in {"spontaneous", "spontaneous_memento", "memento"}:
        return "spontaneous"
    return ""


def _draft_preview(kind: str, payload: Dict[str, Any]) -> str:
    if kind == "waiting":
        text_answer = str(payload.get("text_answer") or "").strip()
        if text_answer:
            return text_answer
        if str(payload.get("voice_note_base64") or "").strip():
            return "[voice note]"
        return ""

    if kind == "bonding":
        text_answer = str(payload.get("text_answer") or "").strip()
        if text_answer:
            return text_answer
        if str(payload.get("voice_note_base64") or "").strip():
            return "[voice note]"
        return ""

    if kind == "spontaneous":
        if str(payload.get("voice_note_base64") or "").strip():
            recipient_id = str(payload.get("recipient_id") or "").strip()
            return f"[spontaneous voice note] recipient_id={recipient_id}".strip()
        return ""

    return ""


def _execute_waiting_bridge_call(payload: Dict[str, Any]) -> Dict[str, Any]:
    text_answer = str(payload.get("text_answer") or "").strip()
    voice_note_base64 = payload.get("voice_note_base64")
    voice_note_filename = payload.get("voice_note_filename")
    voice_note_mime = payload.get("voice_note_mime")
    photo_base64 = payload.get("photo_base64")
    photo_filename = payload.get("photo_filename")
    photo_mime = payload.get("photo_mime")

    voice_bytes = _decode_base64_payload(voice_note_base64)
    if (voice_note_base64 or "").strip() and voice_bytes is None:
        return {"success": False, "error": "retry_invalid_voice_payload"}

    photo_bytes = _decode_base64_payload(photo_base64)
    if (photo_base64 or "").strip() and photo_bytes is None:
        return {"success": False, "error": "retry_invalid_photo_payload"}

    audio_file = None
    if voice_bytes is not None:
        audio_file = (
            (str(voice_note_filename or "voice_note.m4a")),
            voice_bytes,
            str(voice_note_mime or "audio/m4a"),
        )

    photo_file = None
    if photo_bytes is not None:
        photo_file = (
            (str(photo_filename or "photo.jpg")),
            photo_bytes,
            str(photo_mime or "image/jpeg"),
        )

    return _BRIDGE.submit_waiting_reply(
        user_id=int(payload.get("user_id", 0)),
        text_answer=text_answer,
        audio_file=audio_file,
        photo_file=photo_file,
    )


def _execute_bonding_bridge_call(payload: Dict[str, Any]) -> Dict[str, Any]:
    text_answer = str(payload.get("text_answer") or "").strip()
    voice_note_base64 = payload.get("voice_note_base64")
    voice_note_filename = payload.get("voice_note_filename")
    voice_note_mime = payload.get("voice_note_mime")
    photo_base64 = payload.get("photo_base64")
    photo_filename = payload.get("photo_filename")
    photo_mime = payload.get("photo_mime")

    voice_bytes = _decode_base64_payload(voice_note_base64)
    if (voice_note_base64 or "").strip() and voice_bytes is None:
        return {"success": False, "error": "retry_invalid_voice_payload"}

    photo_bytes = _decode_base64_payload(photo_base64)
    if (photo_base64 or "").strip() and photo_bytes is None:
        return {"success": False, "error": "retry_invalid_photo_payload"}

    audio_file = None
    if voice_bytes is not None:
        audio_file = (
            (str(voice_note_filename or "voice_note.m4a")),
            voice_bytes,
            str(voice_note_mime or "audio/m4a"),
        )

    photo_file = None
    if photo_bytes is not None:
        photo_file = (
            (str(photo_filename or "photo.jpg")),
            photo_bytes,
            str(photo_mime or "image/jpeg"),
        )

    return _BRIDGE.submit_bonding_reply(
        user_id=int(payload.get("user_id", 0)),
        asker_id=int(payload.get("asker_id", 0)),
        text_answer=text_answer,
        audio_file=audio_file,
        photo_file=photo_file,
    )


def _execute_spontaneous_bridge_call(payload: Dict[str, Any]) -> Dict[str, Any]:
    voice_note_base64 = payload.get("voice_note_base64")
    voice_note_filename = payload.get("voice_note_filename")
    voice_note_mime = payload.get("voice_note_mime")
    photo_base64 = payload.get("photo_base64")
    photo_filename = payload.get("photo_filename")
    photo_mime = payload.get("photo_mime")

    voice_bytes = _decode_base64_payload(voice_note_base64)
    if voice_bytes is None:
        return {"success": False, "error": "retry_invalid_voice_payload"}

    photo_bytes = _decode_base64_payload(photo_base64)
    if (photo_base64 or "").strip() and photo_bytes is None:
        return {"success": False, "error": "retry_invalid_photo_payload"}

    audio_file = (
        str(voice_note_filename or "voice_note.m4a"),
        voice_bytes,
        str(voice_note_mime or "audio/m4a"),
    )

    photo_file = None
    if photo_bytes is not None:
        photo_file = (
            str(photo_filename or "photo.jpg"),
            photo_bytes,
            str(photo_mime or "image/jpeg"),
        )

    return _BRIDGE.submit_spontaneous_memento(
        user_id=int(payload.get("user_id", 0)),
        recipient_id=int(payload.get("recipient_id", 0)),
        audio_file=audio_file,
        photo_file=photo_file,
    )


def get_schemas() -> List[Dict[str, Any]]:
    """Return GenZbuzz tool schemas."""
    return _SCHEMAS


def _create_send_confirmation_tool(
    *,
    agent_name: str,
    channel: str,
    message_kind: str,
    message_preview: str,
    prompt_text: Optional[str] = None,
    ttl_minutes: int = 120,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    enriched_metadata = dict(metadata or {})
    enriched_metadata.setdefault("message_preview_sha1", _fingerprint_text(message_preview))

    result = _POLICY.create_confirmation(
        agent_name=agent_name,
        channel=channel,
        message_kind=message_kind,
        message_preview=message_preview,
        prompt_text=prompt_text,
        ttl_minutes=ttl_minutes,
        metadata=enriched_metadata,
    )
    _LOG_STORE.record_action(
        agent_name,
        description=f"genzbuzz_create_send_confirmation | id={result.get('confirmation_id', 'unknown')}",
    )
    return result


def _resolve_send_confirmation_tool(
    *,
    agent_name: str,
    confirmation_id: str,
    user_reply: str,
) -> Dict[str, Any]:
    result = _POLICY.resolve_confirmation(
        confirmation_id=confirmation_id,
        user_reply=user_reply,
    )
    _LOG_STORE.record_action(
        agent_name,
        description=(
            "genzbuzz_resolve_send_confirmation "
            f"| id={confirmation_id} | status={result.get('status', 'unknown')}"
        ),
    )
    return result


def _evaluate_send_window_tool(*, agent_name: str) -> Dict[str, Any]:
    result = _POLICY.evaluate_send_window()
    _LOG_STORE.record_action(
        agent_name,
        description=(
            "genzbuzz_evaluate_send_window "
            f"| within_window={result.get('is_within_window', False)}"
        ),
    )
    return result


def _record_delivery_attempt_tool(
    *,
    agent_name: str,
    task_id: str,
    success: bool,
    error_message: Optional[str] = None,
    terminal_failure: bool = False,
) -> Dict[str, Any]:
    result = _POLICY.record_delivery_attempt(
        task_id=task_id,
        success=success,
        error_message=error_message,
        terminal_failure=terminal_failure,
    )
    _LOG_STORE.record_action(
        agent_name,
        description=(
            "genzbuzz_record_delivery_attempt "
            f"| task_id={task_id} | attempt={result.get('attempt_count', 0)} "
            f"| status={result.get('status', 'unknown')}"
        ),
    )
    return result


def _get_active_waiting_cycle_tool(*, agent_name: str, user_id: int) -> Dict[str, Any]:
    result = _BRIDGE.get_active_waiting_cycle(user_id=user_id)
    _LOG_STORE.record_action(
        agent_name,
        description=(
            "genzbuzz_get_active_waiting_cycle "
            f"| user_id={user_id} | success={bool(result.get('success'))}"
        ),
    )
    return result


def _get_waiting_prompt_context_tool(*, agent_name: str, user_id: int) -> Dict[str, Any]:
    result = _BRIDGE.get_waiting_prompt_context(user_id=user_id)
    _LOG_STORE.record_action(
        agent_name,
        description=(
            "genzbuzz_get_waiting_prompt_context "
            f"| user_id={user_id} | success={bool(result.get('success'))}"
        ),
    )
    return result


def _start_new_friend_onboarding_tool(*, agent_name: str, psid: str, user_id: int) -> Dict[str, Any]:
    clean_psid = str(psid or "").strip()
    if clean_psid == "":
        return {"error": "psid is required"}

    # Safety: only allow onboarding start when PSID resolves to the same linked user.
    lookup = _BRIDGE.lookup_messenger_user(psid=clean_psid)
    if not bool(lookup.get("success") or lookup.get("ok")):
        return {"error": "psid_not_linked"}

    resolved_user_id = int((lookup.get("data") or {}).get("user_id") or lookup.get("user_id") or 0)
    if resolved_user_id <= 0 or resolved_user_id != int(user_id):
        return {"error": "psid_user_mismatch"}

    result = _BRIDGE.messenger_new_friend(psid=clean_psid, user_id=int(user_id))
    _LOG_STORE.record_action(
        agent_name,
        description=(
            "genzbuzz_start_new_friend_onboarding "
            f"| user_id={int(user_id)} | success={bool(result.get('success'))}"
        ),
    )
    return result


def _build_task_id(prefix: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"{prefix}-{timestamp}-{uuid4().hex[:8]}"


def _create_draft_tool(
    *,
    agent_name: str,
    draft_kind: str,
    payload: Dict[str, Any],
    channel: str = "imessage",
    title: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    kind = _normalize_draft_kind(draft_kind)
    if kind == "":
        return {"error": "draft_kind must be one of waiting, bonding, spontaneous"}

    clean_payload = payload if isinstance(payload, dict) else {}
    preview = _draft_preview(kind, clean_payload)
    if preview == "":
        return {"error": "payload does not contain a valid draft preview"}

    enriched_metadata = dict(metadata or {})
    enriched_metadata.setdefault("message_preview_sha1", _fingerprint_text(preview))

    result = _POLICY.create_draft(
        agent_name=agent_name,
        channel=(channel or "imessage"),
        draft_kind=kind,
        payload=clean_payload,
        title=title,
        metadata=enriched_metadata,
    )
    _LOG_STORE.record_action(
        agent_name,
        description=(
            "genzbuzz_create_draft "
            f"| kind={kind} | draft_id={result.get('draft_id', 'unknown')}"
        ),
    )
    result["message_preview"] = preview
    result["message_preview_sha1"] = enriched_metadata.get("message_preview_sha1")
    return result


def _get_draft_tool(*, agent_name: str, draft_id: str) -> Dict[str, Any]:
    result = _POLICY.get_draft(draft_id=draft_id)
    _LOG_STORE.record_action(
        agent_name,
        description=(
            "genzbuzz_get_draft "
            f"| draft_id={draft_id} | ok={not bool(result.get('error'))}"
        ),
    )
    return result


def _update_draft_tool(
    *,
    agent_name: str,
    draft_id: str,
    payload: Optional[Dict[str, Any]] = None,
    title: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    expected_version: Optional[int] = None,
) -> Dict[str, Any]:
    next_payload = payload if payload is None or isinstance(payload, dict) else {}
    result = _POLICY.update_draft(
        draft_id=draft_id,
        payload=next_payload,
        title=title,
        metadata=metadata,
        expected_version=expected_version,
    )

    if not result.get("error"):
        refreshed = _POLICY.get_draft(draft_id=draft_id)
        kind = _normalize_draft_kind(str(refreshed.get("draft_kind") or ""))
        draft_payload = refreshed.get("payload") if isinstance(refreshed.get("payload"), dict) else {}
        preview = _draft_preview(kind, draft_payload)
        if preview:
            result["message_preview"] = preview
            result["message_preview_sha1"] = _fingerprint_text(preview)

    _LOG_STORE.record_action(
        agent_name,
        description=(
            "genzbuzz_update_draft "
            f"| draft_id={draft_id} | ok={not bool(result.get('error'))}"
        ),
    )
    return result


def _execute_draft_tool(
    *,
    agent_name: str,
    draft_id: str,
    confirmation_id: str,
    task_id: Optional[str] = None,
) -> Dict[str, Any]:
    draft = _POLICY.get_draft(draft_id=draft_id)
    if draft.get("error"):
        return draft

    draft_status = str(draft.get("status") or "draft").strip().lower()
    if draft_status in {"sent", "archived"}:
        return {
            "draft_id": draft_id,
            "status": "not_executable",
            "reason": f"draft_status:{draft_status}",
        }

    channel = str(draft.get("channel") or "imessage").strip() or "imessage"
    kind = _normalize_draft_kind(str(draft.get("draft_kind") or ""))
    payload = draft.get("payload") if isinstance(draft.get("payload"), dict) else {}
    preview = _draft_preview(kind, payload)
    if preview == "":
        return {
            "draft_id": draft_id,
            "status": "invalid_payload",
            "reason": "payload does not contain a valid draft preview",
        }

    gate = _require_confirmed_send(
        confirmation_id=confirmation_id,
        expected_channel=channel,
        expected_message_kind=kind,
        current_text=preview,
    )
    if not gate.get("ok"):
        return {
            "draft_id": draft_id,
            "status": gate.get("status", "confirmation_required"),
            "reason": gate.get("reason", "confirmation_required"),
            "confirmation": gate.get("confirmation", {}),
        }

    _POLICY.mark_draft_status(draft_id=draft_id, status="ready")

    if kind == "waiting":
        result = _submit_waiting_text_reply_tool(
            agent_name=agent_name,
            user_id=int(payload.get("user_id", 0)),
            text_answer=payload.get("text_answer"),
            voice_note_base64=payload.get("voice_note_base64"),
            voice_note_filename=payload.get("voice_note_filename"),
            voice_note_mime=payload.get("voice_note_mime"),
            photo_base64=payload.get("photo_base64"),
            photo_filename=payload.get("photo_filename"),
            photo_mime=payload.get("photo_mime"),
            confirmation_id=confirmation_id,
            task_id=task_id,
        )
    elif kind == "bonding":
        result = _submit_bonding_text_reply_tool(
            agent_name=agent_name,
            user_id=int(payload.get("user_id", 0)),
            asker_id=int(payload.get("asker_id", 0)),
            text_answer=payload.get("text_answer"),
            voice_note_base64=payload.get("voice_note_base64"),
            voice_note_filename=payload.get("voice_note_filename"),
            voice_note_mime=payload.get("voice_note_mime"),
            photo_base64=payload.get("photo_base64"),
            photo_filename=payload.get("photo_filename"),
            photo_mime=payload.get("photo_mime"),
            confirmation_id=confirmation_id,
            task_id=task_id,
        )
    elif kind == "spontaneous":
        active_task_id = (task_id or "").strip() or _build_task_id("spontaneous")
        voice_note_mime = str(payload.get("voice_note_mime") or "audio/m4a").strip().lower()
        if not voice_note_mime.startswith("audio/"):
            return {
                "draft_id": draft_id,
                "task_id": active_task_id,
                "status": "invalid_payload",
                "reason": "Voice note MIME must be audio/*.",
            }

        photo_mime = str(payload.get("photo_mime") or "image/jpeg").strip().lower()
        if str(payload.get("photo_base64") or "").strip() and not photo_mime.startswith("image/"):
            return {
                "draft_id": draft_id,
                "task_id": active_task_id,
                "status": "invalid_payload",
                "reason": "Photo MIME must be image/*.",
            }

        started_at = time.monotonic()
        send_result = _execute_spontaneous_bridge_call(payload)
        latency_ms = int((time.monotonic() - started_at) * 1000)
        success = bool(send_result.get("success"))
        retry_profile = _classify_retry_profile(send_result=send_result, latency_ms=latency_ms)
        attempt = _POLICY.record_delivery_attempt(
            task_id=active_task_id,
            success=success,
            error_message=None if success else str(send_result.get("message") or send_result.get("error") or "spontaneous_memento_failed"),
            terminal_failure=False if success else bool(retry_profile.get("terminal_failure")),
        )

        retry_job = None
        if not success and attempt.get("status") == "failed_attempt":
            retry_job = _POLICY.schedule_retry(
                task_id=active_task_id,
                action_name=_ACTION_SPONTANEOUS_MEMENTO,
                action_payload=payload,
                attempt_count=int(attempt.get("attempt_count", 1)),
                last_error=str(attempt.get("last_error") or "spontaneous_memento_failed"),
                retry_hint=retry_profile.get("retry_hint"),
            )
        elif not success:
            _POLICY.clear_retry_job(task_id=active_task_id)

        _LOG_STORE.record_action(
            agent_name,
            description=(
                "genzbuzz_execute_draft "
                f"| draft_id={draft_id} | kind=spontaneous | success={success}"
            ),
        )

        result = {
            "draft_id": draft_id,
            "task_id": active_task_id,
            "status": "sent" if success else "failed",
            "response": send_result,
            "attempt": attempt,
            "retry_job": retry_job,
            "retry_profile": retry_profile,
        }
    else:
        return {
            "draft_id": draft_id,
            "status": "invalid_payload",
            "reason": f"Unsupported draft kind: {kind or 'unknown'}",
        }

    if str(result.get("status") or "") == "sent":
        _POLICY.mark_draft_status(draft_id=draft_id, status="sent")

    return {
        "draft_id": draft_id,
        "draft_status": _POLICY.get_draft(draft_id=draft_id).get("status", "unknown"),
        "result": result,
    }


def _require_confirmed_send(
    *,
    confirmation_id: str,
    expected_channel: str,
    expected_message_kind: str,
    current_text: str,
) -> Dict[str, Any]:
    confirmation = _POLICY.get_confirmation(confirmation_id=confirmation_id)
    if confirmation.get("error"):
        return {
            "ok": False,
            "status": "confirmation_required",
            "reason": "unknown_confirmation",
            "confirmation": confirmation,
        }

    status = confirmation.get("status")
    if status != "confirmed":
        return {
            "ok": False,
            "status": "confirmation_required",
            "reason": f"confirmation_not_confirmed:{status}",
            "confirmation": confirmation,
        }

    if str(confirmation.get("channel", "")).strip() != expected_channel:
        return {
            "ok": False,
            "status": "confirmation_required",
            "reason": "confirmation_channel_mismatch",
            "confirmation": confirmation,
        }

    if str(confirmation.get("message_kind", "")).strip() != expected_message_kind:
        return {
            "ok": False,
            "status": "confirmation_required",
            "reason": "confirmation_kind_mismatch",
            "confirmation": confirmation,
        }

    metadata = confirmation.get("metadata", {}) if isinstance(confirmation.get("metadata"), dict) else {}
    expected_sha1 = str(metadata.get("message_preview_sha1", "")).strip()
    if expected_sha1 and expected_sha1 != _fingerprint_text(current_text):
        return {
            "ok": False,
            "status": "confirmation_required",
            "reason": "response_changed_reconfirm_required",
            "confirmation": confirmation,
        }

    return {"ok": True, "confirmation": confirmation}


def _submit_waiting_text_reply_tool(
    *,
    agent_name: str,
    user_id: int,
    text_answer: Optional[str] = None,
    voice_note_base64: Optional[str] = None,
    voice_note_filename: Optional[str] = None,
    voice_note_mime: Optional[str] = None,
    photo_base64: Optional[str] = None,
    photo_filename: Optional[str] = None,
    photo_mime: Optional[str] = None,
    confirmation_id: str,
    task_id: Optional[str] = None,
) -> Dict[str, Any]:
    active_task_id = (task_id or "").strip() or _build_task_id("waiting")

    media_check = _validate_media_inputs(
        text_answer=text_answer,
        voice_note_base64=voice_note_base64,
        voice_note_mime=voice_note_mime,
        photo_base64=photo_base64,
        photo_mime=photo_mime,
    )
    if not media_check.get("ok"):
        return {
            "task_id": active_task_id,
            "status": "invalid_payload",
            "reason": media_check.get("error", "Invalid waiting reply payload"),
        }

    voice_bytes = _decode_base64_payload(voice_note_base64)
    if (voice_note_base64 or "").strip() and voice_bytes is None:
        return {
            "task_id": active_task_id,
            "status": "invalid_payload",
            "reason": "Invalid voice_note_base64 payload",
        }

    photo_bytes = _decode_base64_payload(photo_base64)
    if (photo_base64 or "").strip() and photo_bytes is None:
        return {
            "task_id": active_task_id,
            "status": "invalid_payload",
            "reason": "Invalid photo_base64 payload",
        }

    preview_for_confirmation = (text_answer or "").strip() if (text_answer or "").strip() else "[voice note]"

    confirmation_gate = _require_confirmed_send(
        confirmation_id=confirmation_id,
        expected_channel="imessage",
        expected_message_kind="waiting",
        current_text=preview_for_confirmation,
    )
    if not confirmation_gate.get("ok"):
        return {
            "task_id": active_task_id,
            "status": confirmation_gate.get("status", "confirmation_required"),
            "reason": confirmation_gate.get("reason", "confirmation_required"),
            "confirmation": confirmation_gate.get("confirmation", {}),
        }

    audio_file = None
    if voice_bytes is not None:
        audio_file = (
            (voice_note_filename or "voice_note.m4a"),
            voice_bytes,
            (voice_note_mime or "audio/m4a"),
        )

    photo_file = None
    if photo_bytes is not None:
        photo_file = (
            (photo_filename or "photo.jpg"),
            photo_bytes,
            (photo_mime or "image/jpeg"),
        )

    started_at = time.monotonic()
    result = _BRIDGE.submit_waiting_reply(
        user_id=user_id,
        text_answer=(text_answer or "").strip(),
        audio_file=audio_file,
        photo_file=photo_file,
    )
    latency_ms = int((time.monotonic() - started_at) * 1000)
    success = bool(result.get("success"))
    retry_profile = _classify_retry_profile(send_result=result, latency_ms=latency_ms)
    attempt = _POLICY.record_delivery_attempt(
        task_id=active_task_id,
        success=success,
        error_message=None if success else str(result.get("message") or result.get("error") or "waiting_reply_failed"),
        terminal_failure=False if success else bool(retry_profile.get("terminal_failure")),
    )

    _LOG_STORE.record_action(
        agent_name,
        description=(
            "genzbuzz_submit_waiting_text_reply "
            f"| task_id={active_task_id} | success={success}"
        ),
    )

    retry_job = None
    if not success and attempt.get("status") == "failed_attempt":
        retry_payload = {
            "user_id": int(user_id),
            "text_answer": (text_answer or "").strip(),
            "voice_note_base64": voice_note_base64,
            "voice_note_filename": voice_note_filename,
            "voice_note_mime": voice_note_mime,
            "photo_base64": photo_base64,
            "photo_filename": photo_filename,
            "photo_mime": photo_mime,
        }
        retry_job = _POLICY.schedule_retry(
            task_id=active_task_id,
            action_name=_ACTION_WAITING_REPLY,
            action_payload=retry_payload,
            attempt_count=int(attempt.get("attempt_count", 1)),
            last_error=str(attempt.get("last_error") or "waiting_reply_failed"),
            retry_hint=retry_profile.get("retry_hint"),
        )
    elif not success:
        _POLICY.clear_retry_job(task_id=active_task_id)

    return {
        "task_id": active_task_id,
        "status": "sent" if success else "failed",
        "response": result,
        "attempt": attempt,
        "retry_job": retry_job,
        "retry_profile": retry_profile,
    }


def _submit_bonding_text_reply_tool(
    *,
    agent_name: str,
    user_id: int,
    asker_id: int,
    text_answer: Optional[str] = None,
    voice_note_base64: Optional[str] = None,
    voice_note_filename: Optional[str] = None,
    voice_note_mime: Optional[str] = None,
    photo_base64: Optional[str] = None,
    photo_filename: Optional[str] = None,
    photo_mime: Optional[str] = None,
    confirmation_id: str,
    task_id: Optional[str] = None,
) -> Dict[str, Any]:
    active_task_id = (task_id or "").strip() or _build_task_id("bonding")

    media_check = _validate_media_inputs(
        text_answer=text_answer,
        voice_note_base64=voice_note_base64,
        voice_note_mime=voice_note_mime,
        photo_base64=photo_base64,
        photo_mime=photo_mime,
    )
    if not media_check.get("ok"):
        return {
            "task_id": active_task_id,
            "status": "invalid_payload",
            "reason": media_check.get("error", "Invalid bonding reply payload"),
        }

    voice_bytes = _decode_base64_payload(voice_note_base64)
    if (voice_note_base64 or "").strip() and voice_bytes is None:
        return {
            "task_id": active_task_id,
            "status": "invalid_payload",
            "reason": "Invalid voice_note_base64 payload",
        }

    photo_bytes = _decode_base64_payload(photo_base64)
    if (photo_base64 or "").strip() and photo_bytes is None:
        return {
            "task_id": active_task_id,
            "status": "invalid_payload",
            "reason": "Invalid photo_base64 payload",
        }

    preview_for_confirmation = (text_answer or "").strip() if (text_answer or "").strip() else "[voice note]"

    confirmation_gate = _require_confirmed_send(
        confirmation_id=confirmation_id,
        expected_channel="imessage",
        expected_message_kind="bonding",
        current_text=preview_for_confirmation,
    )
    if not confirmation_gate.get("ok"):
        return {
            "task_id": active_task_id,
            "status": confirmation_gate.get("status", "confirmation_required"),
            "reason": confirmation_gate.get("reason", "confirmation_required"),
            "confirmation": confirmation_gate.get("confirmation", {}),
        }

    window = _POLICY.evaluate_send_window()
    if not window.get("is_within_window", False):
        return {
            "task_id": active_task_id,
            "status": "queued_for_daytime",
            "queue_reason": "outside_daytime_window",
            "send_window": window,
        }

    audio_file = None
    if voice_bytes is not None:
        audio_file = (
            (voice_note_filename or "voice_note.m4a"),
            voice_bytes,
            (voice_note_mime or "audio/m4a"),
        )

    photo_file = None
    if photo_bytes is not None:
        photo_file = (
            (photo_filename or "photo.jpg"),
            photo_bytes,
            (photo_mime or "image/jpeg"),
        )

    started_at = time.monotonic()
    result = _BRIDGE.submit_bonding_reply(
        user_id=user_id,
        asker_id=asker_id,
        text_answer=(text_answer or "").strip(),
        audio_file=audio_file,
        photo_file=photo_file,
    )
    latency_ms = int((time.monotonic() - started_at) * 1000)
    success = bool(result.get("success"))
    retry_profile = _classify_retry_profile(send_result=result, latency_ms=latency_ms)
    attempt = _POLICY.record_delivery_attempt(
        task_id=active_task_id,
        success=success,
        error_message=None if success else str(result.get("message") or result.get("error") or "bonding_reply_failed"),
        terminal_failure=False if success else bool(retry_profile.get("terminal_failure")),
    )

    _LOG_STORE.record_action(
        agent_name,
        description=(
            "genzbuzz_submit_bonding_text_reply "
            f"| task_id={active_task_id} | success={success}"
        ),
    )

    retry_job = None
    if not success and attempt.get("status") == "failed_attempt":
        retry_payload = {
            "user_id": int(user_id),
            "asker_id": int(asker_id),
            "text_answer": (text_answer or "").strip(),
            "voice_note_base64": voice_note_base64,
            "voice_note_filename": voice_note_filename,
            "voice_note_mime": voice_note_mime,
            "photo_base64": photo_base64,
            "photo_filename": photo_filename,
            "photo_mime": photo_mime,
        }
        retry_job = _POLICY.schedule_retry(
            task_id=active_task_id,
            action_name=_ACTION_BONDING_REPLY,
            action_payload=retry_payload,
            attempt_count=int(attempt.get("attempt_count", 1)),
            last_error=str(attempt.get("last_error") or "bonding_reply_failed"),
            retry_hint=retry_profile.get("retry_hint"),
        )
    elif not success:
        _POLICY.clear_retry_job(task_id=active_task_id)

    return {
        "task_id": active_task_id,
        "status": "sent" if success else "failed",
        "response": result,
        "attempt": attempt,
        "retry_job": retry_job,
        "retry_profile": retry_profile,
    }


def _process_due_retries_tool(*, agent_name: str, limit: int = 20) -> Dict[str, Any]:
    due = _POLICY.get_due_retries(limit=limit)
    jobs = due.get("jobs", []) if isinstance(due.get("jobs"), list) else []

    processed: List[Dict[str, Any]] = []
    for job in jobs:
        if not isinstance(job, dict):
            continue

        task_id = str(job.get("task_id") or "").strip()
        action_name = str(job.get("action_name") or "").strip()
        payload = job.get("action_payload", {}) if isinstance(job.get("action_payload"), dict) else {}

        if not task_id or not action_name:
            _POLICY.clear_retry_job(task_id=task_id)
            continue

        if action_name == _ACTION_WAITING_REPLY:
            send_result = _execute_waiting_bridge_call(payload)
        elif action_name == _ACTION_BONDING_REPLY:
            send_result = _execute_bonding_bridge_call(payload)
        elif action_name == _ACTION_SPONTANEOUS_MEMENTO:
            send_result = _execute_spontaneous_bridge_call(payload)
        else:
            send_result = {"success": False, "error": f"unknown_retry_action:{action_name}"}

        success = bool(send_result.get("success"))
        retry_profile = _classify_retry_profile(send_result=send_result, latency_ms=0)
        attempt = _POLICY.record_delivery_attempt(
            task_id=task_id,
            success=success,
            error_message=None if success else str(send_result.get("message") or send_result.get("error") or "retry_send_failed"),
            terminal_failure=False if success else bool(retry_profile.get("terminal_failure")),
        )

        retry_job = None
        if success:
            _POLICY.clear_retry_job(task_id=task_id)
        else:
            if attempt.get("status") == "failed_attempt":
                retry_job = _POLICY.schedule_retry(
                    task_id=task_id,
                    action_name=action_name,
                    action_payload=payload,
                    attempt_count=int(attempt.get("attempt_count", 1)),
                    last_error=str(attempt.get("last_error") or "retry_send_failed"),
                    retry_hint=retry_profile.get("retry_hint"),
                )
            else:
                _POLICY.clear_retry_job(task_id=task_id)

        processed.append(
            {
                "task_id": task_id,
                "action_name": action_name,
                "status": "sent" if success else "failed",
                "attempt": attempt,
                "retry_job": retry_job,
                "retry_profile": retry_profile,
                "response": send_result,
            }
        )

    _LOG_STORE.record_action(
        agent_name,
        description=(
            "genzbuzz_process_due_retries "
            f"| due={len(jobs)} | processed={len(processed)}"
        ),
    )

    return {
        "due_count": len(jobs),
        "processed_count": len(processed),
        "results": processed,
        "generated_at": due.get("generated_at"),
    }


def build_registry(agent_name: str) -> Dict[str, Callable[..., Any]]:
    """Return GenZbuzz tool callables bound to a specific agent."""
    return {
        "genzbuzz_create_draft": partial(_create_draft_tool, agent_name=agent_name),
        "genzbuzz_get_draft": partial(_get_draft_tool, agent_name=agent_name),
        "genzbuzz_update_draft": partial(_update_draft_tool, agent_name=agent_name),
        "genzbuzz_execute_draft": partial(_execute_draft_tool, agent_name=agent_name),
        "genzbuzz_create_send_confirmation": partial(_create_send_confirmation_tool, agent_name=agent_name),
        "genzbuzz_resolve_send_confirmation": partial(_resolve_send_confirmation_tool, agent_name=agent_name),
        "genzbuzz_evaluate_send_window": partial(_evaluate_send_window_tool, agent_name=agent_name),
        "genzbuzz_record_delivery_attempt": partial(_record_delivery_attempt_tool, agent_name=agent_name),
        "genzbuzz_get_waiting_prompt_context": partial(_get_waiting_prompt_context_tool, agent_name=agent_name),
        "genzbuzz_get_active_waiting_cycle": partial(_get_active_waiting_cycle_tool, agent_name=agent_name),
        "genzbuzz_start_new_friend_onboarding": partial(_start_new_friend_onboarding_tool, agent_name=agent_name),
        "genzbuzz_submit_waiting_text_reply": partial(_submit_waiting_text_reply_tool, agent_name=agent_name),
        "genzbuzz_submit_bonding_text_reply": partial(_submit_bonding_text_reply_tool, agent_name=agent_name),
        "genzbuzz_process_due_retries": partial(_process_due_retries_tool, agent_name=agent_name),
    }


__all__ = [
    "build_registry",
    "get_schemas",
]
