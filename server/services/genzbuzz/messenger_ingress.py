"""OpenPoke-first Messenger ingress powered by the interaction agent runtime."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Dict, Optional

from ...services.conversation import (
    begin_outbound_collection,
    get_conversation_log,
    get_outbound_messages,
    reset_delivery_context,
    reset_outbound_collection,
    set_delivery_context,
)
from ...config import get_settings
from ...logging_config import logger
from ...agents.interaction_agent.runtime import InteractionAgentRuntime
from ...openrouter_client import OpenRouterError, request_chat_completion
from .bridge import get_genzbuzz_bridge_service
from .policy_service import get_genzbuzz_policy_service


@dataclass
class MessengerIngressResult:
    handled: bool
    intent: str
    success: bool
    message: str = ""
    user_id: Optional[int] = None
    reply_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "handled": self.handled,
            "intent": self.intent,
            "success": self.success,
            "message": self.message,
            "user_id": self.user_id,
            "reply_text": self.reply_text,
        }


JOURNAL_STATES = {
    "IDLE",
    "PROMPTING",
    "DRAFTING",
    "PENDING_APPROVAL",
    "SENDING",
    "FAILED",
    "WAITING_FOR_INPUT",
    "PAUSED",
    "STALE",
    "SAVED",
}

WAITING_TYPED_INTENTS = {"APPROVE", "CANCEL", "PIVOT", "ACK", "OTHER"}

_waiting_prompt_clarify_counts: Dict[str, int] = {}
_waiting_prompt_offramp_locked: Dict[str, bool] = {}
_lifecycle_sent_correlation_ids: Dict[str, Dict[str, str]] = {}
_onboarding_terminal_state: Dict[str, str] = {}


async def route_messenger_ingress(*, psid: str, text: str, user_id: Optional[int] = None) -> MessengerIngressResult:
    clean_psid = (psid or "").strip()
    clean_text = (text or "").strip()
    is_signup_handoff = clean_text.startswith("[SIGNUP_HANDOFF]")
    if is_signup_handoff:
        clean_text = clean_text.replace("[SIGNUP_HANDOFF]", "", 1).strip()

    if not clean_psid or not clean_text:
        return MessengerIngressResult(
            handled=False,
            intent="other",
            success=False,
            message="missing_psid_or_text",
        )

    # Hard gate: only linked PSIDs can reach the interaction agent.
    # For immediate post-signup handoff, trusted caller may provide user_id.
    resolved_user_id: Optional[int] = int(user_id) if isinstance(user_id, int) and user_id > 0 else None
    if resolved_user_id is None:
        bridge = get_genzbuzz_bridge_service()
        try:
            lookup = bridge.lookup_messenger_user(psid=clean_psid)
            if bool(lookup.get("success")):
                candidate = int((lookup.get("data") or {}).get("user_id") or lookup.get("user_id") or 0)
                if candidate > 0:
                    resolved_user_id = candidate
        except Exception as exc:
            logger.warning("Messenger PSID lookup failed", extra={"error": str(exc), "psid": clean_psid})

    if resolved_user_id is None:
        return MessengerIngressResult(
            handled=True,
            intent="unlinked_gate",
            success=True,
            message="psid_unlinked",
            reply_text="",
        )

    bridge = get_genzbuzz_bridge_service()
    onboarding_state = bridge.messenger_onboarding_state(psid=clean_psid, user_id=resolved_user_id)
    onboarding_data = onboarding_state.get("data") or onboarding_state
    onboarding_stage = str(onboarding_data.get("stage") or "").strip().lower()
    session_key = f"messenger:{clean_psid}"
    if onboarding_stage != "awaiting_frequency":
        _set_onboarding_terminal_state(session_key=session_key, state="")
        _reset_lifecycle_emit_records(session_key=session_key)
    terminal_sealed = _is_onboarding_terminal_sealed(session_key=session_key)
    onboarding_active = bool(onboarding_data.get("active")) or onboarding_stage != ""

    waiting_state = _resolve_waiting_prompt_state(bridge=bridge, user_id=resolved_user_id)
    waiting_has_active_cycle = bool(waiting_state.get("has_active_cycle"))
    waiting_can_accept_reply = bool(waiting_state.get("can_accept_reply"))
    waiting_prompt_text = str(waiting_state.get("waiting_prompt_text") or "").strip()

    policy_service = get_genzbuzz_policy_service()
    pending_waiting_draft = policy_service.get_latest_user_draft(
        draft_kind="waiting",
        user_id=resolved_user_id,
        statuses=("draft", "ready"),
    )
    has_pending_waiting_draft = bool(pending_waiting_draft.get("has_draft"))

    normalized_text = clean_text.lower().strip()
    # Meaning-first routing for frequency turns with semantic ambiguity checks.
    frequency_turn = await _classify_frequency_turn_semantic(clean_text)
    frequency_intent = str(frequency_turn.get("kind") or "").strip().lower()
    frequency_choice = str(frequency_turn.get("choice") or "").strip().lower()
    frequency_should_commit = bool(frequency_turn.get("should_commit"))

    confidence = str(frequency_turn.get("confidence") or "low").strip().lower()
    is_mixed_intent = bool(frequency_turn.get("mixed_intent"))

    # Commit only if semantic confidence is high and intent is not mixed.
    if frequency_intent == "select" and (confidence != "high" or is_mixed_intent):
        frequency_intent = "needs_clarification"
        frequency_should_commit = False

    if (
        onboarding_stage == "awaiting_frequency"
        and not terminal_sealed
        and frequency_choice not in {"weekly", "biweekly", "monthly"}
        and not frequency_should_commit
    ):
        inferred_choice = await _resolve_confirmation_choice_semantic(
            session_key=session_key,
            user_text=clean_text,
        )
        if inferred_choice is not None:
            frequency_intent = "select"
            frequency_choice = inferred_choice
            frequency_should_commit = True
            logger.info(
                "Messenger cadence inferred from confirmation context; psid=%s; inferred_choice=%s",
                clean_psid,
                inferred_choice,
            )

    if frequency_choice not in {"weekly", "biweekly", "monthly"}:
        frequency_choice = None
    if frequency_choice is None:
        frequency_should_commit = False

    logger.info(
        "Messenger frequency semantic turn classification; psid=%s; kind=%s; choice=%s; should_commit=%s; confidence=%s; mixed_intent=%s; text_preview=%s",
        clean_psid,
        frequency_intent,
        frequency_choice or "unknown",
        "true" if frequency_should_commit else "false",
        confidence,
        "true" if is_mixed_intent else "false",
        clean_text[:120].replace("\n", " "),
    )

    is_new_friend_request = _looks_like_new_friend_request(normalized_text)

    lifecycle_intent: Optional[str] = None
    lifecycle_success: Optional[bool] = None
    lifecycle_message: Optional[str] = None
    lifecycle_context: Optional[Dict[str, Any]] = None

    if is_signup_handoff or is_new_friend_request:
        new_friend_result = bridge.messenger_new_friend(
            psid=clean_psid,
            user_id=resolved_user_id,
            state_only=True,
        )
        lifecycle_intent = "new_friend"
        lifecycle_success = bool(new_friend_result.get("success") or new_friend_result.get("ok"))
        if not lifecycle_success:
            bridge_http_status = int(new_friend_result.get("http_status") or 0)
            bridge_success = bool(new_friend_result.get("success"))
            bridge_ok = bool(new_friend_result.get("ok"))
            bridge_message = str(new_friend_result.get("message") or "")
            bridge_error = str(new_friend_result.get("error") or "")
            bridge_has_raw = "raw" in new_friend_result
            bridge_raw_preview = ""
            if bridge_has_raw:
                bridge_raw_preview = str(new_friend_result.get("raw") or "")
                bridge_raw_preview = bridge_raw_preview.replace("\n", " ").replace("\r", " ").strip()[:240]
            logger.warning(
                "Messenger lifecycle new_friend bridge failed; psid=%s; user_id=%s; http_status=%s; success=%s; ok=%s; bridge_message=%s; bridge_error=%s; has_raw=%s; raw_preview=%s",
                clean_psid,
                resolved_user_id,
                bridge_http_status,
                bridge_success,
                bridge_ok,
                bridge_message,
                bridge_error,
                bridge_has_raw,
                bridge_raw_preview,
                extra={
                    "psid": clean_psid,
                    "user_id": resolved_user_id,
                    "http_status": bridge_http_status,
                    "success": bridge_success,
                    "ok": bridge_ok,
                    "bridge_message": bridge_message,
                    "bridge_error": bridge_error,
                    "has_raw": bridge_has_raw,
                    "raw_preview": bridge_raw_preview,
                },
            )
        lifecycle_message = "new_friend_state_started" if lifecycle_success else str(
            new_friend_result.get("message") or "new_friend_failed"
        )
        lifecycle_context = {
            "event": "new_friend",
            "signup_handoff": is_signup_handoff,
            "success": lifecycle_success,
            "state": {
                "stage": str((new_friend_result.get("data") or {}).get("stage") or new_friend_result.get("stage") or ""),
                "theme": str((new_friend_result.get("data") or {}).get("theme") or new_friend_result.get("theme") or ""),
            },
            "error": "" if lifecycle_success else lifecycle_message,
        }

    if frequency_intent == "select" and frequency_choice is not None and frequency_should_commit:
        frequency_result = bridge.messenger_submit_frequency(
            psid=clean_psid,
            user_id=resolved_user_id,
            text=frequency_choice,
            state_only=True,
        )
        frequency_success = bool(frequency_result.get("success") or frequency_result.get("ok"))
        lifecycle_intent = "frequency"
        lifecycle_success = frequency_success
        lifecycle_message = "frequency_saved" if frequency_success else str(
            frequency_result.get("message") or "frequency_failed"
        )
        lifecycle_context = {
            "event": "frequency",
            "normalized_choice": frequency_choice,
            "success": frequency_success,
            "result": {
                "has_active_cycle": bool(
                    (frequency_result.get("data") or {}).get("has_active_cycle")
                    or frequency_result.get("has_active_cycle")
                ),
                "invite_link": str(
                    (frequency_result.get("data") or {}).get("invite_link")
                    or frequency_result.get("invite_link")
                    or ""
                ).strip(),
                "waiting_prompt_text": _sanitize_waiting_prompt_text(
                    (frequency_result.get("data") or {}).get("waiting_prompt_text")
                    or frequency_result.get("waiting_prompt_text")
                    or ""
                ),
            },
            "error": "" if frequency_success else lifecycle_message,
        }
    elif frequency_intent in {"ask_recommendation", "ask_comparison", "ask_outcome"}:
        lifecycle_intent = "frequency"
        lifecycle_success = True
        lifecycle_message = "frequency_guidance"
        lifecycle_context = {
            "event": "frequency_guidance",
            "guidance_kind": frequency_intent,
            "normalized_choice": "",
            "success": True,
            "result": {
                "has_active_cycle": False,
                "invite_link": "",
                "waiting_prompt_text": "",
            },
            "error": "",
        }
    elif frequency_intent == "needs_clarification" and onboarding_stage == "awaiting_frequency":
        lifecycle_intent = "frequency"
        lifecycle_success = False
        lifecycle_message = "frequency_needs_clarification"
        lifecycle_context = {
            "event": "frequency",
            "normalized_choice": "",
            "success": False,
            "result": {
                "has_active_cycle": False,
                "invite_link": "",
                "waiting_prompt_text": "",
            },
            "error": "unrecognized_frequency",
        }

    # If onboarding is actively awaiting frequency, keep the user anchored on
    # cadence selection even when they steer into casual/off-topic turns.
    if lifecycle_context is None:
        unresolved_frequency_turn = (
            frequency_intent in {"none", "select"}
            and frequency_choice is None
            and not frequency_should_commit
        )

        if onboarding_active and onboarding_stage == "awaiting_frequency" and not terminal_sealed and unresolved_frequency_turn:
            logger.info(
                "Messenger onboarding anchor applied; psid=%s; stage=%s; kind=%s",
                clean_psid,
                onboarding_stage,
                frequency_intent or "none",
            )
            lifecycle_intent = "frequency"
            lifecycle_success = False
            lifecycle_message = "frequency_anchor_required"
            lifecycle_context = {
                "event": "frequency_anchor_offtopic",
                "normalized_choice": "",
                "success": False,
                "result": {
                    "has_active_cycle": False,
                    "invite_link": "",
                    "waiting_prompt_text": "",
                },
                "error": "pending_frequency_selection",
            }

    if lifecycle_context is None:
        waiting_flow_eligible = (
            onboarding_stage != "awaiting_frequency"
            and waiting_has_active_cycle
            and waiting_can_accept_reply
        )

        if waiting_flow_eligible:
            waiting_topic = _looks_like_waiting_prompt_topic(normalized_text)
            send_request = _looks_like_send_request(normalized_text)
            typed_gate = await _classify_waiting_intent_gate_semantic(clean_text)
            typed_intent = str(typed_gate.get("intent") or "OTHER")
            typed_confidence = str(typed_gate.get("confidence") or "low")
            pending_turn_kind = typed_intent.lower()
            acknowledgement_only = typed_intent == "ACK"
            execution_outcome = "pending"
            draft_state = "IDLE"
            pending_draft_id = ""
            pending_draft_status = str(pending_waiting_draft.get("status") or "draft")
            pending_draft_age_seconds = int(pending_waiting_draft.get("age_seconds") or 0)
            if has_pending_waiting_draft:
                pending_details = _get_waiting_draft_details(
                    policy_service=policy_service,
                    pending_waiting_draft=pending_waiting_draft,
                )
                pending_draft_id = str(pending_details.get("draft_id") or "")
                draft_state = str(pending_details.get("journal_state") or "PENDING_APPROVAL")

            if has_pending_waiting_draft and typed_intent == "PIVOT" and typed_confidence == "high":
                _set_waiting_draft_state(
                    policy_service=policy_service,
                    draft_id=pending_draft_id,
                    journal_state="PAUSED",
                    updates={
                        "paused_reason": "secondary_intent",
                        "paused_intent_preview": clean_text[:180],
                    },
                )
                draft_state = "PAUSED"
                logger.info(
                    "Messenger waiting flow pivot parked; psid=%s; draft_state=PAUSED; routing=runtime",
                    clean_psid,
                )
                lifecycle_intent = "waiting_prompt"
                lifecycle_success = True
                lifecycle_message = "waiting_draft_paused_unsupported_pivot"
                lifecycle_context = {
                    "event": "waiting_draft_paused_unsupported_pivot",
                    "success": True,
                    "result": {
                        "has_pending_draft": True,
                        "journal_state": "PAUSED",
                        "pending_turn_kind": pending_turn_kind,
                        "intent_confidence": typed_confidence,
                        "draft_status": pending_draft_status,
                        "draft_age_seconds": pending_draft_age_seconds,
                        "waiting_prompt_text": waiting_prompt_text,
                        "execution_outcome": execution_outcome,
                    },
                    "error": "unsupported_secondary_intent",
                }

            elif has_pending_waiting_draft and draft_state == "FAILED":
                if typed_intent == "APPROVE" and typed_confidence == "high":
                    _set_waiting_draft_state(
                        policy_service=policy_service,
                        draft_id=pending_draft_id,
                        journal_state="SENDING",
                        updates={"waiting_for_input": False},
                    )
                    save_result = _execute_pending_waiting_draft_save(
                        bridge=bridge,
                        policy_service=policy_service,
                        user_id=resolved_user_id,
                        pending_waiting_draft=pending_waiting_draft,
                    )
                    save_success = bool(save_result.get("success"))
                    execution_outcome = "success" if save_success else "failure"
                    if save_success:
                        _set_waiting_draft_state(
                            policy_service=policy_service,
                            draft_id=pending_draft_id,
                            journal_state="SAVED",
                            updates={"waiting_for_input": False, "last_error": ""},
                        )
                        policy_service.mark_draft_status(draft_id=pending_draft_id, status="sent")
                    else:
                        _set_waiting_draft_state(
                            policy_service=policy_service,
                            draft_id=pending_draft_id,
                            journal_state="FAILED",
                            updates={
                                "waiting_for_input": True,
                                "wait_state": "WAITING_FOR_INPUT",
                                "last_error": str(save_result.get("error") or "waiting_save_failed"),
                            },
                        )

                    lifecycle_intent = "waiting_prompt"
                    lifecycle_success = True
                    lifecycle_message = "waiting_draft_saved" if save_success else "waiting_draft_save_failed"
                    lifecycle_context = {
                        "event": "waiting_draft_saved" if save_success else "waiting_draft_save_failed",
                        "success": save_success,
                        "result": {
                            "has_pending_draft": not save_success,
                            "journal_state": "SAVED" if save_success else "FAILED",
                            "pending_turn_kind": pending_turn_kind,
                            "intent_confidence": typed_confidence,
                            "draft_status": "sent" if save_success else pending_draft_status,
                            "draft_age_seconds": pending_draft_age_seconds,
                            "waiting_prompt_text": waiting_prompt_text,
                            "execution_outcome": execution_outcome,
                        },
                        "error": "" if save_success else str(save_result.get("error") or "waiting_save_failed"),
                    }
                elif typed_intent == "CANCEL" and typed_confidence == "high":
                    _set_waiting_draft_state(
                        policy_service=policy_service,
                        draft_id=pending_draft_id,
                        journal_state="STALE",
                        updates={"waiting_for_input": False},
                    )
                    policy_service.mark_draft_status(draft_id=pending_draft_id, status="archived")
                    lifecycle_intent = "waiting_prompt"
                    lifecycle_success = True
                    lifecycle_message = "waiting_draft_cancelled"
                    lifecycle_context = {
                        "event": "waiting_draft_cancelled",
                        "success": True,
                        "result": {
                            "has_pending_draft": False,
                            "journal_state": "STALE",
                            "pending_turn_kind": pending_turn_kind,
                            "intent_confidence": typed_confidence,
                            "execution_outcome": execution_outcome,
                        },
                        "error": "",
                    }
                else:
                    return MessengerIngressResult(
                        handled=True,
                        intent="waiting_prompt",
                        success=True,
                        message="waiting_for_input",
                        user_id=resolved_user_id,
                        reply_text="",
                    )

            elif has_pending_waiting_draft and draft_state == "PAUSED":
                if typed_intent == "PIVOT" and typed_confidence == "high":
                    lifecycle_intent = "waiting_prompt"
                    lifecycle_success = True
                    lifecycle_message = "waiting_draft_paused_unsupported_pivot"
                    lifecycle_context = {
                        "event": "waiting_draft_paused_unsupported_pivot",
                        "success": True,
                        "result": {
                            "has_pending_draft": True,
                            "journal_state": "PAUSED",
                            "pending_turn_kind": pending_turn_kind,
                            "intent_confidence": typed_confidence,
                            "draft_status": pending_draft_status,
                            "draft_age_seconds": pending_draft_age_seconds,
                            "waiting_prompt_text": waiting_prompt_text,
                            "execution_outcome": execution_outcome,
                        },
                        "error": "unsupported_secondary_intent",
                    }
                else:
                    _set_waiting_draft_state(
                        policy_service=policy_service,
                        draft_id=pending_draft_id,
                        journal_state="PENDING_APPROVAL",
                        updates={"paused_reason": "", "paused_intent_preview": ""},
                    )
                    lifecycle_intent = "waiting_prompt"
                    lifecycle_success = True
                    lifecycle_message = "waiting_draft_resumed"
                    lifecycle_context = {
                        "event": "waiting_draft_resumed",
                        "success": True,
                        "result": {
                            "has_pending_draft": True,
                            "journal_state": "PENDING_APPROVAL",
                            "pending_turn_kind": pending_turn_kind,
                            "intent_confidence": typed_confidence,
                            "draft_status": pending_draft_status,
                            "draft_age_seconds": pending_draft_age_seconds,
                            "waiting_prompt_text": waiting_prompt_text,
                            "execution_outcome": execution_outcome,
                        },
                        "error": "",
                    }

            elif has_pending_waiting_draft and not acknowledgement_only:
                if typed_intent == "APPROVE" and typed_confidence == "high":
                    _set_waiting_draft_state(
                        policy_service=policy_service,
                        draft_id=pending_draft_id,
                        journal_state="SENDING",
                        updates={"waiting_for_input": False},
                    )
                    save_result = _execute_pending_waiting_draft_save(
                        bridge=bridge,
                        policy_service=policy_service,
                        user_id=resolved_user_id,
                        pending_waiting_draft=pending_waiting_draft,
                    )
                    save_success = bool(save_result.get("success"))
                    execution_outcome = "success" if save_success else "failure"
                    if save_success:
                        _set_waiting_draft_state(
                            policy_service=policy_service,
                            draft_id=pending_draft_id,
                            journal_state="SAVED",
                            updates={"waiting_for_input": False, "last_error": ""},
                        )
                        policy_service.mark_draft_status(draft_id=pending_draft_id, status="sent")
                    else:
                        _set_waiting_draft_state(
                            policy_service=policy_service,
                            draft_id=pending_draft_id,
                            journal_state="FAILED",
                            updates={
                                "waiting_for_input": True,
                                "wait_state": "WAITING_FOR_INPUT",
                                "last_error": str(save_result.get("error") or "waiting_save_failed"),
                            },
                        )

                    lifecycle_intent = "waiting_prompt"
                    lifecycle_success = True
                    lifecycle_message = "waiting_draft_saved" if save_success else "waiting_draft_save_failed"
                    lifecycle_context = {
                        "event": "waiting_draft_saved" if save_success else "waiting_draft_save_failed",
                        "success": save_success,
                        "result": {
                            "has_pending_draft": not save_success,
                            "journal_state": "SAVED" if save_success else "FAILED",
                            "pending_turn_kind": pending_turn_kind,
                            "intent_confidence": typed_confidence,
                            "draft_status": "sent" if save_success else pending_draft_status,
                            "draft_age_seconds": pending_draft_age_seconds,
                            "waiting_prompt_text": waiting_prompt_text,
                            "execution_outcome": execution_outcome,
                        },
                        "error": "" if save_success else str(save_result.get("error") or "waiting_save_failed"),
                    }
                elif typed_intent == "CANCEL" and typed_confidence == "high":
                    _set_waiting_draft_state(
                        policy_service=policy_service,
                        draft_id=pending_draft_id,
                        journal_state="STALE",
                        updates={"waiting_for_input": False},
                    )
                    policy_service.mark_draft_status(draft_id=pending_draft_id, status="archived")
                    lifecycle_intent = "waiting_prompt"
                    lifecycle_success = True
                    lifecycle_message = "waiting_draft_cancelled"
                    lifecycle_context = {
                        "event": "waiting_draft_cancelled",
                        "success": True,
                        "result": {
                            "has_pending_draft": False,
                            "journal_state": "STALE",
                            "pending_turn_kind": pending_turn_kind,
                            "intent_confidence": typed_confidence,
                            "execution_outcome": execution_outcome,
                        },
                        "error": "",
                    }
                else:
                    _set_waiting_draft_state(
                        policy_service=policy_service,
                        draft_id=pending_draft_id,
                        journal_state="PENDING_APPROVAL",
                        updates={"waiting_for_input": False},
                    )
                    lifecycle_intent = "waiting_prompt"
                    lifecycle_success = True
                    lifecycle_message = "waiting_draft_reverify_required"
                    lifecycle_context = {
                        "event": "waiting_draft_reverify",
                        "success": True,
                        "result": {
                            "has_pending_draft": True,
                            "journal_state": "PENDING_APPROVAL",
                            "pending_turn_kind": pending_turn_kind,
                            "intent_confidence": typed_confidence,
                            "draft_status": pending_draft_status,
                            "draft_age_seconds": pending_draft_age_seconds,
                            "waiting_prompt_text": waiting_prompt_text,
                            "execution_outcome": execution_outcome,
                        },
                        "error": "pending_draft_requires_reconfirm",
                    }
            elif has_pending_waiting_draft and acknowledgement_only:
                return MessengerIngressResult(
                    handled=True,
                    intent="waiting_prompt",
                    success=True,
                    message="waiting_ack_hold",
                    user_id=resolved_user_id,
                    reply_text="",
                )
            elif not has_pending_waiting_draft and not acknowledgement_only:
                prompt_input_gate = await _classify_waiting_prompt_input_semantic(
                    user_text=clean_text,
                    waiting_prompt_text=waiting_prompt_text,
                )
                prompt_input_kind = str(prompt_input_gate.get("kind") or "OTHER").upper()
                prompt_input_confidence = str(prompt_input_gate.get("confidence") or "low").lower()
                prompt_input_valid = bool(prompt_input_gate.get("valid_answer"))
                session_key = f"messenger:{clean_psid}"
                answer_accepted = prompt_input_kind == "ANSWER" and prompt_input_confidence != "low"

                if _is_waiting_prompt_offramp_locked(session_key=session_key) and not answer_accepted:
                    return MessengerIngressResult(
                        handled=True,
                        intent="waiting_prompt",
                        success=True,
                        message="waiting_for_input",
                        user_id=resolved_user_id,
                        reply_text="",
                    )

                if prompt_input_kind != "ANSWER" or prompt_input_confidence == "low":
                    clarify_count = _increment_waiting_prompt_clarify_count(session_key=session_key)
                    if clarify_count > 2:
                        lifecycle_intent = "waiting_prompt"
                        lifecycle_success = True
                        lifecycle_message = "waiting_prompt_offramp_wait_input"
                        lifecycle_context = {
                            "event": "waiting_prompt_offramp_wait_input",
                            "success": True,
                            "result": {
                                "has_pending_draft": False,
                                "journal_state": "WAITING_FOR_INPUT",
                                "clarify_loop_count": clarify_count,
                                "waiting_prompt_text": waiting_prompt_text,
                                "execution_outcome": execution_outcome,
                            },
                            "error": "clarify_loop_limit_reached",
                        }
                        _set_waiting_prompt_offramp_lock(session_key=session_key, locked=True)
                    else:
                        clarify_event = "waiting_prompt_reprompt"
                        if prompt_input_kind == "CLARIFY_EXAMPLE":
                            clarify_event = "waiting_prompt_clarify_example"
                        elif prompt_input_kind == "CLARIFY":
                            clarify_event = "waiting_prompt_clarify"

                        lifecycle_intent = "waiting_prompt"
                        lifecycle_success = True
                        lifecycle_message = clarify_event
                        lifecycle_context = {
                            "event": clarify_event,
                            "success": True,
                            "result": {
                                "has_pending_draft": False,
                                "journal_state": "PROMPTING",
                                "clarify_loop_count": clarify_count,
                                "prompt_input_kind": prompt_input_kind,
                                "prompt_input_confidence": prompt_input_confidence,
                                "prompt_input_valid": prompt_input_valid,
                                "waiting_prompt_text": waiting_prompt_text,
                                "execution_outcome": execution_outcome,
                            },
                            "error": "prompt_answer_not_valid",
                        }
                else:
                    created_waiting_draft = _create_waiting_text_draft(
                        user_id=resolved_user_id,
                        text_answer=clean_text,
                        waiting_prompt_text=waiting_prompt_text,
                    )
                    _reset_waiting_prompt_clarify_count(session_key=session_key)
                    _set_waiting_prompt_offramp_lock(session_key=session_key, locked=False)

                    if created_waiting_draft.get("draft_id"):
                        _set_waiting_draft_state(
                            policy_service=policy_service,
                            draft_id=str(created_waiting_draft.get("draft_id") or ""),
                            journal_state="PENDING_APPROVAL",
                            updates={"waiting_for_input": False, "last_error": ""},
                        )
                        lifecycle_intent = "waiting_prompt"
                        lifecycle_success = True
                        lifecycle_message = "waiting_draft_created"
                        lifecycle_context = {
                            "event": "waiting_draft_created",
                            "success": True,
                            "result": {
                                "has_pending_draft": True,
                                "journal_state": "PENDING_APPROVAL",
                                "draft_id": str(created_waiting_draft.get("draft_id") or ""),
                                "draft_status": str(created_waiting_draft.get("status") or "draft"),
                                "draft_preview": str(created_waiting_draft.get("draft_preview") or "").strip(),
                                "waiting_prompt_text": waiting_prompt_text,
                                "execution_outcome": execution_outcome,
                            },
                            "error": "",
                        }
                    else:
                        lifecycle_intent = "waiting_prompt"
                        lifecycle_success = True
                        lifecycle_message = "waiting_unresolved_no_draft"
                        lifecycle_context = {
                            "event": "waiting_unresolved_no_draft",
                            "success": True,
                            "result": {
                                "has_pending_draft": False,
                                "journal_state": "PROMPTING",
                                "waiting_prompt_text": waiting_prompt_text,
                                "execution_outcome": execution_outcome,
                            },
                            "error": "",
                        }
            elif not has_pending_waiting_draft:
                lifecycle_intent = "waiting_prompt"
                lifecycle_success = True
                lifecycle_message = "waiting_unresolved_no_draft"
                lifecycle_context = {
                    "event": "waiting_unresolved_no_draft",
                    "success": True,
                    "result": {
                        "has_pending_draft": False,
                        "journal_state": "PROMPTING",
                        "waiting_prompt_text": waiting_prompt_text,
                        "execution_outcome": execution_outcome,
                    },
                    "error": "",
                }

            logger.info(
                "Messenger waiting flow routing; psid=%s; waiting_topic=%s; send_request=%s; typed_intent=%s; intent_confidence=%s; draft_state=%s; has_pending_draft=%s; lifecycle_selected=%s; execution_outcome=%s",
                clean_psid,
                "true" if waiting_topic else "false",
                "true" if send_request else "false",
                typed_intent,
                typed_confidence,
                draft_state,
                "true" if has_pending_waiting_draft else "false",
                "true" if lifecycle_context is not None else "false",
                execution_outcome,
            )

    if lifecycle_context is not None and lifecycle_intent is not None and lifecycle_success is not None and lifecycle_message is not None:
        logger.info(
            "Messenger lifecycle direct reply path; psid=%s; intent=%s; success=%s; message=%s",
            clean_psid,
            lifecycle_intent,
            "true" if lifecycle_success else "false",
            lifecycle_message,
        )
        correlation_id = ""
        if _is_terminal_frequency_confirmation_event(
            lifecycle_intent=lifecycle_intent,
            lifecycle_success=lifecycle_success,
            lifecycle_context=lifecycle_context,
        ):
            correlation_id = _build_lifecycle_correlation_id(
                session_key=session_key,
                lifecycle_intent=lifecycle_intent,
                lifecycle_context=lifecycle_context,
            )
            if _is_lifecycle_emit_already_sent(session_key=session_key, correlation_id=correlation_id):
                logger.info(
                    "Messenger lifecycle duplicate emit suppressed; psid=%s; correlation_id=%s",
                    clean_psid,
                    correlation_id,
                )
                return MessengerIngressResult(
                    handled=True,
                    intent=lifecycle_intent,
                    success=True,
                    message="frequency_terminal_emit_duplicate_suppressed",
                    user_id=resolved_user_id,
                    reply_text="",
                )
            _mark_lifecycle_emit_sent(session_key=session_key, correlation_id=correlation_id)
            lifecycle_context["correlation_id"] = correlation_id

        sanitized_lifecycle_context = _sanitize_lifecycle_context_for_generation(lifecycle_context)
        lifecycle_reply = _build_waiting_prompt_direct_reply(
            user_text=clean_text,
            lifecycle_context=sanitized_lifecycle_context,
        )
        if lifecycle_reply == "":
            lifecycle_reply = await _generate_lifecycle_reply(
                psid=clean_psid,
                user_id=resolved_user_id,
                user_text=clean_text,
                lifecycle_context=sanitized_lifecycle_context,
            )
        lifecycle_reply = _sanitize_lifecycle_reply_text(lifecycle_reply)
        if correlation_id:
            _set_onboarding_terminal_state(session_key=session_key, state="SENT")
        _record_lifecycle_turn(session_key=f"messenger:{clean_psid}", user_text=clean_text, reply_text=lifecycle_reply)
        return MessengerIngressResult(
            handled=True,
            intent=lifecycle_intent,
            success=lifecycle_success,
            message=lifecycle_message,
            user_id=resolved_user_id,
            reply_text=lifecycle_reply,
        )

    try:
        runtime = InteractionAgentRuntime(session_key=f"messenger:{clean_psid}")
    except Exception as exc:
        logger.error("Messenger ingress runtime init failed", extra={"error": str(exc)})
        return MessengerIngressResult(
            handled=False,
            intent="agent_runtime",
            success=False,
            message="runtime_init_failed",
        )

    delivery_token = set_delivery_context(
        channel="messenger",
        recipient_id=clean_psid,
        mode="collect",
    )
    outbound_token = begin_outbound_collection()
    try:
        # Keep conversation history clean: only store/send the user's actual message text.
        result = await runtime.execute(user_message=clean_text)
        outbound_messages = get_outbound_messages()
    except Exception as exc:
        logger.error("Messenger ingress runtime crashed", extra={"error": str(exc)})
        return MessengerIngressResult(
            handled=False,
            intent="agent_runtime",
            success=False,
            message="runtime_execution_crashed",
        )
    finally:
        reset_outbound_collection(outbound_token)
        reset_delivery_context(delivery_token)

    if not result.success:
        logger.error("Messenger ingress runtime execution failed", extra={"error": result.error or "unknown"})
        return MessengerIngressResult(
            handled=False,
            intent="agent_runtime",
            success=False,
            message=result.error or "runtime_execution_failed",
        )

    reply_text = ""
    if outbound_messages:
        reply_text = (outbound_messages[-1].text or "").strip()
    if not reply_text:
        reply_text = (result.response or "").strip()
    if reply_text == "":
        reply_text = "I hit a temporary issue. Send your message again and I'll continue right away."

    logger.info(
        "Messenger routing fallback to interaction agent; psid=%s; frequency_kind=%s; choice=%s; should_commit=%s",
        clean_psid,
        frequency_intent,
        frequency_choice or "unknown",
        "true" if frequency_should_commit else "false",
    )

    return MessengerIngressResult(
        handled=True,
        intent="agent_runtime",
        success=True,
        message="response_ready",
        reply_text=reply_text,
    )


__all__ = ["MessengerIngressResult", "route_messenger_ingress"]


async def _classify_frequency_turn_semantic(text: str) -> Dict[str, Any]:
    candidate = (text or "").strip()
    if candidate == "" or len(candidate) > 180:
        return {
            "kind": "none",
            "choice": "unknown",
            "should_commit": False,
            "confidence": "low",
            "mixed_intent": False,
        }

    settings = get_settings()
    api_key = (settings.openrouter_api_key or "").strip()
    if not api_key:
        return {
            "kind": "none",
            "choice": "unknown",
            "should_commit": False,
            "confidence": "low",
            "mixed_intent": False,
        }

    system_prompt = (
        "Classify a Messenger onboarding frequency turn. "
        "Return strict JSON only with keys kind, choice, should_commit, confidence, and mixed_intent. "
        "kind must be one of: select, ask_recommendation, ask_comparison, ask_outcome, needs_clarification, none. "
        "choice must be weekly, biweekly, monthly, or unknown. "
        "confidence must be one of: high, medium, low. "
        "mixed_intent must be true when the user both proposes and retracts/hedges, or otherwise expresses conflicting direction in one turn. "
        "should_commit must be true only when the user clearly commits to a cadence now; otherwise false. "
        "Do semantic interpretation, not literal keyword matching. "
        "Map colloquial timing to supported cadences (e.g., every 2 weeks -> biweekly, every month -> monthly). "
        "If the user expresses hesitation, contradiction, or reversal in the same turn, set mixed_intent=true and should_commit=false. "
        "Infer user intent from meaning and discourse purpose, not keyword matching."
    )
    user_prompt = (
        f"User text: {candidate}\n"
        "Interpretation guidance:\n"
        "- select: the user is committing to a cadence now.\n"
        "- ask_recommendation: the user is asking which cadence you suggest.\n"
        "- ask_comparison: the user is asking to compare cadence options.\n"
        "- ask_outcome: the user is asking consequences of a cadence choice.\n"
        "- needs_clarification: the user is discussing cadence but has not picked one yet.\n"
        "- If the message is unrelated to scheduling or cadence, use kind=none and should_commit=false.\n"
        "- If the message includes both a tentative cadence and an immediate hedge/reversal, use kind=needs_clarification, mixed_intent=true, should_commit=false.\n"
        "- Examples:\n"
        "  1) 'let's do weekly' => kind=select, choice=weekly, should_commit=true, confidence=high, mixed_intent=false\n"
        "  2) 'every 2 weeks works' => kind=select, choice=biweekly, should_commit=true, confidence=high, mixed_intent=false\n"
        "  3) 'maybe every 2 weeks... nah not sure' => kind=needs_clarification, choice=biweekly, should_commit=false, confidence=high, mixed_intent=true\n"
        "  4) 'which is better weekly or monthly?' => kind=ask_comparison, choice=unknown, should_commit=false, confidence=high, mixed_intent=false\n"
        "- none: no clear frequency intent.\n"
        "Return JSON only."
    )

    try:
        response = await request_chat_completion(
            model=settings.interaction_agent_model,
            messages=[{"role": "user", "content": user_prompt}],
            system=system_prompt,
            api_key=api_key,
            tools=None,
            temperature=0.0,
        )
        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = str(message.get("content") or "").strip()
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            return {
                "kind": "none",
                "choice": "unknown",
                "should_commit": False,
                "confidence": "low",
                "mixed_intent": False,
            }

        kind = str(parsed.get("kind") or "none").strip().lower()
        cadence = str(parsed.get("choice") or "unknown").strip().lower()
        should_commit = bool(parsed.get("should_commit"))
        confidence = str(parsed.get("confidence") or "low").strip().lower()
        mixed_intent = bool(parsed.get("mixed_intent"))
        if kind not in {"select", "ask_recommendation", "ask_comparison", "ask_outcome", "needs_clarification", "none"}:
            kind = "none"
        if cadence not in {"weekly", "biweekly", "monthly", "unknown"}:
            cadence = "unknown"
        if confidence not in {"high", "medium", "low"}:
            confidence = "low"
        if kind != "select":
            should_commit = False
        if cadence == "unknown":
            should_commit = False
        if mixed_intent:
            should_commit = False
        return {
            "kind": kind,
            "choice": cadence,
            "should_commit": should_commit,
            "confidence": confidence,
            "mixed_intent": mixed_intent,
        }
    except Exception:
        return {
            "kind": "none",
            "choice": "unknown",
            "should_commit": False,
            "confidence": "low",
            "mixed_intent": False,
        }


def _looks_like_new_friend_request(text: str) -> bool:
    candidate = (text or "").lower().strip()
    return bool(
        re.search(
            r"\b(new\s+friend|add\s+(a\s+)?friend|start\s+(a\s+)?new\s+friend|create\s+(a\s+)?new\s+friend|find\s+me\s+(a\s+)?new\s+friend)\b",
            candidate,
        )
    )


async def _resolve_confirmation_choice_semantic(*, session_key: str, user_text: str) -> Optional[str]:
    try:
        log = get_conversation_log(session_key=session_key)
        entries = list(log.iter_entries())
    except Exception:
        return None

    last_reply = ""
    for tag, _timestamp, payload in reversed(entries):
        if tag == "genzbuzz_reply":
            last_reply = str(payload or "").strip()
            break

    if last_reply == "":
        return None

    settings = get_settings()
    api_key = (settings.openrouter_api_key or "").strip()
    if not api_key:
        return None

    system_prompt = (
        "You resolve yes/no confirmations for Messenger onboarding cadence. "
        "Return strict JSON only with keys is_confirmation, choice, confidence. "
        "is_confirmation must be true only if the user message affirms the immediate prior bot message. "
        "choice must be weekly, biweekly, monthly, or unknown. "
        "confidence must be high, medium, or low. "
        "Infer meaning from both turns, not keyword matching. "
        "If the prior bot message presents multiple cadence options without confirming one, choice must be unknown. "
        "If the prior bot message asks to confirm a single cadence and user affirms, return that cadence with high confidence."
    )
    user_prompt = (
        f"Previous bot message: {last_reply}\n"
        f"Current user message: {user_text}\n"
        "Return JSON only."
    )

    try:
        response = await request_chat_completion(
            model=settings.interaction_agent_model,
            messages=[{"role": "user", "content": user_prompt}],
            system=system_prompt,
            api_key=api_key,
            tools=None,
            temperature=0.0,
        )
        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = str(message.get("content") or "").strip()
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            return None

        is_confirmation = bool(parsed.get("is_confirmation"))
        cadence = str(parsed.get("choice") or "unknown").strip().lower()
        confidence = str(parsed.get("confidence") or "low").strip().lower()
        if cadence not in {"weekly", "biweekly", "monthly"}:
            return None
        if confidence != "high" or not is_confirmation:
            return None
        return cadence
    except Exception:
        return None


async def _generate_lifecycle_reply(
    *,
    psid: str,
    user_id: int,
    user_text: str,
    lifecycle_context: Dict[str, Any],
) -> str:
    settings = get_settings()
    api_key = (settings.openrouter_api_key or "").strip()
    if not api_key:
        return "I hit a temporary issue. Send your message again and I'll continue right away."

    system_prompt = (
        "You're GenZbuzz and you write replies for GenZbuzz onboarding lifecycle events. "
        "GenZbuzz helps adults deepen existing friendships through meaningful check-ins, not generic app updates. "
        "Position GenZbuzz as a facilitator for the user's real-world friendships, not as a friend itself. "
        "Never imply GenZbuzz will become the user's friend, replace human friendship, or find/match new friends. "
        "Never describe the relationship as 'our friendship' between GenZbuzz and the user. "
        "Use wording like 'your friendship with your friend' or 'your friendships'. "
        "Keep replies concise, lowercase, natural, and user-facing. "
        "Avoid casual reassurance fillers, chirpy exclamatory openers, and restart-loop phrasing. "
        "Never mention tools, APIs, system internals, execution agents, or hidden context. "
        "When asking for cadence, supported options are exactly weekly, biweekly, or monthly. "
        "When discussing frequency, frame it as check-ins/messages between friends and relationship-building touchpoints. "
        "For new_friend onboarding, clearly state that check-ins are between the user and their friend. "
        "If lifecycle_context.event is frequency_guidance, answer the user's recommendation/comparison/outcome question directly, then ask them to choose weekly, biweekly, or monthly. "
        "For recommendation questions, default recommendation is weekly and explain briefly why. "
        "Do not describe onboarding as optional. "
        "Do not claim frequency was saved unless lifecycle_context.event is frequency and success is true. "
        "If lifecycle_context.event is frequency and success is true, confirm the saved cadence and do not ask the user to choose cadence options again in that same message. "
        "If lifecycle_context.event is frequency and success is false, ask a natural clarifying question and do not claim frequency was saved. "
        "If lifecycle_context.event is frequency_anchor_offtopic, briefly acknowledge the user's message, then steer back to the required cadence choice with exactly weekly, biweekly, or monthly. "
        "If lifecycle_context.event is waiting_unresolved_no_draft, reply naturally to the user first and then bring up the unresolved waiting prompt in one gentle sentence. "
        "If lifecycle_context.event is waiting_prompt_clarify, answer the user's meta-question briefly and re-emit the original waiting prompt text in the same message. "
        "If lifecycle_context.event is waiting_prompt_clarify_example, provide one or two concrete sample answers for the active prompt, then re-emit the prompt. "
        "If lifecycle_context.event is waiting_prompt_reprompt, ask for a concrete personal proud moment answer and re-emit the original waiting prompt text in the same message. "
        "If lifecycle_context.event is waiting_prompt_offramp_wait_input, provide a short off-ramp and invite the user to send text or voice note when ready. "
        "For waiting prompt events, do not describe the action as sending to a friend. "
        "For waiting prompt events, describe the action as saving to the user's Memento Album journal page. "
        "If lifecycle_context.event is waiting_draft_created, confirm you captured the draft and ask for explicit approval before saving it to the user's Memento Album. "
        "If lifecycle_context.event is waiting_draft_reverify, ask the user to reconfirm or revise the draft before any send action. "
        "If lifecycle_context.event is waiting_draft_resumed, confirm the journal draft is active again and ask whether to save or revise it. "
        "If lifecycle_context.event is waiting_draft_paused_unsupported_pivot, explain that this side-task is not supported yet, keep the draft paused, and ask whether to resume journal save or cancel. "
        "If lifecycle_context.event is waiting_draft_saved, confirm it was saved to the user's Memento Album and offer adding a photo. "
        "If lifecycle_context.event is waiting_draft_save_failed, state the concrete failure reason and offer only two options: retry or cancel. "
        "If lifecycle_context.event is waiting_draft_cancelled, confirm the draft was cancelled and will not be saved. "
        "Use lifecycle_context.result values exactly as provided. "
        "If invite_link is empty, do not invent or imply a link. "
        "If waiting_prompt_text is present, include it directly in your reply."
    )
    user_prompt = (
        "Channel: Messenger\n"
        f"Sender PSID: {psid}\n"
        f"Linked User ID: {user_id}\n"
        f"User message: {user_text}\n"
        f"Lifecycle context: {lifecycle_context}\n"
        "Write one natural reply message only like a lowercase text message from a friend."
    )

    try:
        response = await request_chat_completion(
            model=settings.interaction_agent_model,
            messages=[{"role": "user", "content": user_prompt}],
            system=system_prompt,
            api_key=api_key,
            tools=None,
            temperature=0.3,
        )
        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = str(message.get("content") or "").strip()
        if len(content) >= 2 and ((content[0] == '"' and content[-1] == '"') or (content[0] == "'" and content[-1] == "'")):
            content = content[1:-1].strip()
        if content:
            return content
    except OpenRouterError as exc:
        logger.warning("Lifecycle reply generation failed", extra={"error": str(exc)})
    except Exception as exc:
        logger.warning("Lifecycle reply generation crashed", extra={"error": str(exc)})

    return "I hit a temporary issue. Send your message again and I'll continue right away."


def _sanitize_waiting_prompt_text(raw_text: str) -> str:
    text = (raw_text or "").strip()
    if text == "":
        return ""

    expected_prefix = "Personal Journal Prompt:"
    expected_suffix = "Journal your response with text or a voice note"

    if len(text) > 800:
        logger.warning(
            "Dropping waiting prompt text from frequency success reply: too long",
            extra={"length": len(text), "preview": text[:160]},
        )
        return ""

    # Accept only the expected waiting prompt shape returned by WordPress helper.
    if not text.startswith(expected_prefix) or expected_suffix not in text:
        logger.warning(
            "Dropping waiting prompt text from frequency success reply: unexpected format",
            extra={"preview": text[:200]},
        )
        return ""

    return text


def _resolve_waiting_prompt_state(*, bridge: Any, user_id: int) -> Dict[str, Any]:
    fallback = {
        "has_active_cycle": False,
        "can_accept_reply": False,
        "current_prompt_text": "",
        "waiting_prompt_text": "",
    }

    try:
        result = bridge.get_waiting_prompt_context(user_id=user_id)
    except Exception as exc:
        logger.warning(
            "Messenger waiting prompt context lookup failed",
            extra={"error": str(exc), "user_id": user_id},
        )
        return fallback

    payload = result.get("data") if isinstance(result.get("data"), dict) else result
    if not isinstance(payload, dict):
        return fallback

    has_active_cycle = bool(payload.get("has_active_cycle"))
    can_accept_reply = bool(payload.get("can_accept_reply"))
    current_prompt_text = str(payload.get("current_prompt_text") or "").strip()

    waiting_prompt_text = ""
    if current_prompt_text:
        waiting_prompt_text = (
            "Personal Journal Prompt:\n"
            f"{current_prompt_text}\n\n"
            "Journal your response with text or a voice note"
        )

    return {
        "has_active_cycle": has_active_cycle,
        "can_accept_reply": can_accept_reply,
        "current_prompt_text": current_prompt_text,
        "waiting_prompt_text": waiting_prompt_text,
    }


def _looks_like_waiting_prompt_topic(text: str) -> bool:
    candidate = (text or "").strip().lower()
    if candidate == "":
        return False
    return bool(
        re.search(
            r"\b(waiting|journal|prompt|response|reply|voice\s*note|draft|answer)\b",
            candidate,
        )
    )


def _looks_like_send_request(text: str) -> bool:
    candidate = (text or "").strip().lower()
    return bool(re.search(r"\b(send|submit|post|ship|finalize|approve)\b", candidate))


def _looks_like_acknowledgement(text: str) -> bool:
    candidate = (text or "").strip().lower()
    if candidate == "":
        return False

    normalized = re.sub(r"[^a-z0-9\s]", "", candidate).strip()
    if normalized in {"thanks", "thank you", "ok", "okay", "cool", "got it", "k", "kk", "nice"}:
        return True

    if len(normalized.split()) <= 3 and normalized.startswith("thank"):
        return True

    return False


def _normalize_journal_state(raw_state: str) -> str:
    state = (raw_state or "").strip().upper()
    if state in JOURNAL_STATES:
        return state
    return "PENDING_APPROVAL"


def _get_waiting_draft_details(*, policy_service: Any, pending_waiting_draft: Dict[str, Any]) -> Dict[str, Any]:
    draft_id = str(pending_waiting_draft.get("draft_id") or "").strip()
    if draft_id == "":
        return {"draft_id": "", "journal_state": "PENDING_APPROVAL"}

    record = policy_service.get_draft(draft_id=draft_id)
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    payload = record.get("payload") if isinstance(record.get("payload"), dict) else {}
    state = _normalize_journal_state(str(metadata.get("journal_state") or "PENDING_APPROVAL"))
    return {
        "draft_id": draft_id,
        "metadata": metadata,
        "payload": payload,
        "status": str(record.get("status") or "draft"),
        "journal_state": state,
    }


def _set_waiting_draft_state(
    *,
    policy_service: Any,
    draft_id: str,
    journal_state: str,
    updates: Optional[Dict[str, Any]] = None,
) -> None:
    key = (draft_id or "").strip()
    if key == "":
        return

    record = policy_service.get_draft(draft_id=key)
    payload = record.get("payload") if isinstance(record.get("payload"), dict) else {}
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    metadata["journal_state"] = _normalize_journal_state(journal_state)
    metadata["state_updated_by"] = "messenger_ingress"
    if isinstance(updates, dict):
        for field, value in updates.items():
            metadata[str(field)] = value

    policy_service.update_draft(
        draft_id=key,
        payload=payload,
        metadata=metadata,
    )


def _looks_like_pivot_intent(text: str) -> bool:
    candidate = (text or "").strip().lower()
    if candidate == "":
        return False
    return bool(
        re.search(
            r"\b(reminder|alarm|calendar|schedule|set\s+a\s+reminder|remind\s+me|todo|to\s*do|task)\b",
            candidate,
        )
    )


async def _classify_waiting_intent_gate_semantic(text: str) -> Dict[str, str]:
    candidate = (text or "").strip()
    if candidate == "":
        return {"intent": "OTHER", "confidence": "low"}

    if _looks_like_acknowledgement(candidate):
        return {"intent": "ACK", "confidence": "high"}

    if _looks_like_pivot_intent(candidate):
        return {"intent": "PIVOT", "confidence": "high"}

    strict_kind = _classify_pending_waiting_turn(candidate)
    strict_map = {
        "approve": "APPROVE",
        "cancel": "CANCEL",
        "ack": "ACK",
        "edit_or_clarify": "OTHER",
        "other": "OTHER",
    }
    strict_intent = strict_map.get(strict_kind, "OTHER")

    if len(candidate) > 240:
        return {"intent": strict_intent, "confidence": "medium"}

    settings = get_settings()
    api_key = (settings.openrouter_api_key or "").strip()
    if not api_key:
        return {"intent": strict_intent, "confidence": "medium"}

    system_prompt = (
        "Classify intent for a Messenger journal pending-draft state machine. "
        "Return strict JSON only with keys intent and confidence. "
        "intent must be one of: APPROVE, CANCEL, PIVOT, ACK, OTHER. "
        "confidence must be one of: high, medium, low. "
        "APPROVE means explicit approval to save now. "
        "CANCEL means explicit discard/cancel instruction. "
        "PIVOT means switching to a different task or intent outside journal save. "
        "ACK means short acknowledgement only. "
        "Use semantic interpretation, not keyword-only matching."
    )
    user_prompt = f"User text: {candidate}\nReturn JSON only."

    try:
        response = await request_chat_completion(
            model=settings.interaction_agent_model,
            messages=[{"role": "user", "content": user_prompt}],
            system=system_prompt,
            api_key=api_key,
            tools=None,
            temperature=0.0,
        )
        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = str(message.get("content") or "").strip()
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            return {"intent": strict_intent, "confidence": "medium"}

        intent = str(parsed.get("intent") or "OTHER").strip().upper()
        confidence = str(parsed.get("confidence") or "low").strip().lower()
        if intent not in WAITING_TYPED_INTENTS:
            intent = strict_intent
        if confidence not in {"high", "medium", "low"}:
            confidence = "medium"

        if confidence == "low":
            return {"intent": strict_intent, "confidence": "medium"}
        return {"intent": intent, "confidence": confidence}
    except Exception:
        return {"intent": strict_intent, "confidence": "medium"}


def _looks_like_meta_prompt_question(text: str) -> bool:
    candidate = (text or "").strip().lower()
    if candidate == "":
        return False
    return bool(
        re.search(
            r"\b(how\s+do\s+i\s+answer|what\s+should\s+i\s+write|what\s+do\s+you\s+mean|how\s+should\s+i\s+respond|can\s+you\s+explain\s+the\s+prompt)\b",
            candidate,
        )
    )


def _looks_like_example_request(text: str) -> bool:
    candidate = (text or "").strip().lower()
    if candidate == "":
        return False
    return bool(
        re.search(
            r"\b(example|sample|show\s+me|like\s+what|for\s+instance|for\s+example|give\s+me\s+an\s+example|demo)\b",
            candidate,
        )
    )


def _heuristic_proud_moment_answer_validity(text: str) -> bool:
    candidate = (text or "").strip().lower()
    if candidate == "":
        return False

    if _looks_like_meta_prompt_question(candidate):
        return False

    if "?" in candidate and len(candidate.split()) <= 14:
        return False

    first_person = bool(re.search(r"\b(i|my|me|mine|we|our|us|met|had|did|went|saw)\b", candidate))
    positive_valence = bool(re.search(r"\b(proud|happy|accomplished|grateful|excited|confident|achieved|success|succeeded)\b", candidate))
    experience = bool(re.search(r"\b(when|last|first|today|yesterday|moment|job|school|project|family|friend|work|career|graduat|built|made|helped)\b", candidate))

    short_memory_pattern = bool(re.search(r"\b(i\s+met|i\s+had|i\s+did|i\s+was|i\s+got|i\s+made)\b", candidate))

    return (first_person and (positive_valence or experience)) or short_memory_pattern


async def _classify_waiting_prompt_input_semantic(*, user_text: str, waiting_prompt_text: str) -> Dict[str, Any]:
    candidate = (user_text or "").strip()
    if candidate == "":
        return {"kind": "OTHER", "confidence": "low", "valid_answer": False}

    # Priority order: CLARIFY_EXAMPLE -> CLARIFY -> ANSWER -> OTHER.
    if _looks_like_example_request(candidate):
        return {"kind": "CLARIFY_EXAMPLE", "confidence": "high", "valid_answer": False}

    if _looks_like_meta_prompt_question(candidate):
        return {"kind": "CLARIFY", "confidence": "high", "valid_answer": False}

    heuristic_valid = _heuristic_proud_moment_answer_validity(candidate)

    settings = get_settings()
    api_key = (settings.openrouter_api_key or "").strip()
    if not api_key:
        return {
            "kind": "ANSWER" if heuristic_valid else "OTHER",
            "confidence": "medium",
            "valid_answer": heuristic_valid,
        }

    system_prompt = (
        "Classify a user reply to a personal journal prompt. "
        "Return strict JSON only with keys kind, confidence, valid_answer. "
        "kind must be one of: CLARIFY_EXAMPLE, CLARIFY, ANSWER, OTHER. "
        "confidence must be one of: high, medium, low. "
        "valid_answer must be true only if the message contains a personal memory/experience that answers the prompt. "
        "Apply this strict priority order before deciding: CLARIFY_EXAMPLE, then CLARIFY, then ANSWER, then OTHER. "
        "CLARIFY_EXAMPLE means the user is asking for an example response or demonstration. "
        "For the prompt class 'what's a moment you feel proud of', valid answers describe personal experience with positive valence or accomplishment. "
        "Meta-questions like how to answer are CLARIFY and valid_answer=false. "
        "Short conversational memories are still valid ANSWER entries."
    )
    user_prompt = (
        f"Prompt: {waiting_prompt_text}\n"
        f"User reply: {candidate}\n"
        "Examples:\n"
        "- 'sample answer' => kind=CLARIFY_EXAMPLE, confidence=high, valid_answer=false\n"
        "- 'show me what to write' => kind=CLARIFY_EXAMPLE, confidence=high, valid_answer=false\n"
        "- 'how do i answer this?' => kind=CLARIFY, confidence=high, valid_answer=false\n"
        "- 'uhm, i met a long time school friend from 10 years' => kind=ANSWER, confidence=high, valid_answer=true\n"
        "- 'i finally finished my project at work' => kind=ANSWER, confidence=high, valid_answer=true\n"
        "- 'ok' => kind=OTHER, confidence=medium, valid_answer=false\n"
        "Return JSON only."
    )

    try:
        response = await request_chat_completion(
            model=settings.interaction_agent_model,
            messages=[{"role": "user", "content": user_prompt}],
            system=system_prompt,
            api_key=api_key,
            tools=None,
            temperature=0.0,
        )
        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = str(message.get("content") or "").strip()
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            return {
                "kind": "ANSWER" if heuristic_valid else "OTHER",
                "confidence": "medium",
                "valid_answer": heuristic_valid,
            }

        kind = str(parsed.get("kind") or "OTHER").strip().upper()
        confidence = str(parsed.get("confidence") or "low").strip().lower()
        valid_answer = bool(parsed.get("valid_answer"))

        if kind not in {"CLARIFY_EXAMPLE", "ANSWER", "CLARIFY", "OTHER"}:
            kind = "OTHER"
        if confidence not in {"high", "medium", "low"}:
            confidence = "low"

        # Deterministic anti-loop guard: if heuristic says this is a valid memory,
        # do not allow semantic drift into CLARIFY/OTHER.
        if heuristic_valid and not _looks_like_meta_prompt_question(candidate):
            return {"kind": "ANSWER", "confidence": "high" if confidence == "low" else confidence, "valid_answer": True}

        if kind == "CLARIFY_EXAMPLE":
            return {"kind": "CLARIFY_EXAMPLE", "confidence": confidence, "valid_answer": False}

        if kind == "CLARIFY":
            return {"kind": "CLARIFY", "confidence": confidence, "valid_answer": False}

        if kind != "ANSWER":
            return {"kind": "OTHER", "confidence": confidence, "valid_answer": False}

        return {
            "kind": "ANSWER",
            "confidence": confidence,
            "valid_answer": bool(valid_answer and heuristic_valid),
        }
    except Exception:
        return {
            "kind": "ANSWER" if heuristic_valid else "OTHER",
            "confidence": "medium",
            "valid_answer": heuristic_valid,
        }


def _execute_pending_waiting_draft_save(
    *,
    bridge: Any,
    policy_service: Any,
    user_id: int,
    pending_waiting_draft: Dict[str, Any],
) -> Dict[str, Any]:
    draft_id = str(pending_waiting_draft.get("draft_id") or "").strip()
    if draft_id == "":
        return {"success": False, "error": "missing_pending_draft_id"}

    draft_record = policy_service.get_draft(draft_id=draft_id)
    payload = draft_record.get("payload") if isinstance(draft_record.get("payload"), dict) else {}
    text_answer = str(payload.get("text_answer") or "").strip()
    if text_answer == "":
        return {"success": False, "error": "missing_waiting_text_answer"}

    save_result = bridge.submit_waiting_reply(
        user_id=user_id,
        text_answer=text_answer,
    )
    save_success = bool(save_result.get("success") or save_result.get("ok"))
    if save_success:
        return {"success": True}

    return {
        "success": False,
        "error": str(save_result.get("message") or save_result.get("error") or "waiting_save_failed"),
    }


def _classify_pending_waiting_turn(text: str) -> str:
    candidate = (text or "").strip().lower()
    if candidate == "":
        return "other"

    if _looks_like_acknowledgement(candidate):
        return "ack"

    normalized = re.sub(r"[^a-z0-9\s]", " ", candidate)
    normalized = " ".join(normalized.split())

    if normalized in {"yes", "y", "sure", "yep", "yeah", "ok send", "send it", "go ahead", "do it", "approve"}:
        return "approve"

    if normalized in {"cancel", "never mind", "nevermind", "stop", "dont save", "do not save"}:
        return "cancel"

    if any(
        phrase in normalized
        for phrase in {"what is it", "what draft", "change it", "edit", "revise", "update", "show draft", "have you added"}
    ):
        return "edit_or_clarify"

    return "other"


def _increment_waiting_prompt_clarify_count(*, session_key: str) -> int:
    key = (session_key or "").strip() or "default"
    updated = int(_waiting_prompt_clarify_counts.get(key, 0)) + 1
    _waiting_prompt_clarify_counts[key] = updated
    return updated


def _set_onboarding_terminal_state(*, session_key: str, state: str) -> None:
    key = (session_key or "").strip() or "default"
    normalized = (state or "").strip().upper()
    if normalized:
        _onboarding_terminal_state[key] = normalized
    else:
        _onboarding_terminal_state.pop(key, None)


def _is_onboarding_terminal_sealed(*, session_key: str) -> bool:
    key = (session_key or "").strip() or "default"
    return _onboarding_terminal_state.get(key) in {"SENT", "COMPLETE", "IDLE"}


def _is_terminal_frequency_confirmation_event(
    *,
    lifecycle_intent: str,
    lifecycle_success: bool,
    lifecycle_context: Dict[str, Any],
) -> bool:
    if lifecycle_intent != "frequency" or not lifecycle_success:
        return False
    event = str(lifecycle_context.get("event") or "").strip().lower()
    return event == "frequency"


def _build_lifecycle_correlation_id(*, session_key: str, lifecycle_intent: str, lifecycle_context: Dict[str, Any]) -> str:
    normalized_choice = str(lifecycle_context.get("normalized_choice") or "").strip().lower()
    event = str(lifecycle_context.get("event") or "").strip().lower()
    payload = f"{session_key}|{lifecycle_intent}|{event}|{normalized_choice}|terminal_confirmation"
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return f"terminal:{digest}"


def _mark_lifecycle_emit_sent(*, session_key: str, correlation_id: str) -> None:
    key = (session_key or "").strip() or "default"
    corr = (correlation_id or "").strip()
    if corr == "":
        return
    sent_map = _lifecycle_sent_correlation_ids.setdefault(key, {})
    sent_map[corr] = "sent"


def _reset_lifecycle_emit_records(*, session_key: str) -> None:
    key = (session_key or "").strip() or "default"
    _lifecycle_sent_correlation_ids.pop(key, None)


def _is_lifecycle_emit_already_sent(*, session_key: str, correlation_id: str) -> bool:
    key = (session_key or "").strip() or "default"
    corr = (correlation_id or "").strip()
    if corr == "":
        return False
    sent_map = _lifecycle_sent_correlation_ids.get(key) or {}
    return corr in sent_map


def _reset_waiting_prompt_clarify_count(*, session_key: str) -> None:
    key = (session_key or "").strip() or "default"
    _waiting_prompt_clarify_counts.pop(key, None)


def _set_waiting_prompt_offramp_lock(*, session_key: str, locked: bool) -> None:
    key = (session_key or "").strip() or "default"
    if locked:
        _waiting_prompt_offramp_locked[key] = True
    else:
        _waiting_prompt_offramp_locked.pop(key, None)


def _is_waiting_prompt_offramp_locked(*, session_key: str) -> bool:
    key = (session_key or "").strip() or "default"
    return bool(_waiting_prompt_offramp_locked.get(key))


def _sanitize_lifecycle_context_for_generation(lifecycle_context: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(lifecycle_context, dict):
        return {}

    event = str(lifecycle_context.get("event") or "").strip()
    result = lifecycle_context.get("result") if isinstance(lifecycle_context.get("result"), dict) else {}

    # Global ephemeral flag flush.
    blocked_keys = {"failed_save", "validation_error", "previous_clarification_attempt"}
    filtered_result = {k: v for k, v in result.items() if str(k) not in blocked_keys}

    if event in {"waiting_prompt_clarify", "waiting_prompt_clarify_example", "waiting_prompt_reprompt"}:
        return {
            "event": event,
            "success": bool(lifecycle_context.get("success")),
            "result": {
                "waiting_prompt_text": str(filtered_result.get("waiting_prompt_text") or "").strip(),
                "journal_state": str(filtered_result.get("journal_state") or "PROMPTING"),
                "clarify_loop_count": int(filtered_result.get("clarify_loop_count") or 0),
            },
            "error": "",
        }

    sanitized = dict(lifecycle_context)
    sanitized["result"] = filtered_result
    return sanitized


def _build_waiting_prompt_direct_reply(*, user_text: str, lifecycle_context: Dict[str, Any]) -> str:
    event = str(lifecycle_context.get("event") or "").strip()
    result = lifecycle_context.get("result") if isinstance(lifecycle_context.get("result"), dict) else {}
    prompt_text = str(result.get("waiting_prompt_text") or "").strip()

    if event == "waiting_prompt_clarify":
        return (
            "Answer with one real moment that made you feel proud this week. "
            "A short text or voice note works.\n\n"
            f"{prompt_text}" if prompt_text else "Answer with one real moment that made you feel proud this week. A short text or voice note works."
        )

    if event == "waiting_prompt_clarify_example":
        examples = (
            "Example 1: I reconnected with a school friend after years and felt proud for keeping that friendship alive.\n"
            "Example 2: I helped my teammate finish a hard task and felt proud of how calm I stayed."
        )
        if prompt_text:
            return f"Here are sample answers:\n{examples}\n\n{prompt_text}"
        return f"Here are sample answers:\n{examples}\n\nShare your own version in text or voice note."

    if event == "waiting_prompt_reprompt":
        return (
            "Share one specific proud moment from this week. Keep it short and personal.\n\n"
            f"{prompt_text}" if prompt_text else "Share one specific proud moment from this week. Keep it short and personal."
        )

    if event == "waiting_prompt_offramp_wait_input":
        return "Take your time. Send a text or voice note whenever you are ready."

    return ""


def _sanitize_lifecycle_reply_text(reply: str) -> str:
    text = str(reply or "").strip()
    if text == "":
        return text

    replacements = {
        r"\bno\s+worries\b[:,!]?\s*": "",
        r"\bhey!?\b[:,!]?\s*": "",
        r"\blet'?s\s+try\s+again\b[:,!]?\s*": "",
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    text = re.sub(r"\s{2,}", " ", text).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _create_waiting_text_draft(*, user_id: int, text_answer: str, waiting_prompt_text: str) -> Dict[str, Any]:
    if user_id <= 0:
        return {"error": "invalid_user_id"}

    preview = (text_answer or "").strip()
    if preview == "":
        return {"error": "empty_text_answer"}

    try:
        result = get_genzbuzz_policy_service().create_draft(
            agent_name="Messenger Waiting Flow",
            channel="messenger",
            draft_kind="waiting",
            payload={
                "user_id": int(user_id),
                "text_answer": preview,
                "waiting_prompt_text": str(waiting_prompt_text or "").strip(),
            },
            title="Messenger waiting prompt reply",
            metadata={
                "source": "messenger_ingress_waiting",
                "journal_state": "DRAFTING",
            },
        )
        return {
            "draft_id": str(result.get("draft_id") or "").strip(),
            "status": str(result.get("status") or "draft").strip(),
            "draft_preview": preview,
        }
    except Exception as exc:
        logger.warning(
            "Messenger waiting draft creation failed",
            extra={"error": str(exc), "user_id": user_id},
        )
        return {"error": "draft_create_failed"}


def _record_lifecycle_turn(*, session_key: str, user_text: str, reply_text: str) -> None:
    try:
        log = get_conversation_log(session_key=session_key)
        log.record_user_message(user_text)
        log.record_reply(reply_text)
    except Exception as exc:
        logger.warning(
            "Messenger lifecycle turn log append failed",
            extra={"error": str(exc), "session_key": session_key},
        )


