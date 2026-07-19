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
    
    # Context Injection: Fetch saved cadence for system prompt framing.
    saved_cadence = str(onboarding_data.get("frequency") or "").strip().lower()

    session_key = f"messenger:{clean_psid}"
    if onboarding_stage != "awaiting_frequency":
        _set_onboarding_terminal_state(session_key=session_key, state="")
        _reset_lifecycle_emit_records(session_key=session_key)
    terminal_sealed = _is_onboarding_terminal_sealed(session_key=session_key)
    terminal_sent = _get_onboarding_terminal_state(session_key=session_key) == "SENT"
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

    # Pre-check for acknowledgement in post-frequency SENT state.
    # This prevents 'Thanks'/'Cool' from falling back to runtime immediately after save.
    if terminal_sent and _looks_like_acknowledgement(normalized_text):
        lifecycle_intent = "frequency"
        lifecycle_success = True
        lifecycle_message = "frequency_complete_ack"
        lifecycle_context = {
            "event": "frequency_complete_ack",
            "success": True,
            "result": {},
            "error": "",
        }
        _set_onboarding_terminal_state(session_key=session_key, state="COMPLETE")
        logger.info("Messenger onboarding final ack handled; psid=%s", clean_psid)
        return await _finish_lifecycle_turn(
            psid=clean_psid,
            user_id=resolved_user_id,
            user_text=clean_text,
            lifecycle_intent=lifecycle_intent,
            lifecycle_success=lifecycle_success,
            lifecycle_message=lifecycle_message,
            lifecycle_context=lifecycle_context,
            session_key=session_key,
        )

    # Meaning-first routing for frequency turns with semantic ambiguity checks.
    frequency_turn = await _classify_frequency_turn_semantic(clean_text)
    frequency_intent = str(frequency_turn.get("kind") or "").strip().lower()
    frequency_choice = str(frequency_turn.get("choice") or "").strip().lower()
    frequency_should_commit = bool(frequency_turn.get("should_commit"))

    confidence = str(frequency_turn.get("confidence") or "low").strip().lower()
    is_mixed_intent = bool(frequency_turn.get("mixed_intent"))

    # Prefer current state over generic fallback when in a guided onboarding stage.
    # Accepts common replies inside frequency choice flow without requiring 'high' confidence.
    if onboarding_stage == "awaiting_frequency" and not terminal_sealed:
        if frequency_intent == "select" and frequency_choice != "unknown":
            # Loosen confidence requirement for guided frequency turns.
            frequency_should_commit = True
        elif frequency_intent == "none":
            # If truly unknown/off-topic during onboarding, route to clarification instead of generic fallback.
            frequency_intent = "needs_clarification"
            frequency_should_commit = False

    # Commit only if semantic confidence is high and intent is not mixed.
    if frequency_intent == "select" and (confidence == "low" or is_mixed_intent) and onboarding_stage != "awaiting_frequency":
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

    if lifecycle_context is not None:
        return await _finish_lifecycle_turn(
            psid=clean_psid,
            user_id=resolved_user_id,
            user_text=clean_text,
            lifecycle_intent=lifecycle_intent,
            lifecycle_success=lifecycle_success,
            lifecycle_message=lifecycle_message,
            lifecycle_context=lifecycle_context,
            session_key=session_key,
        )

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
                if typed_intent == "APPROVE" and typed_confidence != "low":
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
                elif typed_intent == "CANCEL" and typed_confidence != "low":
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

            elif acknowledgement_only and typed_confidence == "high":
                lifecycle_intent = "acknowledgement"
                lifecycle_success = True
                lifecycle_message = "acknowledgement_handled"
                lifecycle_context = {
                    "event": "acknowledgement",
                    "success": True,
                    "result": {},
                    "error": "",
                }
                logger.info("Messenger acknowledgement handled; psid=%s", clean_psid)

            elif waiting_topic or send_request or typed_intent != "OTHER":
                # Route specialized waiting/sending intents to runtime for draft/action.
                lifecycle_intent = "waiting_prompt"
                lifecycle_success = True
                lifecycle_message = "waiting_intent_routed_to_runtime"
                lifecycle_context = {
                    "event": "waiting_intent_routed_to_runtime",
                    "success": True,
                    "result": {
                        "has_pending_draft": has_pending_waiting_draft,
                        "journal_state": draft_state,
                        "pending_turn_kind": pending_turn_kind,
                        "intent_confidence": typed_confidence,
                        "draft_status": pending_draft_status,
                        "draft_age_seconds": pending_draft_age_seconds,
                        "waiting_prompt_text": waiting_prompt_text,
                        "execution_outcome": execution_outcome,
                    },
                    "error": "",
                }

    # Fallback to general interaction agent runtime for mixed/unstructured turns.
    try:
        runtime = InteractionAgentRuntime(session_key=session_key, cadence=saved_cadence)
        result = await runtime.execute(user_message=clean_text)
        return MessengerIngressResult(
            handled=True,
            intent="runtime",
            success=result.success,
            message=result.error or "executed",
            user_id=resolved_user_id,
            reply_text=result.response,
        )
    except Exception as exc:
        logger.error("Messenger runtime fallback failed", extra={"error": str(exc), "psid": clean_psid})
        return MessengerIngressResult(
            handled=True,
            intent="runtime",
            success=False,
            message=f"runtime_error: {exc}",
            user_id=resolved_user_id,
        )


async def _finish_lifecycle_turn(
    *,
    psid: str,
    user_id: int,
    user_text: str,
    lifecycle_intent: str,
    lifecycle_success: bool,
    lifecycle_message: str,
    lifecycle_context: Dict[str, Any],
    session_key: str,
) -> MessengerIngressResult:
    # Record metadata for the turn in the conversation log for runtime awareness.
    log = get_conversation_log(session_key=session_key)
    log.record_user_message(user_text)

    agent_message = f"<lifecycle_event intent=\"{lifecycle_intent}\" message=\"{lifecycle_message}\" success=\"{'true' if lifecycle_success else 'false'}\">\n{json.dumps(lifecycle_context)}\n</lifecycle_event>"
    
    # We always resolve terminal frequency state here so that if the runtime is called
    # it knows the frequency selection is already committed/handled.
    if lifecycle_intent == "frequency" and lifecycle_success and lifecycle_message == "frequency_saved":
         _set_onboarding_terminal_state(session_key=session_key, state="SENT")

    try:
        # Re-fetch cadence in case it was just updated in this turn.
        bridge = get_genzbuzz_bridge_service()
        onboarding_state = bridge.messenger_onboarding_state(psid=psid, user_id=user_id)
        saved_cadence = str((onboarding_state.get("data") or onboarding_state).get("frequency") or "").strip().lower()

        runtime = InteractionAgentRuntime(session_key=session_key, cadence=saved_cadence)
        result = await runtime.handle_agent_message(agent_message=agent_message)
        return MessengerIngressResult(
            handled=True,
            intent=lifecycle_intent,
            success=result.success,
            message=result.error or lifecycle_message,
            user_id=user_id,
            reply_text=result.response,
        )
    except Exception as exc:
        logger.error("Messenger lifecycle handoff failed", extra={"error": str(exc), "psid": psid})
        return MessengerIngressResult(
            handled=True,
            intent=lifecycle_intent,
            success=False,
            message=f"handoff_error: {exc}",
            user_id=user_id,
        )

# (Rest of file truncated for brevity in this mock-up - the push will include the full file)
