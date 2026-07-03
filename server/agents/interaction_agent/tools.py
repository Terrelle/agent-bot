"""Tool definitions for interaction agent."""

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Optional

from ...logging_config import logger
from ...services.conversation import (
    collect_outbound_message,
    dispatch_message,
    get_active_session_key,
    get_conversation_log,
    get_delivery_context,
)
from ...services.execution import get_agent_roster, get_execution_agent_logs
from ..execution_agent.batch_manager import ExecutionBatchManager


@dataclass
class ToolResult:
    """Standardized payload returned by interaction-agent tools."""

    success: bool
    payload: Any = None
    user_message: Optional[str] = None
    recorded_reply: bool = False

# Tool schemas for OpenRouter
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "send_message_to_agent",
            "description": "Deliver instructions to a specific execution agent. Creates a new agent if the name doesn't exist in the roster, or reuses an existing one.",
            "parameters": {
                "type": "object",
                "properties": {
                    "agent_name": {
                        "type": "string",
                        "description": "Human-readable agent name describing its purpose (e.g., 'Bonding Cycle Follow-up', 'Waiting Prompt Queue'). This name will be used to identify and potentially reuse the agent."
                    },
                    "instructions": {"type": "string", "description": "Instructions for the agent to execute."},
                },
                "required": ["agent_name", "instructions"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_message_to_user",
            "description": "Deliver a natural-language response directly to the user. Use this for updates, confirmations, or any assistant response the user should see immediately.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Plain-text message that will be shown to the user and recorded in the conversation log.",
                    },
                },
                "required": ["message"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_draft",
            "description": "Record a GenZbuzz draft (waiting prompt, bonding prompt, or spontaneous memento) for user review before final send.",
            "parameters": {
                "type": "object",
                "properties": {
                    "draft_kind": {
                        "type": "string",
                        "enum": ["waiting_prompt", "bonding_prompt", "spontaneous_memento"],
                        "description": "Type of GenZbuzz draft to stage for review.",
                    },
                    "draft_title": {
                        "type": "string",
                        "description": "Short draft title shown in review output.",
                    },
                    "draft_body": {
                        "type": "string",
                        "description": "Draft content that will be shown to the user for confirmation.",
                    },
                    "target_user_id": {
                        "type": "string",
                        "description": "Optional user id for the draft recipient (for bonding/spontaneous use cases).",
                    },
                    "friend_user_id": {
                        "type": "string",
                        "description": "Optional friend user id for bonding/spontaneous drafts.",
                    },
                },
                "required": ["draft_kind", "draft_body"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wait",
            "description": "Wait silently when a message is already in conversation history to avoid duplicating responses. Adds a <wait> log entry that is not visible to the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Brief explanation of why waiting (e.g., 'Message already sent', 'Draft already created').",
                    },
                },
                "required": ["reason"],
                "additionalProperties": False,
            },
        },
    },
]

_EXECUTION_BATCH_MANAGER = ExecutionBatchManager()


# Create or reuse execution agent and dispatch instructions asynchronously
def send_message_to_agent(agent_name: str, instructions: str) -> ToolResult:
    """Send instructions to an execution agent."""
    roster = get_agent_roster()
    roster.load()
    existing_agents = set(roster.get_agents())
    is_new = agent_name not in existing_agents

    if is_new:
        roster.add_agent(agent_name)

    get_execution_agent_logs().record_request(agent_name, instructions)

    action = "Created" if is_new else "Reused"
    logger.info(f"{action} agent: {agent_name}")

    async def _execute_async() -> None:
        try:
            result = await _EXECUTION_BATCH_MANAGER.execute_agent(
                agent_name,
                instructions,
                session_key=get_active_session_key(),
            )
            status = "SUCCESS" if result.success else "FAILED"
            logger.info(f"Agent '{agent_name}' completed: {status}")
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(f"Agent '{agent_name}' failed: {str(exc)}")

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.error("No running event loop available for async execution")
        return ToolResult(success=False, payload={"error": "No event loop available"})

    loop.create_task(_execute_async())

    return ToolResult(
        success=True,
        payload={
            "status": "submitted",
            "agent_name": agent_name,
            "new_agent_created": is_new,
        },
    )


# Send immediate message to user and record in conversation history
def send_message_to_user(message: str) -> ToolResult:
    """Record a user-visible reply in the conversation log."""
    text = str(message or "").strip()
    if text == "":
        return ToolResult(success=False, payload={"error": "message is required"})

    context = get_delivery_context()
    delivered = False
    delivery_payload: dict[str, Any] = {
        "mode": context.mode,
        "channel": context.channel,
        "recipient_id": context.recipient_id,
    }

    if context.mode == "collect":
        collect_outbound_message(
            channel=context.channel,
            recipient_id=context.recipient_id,
            text=text,
        )
        delivered = True
        delivery_payload["status"] = "collected"
    elif context.mode == "dispatch":
        result = dispatch_message(
            channel=context.channel,
            recipient_id=context.recipient_id,
            text=text,
        )
        delivered = bool(result.ok)
        delivery_payload["status"] = result.status
        if result.detail:
            delivery_payload["detail"] = result.detail

    # Always persist assistant replies for transcript/history, regardless of delivery mode.
    log = get_conversation_log()
    log.record_reply(text)

    return ToolResult(
        success=True,
        payload={"status": "delivered" if delivered else "logged_only", **delivery_payload},
        user_message=text,
        recorded_reply=True,
    )


# Format and record GenZbuzz draft for user review
def send_draft(
    draft_kind: Optional[str] = None,
    draft_body: Optional[str] = None,
    draft_title: Optional[str] = None,
    target_user_id: Optional[str] = None,
    friend_user_id: Optional[str] = None,
    to: Optional[str] = None,
    subject: Optional[str] = None,
    body: Optional[str] = None,
) -> ToolResult:
    """Record a user-reviewable draft for GenZbuzz message flows."""
    kind = (draft_kind or "").strip().lower()
    title = (draft_title or "").strip()
    text = (draft_body or "").strip()

    # Backward compatibility: map legacy email-style draft payloads into generic draft fields.
    if not text and body:
        text = str(body).strip()
    if not title and subject:
        title = str(subject).strip()
    if not kind:
        kind = "spontaneous_memento" if (to or subject or body) else ""

    allowed_kinds = {"waiting_prompt", "bonding_prompt", "spontaneous_memento"}
    if kind not in allowed_kinds:
        return ToolResult(
            success=False,
            payload={
                "error": "draft_kind must be one of waiting_prompt, bonding_prompt, spontaneous_memento",
            },
        )
    if text == "":
        return ToolResult(success=False, payload={"error": "draft_body is required"})

    header_lines = [f"Draft Type: {kind}"]
    if title:
        header_lines.append(f"Draft Title: {title}")
    if target_user_id:
        header_lines.append(f"Target User ID: {str(target_user_id).strip()}")
    if friend_user_id:
        header_lines.append(f"Friend User ID: {str(friend_user_id).strip()}")
    message = "\n".join(header_lines) + "\n\n" + text

    context = get_delivery_context()
    delivered = False
    delivery_payload: dict[str, Any] = {
        "mode": context.mode,
        "channel": context.channel,
        "recipient_id": context.recipient_id,
    }

    if context.mode == "collect":
        collect_outbound_message(
            channel=context.channel,
            recipient_id=context.recipient_id,
            text=message,
        )
        delivered = True
        delivery_payload["status"] = "collected"
    elif context.mode == "dispatch":
        result = dispatch_message(
            channel=context.channel,
            recipient_id=context.recipient_id,
            text=message,
        )
        delivered = bool(result.ok)
        delivery_payload["status"] = result.status
        if result.detail:
            delivery_payload["detail"] = result.detail

    log = get_conversation_log()
    log.record_reply(message)
    logger.info("GenZbuzz draft recorded", extra={"draft_kind": kind})

    return ToolResult(
        success=True,
        payload={
            "status": "draft_recorded",
            "draft_kind": kind,
            "draft_title": title,
            "target_user_id": str(target_user_id or "").strip(),
            "friend_user_id": str(friend_user_id or "").strip(),
            "delivery": {
                "status": "delivered" if delivered else "logged_only",
                **delivery_payload,
            },
        },
        user_message=message,
        recorded_reply=True,
    )


# Record silent wait state to avoid duplicate responses
def wait(reason: str) -> ToolResult:
    """Wait silently and add a wait log entry that is not visible to the user."""
    log = get_conversation_log()
    
    # Record a dedicated wait entry so the UI knows to ignore it
    log.record_wait(reason)
    

    return ToolResult(
        success=True,
        payload={
            "status": "waiting",
            "reason": reason,
        },
        recorded_reply=True,
    )


# Return predefined tool schemas for LLM function calling
def get_tool_schemas():
    """Return OpenAI-compatible tool schemas."""
    return TOOL_SCHEMAS


# Route tool calls to appropriate handlers with argument validation and error handling
def handle_tool_call(name: str, arguments: Any) -> ToolResult:
    """Handle tool calls from interaction agent."""
    try:
        if isinstance(arguments, str):
            args = json.loads(arguments) if arguments.strip() else {}
        elif isinstance(arguments, dict):
            args = arguments
        else:
            return ToolResult(success=False, payload={"error": "Invalid arguments format"})

        if name == "send_message_to_agent":
            return send_message_to_agent(**args)
        if name == "send_message_to_user":
            return send_message_to_user(**args)
        if name == "send_draft":
            return send_draft(**args)
        if name == "wait":
            return wait(**args)

        logger.warning("unexpected tool", extra={"tool": name})
        return ToolResult(success=False, payload={"error": f"Unknown tool: {name}"})
    except json.JSONDecodeError:
        return ToolResult(success=False, payload={"error": "Invalid JSON"})
    except TypeError as exc:
        return ToolResult(success=False, payload={"error": f"Missing required arguments: {exc}"})
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("tool call failed", extra={"tool": name, "error": str(exc)})
        return ToolResult(success=False, payload={"error": "Failed to execute"})
