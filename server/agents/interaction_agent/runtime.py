"""Interaction Agent Runtime - handles LLM calls for user and agent turns."""

from html import unescape
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from .agent import build_system_prompt, prepare_message_with_history
from .tools import ToolResult, get_tool_schemas, handle_tool_call
from ...config import get_settings
from ...services.conversation import (
    get_conversation_log,
    get_working_memory_log,
    reset_active_session_key,
    set_active_session_key,
)
from ...openrouter_client import OpenRouterError, request_chat_completion
from ...logging_config import logger


@dataclass
class InteractionResult:
    """Result from the interaction agent."""

    success: bool
    response: str
    error: Optional[str] = None
    execution_agents_used: int = 0


@dataclass
class _ToolCall:
    """Parsed tool invocation from an LLM response."""

    identifier: Optional[str]
    name: str
    arguments: Dict[str, Any]


@dataclass
class _LoopSummary:
    """Aggregate information produced by the interaction loop."""

    last_assistant_text: str = ""
    user_messages: List[str] = field(default_factory=list)
    tool_names: List[str] = field(default_factory=list)
    execution_agents: Set[str] = field(default_factory=set)
    had_tool_rejection: bool = False


class InteractionAgentRuntime:
    """Manages the interaction agent's request processing."""

    MAX_TOOL_ITERATIONS = 8
    MAX_EMPTY_ASSISTANT_RETRIES = 1
    FALLBACK_MESSAGE_MAX_CHARS = 1000
    MESSENGER_HISTORY_TURNS = 12
    SAFE_FALLBACK_RESPONSE = "Sorry about that. I can help with that - what do you want to do next?"
    FALLBACK_SYSTEM_PROMPT = (
        "You are GenZbuzz. Reply briefly and clearly to the user's latest message. "
        "Keep identity/capability claims strictly within GenZbuzz messaging flows "
        "(waiting prompts, bonding prompts, spontaneous mementos, reminders). "
        "Do not claim generic cross-platform integration setup or unsupported capabilities."
    )
    INTERNAL_META_PATTERNS = (
        "no function call was made",
        "tool call",
        "function call",
        "execution agent",
        "hidden tools",
        "internal process",
        "<send_message_to_user>",
        "</send_message_to_user>",
    )
    TOOL_ENVELOPE_NAME_PATTERN = re.compile(
        r'"name"\s*:\s*"(send_message_to_user|send_draft|send_message_to_agent|wait)"',
        flags=re.IGNORECASE,
    )

    # Initialize interaction agent runtime with settings and service dependencies
    def __init__(self, session_key: Optional[str] = None) -> None:
        settings = get_settings()
        self.api_key = settings.openrouter_api_key
        self.model = settings.interaction_agent_model
        self.settings = settings
        self.session_key = session_key
        self.conversation_log = get_conversation_log(session_key=session_key)
        self.working_memory_log = get_working_memory_log(session_key=session_key)
        self.tool_schemas = self._resolve_tool_schemas()

        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not configured. Set OPENROUTER_API_KEY environment variable."
            )

    # Main entry point for processing user messages through the LLM interaction loop
    async def execute(self, user_message: str) -> InteractionResult:
        """Handle a user-authored message."""

        token = set_active_session_key(self.session_key)
        try:
            transcript_before = self._load_conversation_transcript()
            self.conversation_log.record_user_message(user_message)

            system_prompt = build_system_prompt()
            messages = prepare_message_with_history(
                user_message, transcript_before, message_type="user"
            )

            logger.info("Processing user message through interaction agent")
            summary = await self._run_interaction_loop(system_prompt, messages)

            final_response = self._finalize_response(summary)

            # send_message_to_user already records the visible reply.
            if final_response and not summary.user_messages:
                self.conversation_log.record_reply(final_response)

            return InteractionResult(
                success=True,
                response=final_response,
                execution_agents_used=len(summary.execution_agents),
            )

        except Exception as exc:
            logger.error("Interaction agent failed", extra={"error": str(exc)})
            return InteractionResult(
                success=False,
                response="",
                error=str(exc),
            )
        finally:
            reset_active_session_key(token)

    # Handle incoming messages from execution agents and generate appropriate responses
    async def handle_agent_message(self, agent_message: str) -> InteractionResult:
        """Process a status update emitted by an execution agent."""

        if self._is_messenger_session():
            # Messenger ingress is user-turn driven; execution-agent status chatter
            # should not create unsolicited user-visible replies.
            return InteractionResult(
                success=True,
                response="",
                execution_agents_used=0,
            )

        token = set_active_session_key(self.session_key)
        try:
            transcript_before = self._load_conversation_transcript()
            self.conversation_log.record_agent_message(agent_message)

            system_prompt = build_system_prompt()
            messages = prepare_message_with_history(
                agent_message, transcript_before, message_type="agent"
            )

            logger.info("Processing execution agent results")
            summary = await self._run_interaction_loop(system_prompt, messages)

            final_response = self._finalize_response(summary)

            if final_response and not summary.user_messages:
                self.conversation_log.record_reply(final_response)

            return InteractionResult(
                success=True,
                response=final_response,
                execution_agents_used=len(summary.execution_agents),
            )

        except Exception as exc:
            logger.error("Interaction agent (agent message) failed", extra={"error": str(exc)})
            return InteractionResult(
                success=False,
                response="",
                error=str(exc),
            )
        finally:
            reset_active_session_key(token)

    # Core interaction loop that handles LLM calls and tool executions until completion
    async def _run_interaction_loop(
        self,
        system_prompt: str,
        messages: List[Dict[str, Any]],
    ) -> _LoopSummary:
        """Iteratively query the LLM until it issues a final response."""

        summary = _LoopSummary()
        user_visible_message_emitted = False
        empty_assistant_retries = 0

        for iteration in range(self.MAX_TOOL_ITERATIONS):
            response = await self._make_llm_call(system_prompt, messages)
            assistant_message = self._extract_assistant_message(response)
            choice = (response.get("choices") or [{}])[0]
            finish_reason = str(choice.get("finish_reason") or "")

            assistant_content = (assistant_message.get("content") or "").strip()
            if assistant_content:
                summary.last_assistant_text = assistant_content

            raw_tool_calls = assistant_message.get("tool_calls") or []
            parsed_tool_calls = self._parse_tool_calls(raw_tool_calls)

            if not assistant_content and not raw_tool_calls:
                logger.warning(
                    "LLM returned empty assistant turn; finish_reason=%s; iteration=%s",
                    finish_reason or "unknown",
                    iteration + 1,
                )
                if empty_assistant_retries < self.MAX_EMPTY_ASSISTANT_RETRIES:
                    empty_assistant_retries += 1
                    logger.warning(
                        "Retrying LLM call after empty assistant turn; retry=%s",
                        empty_assistant_retries,
                    )
                    continue
            elif not assistant_content and raw_tool_calls and not parsed_tool_calls:
                logger.warning(
                    "LLM returned tool-calls but none were parseable; finish_reason=%s; raw_tool_calls=%s; iteration=%s",
                    finish_reason or "unknown",
                    len(raw_tool_calls),
                    iteration + 1,
                )

            assistant_entry: Dict[str, Any] = {
                "role": "assistant",
                "content": assistant_message.get("content", "") or "",
            }
            if raw_tool_calls:
                assistant_entry["tool_calls"] = raw_tool_calls
            messages.append(assistant_entry)

            if not parsed_tool_calls:
                break

            for tool_call in parsed_tool_calls:
                summary.tool_names.append(tool_call.name)

                if tool_call.name == "send_message_to_agent":
                    agent_name = tool_call.arguments.get("agent_name")
                    if isinstance(agent_name, str) and agent_name:
                        summary.execution_agents.add(agent_name)

                is_user_visible_tool = tool_call.name in {"send_message_to_user", "send_draft"}

                if is_user_visible_tool and user_visible_message_emitted:
                    logger.warning(
                        "Suppressing duplicate user-visible tool call in same turn",
                        extra={"tool": tool_call.name},
                    )
                    result = ToolResult(
                        success=True,
                        payload={"status": "suppressed_duplicate_user_visible_message"},
                    )
                else:
                    result = self._execute_tool(tool_call)

                if not result.success:
                    summary.had_tool_rejection = True

                if result.user_message:
                    summary.user_messages.append(result.user_message)
                    if is_user_visible_tool:
                        user_visible_message_emitted = True
                        if self._is_messenger_session():
                            # Messenger ingress expects one immediate user-visible reply per turn.
                            # Stop tool-looping once we have that reply to avoid post-reply churn.
                            return summary

                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call.identifier or tool_call.name,
                    "content": self._format_tool_result(tool_call, result),
                }
                messages.append(tool_message)
        else:
            if summary.user_messages:
                logger.warning(
                    "Reached tool iteration limit after user-visible reply; returning existing reply"
                )
                return summary
            raise RuntimeError("Reached tool iteration limit without final response")

        if not summary.user_messages and not summary.last_assistant_text:
            logger.warning(
                "Interaction loop exited without assistant content; had_tool_rejection=%s; tools_seen=%s",
                "true" if summary.had_tool_rejection else "false",
                len(summary.tool_names),
            )

        return summary

    # Load conversation history, preferring summarized version if available
    def _load_conversation_transcript(self) -> str:
        if self._is_messenger_session():
            transcript = self.conversation_log.load_transcript()
            return self._compact_messenger_transcript(transcript)

        if self.settings.summarization_enabled:
            rendered = self.working_memory_log.render_transcript()
            if rendered.strip():
                return rendered
        return self.conversation_log.load_transcript()

    # Execute API call to OpenRouter with system prompt, messages, and tool schemas
    async def _make_llm_call(
        self,
        system_prompt: str,
        messages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Make an LLM call via OpenRouter."""

        logger.debug(
            "Interaction agent calling LLM",
            extra={"model": self.model, "tools": len(self.tool_schemas)},
        )
        try:
            return await request_chat_completion(
                model=self.model,
                messages=messages,
                system=system_prompt,
                api_key=self.api_key,
                tools=self.tool_schemas,
            )
        except OpenRouterError as exc:
            message = str(exc)
            if "Prompt tokens limit exceeded" not in message:
                raise

            logger.warning(
                "Prompt exceeded key budget; retrying with compact context",
                extra={"error": message},
            )

            compact_messages = self._build_budget_fallback_messages(messages)
            return await request_chat_completion(
                model=self.model,
                messages=compact_messages,
                system=self.FALLBACK_SYSTEM_PROMPT,
                api_key=self.api_key,
                tools=None,
            )

    def _build_budget_fallback_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Extract only the latest user/agent turn to keep prompt tokens low."""

        latest_content = ""
        for message in reversed(messages):
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                latest_content = content
                break

        extracted = self._extract_latest_turn_text(latest_content)
        if len(extracted) > self.FALLBACK_MESSAGE_MAX_CHARS:
            extracted = extracted[-self.FALLBACK_MESSAGE_MAX_CHARS :]

        return [{"role": "user", "content": extracted or "Please respond to my latest message."}]

    def _extract_latest_turn_text(self, content: str) -> str:
        """Pull plain text from structured payloads like <new_user_message>...</new_user_message>."""

        for tag in ("new_user_message", "new_agent_message"):
            pattern = rf"<{tag}>\\s*(.*?)\\s*</{tag}>"
            match = re.search(pattern, content, flags=re.DOTALL)
            if match:
                return match.group(1).strip()

        return content.strip()

    # Extract the assistant's message from the OpenRouter API response structure
    def _extract_assistant_message(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Return the assistant message from the raw response payload."""

        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message")
        if not isinstance(message, dict):
            raise RuntimeError("LLM response did not include an assistant message")
        return message

    # Convert raw LLM tool calls into structured _ToolCall objects with validation
    def _parse_tool_calls(self, raw_tool_calls: List[Dict[str, Any]]) -> List[_ToolCall]:
        """Normalize tool call payloads from the LLM."""

        parsed: List[_ToolCall] = []
        for raw in raw_tool_calls:
            function_block = raw.get("function") or {}
            name = function_block.get("name")
            if not isinstance(name, str) or not name:
                logger.warning("Skipping tool call without name", extra={"tool": raw})
                continue

            raw_arguments = function_block.get("arguments")
            arguments, error = self._parse_tool_arguments(raw_arguments)
            if error:
                arg_preview = str(raw_arguments).replace("\n", " ").replace("\r", " ").strip()[:240]
                logger.warning(
                    "Tool call arguments invalid; tool=%s; error=%s; arg_preview=%s",
                    name,
                    error,
                    arg_preview,
                    extra={"tool": name, "error": error, "arg_preview": arg_preview},
                )
                parsed.append(
                    _ToolCall(
                        identifier=raw.get("id"),
                        name=name,
                        arguments={"__invalid_arguments__": error},
                    )
                )
                continue

            parsed.append(
                _ToolCall(identifier=raw.get("id"), name=name, arguments=arguments)
            )

        return parsed

    # Parse and validate tool arguments from various formats (dict, JSON string, etc.)
    def _parse_tool_arguments(
        self, raw_arguments: Any
    ) -> tuple[Dict[str, Any], Optional[str]]:
        """Convert tool arguments into a dictionary, reporting errors."""

        if raw_arguments is None:
            return {}, None

        if isinstance(raw_arguments, dict):
            return raw_arguments, None

        if isinstance(raw_arguments, str):
            raw_text = raw_arguments.strip()
            if not raw_text:
                return {}, None
            try:
                parsed = json.loads(raw_text)
            except json.JSONDecodeError as exc:
                recovered = self._recover_first_json_object(raw_text)
                if recovered is not None:
                    return recovered, None
                return {}, f"invalid json: {exc}"
            if isinstance(parsed, dict):
                return parsed, None
            return {}, "decoded arguments were not an object"

        return {}, f"unsupported argument type: {type(raw_arguments).__name__}"

    def _recover_first_json_object(self, raw_text: str) -> Optional[Dict[str, Any]]:
        """Recover first JSON object from concatenated JSON payloads."""

        decoder = json.JSONDecoder()
        try:
            first_value, end_index = decoder.raw_decode(raw_text)
        except json.JSONDecodeError:
            return None

        if not isinstance(first_value, dict):
            return None

        trailing = raw_text[end_index:].strip()
        if trailing == "":
            return first_value

        # Common model failure mode: two JSON objects concatenated with no delimiter.
        if trailing.startswith("{"):
            try:
                second_value, second_end = decoder.raw_decode(trailing)
                if isinstance(second_value, dict) and trailing[second_end:].strip() == "":
                    logger.warning(
                        "Recovered tool call arguments from concatenated JSON objects",
                        extra={
                            "first_keys": sorted(first_value.keys()),
                            "second_keys": sorted(second_value.keys()),
                        },
                    )
                    return first_value
            except json.JSONDecodeError:
                return None

        return None

    # Execute tool calls with error handling and logging, returning standardized results
    def _execute_tool(self, tool_call: _ToolCall) -> ToolResult:
        """Execute a tool call and convert low-level errors into structured results."""

        if "__invalid_arguments__" in tool_call.arguments:
            error = tool_call.arguments["__invalid_arguments__"]
            self._log_tool_invocation(tool_call, stage="rejected", detail={"error": error})
            return ToolResult(success=False, payload={"error": error})

        try:
            self._log_tool_invocation(tool_call, stage="start")
            result = handle_tool_call(tool_call.name, tool_call.arguments)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(
                "Tool execution crashed",
                extra={"tool": tool_call.name, "error": str(exc)},
            )
            self._log_tool_invocation(
                tool_call,
                stage="error",
                detail={"error": str(exc)},
            )
            return ToolResult(success=False, payload={"error": str(exc)})

        if not isinstance(result, ToolResult):
            logger.warning(
                "Tool did not return ToolResult; coercing",
                extra={"tool": tool_call.name},
            )
            wrapped = ToolResult(success=True, payload=result)
            self._log_tool_invocation(tool_call, stage="done", result=wrapped)
            return wrapped

        status = "success" if result.success else "error"
        logger.debug(
            "Tool executed",
            extra={
                "tool": tool_call.name,
                "status": status,
            },
        )
        self._log_tool_invocation(tool_call, stage="done", result=result)
        return result

    # Format tool execution results into JSON for LLM consumption
    def _format_tool_result(self, tool_call: _ToolCall, result: ToolResult) -> str:
        """Render a tool execution result back to the LLM."""

        payload: Dict[str, Any] = {
            "tool": tool_call.name,
            "status": "success" if result.success else "error",
            "arguments": {
                key: value
                for key, value in tool_call.arguments.items()
                if key != "__invalid_arguments__"
            },
        }

        if result.payload is not None:
            key = "result" if result.success else "error"
            payload[key] = result.payload

        return self._safe_json_dump(payload)

    # Safely serialize objects to JSON with fallback to string representation
    def _safe_json_dump(self, payload: Any) -> str:
        """Serialize payload to JSON, falling back to repr on failure."""

        try:
            return json.dumps(payload, default=str)
        except TypeError:
            return repr(payload)

    # Log tool execution stages (start, done, error) with structured metadata
    def _log_tool_invocation(
        self,
        tool_call: _ToolCall,
        *,
        stage: str,
        result: Optional[ToolResult] = None,
        detail: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit structured logs for tool lifecycle events."""

        cleaned_args = {
            key: value
            for key, value in tool_call.arguments.items()
            if key != "__invalid_arguments__"
        }

        log_payload: Dict[str, Any] = {
            "tool": tool_call.name,
            "stage": stage,
            "arguments": cleaned_args,
        }

        if result is not None:
            log_payload["success"] = result.success
            if result.payload is not None:
                log_payload["payload"] = result.payload

        if detail:
            log_payload.update(detail)

        if stage == "done":
            logger.info(f"Tool '{tool_call.name}' completed")
        elif stage in {"error", "rejected"}:
            logger.warning(f"Tool '{tool_call.name}' {stage}")
        else:
            logger.debug(f"Tool '{tool_call.name}' {stage}")

    # Determine final user-facing response from interaction loop summary
    def _finalize_response(self, summary: _LoopSummary) -> str:
        """Decide what text should be exposed to the user as the final reply."""

        if summary.user_messages:
            # Structural policy: first user-visible message in a turn is authoritative.
            return summary.user_messages[0]

        candidate = summary.last_assistant_text.strip()
        normalized = self._normalize_assistant_response(
            candidate,
            had_tool_rejection=summary.had_tool_rejection,
        )
        if normalized:
            return normalized

        logger.warning("Suppressing unsafe assistant text; returning safe fallback")
        return self.SAFE_FALLBACK_RESPONSE

    def _normalize_assistant_response(self, text: str, *, had_tool_rejection: bool) -> str:
        candidate = unescape((text or "").strip())
        if candidate == "":
            return ""

        # Strip common parser debris prefixes like '?:>' while preserving meaning.
        candidate = re.sub(r"^[\s:;>\?]+", "", candidate).strip()
        candidate = self._strip_wrapping_quotes(candidate)

        extracted = self._extract_message_from_structured_text(candidate)
        if extracted:
            logger.warning("Recovered user message from structured assistant payload")
            return extracted

        looks_structural = self._looks_like_function_payload(candidate) or self._looks_like_tool_markup(candidate)
        if looks_structural:
            return ""

        if self._looks_like_internal_meta(candidate):
            return ""

        if had_tool_rejection and self._contains_structural_artifacts(candidate):
            logger.warning("Dropping assistant fallback after tool rejection due to structural artifacts")
            return ""

        return candidate

    def _strip_wrapping_quotes(self, text: str) -> str:
        candidate = (text or "").strip()
        if len(candidate) >= 2 and ((candidate[0] == '"' and candidate[-1] == '"') or (candidate[0] == "'" and candidate[-1] == "'")):
            return candidate[1:-1].strip()
        return candidate

    def _extract_message_from_structured_text(self, text: str) -> str:
        candidate = (text or "").strip()
        if candidate == "":
            return ""

        # XML-like pseudo-tool payloads: <send_message_to_user>{...}</send_message_to_user>
        markup_match = re.search(
            r"<([a-z_][a-z0-9_]*)>\s*(\{.*?\})\s*</\1>",
            candidate,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if markup_match:
            payload = self._try_parse_json_object(markup_match.group(2))
            message = self._extract_user_message_from_payload(payload)
            if message:
                return message

        # Whole-text JSON payloads.
        payload = self._try_parse_json_object(candidate)
        message = self._extract_user_message_from_payload(payload)
        if message:
            return message

        # Scan for embedded JSON objects inside surrounding noise and recover message.
        for embedded in self._iter_json_objects(candidate):
            message = self._extract_user_message_from_payload(embedded)
            if message:
                return message

        return ""

    def _try_parse_json_object(self, text: str) -> Optional[Dict[str, Any]]:
        raw = (text or "").strip()
        if not raw.startswith("{"):
            return None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return parsed
        return None

    def _iter_json_objects(self, text: str) -> List[Dict[str, Any]]:
        source = text or ""
        decoder = json.JSONDecoder()
        found: List[Dict[str, Any]] = []
        for match in re.finditer(r"\{", source):
            idx = match.start()
            try:
                parsed, _ = decoder.raw_decode(source[idx:])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                found.append(parsed)
        return found

    def _extract_user_message_from_payload(self, payload: Optional[Dict[str, Any]]) -> str:
        if not isinstance(payload, dict):
            return ""

        for key in ("message", "text", "response"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        for container_key in ("parameters", "arguments"):
            nested = payload.get(container_key)
            if isinstance(nested, dict):
                for key in ("message", "text", "response"):
                    value = nested.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()

        return ""

    def _looks_like_internal_meta(self, text: str) -> bool:
        lowered = (text or "").lower().strip()
        if not lowered:
            return False
        return any(token in lowered for token in self.INTERNAL_META_PATTERNS)

    def _looks_like_function_payload(self, text: str) -> bool:
        lowered = (text or "").lower()
        if not lowered:
            return False

        # Compact tool envelope form seen in some model outputs:
        # {"name": "send_message_to_user", "parameters": {...}}
        has_compact_tool_name = bool(self.TOOL_ENVELOPE_NAME_PATTERN.search(lowered))
        has_compact_params = '"parameters"' in lowered or '"arguments"' in lowered
        if has_compact_tool_name and has_compact_params:
            return True

        # OpenAI function-wrapper form:
        # {"type":"function","name":"send_message_to_user","parameters":{...}}
        has_type = '"type"' in lowered and '"function"' in lowered
        has_name = '"name"' in lowered and (
            '"send_message_to_user"' in lowered
            or '"send_draft"' in lowered
            or '"send_message_to_agent"' in lowered
            or '"wait"' in lowered
        )
        has_parameters = '"parameters"' in lowered or '"arguments"' in lowered
        return has_type and has_name and has_parameters

    def _looks_like_tool_markup(self, text: str) -> bool:
        lowered = (text or "").lower()
        if not lowered:
            return False
        return bool(
            re.search(r"</?(send_message_to_user|send_draft|send_message_to_agent|wait)>", lowered)
        )

    def _contains_structural_artifacts(self, text: str) -> bool:
        lowered = (text or "").lower()
        if not lowered:
            return False

        has_openai_wrapper = (
            '"type"' in lowered
            and '"function"' in lowered
            and ('"parameters"' in lowered or '"arguments"' in lowered)
        )
        has_compact_tool_envelope = bool(self.TOOL_ENVELOPE_NAME_PATTERN.search(lowered)) and (
            '"parameters"' in lowered or '"arguments"' in lowered
        )

        return has_openai_wrapper or has_compact_tool_envelope or self._looks_like_tool_markup(text)

    def _is_messenger_session(self) -> bool:
        key = (self.session_key or "").strip().lower()
        return key.startswith("messenger:")

    def _resolve_tool_schemas(self) -> List[Dict[str, Any]]:
        schemas = get_tool_schemas()
        if not self._is_messenger_session():
            return schemas

        # For Messenger, keep interaction replies user-facing and synchronous.
        # Disable agent fan-out and draft plumbing that can leak internal chatter.
        allowed = {"send_message_to_user"}
        filtered: List[Dict[str, Any]] = []
        for schema in schemas:
            function_block = (schema or {}).get("function") or {}
            name = function_block.get("name")
            if isinstance(name, str) and name in allowed:
                filtered.append(schema)

        return filtered

    def _compact_messenger_transcript(self, transcript: str) -> str:
        source = (transcript or "").strip()
        if not source:
            return ""

        # Keep only human-visible turns so stale agent/wait/debug chatter
        # cannot steer replies in active Messenger sessions.
        pattern = re.compile(
            r"<(user_message|genzbuzz_reply)(?:\s+[^>]*)?>.*?</\1>",
            flags=re.DOTALL,
        )
        matches = pattern.finditer(source)
        selected: List[str] = []
        for match in matches:
            selected.append(match.group(0).strip())

        if not selected:
            return ""

        # Drop consecutive duplicate bot replies to avoid amplifying stale phrasing.
        deduped: List[str] = []
        last_reply_body = ""
        for entry in selected:
            if entry.startswith("<genzbuzz_reply"):
                body = re.sub(r"^<genzbuzz_reply(?:\s+[^>]*)?>|</genzbuzz_reply>$", "", entry, flags=re.DOTALL).strip()
                if body == last_reply_body:
                    continue
                last_reply_body = body
            deduped.append(entry)

        selected = deduped

        return "\n".join(selected[-self.MESSENGER_HISTORY_TURNS :])
