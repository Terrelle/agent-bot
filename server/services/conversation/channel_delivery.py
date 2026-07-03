from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional


@dataclass(frozen=True)
class DeliveryContext:
    channel: str = "default"
    recipient_id: str = ""
    mode: str = "log"  # log | collect | dispatch


@dataclass(frozen=True)
class OutboundMessage:
    channel: str
    recipient_id: str
    text: str


@dataclass(frozen=True)
class DeliveryResult:
    ok: bool
    status: str
    detail: str = ""


ChannelDispatcher = Callable[[str, str], DeliveryResult]


_DEFAULT_CONTEXT = DeliveryContext()
_delivery_context: ContextVar[DeliveryContext] = ContextVar(
    "openpoke_delivery_context", default=_DEFAULT_CONTEXT
)
_outbound_messages: ContextVar[List[OutboundMessage]] = ContextVar(
    "openpoke_outbound_messages", default=[]
)

_dispatchers: Dict[str, ChannelDispatcher] = {}


def get_delivery_context() -> DeliveryContext:
    return _delivery_context.get()


def set_delivery_context(*, channel: str, recipient_id: str, mode: str) -> Token[DeliveryContext]:
    clean_channel = (channel or "default").strip().lower() or "default"
    clean_recipient = (recipient_id or "").strip()
    clean_mode = (mode or "log").strip().lower() or "log"
    return _delivery_context.set(
        DeliveryContext(channel=clean_channel, recipient_id=clean_recipient, mode=clean_mode)
    )


def reset_delivery_context(token: Token[DeliveryContext]) -> None:
    _delivery_context.reset(token)


def begin_outbound_collection() -> Token[List[OutboundMessage]]:
    return _outbound_messages.set([])


def reset_outbound_collection(token: Token[List[OutboundMessage]]) -> None:
    _outbound_messages.reset(token)


def collect_outbound_message(*, channel: str, recipient_id: str, text: str) -> None:
    existing = list(_outbound_messages.get())
    existing.append(
        OutboundMessage(
            channel=(channel or "default").strip().lower() or "default",
            recipient_id=(recipient_id or "").strip(),
            text=(text or "").strip(),
        )
    )
    _outbound_messages.set(existing)


def get_outbound_messages() -> List[OutboundMessage]:
    return list(_outbound_messages.get())


def register_dispatcher(channel: str, dispatcher: ChannelDispatcher) -> None:
    key = (channel or "").strip().lower()
    if not key:
        raise ValueError("channel is required")
    _dispatchers[key] = dispatcher


def dispatch_message(*, channel: str, recipient_id: str, text: str) -> DeliveryResult:
    key = (channel or "").strip().lower()
    dispatcher = _dispatchers.get(key)
    if dispatcher is None:
        return DeliveryResult(ok=False, status="no_dispatcher", detail=f"channel={key}")
    try:
        return dispatcher(recipient_id, text)
    except Exception as exc:  # pragma: no cover - defensive
        return DeliveryResult(ok=False, status="dispatch_error", detail=str(exc))


__all__ = [
    "DeliveryContext",
    "DeliveryResult",
    "OutboundMessage",
    "begin_outbound_collection",
    "collect_outbound_message",
    "dispatch_message",
    "get_delivery_context",
    "get_outbound_messages",
    "register_dispatcher",
    "reset_delivery_context",
    "reset_outbound_collection",
    "set_delivery_context",
]
