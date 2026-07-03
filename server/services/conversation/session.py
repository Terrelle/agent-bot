from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Optional

_DEFAULT_SESSION_KEY = "default"
_active_session_key: ContextVar[str] = ContextVar(
    "openpoke_active_session_key", default=_DEFAULT_SESSION_KEY
)


def normalize_session_key(session_key: Optional[str]) -> str:
    candidate = (session_key or "").strip()
    return candidate or _DEFAULT_SESSION_KEY


def get_active_session_key() -> str:
    return normalize_session_key(_active_session_key.get())


def set_active_session_key(session_key: Optional[str]) -> Token[str]:
    normalized = normalize_session_key(session_key)
    return _active_session_key.set(normalized)


def reset_active_session_key(token: Token[str]) -> None:
    _active_session_key.reset(token)


__all__ = [
    "get_active_session_key",
    "normalize_session_key",
    "reset_active_session_key",
    "set_active_session_key",
]
