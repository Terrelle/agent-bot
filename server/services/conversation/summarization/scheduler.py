from __future__ import annotations

import asyncio
from typing import Dict, Set

from ....logging_config import logger
from ..session import get_active_session_key, normalize_session_key
from .summarizer import summarize_conversation

_pending: Set[str] = set()
_running: Dict[str, asyncio.Task] = {}


def schedule_summarization(session_key: str | None = None) -> None:
    """Schedule a background summarization pass if not already queued."""
    resolved = normalize_session_key(session_key or get_active_session_key())
    _pending.add(resolved)
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.debug("summarization skipped (no running event loop)")
        return

    existing = _running.get(resolved)
    if existing is None or existing.done():
        _running[resolved] = loop.create_task(_run_worker(resolved))


async def _run_worker(session_key: str) -> None:
    try:
        while session_key in _pending:
            _pending.discard(session_key)
            try:
                await summarize_conversation(session_key=session_key)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error(
                    "summarization worker failed",
                    extra={"error": str(exc)},
                )
    finally:
        _running.pop(session_key, None)


__all__ = ["schedule_summarization"]
