"""Background watcher that surfaces important memento updates proactively."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .bridge import get_genzbuzz_bridge_service
from .memento_classifier import MementoCandidate, classify_memento_importance
from ..gmail.seen_store import GmailSeenStore
from ...config import get_settings
from ...logging_config import logger


def _resolve_interaction_runtime(session_key: str):
    from ...agents.interaction_agent.runtime import InteractionAgentRuntime

    return InteractionAgentRuntime(session_key=session_key)


UTC = timezone.utc
DEFAULT_POLL_INTERVAL_SECONDS = 75.0
DEFAULT_LOOKBACK_MINUTES = 30
DEFAULT_MAX_RESULTS = 50
DEFAULT_SEEN_LIMIT = 500

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_DEFAULT_SEEN_PATH = _DATA_DIR / "memento_watcher_seen.json"


class MementoWatcher:
    """Poll GenZbuzz memento queue and surface high-signal updates."""

    def __init__(
        self,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
        lookback_minutes: int = DEFAULT_LOOKBACK_MINUTES,
        *,
        seen_store: Optional[GmailSeenStore] = None,
    ) -> None:
        self._poll_interval = poll_interval_seconds
        self._lookback_minutes = lookback_minutes
        self._lock = asyncio.Lock()
        self._task: Optional[asyncio.Task[None]] = None
        self._running = False
        self._bridge = get_genzbuzz_bridge_service()
        self._seen_store = seen_store or GmailSeenStore(_DEFAULT_SEEN_PATH, DEFAULT_SEEN_LIMIT)
        self._has_seeded_initial_snapshot = False

    async def start(self) -> None:
        async with self._lock:
            if self._task and not self._task.done():
                return
            loop = asyncio.get_running_loop()
            self._running = True
            self._has_seeded_initial_snapshot = False
            self._task = loop.create_task(self._run(), name="memento-watcher")
            logger.info(
                "Memento watcher started",
                extra={"interval_seconds": self._poll_interval, "lookback_minutes": self._lookback_minutes},
            )

    async def stop(self) -> None:
        async with self._lock:
            self._running = False
            if self._task:
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
                finally:
                    self._task = None
                logger.info("Memento watcher stopped")

    async def _run(self) -> None:
        try:
            while self._running:
                try:
                    await self._poll_once()
                except Exception as exc:  # pragma: no cover - defensive
                    logger.exception("Memento watcher poll failed", extra={"error": str(exc)})
                await asyncio.sleep(self._poll_interval)
        except asyncio.CancelledError:
            raise

    async def _poll_once(self) -> None:
        since_iso = (datetime.now(UTC) - timedelta(minutes=self._lookback_minutes)).isoformat()
        payload = self._bridge.get_recent_memento_notifications(
            since_iso=since_iso,
            limit=DEFAULT_MAX_RESULTS,
        )

        if not bool(payload.get("success") or payload.get("ok")):
            logger.debug(
                "Memento watcher fetch skipped",
                extra={"error": payload.get("error") or payload.get("message") or "unknown"},
            )
            return

        data = payload.get("data")
        if isinstance(data, dict):
            rows = data.get("notifications") or []
        else:
            rows = payload.get("notifications") or []

        if not isinstance(rows, list) or not rows:
            return

        candidates = self._parse_candidates(rows)
        if not candidates:
            return

        if not self._has_seeded_initial_snapshot:
            self._seen_store.mark_seen(candidate.queue_id for candidate in candidates)
            self._has_seeded_initial_snapshot = True
            logger.info(
                "Memento watcher completed initial warmup",
                extra={"skipped_ids": len(candidates)},
            )
            return

        unseen = [candidate for candidate in candidates if not self._seen_store.is_seen(candidate.queue_id)]
        if not unseen:
            return

        surfaced = 0
        processed: List[str] = []
        for candidate in unseen:
            processed.append(candidate.queue_id)
            classified = await classify_memento_importance(candidate)
            if not classified:
                continue
            surfaced += 1
            await self._dispatch_summary(candidate, classified)

        if processed:
            self._seen_store.mark_seen(processed)

        logger.info(
            "Memento watcher check complete",
            extra={"reviewed": len(unseen), "surfaced": surfaced},
        )

    def _parse_candidates(self, rows: List[Dict[str, object]]) -> List[MementoCandidate]:
        parsed: List[MementoCandidate] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            queue_id = str(row.get("queue_id") or "").strip()
            if not queue_id:
                continue

            queued_at = None
            raw_time = str(row.get("queued_at") or "").strip()
            if raw_time:
                try:
                    queued_at = datetime.fromisoformat(raw_time.replace("Z", "+00:00"))
                    if queued_at.tzinfo is None:
                        queued_at = queued_at.replace(tzinfo=UTC)
                except ValueError:
                    queued_at = None

            parsed.append(
                MementoCandidate(
                    queue_id=queue_id,
                    note_id=str(row.get("note_id") or "").strip(),
                    recipient_email=str(row.get("recipient_email") or "").strip(),
                    sender_name=str(row.get("sender_name") or "").strip(),
                    sender_id=int(row.get("sender_id") or 0),
                    queued_at=queued_at,
                    note_excerpt=str(row.get("note_excerpt") or "").strip(),
                )
            )

        return parsed

    async def _dispatch_summary(self, candidate: MementoCandidate, classified: Dict[str, str]) -> None:
        recipient_email = candidate.recipient_email or "unknown"
        session_key = f"memento:{recipient_email}"
        runtime = _resolve_interaction_runtime(session_key=session_key)
        category = classified.get("category", "other")
        summary = classified.get("summary", "")

        message = (
            "Memento watcher notification:\n"
            f"Recipient: {recipient_email}\n"
            f"Category: {category}\n"
            f"Summary: {summary}"
        )

        try:
            await runtime.handle_agent_message(message)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(
                "Failed to dispatch memento summary",
                extra={"queue_id": candidate.queue_id, "error": str(exc)},
            )


_watcher_instance: Optional[MementoWatcher] = None


def get_memento_watcher() -> MementoWatcher:
    global _watcher_instance
    if _watcher_instance is None:
        settings = get_settings()
        _watcher_instance = MementoWatcher(
            poll_interval_seconds=float(settings.genzbuzz_memento_watcher_poll_interval_seconds),
            lookback_minutes=int(settings.genzbuzz_memento_watcher_lookback_minutes),
        )
    return _watcher_instance


__all__ = ["MementoWatcher", "get_memento_watcher"]
