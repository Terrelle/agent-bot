"""Background scheduler for GenZbuzz due retry processing."""

from __future__ import annotations

import asyncio
from typing import Optional

from ..agents.execution_agent.tools.genzbuzz import build_registry
from ..config import get_settings
from ..logging_config import logger


class GenZbuzzRetryScheduler:
    """Poll due retry jobs and process them in bounded batches."""

    def __init__(self, poll_interval_seconds: Optional[float] = None, batch_limit: Optional[int] = None) -> None:
        settings = get_settings()
        self._poll_interval = float(
            poll_interval_seconds
            if poll_interval_seconds is not None
            else settings.genzbuzz_retry_poll_interval_seconds
        )
        configured_limit = int(batch_limit if batch_limit is not None else settings.genzbuzz_retry_batch_limit)
        self._batch_limit = max(1, min(configured_limit, 200))
        self._task: Optional[asyncio.Task[None]] = None
        self._running = False
        self._lock = asyncio.Lock()
        self._tool = build_registry("genzbuzz_retry_scheduler").get("genzbuzz_process_due_retries")

    async def start(self) -> None:
        async with self._lock:
            if self._task and not self._task.done():
                return
            if self._tool is None:
                logger.warning("GenZbuzz retry scheduler disabled; tool not found")
                return
            loop = asyncio.get_running_loop()
            self._running = True
            self._task = loop.create_task(self._run(), name="genzbuzz-retry-scheduler")
            logger.info(
                "GenZbuzz retry scheduler started",
                extra={
                    "interval_seconds": self._poll_interval,
                    "batch_limit": self._batch_limit,
                },
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
                self._task = None
                logger.info("GenZbuzz retry scheduler stopped")

    async def _run(self) -> None:
        try:
            while self._running:
                await self._poll_once()
                await asyncio.sleep(self._poll_interval)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("GenZbuzz retry scheduler loop crashed", extra={"error": str(exc)})

    async def _poll_once(self) -> None:
        if self._tool is None:
            return

        try:
            result = self._tool(limit=self._batch_limit)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("GenZbuzz retry scheduler poll failed", extra={"error": str(exc)})
            return

        if not isinstance(result, dict):
            return

        processed_count = int(result.get("processed_count", 0) or 0)
        due_count = int(result.get("due_count", 0) or 0)
        if due_count > 0 or processed_count > 0:
            logger.info(
                "GenZbuzz retry scheduler processed batch",
                extra={
                    "due_count": due_count,
                    "processed_count": processed_count,
                },
            )


_scheduler_instance: Optional[GenZbuzzRetryScheduler] = None


def get_genzbuzz_retry_scheduler() -> GenZbuzzRetryScheduler:
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = GenZbuzzRetryScheduler()
    return _scheduler_instance


__all__ = ["GenZbuzzRetryScheduler", "get_genzbuzz_retry_scheduler"]
