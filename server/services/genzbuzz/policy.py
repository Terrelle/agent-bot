"""GenZbuzz send policy helpers.

This module enforces daytime-only delivery windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo


UTC = timezone.utc


@dataclass(frozen=True)
class DaytimeWindow:
    """Daytime send window in local user time."""

    start_hour: int
    end_hour: int


@dataclass(frozen=True)
class SendPolicyResult:
    """Evaluation result for a proposed send time."""

    should_send_now: bool
    scheduled_at_utc: datetime
    scheduled_at_local: datetime
    timezone_name: str


def _validate_window(window: DaytimeWindow) -> None:
    if not (0 <= window.start_hour <= 23):
        raise ValueError("start_hour must be between 0 and 23")
    if not (1 <= window.end_hour <= 24):
        raise ValueError("end_hour must be between 1 and 24")
    if window.start_hour >= window.end_hour:
        raise ValueError("start_hour must be earlier than end_hour")


def _next_window_open(local_now: datetime, window: DaytimeWindow) -> datetime:
    local_open_today = local_now.replace(
        hour=window.start_hour,
        minute=0,
        second=0,
        microsecond=0,
    )
    local_close_today = local_now.replace(
        hour=window.end_hour % 24,
        minute=0,
        second=0,
        microsecond=0,
    )

    if window.end_hour == 24:
        local_close_today = local_close_today + timedelta(days=1)

    if local_now < local_open_today:
        return local_open_today

    if local_open_today <= local_now < local_close_today:
        return local_now

    return local_open_today + timedelta(days=1)


def evaluate_daytime_policy(
    *,
    ready_at_utc: Optional[datetime] = None,
    timezone_name: str = "America/New_York",
    window: Optional[DaytimeWindow] = None,
) -> SendPolicyResult:
    """Return whether a message can send now or when it should be queued."""

    active_window = window or DaytimeWindow(start_hour=10, end_hour=20)
    _validate_window(active_window)

    now_utc = ready_at_utc or datetime.now(UTC)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=UTC)
    else:
        now_utc = now_utc.astimezone(UTC)

    tz = ZoneInfo(timezone_name)
    local_now = now_utc.astimezone(tz)
    local_scheduled = _next_window_open(local_now, active_window)
    scheduled_utc = local_scheduled.astimezone(UTC)

    return SendPolicyResult(
        should_send_now=(local_scheduled == local_now),
        scheduled_at_utc=scheduled_utc,
        scheduled_at_local=local_scheduled,
        timezone_name=timezone_name,
    )


__all__ = [
    "DaytimeWindow",
    "SendPolicyResult",
    "evaluate_daytime_policy",
]
