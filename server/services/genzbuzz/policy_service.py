"""GenZbuzz conversational send policy and confirmation state management."""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from zoneinfo import ZoneInfo


UTC = timezone.utc


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _to_iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _from_iso(raw: str) -> datetime:
    normalized = (raw or "").strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


@dataclass
class ConfirmationRecord:
    confirmation_id: str
    agent_name: str
    channel: str
    message_kind: str
    message_preview: str
    prompt_text: str
    metadata: Dict[str, Any]
    status: str
    created_at: str
    expires_at: str
    resolved_at: Optional[str] = None
    resolution: Optional[str] = None
    resolution_note: Optional[str] = None


@dataclass
class DraftRecord:
    draft_id: str
    agent_name: str
    channel: str
    draft_kind: str
    payload: Dict[str, Any]
    title: str
    metadata: Dict[str, Any]
    status: str
    version: int
    created_at: str
    updated_at: str
    sent_at: Optional[str] = None
    archived_at: Optional[str] = None


class GenZbuzzPolicyService:
    """OpenPoke-native policy service for send windows, confirmations, and retry escalation."""

    def __init__(
        self,
        *,
        path: Path,
        timezone_name: str = "America/New_York",
        start_hour: int = 10,
        end_hour: int = 20,
        max_retries: int = 3,
        default_confirmation_prompt: str = "Ready to send this now?",
    ) -> None:
        self._path = path
        self._timezone = ZoneInfo(timezone_name)
        self._start_hour = start_hour
        self._end_hour = end_hour
        self._max_retries = max_retries
        self._default_confirmation_prompt = default_confirmation_prompt.strip() or "Ready to send this now?"
        self._lock = threading.Lock()
        self._state: Dict[str, Any] = {
            "confirmations": {},
            "tasks": {},
            "retry_jobs": {},
            "drafts": {},
        }
        self._load()

    def create_draft(
        self,
        *,
        agent_name: str,
        channel: str,
        draft_kind: str,
        payload: Dict[str, Any],
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        now_iso = _to_iso(_utc_now())
        draft_id = str(uuid4())
        normalized_kind = (draft_kind or "").strip()
        normalized_channel = (channel or "imessage").strip() or "imessage"

        record = DraftRecord(
            draft_id=draft_id,
            agent_name=(agent_name or "").strip(),
            channel=normalized_channel,
            draft_kind=normalized_kind,
            payload=payload if isinstance(payload, dict) else {},
            title=(title or "").strip(),
            metadata=metadata if isinstance(metadata, dict) else {},
            status="draft",
            version=1,
            created_at=now_iso,
            updated_at=now_iso,
        )

        with self._lock:
            self._state["drafts"][draft_id] = asdict(record)
            self._save_locked()

        return {
            "draft_id": draft_id,
            "status": record.status,
            "version": record.version,
            "channel": record.channel,
            "draft_kind": record.draft_kind,
            "title": record.title,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
        }

    def get_draft(self, *, draft_id: str) -> Dict[str, Any]:
        key = (draft_id or "").strip()
        if not key:
            return {"error": "draft_id is required"}

        with self._lock:
            record = self._state.get("drafts", {}).get(key)
            if not isinstance(record, dict):
                return {"error": f"Unknown draft_id: {key}"}
            return dict(record)

    def get_latest_user_draft(
        self,
        *,
        draft_kind: str,
        user_id: int,
        statuses: Optional[tuple[str, ...]] = None,
    ) -> Dict[str, Any]:
        if user_id <= 0:
            return {"error": "invalid_user_id"}

        normalized_kind = (draft_kind or "").strip().lower()
        if normalized_kind == "":
            return {"error": "draft_kind is required"}

        accepted_statuses = tuple(s.strip().lower() for s in (statuses or ("draft", "ready")) if s)
        latest: Optional[Dict[str, Any]] = None
        latest_dt: Optional[datetime] = None

        with self._lock:
            drafts = self._state.get("drafts", {})
            if not isinstance(drafts, dict):
                return {"has_draft": False}

            for draft_id, record in drafts.items():
                if not isinstance(record, dict):
                    continue
                status = str(record.get("status") or "").strip().lower()
                if accepted_statuses and status not in accepted_statuses:
                    continue

                record_kind = str(record.get("draft_kind") or "").strip().lower()
                if record_kind != normalized_kind:
                    continue

                payload = record.get("payload") if isinstance(record.get("payload"), dict) else {}
                payload_user_id = int(payload.get("user_id") or 0)
                if payload_user_id != user_id:
                    continue

                updated_at = str(record.get("updated_at") or record.get("created_at") or "").strip()
                try:
                    updated_dt = _from_iso(updated_at) if updated_at else _utc_now()
                except Exception:
                    updated_dt = _utc_now()

                if latest_dt is None or updated_dt > latest_dt:
                    latest_dt = updated_dt
                    latest = {
                        "draft_id": str(draft_id),
                        "status": status,
                        "updated_at": updated_at,
                    }

        if latest is None:
            return {"has_draft": False}

        age_seconds = max(0, int((_utc_now() - (latest_dt or _utc_now())).total_seconds()))
        return {
            "has_draft": True,
            "draft_id": latest.get("draft_id", ""),
            "status": latest.get("status", "unknown"),
            "updated_at": latest.get("updated_at", ""),
            "age_seconds": age_seconds,
        }

    def update_draft(
        self,
        *,
        draft_id: str,
        payload: Optional[Dict[str, Any]] = None,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        expected_version: Optional[int] = None,
    ) -> Dict[str, Any]:
        key = (draft_id or "").strip()
        if not key:
            return {"error": "draft_id is required"}

        with self._lock:
            record = self._state.get("drafts", {}).get(key)
            if not isinstance(record, dict):
                return {"error": f"Unknown draft_id: {key}"}

            if expected_version is not None and int(record.get("version", 0)) != int(expected_version):
                return {
                    "error": "version_conflict",
                    "draft_id": key,
                    "current_version": int(record.get("version", 0)),
                }

            current_status = str(record.get("status", "draft"))
            if current_status not in {"draft", "ready"}:
                return {
                    "error": "draft_not_editable",
                    "draft_id": key,
                    "status": current_status,
                }

            if payload is not None:
                record["payload"] = payload if isinstance(payload, dict) else {}
            if title is not None:
                record["title"] = str(title).strip()
            if metadata is not None:
                record["metadata"] = metadata if isinstance(metadata, dict) else {}

            record["version"] = int(record.get("version", 1)) + 1
            record["status"] = "draft"
            record["updated_at"] = _to_iso(_utc_now())
            self._save_locked()

            return {
                "draft_id": key,
                "status": record["status"],
                "version": record["version"],
                "updated_at": record["updated_at"],
            }

    def mark_draft_status(
        self,
        *,
        draft_id: str,
        status: str,
    ) -> Dict[str, Any]:
        key = (draft_id or "").strip()
        normalized = (status or "").strip().lower()
        if not key:
            return {"error": "draft_id is required"}
        if normalized not in {"ready", "sent", "archived", "draft"}:
            return {"error": f"Unsupported draft status: {status}"}

        now_iso = _to_iso(_utc_now())
        with self._lock:
            record = self._state.get("drafts", {}).get(key)
            if not isinstance(record, dict):
                return {"error": f"Unknown draft_id: {key}"}

            record["status"] = normalized
            record["updated_at"] = now_iso
            if normalized == "sent":
                record["sent_at"] = now_iso
            if normalized == "archived":
                record["archived_at"] = now_iso
            self._save_locked()

            return {
                "draft_id": key,
                "status": record["status"],
                "version": int(record.get("version", 1)),
                "updated_at": record["updated_at"],
                "sent_at": record.get("sent_at"),
                "archived_at": record.get("archived_at"),
            }

    def evaluate_send_window(self, *, now_utc: Optional[datetime] = None) -> Dict[str, Any]:
        now = (now_utc or _utc_now()).astimezone(UTC)
        local_now = now.astimezone(self._timezone)

        in_window = self._start_hour <= local_now.hour < self._end_hour
        if in_window:
            send_at_local = local_now
        else:
            send_day = local_now.date()
            if local_now.hour >= self._end_hour:
                send_day = send_day + timedelta(days=1)
            send_at_local = datetime(
                send_day.year,
                send_day.month,
                send_day.day,
                self._start_hour,
                0,
                0,
                tzinfo=self._timezone,
            )

        return {
            "timezone": self._timezone.key,
            "window_start_hour": self._start_hour,
            "window_end_hour": self._end_hour,
            "is_within_window": in_window,
            "evaluated_at": _to_iso(now),
            "evaluated_at_local": send_at_local.isoformat(timespec="seconds") if in_window else local_now.isoformat(timespec="seconds"),
            "recommended_send_at": _to_iso(send_at_local.astimezone(UTC)),
        }

    def create_confirmation(
        self,
        *,
        agent_name: str,
        channel: str,
        message_kind: str,
        message_preview: str,
        metadata: Optional[Dict[str, Any]] = None,
        prompt_text: Optional[str] = None,
        ttl_minutes: int = 120,
    ) -> Dict[str, Any]:
        now = _utc_now()
        confirmation_id = str(uuid4())
        prompt = (prompt_text or self._default_confirmation_prompt).strip()
        expires_at = now + timedelta(minutes=max(1, ttl_minutes))

        record = ConfirmationRecord(
            confirmation_id=confirmation_id,
            agent_name=agent_name,
            channel=(channel or "imessage").strip() or "imessage",
            message_kind=(message_kind or "general").strip() or "general",
            message_preview=(message_preview or "").strip(),
            prompt_text=prompt,
            metadata=metadata or {},
            status="pending",
            created_at=_to_iso(now),
            expires_at=_to_iso(expires_at),
        )

        with self._lock:
            self._state["confirmations"][confirmation_id] = asdict(record)
            self._save_locked()

        return {
            "confirmation_id": confirmation_id,
            "status": "pending",
            "prompt_text": prompt,
            "expires_at": record.expires_at,
        }

    def resolve_confirmation(self, *, confirmation_id: str, user_reply: str) -> Dict[str, Any]:
        decision = self._classify_confirmation_reply(user_reply)
        now_iso = _to_iso(_utc_now())

        with self._lock:
            record = self._state["confirmations"].get(confirmation_id)
            if not record:
                return {"error": f"Unknown confirmation_id: {confirmation_id}"}

            if record.get("status") != "pending":
                return {
                    "confirmation_id": confirmation_id,
                    "status": record.get("status"),
                    "resolution": record.get("resolution"),
                    "already_resolved": True,
                }

            expires_at = _from_iso(record.get("expires_at", ""))
            if expires_at < _utc_now():
                record["status"] = "expired"
                record["resolved_at"] = now_iso
                record["resolution"] = "expired"
                record["resolution_note"] = "No action taken before expiry"
                self._save_locked()
                return {
                    "confirmation_id": confirmation_id,
                    "status": "expired",
                    "resolution": "expired",
                }

            if decision == "confirm":
                record["status"] = "confirmed"
            elif decision == "cancel":
                record["status"] = "cancelled"
            elif decision == "edit":
                record["status"] = "needs_edit"
            else:
                return {
                    "confirmation_id": confirmation_id,
                    "status": "pending",
                    "resolution": "unknown",
                    "message": "Reply unclear; ask user to confirm, cancel, or edit.",
                }

            record["resolved_at"] = now_iso
            record["resolution"] = decision
            record["resolution_note"] = (user_reply or "").strip()
            self._save_locked()

            return {
                "confirmation_id": confirmation_id,
                "status": record["status"],
                "resolution": decision,
            }

    def get_confirmation(self, *, confirmation_id: str) -> Dict[str, Any]:
        """Return confirmation state for execution-time gating."""
        now_iso = _to_iso(_utc_now())
        with self._lock:
            record = self._state["confirmations"].get(confirmation_id)
            if not record:
                return {"error": f"Unknown confirmation_id: {confirmation_id}"}

            if record.get("status") == "pending":
                expires_at = _from_iso(record.get("expires_at", ""))
                if expires_at < _utc_now():
                    record["status"] = "expired"
                    record["resolved_at"] = now_iso
                    record["resolution"] = "expired"
                    record["resolution_note"] = "No action taken before expiry"
                    self._save_locked()

            return {
                "confirmation_id": confirmation_id,
                "status": record.get("status", "unknown"),
                "channel": record.get("channel", ""),
                "message_kind": record.get("message_kind", ""),
                "message_preview": record.get("message_preview", ""),
                "metadata": record.get("metadata", {}) if isinstance(record.get("metadata"), dict) else {},
                "created_at": record.get("created_at"),
                "expires_at": record.get("expires_at"),
                "resolved_at": record.get("resolved_at"),
                "resolution": record.get("resolution"),
            }

    def record_delivery_attempt(
        self,
        *,
        task_id: str,
        success: bool,
        error_message: Optional[str] = None,
        terminal_failure: bool = False,
    ) -> Dict[str, Any]:
        task_key = (task_id or "").strip()
        if not task_key:
            return {"error": "task_id is required"}

        now_iso = _to_iso(_utc_now())
        with self._lock:
            tasks = self._state["tasks"]
            task = tasks.get(task_key) or {
                "task_id": task_key,
                "attempt_count": 0,
                "max_retries": self._max_retries,
                "status": "pending",
                "last_error": None,
                "attempts": [],
            }

            task["attempt_count"] += 1
            task["attempts"].append(
                {
                    "attempt_number": task["attempt_count"],
                    "success": bool(success),
                    "error_message": (error_message or "").strip() or None,
                    "at": now_iso,
                }
            )

            if success:
                task["status"] = "sent"
                task["last_error"] = None
                escalation = None
                self._state["retry_jobs"].pop(task_key, None)
            else:
                task["status"] = "failed_attempt"
                task["last_error"] = (error_message or "Delivery failed").strip()
                if terminal_failure:
                    task["status"] = "failed_permanent"
                    escalation = {
                        "alert_user": True,
                        "alert_admin": True,
                        "reason": "Delivery failed with non-retriable error",
                        "task_log_required": True,
                    }
                    self._state["retry_jobs"].pop(task_key, None)
                elif task["attempt_count"] >= self._max_retries:
                    task["status"] = "failed_permanent"
                    escalation = {
                        "alert_user": True,
                        "alert_admin": True,
                        "reason": f"Delivery failed after {self._max_retries} attempts",
                        "task_log_required": True,
                    }
                    self._state["retry_jobs"].pop(task_key, None)
                else:
                    escalation = None

            tasks[task_key] = task
            self._save_locked()

        return {
            "task_id": task_key,
            "attempt_count": task["attempt_count"],
            "max_retries": self._max_retries,
            "status": task["status"],
            "last_error": task["last_error"],
            "escalation": escalation,
        }

    def schedule_retry(
        self,
        *,
        task_id: str,
        action_name: str,
        action_payload: Dict[str, Any],
        attempt_count: int,
        last_error: Optional[str] = None,
        retry_hint: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        task_key = (task_id or "").strip()
        if not task_key:
            return {"error": "task_id is required"}
        if attempt_count >= self._max_retries:
            return {
                "task_id": task_key,
                "scheduled": False,
                "reason": "max_retries_reached",
            }

        delay_seconds = self._retry_delay_seconds(attempt_count, retry_hint=retry_hint)
        next_retry = _utc_now() + timedelta(seconds=delay_seconds)
        hint = retry_hint if isinstance(retry_hint, dict) else {}
        job = {
            "task_id": task_key,
            "action_name": (action_name or "").strip(),
            "action_payload": action_payload if isinstance(action_payload, dict) else {},
            "attempt_count": int(attempt_count),
            "next_retry_at": _to_iso(next_retry),
            "last_error": (last_error or "").strip() or None,
            "retry_hint": hint,
            "updated_at": _to_iso(_utc_now()),
        }

        with self._lock:
            self._state["retry_jobs"][task_key] = job
            self._save_locked()

        return {
            "task_id": task_key,
            "scheduled": True,
            "next_retry_at": job["next_retry_at"],
            "delay_seconds": delay_seconds,
            "attempt_count": int(attempt_count),
            "retry_hint": hint,
        }

    def get_due_retries(self, *, limit: int = 20) -> Dict[str, Any]:
        effective_limit = max(1, min(int(limit), 200))
        now = _utc_now()

        with self._lock:
            jobs = []
            for job in self._state.get("retry_jobs", {}).values():
                if not isinstance(job, dict):
                    continue
                next_raw = str(job.get("next_retry_at", "")).strip()
                if not next_raw:
                    continue
                try:
                    next_dt = _from_iso(next_raw)
                except Exception:
                    continue
                if next_dt <= now:
                    jobs.append(job)

            jobs.sort(key=lambda item: str(item.get("next_retry_at", "")))
            jobs = jobs[:effective_limit]

        return {
            "count": len(jobs),
            "jobs": jobs,
            "generated_at": _to_iso(now),
        }

    def clear_retry_job(self, *, task_id: str) -> Dict[str, Any]:
        task_key = (task_id or "").strip()
        if not task_key:
            return {"error": "task_id is required"}

        with self._lock:
            existed = self._state.get("retry_jobs", {}).pop(task_key, None) is not None
            self._save_locked()

        return {
            "task_id": task_key,
            "removed": existed,
        }

    def _retry_delay_seconds(self, attempt_count: int, *, retry_hint: Optional[Dict[str, Any]] = None) -> int:
        # attempt_count refers to the count that just failed.
        # Baseline grows with each attempt; hints can raise or lower delay based on service health.
        hint = retry_hint if isinstance(retry_hint, dict) else {}
        retry_after_seconds = int(hint.get("retry_after_seconds") or 0)
        latency_ms = int(hint.get("latency_ms") or 0)
        service_health = str(hint.get("service_health") or "unknown").strip().lower()

        if retry_after_seconds > 0:
            # Respect explicit server backoff guidance when available.
            return max(5, min(retry_after_seconds, 3600))

        # Exponential baseline with cap.
        baseline = min(3600, 30 * (2 ** max(0, attempt_count - 1)))

        if latency_ms >= 8000:
            baseline = int(baseline * 2.5)
        elif latency_ms >= 4000:
            baseline = int(baseline * 2.0)
        elif latency_ms >= 2000:
            baseline = int(baseline * 1.5)

        if service_health in {"flapping", "degraded"}:
            baseline = int(baseline * 1.5)

        return max(20, min(baseline, 3600))

    def _classify_confirmation_reply(self, raw_reply: str) -> str:
        normalized = (raw_reply or "").strip().lower()
        if not normalized:
            return "unknown"

        confirm_tokens = {"yes", "y", "confirm", "send", "send it", "go ahead", "do it"}
        cancel_tokens = {"no", "n", "cancel", "stop", "dont send", "do not send"}
        edit_tokens = {"edit", "change", "rewrite", "update", "not yet", "fix it"}

        if normalized in confirm_tokens:
            return "confirm"
        if normalized in cancel_tokens:
            return "cancel"
        if normalized in edit_tokens:
            return "edit"
        return "unknown"

    def _load(self) -> None:
        try:
            text = self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return
        except Exception:
            return

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return

        if isinstance(parsed, dict):
            self._state["confirmations"] = parsed.get("confirmations", {}) or {}
            self._state["tasks"] = parsed.get("tasks", {}) or {}
            self._state["retry_jobs"] = parsed.get("retry_jobs", {}) or {}
            self._state["drafts"] = parsed.get("drafts", {}) or {}

    def _save_locked(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._state, indent=2, ensure_ascii=True), encoding="utf-8")


_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_STATE_PATH = _DATA_DIR / "genzbuzz_policy_state.json"

_genzbuzz_policy_service = GenZbuzzPolicyService(path=_STATE_PATH)


def get_genzbuzz_policy_service() -> GenZbuzzPolicyService:
    return _genzbuzz_policy_service


__all__ = ["GenZbuzzPolicyService", "get_genzbuzz_policy_service"]
