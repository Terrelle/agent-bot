"""WordPress bridge for GenZbuzz conversational actions."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import httpx

from ...config import get_settings


class GenZbuzzBridgeService:
    """Lightweight client for GenZbuzz WordPress AJAX actions."""

    def __init__(self) -> None:
        settings = get_settings()
        self._ajax_url = settings.genzbuzz_wp_ajax_url
        self._referer = settings.genzbuzz_wp_referer
        self._timeout = settings.genzbuzz_wp_timeout_seconds

    def get_active_waiting_cycle(self, *, user_id: int) -> Dict[str, Any]:
        return self._post_form(
            {
                "action": "get_active_waiting_cycle",
                "user_id": str(user_id),
            }
        )

    def get_waiting_prompt_context(self, *, user_id: int) -> Dict[str, Any]:
        return self._post_form(
            {
                "action": "get_waiting_prompt_context",
                "user_id": str(user_id),
            }
        )

    def submit_waiting_reply(
        self,
        *,
        user_id: int,
        text_answer: Optional[str] = None,
        audio_file: Optional[tuple[str, bytes, str]] = None,
        photo_file: Optional[tuple[str, bytes, str]] = None,
    ) -> Dict[str, Any]:
        files: Dict[str, tuple[str, bytes, str]] = {}
        if audio_file is not None:
            files["audioFile"] = audio_file
            action = "receive_imessage_waiting_voice_update"
            data = {
                "action": action,
                "user_id": str(user_id),
            }
        else:
            action = "receive_imessage_waiting_text_update"
            data = {
                "action": action,
                "user_id": str(user_id),
                "text_answer": (text_answer or ""),
            }

        if photo_file is not None:
            files["photoFile"] = photo_file

        return self._post_form(data, files=files or None)

    def submit_bonding_reply(
        self,
        *,
        user_id: int,
        asker_id: int,
        text_answer: Optional[str] = None,
        audio_file: Optional[tuple[str, bytes, str]] = None,
        photo_file: Optional[tuple[str, bytes, str]] = None,
    ) -> Dict[str, Any]:
        files: Dict[str, tuple[str, bytes, str]] = {}
        if audio_file is not None:
            files["audioFile"] = audio_file
            action = "receive_imessage_voice_update"
            data = {
                "action": action,
                "user_id": str(user_id),
                "asker_id": str(asker_id),
            }
        else:
            action = "receive_imessage_text_update"
            data = {
                "action": action,
                "user_id": str(user_id),
                "asker_id": str(asker_id),
                "text_answer": (text_answer or ""),
            }

        if photo_file is not None:
            files["photoFile"] = photo_file

        return self._post_form(data, files=files or None)

    def submit_spontaneous_memento(
        self,
        *,
        user_id: int,
        recipient_id: int,
        audio_file: tuple[str, bytes, str],
        photo_file: Optional[tuple[str, bytes, str]] = None,
    ) -> Dict[str, Any]:
        files: Dict[str, tuple[str, bytes, str]] = {
            "audioFile": audio_file,
        }
        if photo_file is not None:
            files["photoFile"] = photo_file

        data = {
            "action": "receive_spontaneous_memento_batch",
            "user_id": str(user_id),
            "recipient_id": str(recipient_id),
        }
        return self._post_form(data, files=files)

    def lookup_messenger_user(self, *, psid: str) -> Dict[str, Any]:
        return self._post_form(
            {
                "action": "openpoke_messenger_lookup_user",
                "psid": str(psid or ""),
            }
        )

    def messenger_verify(self, *, psid: str, user_id: int) -> Dict[str, Any]:
        return self._post_form(
            {
                "action": "openpoke_messenger_verify",
                "psid": str(psid or ""),
                "user_id": str(int(user_id)),
            }
        )

    def messenger_new_friend(self, *, psid: str, user_id: int, state_only: bool = False) -> Dict[str, Any]:
        payload = {
            "action": "openpoke_messenger_new_friend",
            "psid": str(psid or ""),
            "user_id": str(int(user_id)),
        }
        if state_only:
            payload["state_only"] = "1"
        return self._post_form(payload)

    def messenger_onboarding_state(self, *, psid: str, user_id: int) -> Dict[str, Any]:
        return self._post_form(
            {
                "action": "openpoke_messenger_onboarding_state",
                "psid": str(psid or ""),
                "user_id": str(int(user_id)),
            }
        )

    def messenger_submit_frequency(
        self,
        *,
        psid: str,
        user_id: int,
        text: str,
        state_only: bool = False,
    ) -> Dict[str, Any]:
        payload = {
            "action": "openpoke_messenger_frequency",
            "psid": str(psid or ""),
            "user_id": str(int(user_id)),
            "text": str(text or ""),
        }
        if state_only:
            payload["state_only"] = "1"
        return self._post_form(payload)

    def messenger_who_status(self, *, psid: str, user_id: int) -> Dict[str, Any]:
        return self._post_form(
            {
                "action": "openpoke_messenger_who",
                "psid": str(psid or ""),
                "user_id": str(int(user_id)),
            }
        )

    def messenger_reply_to(self, *, psid: str, user_id: int, text: str) -> Dict[str, Any]:
        return self._post_form(
            {
                "action": "openpoke_messenger_reply_to",
                "psid": str(psid or ""),
                "user_id": str(int(user_id)),
                "text": str(text or ""),
            }
        )

    def get_recent_memento_notifications(
        self,
        *,
        since_iso: Optional[str] = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        payload = {
            "action": "openpoke_memento_notifications",
            "limit": str(max(1, min(int(limit), 200))),
        }
        if isinstance(since_iso, str) and since_iso.strip():
            payload["since_iso"] = since_iso.strip()
        return self._post_form(payload)

    def _post_form(
        self,
        data: Dict[str, str],
        *,
        files: Optional[Dict[str, tuple[str, bytes, str]]] = None,
    ) -> Dict[str, Any]:
        headers = {
            "User-Agent": "OpenPoke-GenZbuzz/1.0",
            "Referer": self._referer,
        }

        try:
            response = httpx.post(
                self._ajax_url,
                data=data,
                files=files,
                headers=headers,
                timeout=self._timeout,
            )
        except Exception as exc:
            return {
                "ok": False,
                "success": False,
                "error": f"request_failed: {exc}",
            }

        try:
            payload = response.json()
        except Exception:
            raw_text = response.text or ""
            recovered_payload: Optional[Dict[str, Any]] = None

            # WordPress/PHP may prepend notices before a JSON response.
            # Attempt to recover the JSON object from the raw body.
            start = raw_text.find("{")
            end = raw_text.rfind("}")
            if start != -1 and end != -1 and end > start:
                candidate = raw_text[start:end + 1]
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        recovered_payload = parsed
                        recovered_payload.setdefault("json_recovered", True)
                except Exception:
                    recovered_payload = None

            if recovered_payload is not None:
                payload = recovered_payload
            else:
                payload = {
                    "success": False,
                    "message": "invalid_json",
                    "raw": raw_text,
                }

        if isinstance(payload, dict):
            payload.setdefault("ok", bool(payload.get("success")))
            payload.setdefault("http_status", response.status_code)
            retry_after = response.headers.get("Retry-After")
            if retry_after is not None:
                payload.setdefault("retry_after", str(retry_after).strip())
            return payload

        return {
            "ok": False,
            "success": False,
            "http_status": response.status_code,
            "error": "invalid_payload_type",
        }


_genzbuzz_bridge_service: Optional[GenZbuzzBridgeService] = None


def get_genzbuzz_bridge_service() -> GenZbuzzBridgeService:
    global _genzbuzz_bridge_service
    if _genzbuzz_bridge_service is None:
        _genzbuzz_bridge_service = GenZbuzzBridgeService()
    return _genzbuzz_bridge_service


__all__ = ["GenZbuzzBridgeService", "get_genzbuzz_bridge_service"]
