from __future__ import annotations

from pydantic import BaseModel, Field
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ..services.genzbuzz.messenger_ingress import route_messenger_ingress


router = APIRouter(prefix="/messenger", tags=["messenger"])


class MessengerIngressPayload(BaseModel):
    psid: str = Field(min_length=1)
    text: str = Field(min_length=1)
    user_id: int | None = Field(default=None, ge=1)


@router.post("/ingress", response_class=JSONResponse, summary="OpenPoke-first Messenger onboarding ingress")
async def messenger_ingress(payload: MessengerIngressPayload) -> JSONResponse:
    result = await route_messenger_ingress(psid=payload.psid, text=payload.text, user_id=payload.user_id)
    return JSONResponse({"ok": result.success, **result.to_dict()})


__all__ = ["router"]
