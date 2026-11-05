from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from apps.api_gateway.auth import require_matching_user, require_user
from apps.api_gateway.config.setting import settings
from services.auth.identity import AuthenticatedUser
from services.daily_briefing.force import force_generate_daily_briefing
from services.daily_briefing.store import DailyBriefingStore
from services.db.mongo import get_database

router = APIRouter(dependencies=[Depends(require_user)])


class ForceGenerateBody(BaseModel):
    userId: str = Field(min_length=1)
    date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    # yesterday = same day the midnight scheduler would use
    # today = manual testing only; a READY "today" briefing blocks the midnight run for that day
    period: str = Field(default="yesterday", pattern=r"^(today|yesterday)$")


@router.get("")
async def get_daily_briefing(
    actor: AuthenticatedUser = Depends(require_user),
    user_id: str = Query(..., alias="userId", min_length=1),
    date: str | None = Query(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
):
    verified_user_id = require_matching_user(actor, user_id)
    store = DailyBriefingStore(get_database())
    document = await store.get(verified_user_id, date) if date else await store.get_latest(verified_user_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Daily briefing not found.")
    document.pop("missedCandidates", None)
    document.pop("error", None)
    document.pop("claimedBy", None)
    document.pop("claimedAt", None)
    document["_id"] = str(document.get("_id") or "")
    return {"briefing": document}


@router.post("/force-generate")
async def force_generate_daily_briefing_route(
    body: ForceGenerateBody,
    actor: AuthenticatedUser = Depends(require_user),
):
    """
    Manual regeneration for the signed-in user.
    Node forwards that user's verified bearer token. A session-only Node request
    mints a short-lived token for the same verified user and does not return it.
    The midnight scheduler does not call this route. It enqueues Redis events in-process
    and must not present an end-user token or a queue credential here.
    """
    if not settings.DAILY_BRIEFING_ALLOW_FORCE_GENERATE:
        raise HTTPException(status_code=403, detail="Force generate is disabled.")
    user_id = require_matching_user(actor, body.userId)
    try:
        result = await force_generate_daily_briefing(
            get_database(),
            user_id=user_id,
            date_key=body.date,
            period=body.period,
        )
    except PermissionError as error:
        raise HTTPException(status_code=403, detail=str(error)) from error
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail="Force generate failed.") from error
    return {"success": True, "result": result}
