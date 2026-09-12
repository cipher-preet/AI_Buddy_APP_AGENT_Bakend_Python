from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from apps.api_gateway.config.setting import settings
from services.daily_briefing.force import force_generate_daily_briefing
from services.daily_briefing.store import DailyBriefingStore
from services.db.mongo import get_database

router = APIRouter()


class ForceGenerateBody(BaseModel):
    userId: str = Field(min_length=1)
    date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    # today = best for manual testing with recent activity
    # yesterday = same day the midnight scheduler would use
    period: str = Field(default="today", pattern=r"^(today|yesterday)$")


@router.get("")
async def get_daily_briefing(
    user_id: str = Query(..., alias="userId", min_length=1),
    date: str | None = Query(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
):
    store = DailyBriefingStore(get_database())
    document = await store.get(user_id, date) if date else await store.get_latest(user_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Daily briefing not found.")
    document.pop("missedCandidates", None)
    document.pop("error", None)
    document.pop("claimedBy", None)
    document.pop("claimedAt", None)
    document["_id"] = str(document.get("_id") or "")
    return {"briefing": document}


@router.post("/force-generate")
async def force_generate_daily_briefing_route(body: ForceGenerateBody):
    """
    TEMPORARY test endpoint used by the app "Generate now" button.
    Runs the full pipeline for the signed-in user (via Node proxy).
    """
    if not settings.DAILY_BRIEFING_ALLOW_FORCE_GENERATE:
        raise HTTPException(status_code=403, detail="Force generate is disabled.")
    try:
        result = await force_generate_daily_briefing(
            get_database(),
            user_id=body.userId,
            date_key=body.date,
            period=body.period,
        )
    except PermissionError as error:
        raise HTTPException(status_code=403, detail=str(error)) from error
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Force generate failed: {type(error).__name__}: {error}",
        ) from error
    return {"success": True, "result": result}
