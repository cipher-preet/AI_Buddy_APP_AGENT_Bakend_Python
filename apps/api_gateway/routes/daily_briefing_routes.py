from fastapi import APIRouter, HTTPException, Query

from services.daily_briefing.store import DailyBriefingStore
from services.db.mongo import get_database

router = APIRouter()


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
