from __future__ import annotations


async def force_generate_daily_briefing(
    database,
    *,
    user_id: str,
    date_key: str | None = None,
    period: str = "yesterday",
) -> dict:
    """Daily briefing generation is off. This does not queue a model job."""
    del database, period
    return {
        "userId": user_id,
        "dateKey": date_key or "",
        "status": "SKIPPED",
        "forced": False,
        "message": "Daily briefing is turned off.",
    }
