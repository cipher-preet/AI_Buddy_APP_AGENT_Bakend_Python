"""Mongo writes that cannot land after a newer attempt has taken the job.

require_active_lease() only proves ownership at the moment it returns. The
write itself has to carry the fence token. A newer token stored on the
document makes an older update match nothing.
"""

from __future__ import annotations

import contextvars
from typing import Any, Awaitable, Callable

from services.queue.job_lease import StaleOwnershipError, current_lease

_before_fenced_write: contextvars.ContextVar[Callable[[], Awaitable[None]] | None] = contextvars.ContextVar(
    "kukunotes_before_fenced_write",
    default=None,
)

FENCE_FIELD = "processingFence"


def push_fenced_write_hook(callback: Callable[[], Awaitable[None]] | None):
    """Test hook that runs after the token is chosen and before the write."""
    return _before_fenced_write.set(callback)


def reset_fenced_write_hook(token: contextvars.Token) -> None:
    _before_fenced_write.reset(token)


def lease_fence_token() -> int | None:
    """None when this call is not inside a leased worker. Raises if the lease is already lost."""
    lease = current_lease()
    if lease is None:
        return None
    if lease.lost or lease.released:
        raise StaleOwnershipError(lease.logical_key, reason=lease.reason or "lease_lost")
    try:
        return int(lease.token)
    except (TypeError, ValueError) as error:
        raise StaleOwnershipError(lease.logical_key, reason="bad_fence") from error


def _fence_clause(token: int) -> dict[str, Any]:
    return {
        "$or": [
            {FENCE_FIELD: {"$exists": False}},
            {FENCE_FIELD: None},
            {FENCE_FIELD: {"$lte": token}},
        ]
    }


def _fenced_filter(base_filter: dict[str, Any], token: int) -> dict[str, Any]:
    return {"$and": [base_filter, _fence_clause(token)]}


def _with_fence(update: dict[str, Any], token: int) -> dict[str, Any]:
    cloned = {key: value for key, value in update.items()}
    set_doc = dict(cloned.get("$set") or {})
    set_doc[FENCE_FIELD] = token
    cloned["$set"] = set_doc
    return cloned


def _document_fence(document: dict[str, Any] | None) -> int:
    if not document:
        return 0
    try:
        return int(document.get(FENCE_FIELD) or 0)
    except (TypeError, ValueError):
        return 0


def _probe_filter(base_filter: dict[str, Any]) -> dict[str, Any]:
    if "_id" in base_filter:
        return {"_id": base_filter["_id"]}
    ident = {
        key: base_filter[key]
        for key in ("userId", "dateKey", "conversationId", "fingerprint")
        if key in base_filter
    }
    return ident or base_filter


async def _reject_if_newer(collection: Any, base_filter: dict[str, Any], token: int) -> None:
    probe = _probe_filter(base_filter)
    existing = await collection.find_one(probe) if hasattr(collection, "find_one") else None
    if existing is not None and _document_fence(existing) > token:
        lease = current_lease()
        raise StaleOwnershipError(
            lease.logical_key if lease is not None else "job",
            reason="stale_fence",
        )


async def fenced_update_one(collection: Any, base_filter: dict[str, Any], update: dict[str, Any], *, upsert: bool = False):
    token = lease_fence_token()
    hook = _before_fenced_write.get()
    if hook is not None:
        await hook()
    if token is None:
        if upsert:
            return await collection.update_one(base_filter, update, upsert=True)
        return await collection.update_one(base_filter, update)
    result = await collection.update_one(_fenced_filter(base_filter, token), _with_fence(update, token))
    if int(getattr(result, "matched_count", 0) or 0) > 0:
        return result
    await _reject_if_newer(collection, base_filter, token)
    if upsert:
        try:
            return await collection.update_one(base_filter, _with_fence(update, token), upsert=True)
        except Exception as error:
            if "duplicate" in type(error).__name__.lower() or "E11000" in str(error):
                await _reject_if_newer(collection, base_filter, token)
            raise
    return result


async def fenced_find_one_and_update(
    collection: Any,
    base_filter: dict[str, Any],
    update: dict[str, Any],
    *,
    upsert: bool = False,
    return_document: Any = None,
):
    token = lease_fence_token()
    hook = _before_fenced_write.get()
    if hook is not None:
        await hook()
    kwargs: dict[str, Any] = {}
    if return_document is not None:
        kwargs["return_document"] = return_document
    if token is None:
        if upsert:
            return await collection.find_one_and_update(base_filter, update, upsert=True, **kwargs)
        return await collection.find_one_and_update(base_filter, update, **kwargs)
    result = await collection.find_one_and_update(
        _fenced_filter(base_filter, token),
        _with_fence(update, token),
        **kwargs,
    )
    if result is not None:
        return result
    await _reject_if_newer(collection, base_filter, token)
    if not upsert:
        return None
    try:
        return await collection.find_one_and_update(base_filter, _with_fence(update, token), upsert=True, **kwargs)
    except Exception as error:
        if "duplicate" in type(error).__name__.lower() or "E11000" in str(error):
            await _reject_if_newer(collection, base_filter, token)
        raise
