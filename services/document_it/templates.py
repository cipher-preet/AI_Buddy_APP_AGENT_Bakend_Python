from __future__ import annotations

from typing import Any

from services.document_it.schemas import TemplateSpec, get_template_spec


async def load_template_spec(database, template_code: str) -> TemplateSpec:
    """Prefer Mongo catalog metadata; fall back to built-in professional specs."""
    code = str(template_code or "").strip().lower()
    spec = get_template_spec(code)
    if not code:
        return spec

    doc: dict[str, Any] | None = None
    for collection_name in ("documenttemplates", "documentTemplates"):
        collection = database[collection_name]
        doc = await collection.find_one({"code": code, "isActive": {"$ne": False}})
        if doc:
            break

    if not doc:
        return spec

    title = str(doc.get("title") or spec.title)
    return TemplateSpec(
        code=code,
        title=title,
        tagline=str(doc.get("tagline") or spec.tagline),
        description=str(doc.get("description") or spec.description),
        requiredSections=spec.requiredSections,
        guidance=spec.guidance,
        tableHints=spec.tableHints,
    )
