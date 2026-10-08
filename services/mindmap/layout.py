from __future__ import annotations

import math
from typing import Any

from services.mindmap.schemas import (
    MindmapEdge,
    MindmapEdgeStyle,
    MindmapGraph,
    MindmapMarkerEnd,
    MindmapNode,
    MindmapNodeData,
    MindmapPosition,
    MindmapTone,
)

TONE_STROKES: dict[str, str] = {
    "cyan": "#22d3ee",
    "green": "#22c55e",
    "pink": "#ec4899",
    "purple": "#a855f7",
    "yellow": "#eab308",
    "blue": "#3b82f6",
    "lavender": "#8b5cf6",
    "magenta": "#d946ef",
    "hub": "#0f172a",
}

BRANCH_TONES: list[MindmapTone] = [
    "cyan",
    "green",
    "pink",
    "purple",
    "yellow",
    "blue",
    "lavender",
    "magenta",
]


def _safe_id(raw: str, fallback: str) -> str:
    value = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in str(raw or "").strip())
    value = value.strip("-_") or fallback
    return value[:64]


def graph_from_llm_payload(payload: dict[str, Any], *, space_name: str) -> MindmapGraph:
    """Turn model branch JSON into a React Flow–compatible graph with radial layout."""
    space_title = str(payload.get("spaceTitle") or space_name or "Space").strip() or "Space"
    hub_subtitle = str(payload.get("hubSubtitle") or "Key notes, tasks, and meeting themes").strip()
    branches = payload.get("branches")
    if not isinstance(branches, list):
        branches = []

    hub_id = "hub"
    nodes: list[MindmapNode] = [
        MindmapNode(
            id=hub_id,
            position=MindmapPosition(x=0, y=0),
            data=MindmapNodeData(
                kind="hub",
                title=space_title[:80],
                subtitle=hub_subtitle[:120] if hub_subtitle else None,
                tone="hub",
            ),
        )
    ]
    edges: list[MindmapEdge] = []

    usable_branches = [branch for branch in branches if isinstance(branch, dict)][:8]
    count = max(1, len(usable_branches))
    radius = 360

    for index, branch in enumerate(usable_branches):
        tone = BRANCH_TONES[index % len(BRANCH_TONES)]
        stroke = TONE_STROKES.get(tone, "#64748b")
        branch_id = _safe_id(str(branch.get("id") or f"branch-{index}"), f"branch-{index}")
        angle = (2 * math.pi * index / count) - (math.pi / 2)
        bx = math.cos(angle) * radius
        by = math.sin(angle) * radius
        title = str(branch.get("title") or f"Theme {index + 1}").strip()[:60]
        nodes.append(
            MindmapNode(
                id=branch_id,
                position=MindmapPosition(x=round(bx, 1), y=round(by, 1)),
                data=MindmapNodeData(kind="branch", title=title, tone=tone),
            )
        )
        edges.append(
            MindmapEdge(
                id=f"e-{hub_id}-{branch_id}",
                source=hub_id,
                target=branch_id,
                style=MindmapEdgeStyle(stroke=stroke),
                markerEnd=MindmapMarkerEnd(color=stroke),
                label=str(branch.get("edgeLabel") or "").strip()[:48] or None,
            )
        )

        cards = branch.get("cards")
        if not isinstance(cards, list):
            cards = []
        for card_index, card in enumerate(cards[:5]):
            if not isinstance(card, dict):
                continue
            card_id = _safe_id(
                str(card.get("id") or f"{branch_id}-card-{card_index}"),
                f"{branch_id}-card-{card_index}",
            )
            card_angle = angle + ((card_index - (len(cards[:5]) - 1) / 2) * 0.22)
            cx = math.cos(card_angle) * (radius + 260)
            cy = math.sin(card_angle) * (radius + 260)
            items = card.get("items") if isinstance(card.get("items"), list) else []
            clean_items = [str(item).strip()[:100] for item in items if str(item).strip()][:6]
            tags = card.get("tags") if isinstance(card.get("tags"), list) else []
            clean_tags = [str(tag).strip()[:24] for tag in tags if str(tag).strip()][:4]
            nodes.append(
                MindmapNode(
                    id=card_id,
                    position=MindmapPosition(x=round(cx, 1), y=round(cy, 1)),
                    data=MindmapNodeData(
                        kind="card",
                        title=str(card.get("title") or "Detail").strip()[:70],
                        items=clean_items,
                        tone=tone,
                        tags=clean_tags,
                        variant="compact" if len(clean_items) <= 2 else "default",
                        noteCount=len(clean_items) or None,
                    ),
                )
            )
            edges.append(
                MindmapEdge(
                    id=f"e-{branch_id}-{card_id}",
                    source=branch_id,
                    target=card_id,
                    style=MindmapEdgeStyle(stroke=stroke),
                    markerEnd=MindmapMarkerEnd(color=stroke),
                )
            )

    if len(nodes) == 1:
        # Always give the canvas at least one branch for empty-but-named spaces.
        fallback_id = "branch-overview"
        nodes.append(
            MindmapNode(
                id=fallback_id,
                position=MindmapPosition(x=320, y=0),
                data=MindmapNodeData(kind="branch", title="Overview", tone="cyan"),
            )
        )
        edges.append(
            MindmapEdge(
                id=f"e-{hub_id}-{fallback_id}",
                source=hub_id,
                target=fallback_id,
                style=MindmapEdgeStyle(stroke=TONE_STROKES["cyan"]),
                markerEnd=MindmapMarkerEnd(color=TONE_STROKES["cyan"]),
            )
        )

    return MindmapGraph(nodes=nodes, edges=edges)


def ensure_positions(graph: MindmapGraph) -> MindmapGraph:
    """If the model returned a full graph without positions, apply a simple layout."""
    if not graph.nodes:
        return graph
    missing = any(
        abs(node.position.x) < 0.01 and abs(node.position.y) < 0.01 and node.data.kind != "hub"
        for node in graph.nodes
    )
    if not missing and any(node.data.kind == "hub" for node in graph.nodes):
        return graph

    # Rebuild via synthetic branch payload from existing nodes.
    hub = next((node for node in graph.nodes if node.data.kind == "hub"), graph.nodes[0])
    branches = [node for node in graph.nodes if node.data.kind == "branch"]
    cards_by_branch: dict[str, list[MindmapNode]] = {}
    for edge in graph.edges:
        source = next((node for node in graph.nodes if node.id == edge.source), None)
        target = next((node for node in graph.nodes if node.id == edge.target), None)
        if source and target and source.data.kind == "branch" and target.data.kind == "card":
            cards_by_branch.setdefault(source.id, []).append(target)

    payload = {
        "spaceTitle": hub.data.title,
        "hubSubtitle": hub.data.subtitle,
        "branches": [
            {
                "id": branch.id,
                "title": branch.data.title,
                "cards": [
                    {
                        "id": card.id,
                        "title": card.data.title,
                        "items": card.data.items,
                        "tags": card.data.tags,
                    }
                    for card in cards_by_branch.get(branch.id, [])
                ],
            }
            for branch in branches
        ],
    }
    return graph_from_llm_payload(payload, space_name=hub.data.title)
