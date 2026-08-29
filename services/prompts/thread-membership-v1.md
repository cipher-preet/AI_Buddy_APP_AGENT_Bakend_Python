Decide whether this atomic event belongs to the candidate thread.

Use only the event meaning, entities, object, and the compact thread context.
Do not use raw meeting transcripts. Do not invent a new thread label.

semanticRelation must be exactly one of:
- SAME_THREAD: the same semantic object, the same issue/goal, or the same evolving thread.
- RELATED_BUT_DISTINCT: same broad technical domain, but a different object or issue. May keep a graph edge. Must NOT merge.
- UNRELATED: no meaningful relation.

sameThread=true only when semanticRelation=SAME_THREAD.

Shared words or entities such as "server" or "ID" are not enough.
These must remain RELATED_BUT_DISTINCT or UNRELATED, not SAME_THREAD:
- server ID creation
- server connection failure
- database connection string
- Port ID tracking

If the relationship is unclear, set ambiguous=true rather than guessing.
Return only output matching the required schema.
