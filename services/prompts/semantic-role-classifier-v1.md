Classify the bounded transcript evidence into universal conversational roles.

For each meaningful evidence unit, return only the roles supported by that unit, a concise dynamic topic, an opaque threadKey shared only by units that express the same underlying topic/context, grounded normalized meaning, exact evidence sequence IDs, confidence, and uncertain=true when the meaning remains conditional, speculative, or unresolved.

Allowed roles: fact, claim, explanation, decision, action, commitment, request, question, answer, problem, solution, requirement, instruction, definition, example, important_point, disagreement, conclusion, follow_up, deadline, assignment, reference, unresolved.

The transcript may be in any language or mixed languages. Infer meaning from the language itself; do not rely on English trigger words. Do not invent roles, owners, dates, topics, thread relationships, or evidence IDs. Omit filler and unsupported units.

Return only the required JSON schema.
