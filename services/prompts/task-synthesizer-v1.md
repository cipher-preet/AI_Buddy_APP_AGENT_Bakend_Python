Improve wording of this already-grounded action event into a Task.

Do not create new semantic facts. Do not add owners, deadlines, systems, or objects that are not in the event or its evidence.

If the action object cannot be recovered, return a generic title that the validator can reject rather than inventing one.

Keep evidence unchanged. Title should name the actual action object.
Body must explain the task, not copy the title. Use only grounded details from the event and its evidence: objective, current status, constraints, and owner/date if present. If related thread context adds a confirmed constraint or status, you may mention it in the body without adding new evidence IDs.
Do not widen a narrow object. Do not add thread-context events into evidence.

Return only output matching the required schema.
