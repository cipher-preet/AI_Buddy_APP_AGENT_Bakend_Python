Repair missing extraction items using only the CURRENT CONVERSATION spans provided.

Do not use BACKGROUND SPACE CONTEXT as evidence.
Create only items that are directly supported by the missing spans.
Create tasks for explicit actions, commitments, deadlines, task updates, or task completions mentioned in the missing spans.
Create notes for durable facts, requirements, ideas, strategy points, preferences, or important context mentioned in the missing spans.
If a user is listening inside a space, treat meaningful space-related work and notes as eligible even when they introduce a new task or topic.
Ignore only general talk, greetings, examples, hypotheticals, and non-actionable chatter.
Preserve uncertainty and set needsConfirmation=true when needed.
Return only output matching the required schema.
