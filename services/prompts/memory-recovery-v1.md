Recover evidence-supported items missed from a noisy conversation window, regardless of its subject.

Use SEMANTIC THREADS when provided. They are hints, not approved outputs. Treat raw WINDOW lines as the only authority.

Stages:
1. Classify supported conversational roles: fact, claim, explanation, decision, question, action, commitment, request, conclusion, or unresolved point.
2. Turn durable information into notes.
3. Turn explicit actions, commitments, requests, or strongly supported next steps into task candidates.
4. Reject anything not directly justified by cited evidence.

Do not infer owners, dates, deadlines, priority, subjects, or details. Noisy Hindi/Hinglish/STT wording can still support an item when repeated evidence is consistent.
Do not create a task merely because a role appears in SEMANTIC TOPIC PACKETS.

Every output must include exact sequence evidence from this window. Omit casual chatter and duplicates. Do not return NO_ACTION tasks.

For notes, synthesize related evidence into one specific 2-5 sentence professional body. For tasks, state a concrete supported action. Classifier labels must never appear in a title or body.
For every task and note, include semanticArtifactKey and quality with grounded=true and independentlyUseful=true only when the cited evidence supports the complete artifact.

Return only output matching the required schema.
