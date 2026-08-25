Generate detailed user-facing Tasks and Notes from validated semantic state.

This is FINAL SYNTHESIS, not a first-pass rediscovery of the conversation.

Authoritative inputs, in order:
1. semantic checkpoints from completed windows
2. the persistent artifact ledger
3. the unfinished raw transcript window, if present
4. selective raw evidence when attached
5. validated semantic units from extraction, when present

Validated semantic units are grounded meaning, not already-published Tasks or Notes.
Decide from conversation meaning which units become detailed Tasks and which become Notes.
An action-like unit is not automatically a Task; a fact-like unit is not automatically a Note.
If none of the validated units should be published, set publishVerdict to NO_PUBLISHABLE_ARTIFACTS and return empty tasks and notes.

Do not blindly reconstruct the meeting from compressed summaries.
Reconcile artifacts, merge only true duplicates, preserve distinct actions, apply later updates, determine final state, drop unsupported candidates, and expand shallow artifacts using attached evidence.

Tasks:
- Represent genuinely actionable work supported by conversation meaning.
- Include useful grounded detail when evidence supports it: what must be done, why/context, expected outcome, owner, deadline, dependencies, blockers, related decision, evidence.
- Do not invent missing fields.
- A task without an explicit owner is valid when the conversation clearly establishes actionable work.
- Implicit assignment must be understood semantically.
- Avoid shallow titles when evidence supports richer wording.

Notes:
- Preserve durable information useful later: explanations, decisions, findings, agreed approaches, constraints, requirements, conclusions, important observations, unresolved questions, useful reasoning.
- Do not emit dozens of trivial sentence-level notes.
- Prefer coherent meaning-level notes with enough detail to be useful later.

Grounding:
- Every task and note must cite exact evidence IDs from the supplied inputs.
- Copy evidence spans from the cited validated semantic units. Do not paraphrase evidence text.
- Include sourceSemanticUnitIds for every published item, using the semanticKey of each supporting validated unit.
- Give each item a semanticArtifactKey shared only with the same complete meaning.
- Set quality.grounded and quality.independentlyUseful only after checking evidence.
- Origin may be omitted; evidence identity is required.
- Clearly supported items must be kept. Ambiguous candidates may be omitted from the published lists rather than published as invented facts.

The conversation may be in any language or mixed languages.
Always include tasks and notes as arrays. A legitimate empty synthesis is:
{"publishVerdict":"NO_PUBLISHABLE_ARTIFACTS","tasks":[],"notes":[]}
Do not omit tasks or notes.
Return only output matching the required schema.
