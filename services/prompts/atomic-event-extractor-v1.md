Extract atomic grounded events from this local topic only.

Do NOT generate Tasks or Notes. Do NOT merge two meanings into one event.

Extract ALL distinct supported semantic units from every supplied micro-block, not only the dominant topic or the most actionable statement.

One event = one meaning. A content-rich micro-block that contains several independent facts, requirements, decisions, plans, states, issues, open questions, or actions must emit several events.

Do not combine independent useful semantics into one vague event when the local evidence supports separate meanings. Do not split trivial paraphrases or restatements of the same meaning.

Required principle:
- distinct meaning → distinct atomic event
- same meaning repeated → one event (later coverage may mark the rest DUPLICATE)

Each event expresses exactly one meaning. Supported kinds:
REQUEST, COMMITMENT, ASSIGNMENT, DECISION, PROPOSAL, REQUIREMENT, ISSUE, STATE, RESULT, FACT, FOLLOW_UP, DEADLINE, COMPLETION, CANCELLATION, CONTRADICTION, CONSTRAINT, IMPORTANT_CONTEXT, OPEN_QUESTION, NOISE.

Kind is the semantic type. It is NOT the Task/Note classifier.

Independently fill:

actionSignal:
- isActionable: true only when a real action exists (request, commitment, assignment, instruction, or follow-up).
- role: REQUEST | COMMITMENT | ASSIGNMENT | INSTRUCTION | FOLLOW_UP | null
- actionStrength: NONE | POSSIBLE | EXPLICIT
- verb: the supported action verb (create, use, track, enable, request, fix, document, build, investigate, check, …). This is a semantic label. It does not have to appear as the same string in the transcript.
- object: the specific action object. Prefer the narrowest phrase supported by evidence.
- objectGroundingType: EXPLICIT | LOCAL_COREFERENCE | INFERRED | UNRESOLVED
- actor: only if explicitly supported, else null
- deadline: only if explicitly supported, else null

actionStrength rules:
- EXPLICIT: someone is actually asked, assigned, instructed, or commits to do the action. This includes obligation and intended-work meanings, in any language: a speaker saying they will do X, need to do X, have to do X, or will implement/build/check/investigate X with a grounded object.
- POSSIBLE: a proposal, discussion, suggestion, or hypothetical. Keep as memory/context. isActionable=false.
- NONE: information, state, issue, or possibility with no action.

isActionable=true only with actionStrength=EXPLICIT.
POSSIBLE must not become a Task.
Always fill role when isActionable=true. Kind FACT/DECISION/STATE does not cancel an EXPLICIT action.

Distinguish:
- PROPOSAL / DISCUSSION / POSSIBILITY / INFORMATION → not ACTION
- REQUEST / COMMITMENT / ASSIGNMENT / INSTRUCTION / FOLLOW_UP → ACTION if EXPLICIT
- Obligation or intended future work on a grounded object is COMMITMENT or REQUEST, even if phrased as a fact ("X banana hai", "X karna hai", "we will build X", "isko dekhna hai", "we need to investigate X"). These are semantic patterns, not phrase lists.
- An open investigation ("need to determine whether/how X") is FOLLOW_UP or OPEN_QUESTION, not a confirmed FACT that X already holds.

memorySignal:
- isMemoryWorthy: true only when the meaning is worth remembering as Buddy memory, independent of whether it is factually grounded.
- importance: HIGH | MEDIUM | LOW
- reason: short semantic label such as DECISION, REQUIREMENT, ISSUE, STATUS, RESULT, CONSTRAINT, PLAN, CONTEXT, OPEN_QUESTION. Not a domain category.
- HIGH: decisions, requirements, important issues, meaningful status, results, constraints, important product/technical context.
- MEDIUM: useful context that is not the main decision or issue.
- LOW: backchannel, audio-check, filler, minor conversational fragments. Grounded is not enough.
- Do not suppress memory just because the event is actionable.
- A REQUIREMENT, DECISION, ISSUE, or STATE may still be actionable if it contains a real instruction.
- Low-information filler is not memory. Set isMemoryWorthy=false and importance=LOW.

Preserve epistemic status in meaning:
- confirmed fact → FACT
- decision → DECISION
- proposal → PROPOSAL
- open question / unresolved design concern → OPEN_QUESTION or ISSUE
- issue → ISSUE
- future plan → may be FACT/STATE memory AND an EXPLICIT COMMITMENT if someone is to do it
- commitment → COMMITMENT with actionSignal
Never rewrite "should we / whether to / need to look into" as "it is already true".

fieldEvidence:
- actionVerb, actionObject, actor, deadline: cite the exact transcript spans that support each field.
- Ground each field independently. Do not copy an object from JSON if the transcript does not support it.
- actionVerb evidence may be the local obligation/intention span even when the verb label is a paraphrase.

objectGroundingType:
- EXPLICIT: the object phrase (or a safe abbreviation like mic→microphone) appears in the local evidence.
- LOCAL_COREFERENCE: a pronoun (it, that, usko, isko) resolves to exactly one antecedent in the same micro-block or the immediately preceding coherent micro-block.
- INFERRED: a broader paraphrase not in the evidence (reject this object).
- UNRESOLVED: no unique supported object.

Rules:
- Preserve exact evidence spans with sequenceStart, sequenceEnd, and the original transcript text.
- Never invent actor, deadline, object, names, or evidence IDs.
- Never infer an action merely because a problem exists. An ISSUE or STATE is not a REQUEST unless an action is stated.
- Never omit an ISSUE, STATE, REQUIREMENT, DECISION, or FACT because a REQUEST or COMMITMENT follows. Emit both events. Action does not consume memory.
- Never omit a second supported FACT, REQUIREMENT, PLAN, or STATE because another meaning in the same micro-block was already extracted. Sequence overlap is not semantic coverage.
- Never merge an observed problem and the request to act on it into a single REQUEST/COMMITMENT. They are two meanings.
- Never merge independent steps, independent requirements, or independent facts that share a subject. Related is not the same as identical.
- If one utterance both states a fact and asks for an action, emit the memory event and the action event separately. Both may be valid.
- Instructions and requirements MAY still be actionable. Example: "Use GPT for coordinate action" is a REQUIREMENT with actionSignal.isActionable=true, role=INSTRUCTION, actionStrength=EXPLICIT, verb=use, object=GPT for coordinate action.
- Prefer abstention over a generic action. If the object cannot be grounded, keep isActionable=true if an EXPLICIT action exists, set object=null, objectGroundingType=UNRESOLVED, and do not invent "pending task" / "it" / "the issue".
- Resolve pronouns (it, that, usko, uska, isko) ONLY from the same micro-block or the immediately preceding coherent micro-block, and only when one antecedent is clearly supported. If multiple plausible objects exist, object=null and objectGroundingType=UNRESOLVED.
- Do not resolve action objects from distant thread similarity.
- Do not widen a narrow object. Evidence "server ID create karna hai" → object="server ID", objectGroundingType=EXPLICIT. Not "server infrastructure configuration" (INFERRED).
- Do not merge separate technical objects merely because they are related. "Use GPT for coordinates" and "Use OpenCV for room dimensions" are two events.
- Bad grammar, Hindi/Hinglish, incomplete sentences, or overlapping speakers are not reasons to drop meaning.
- If a span is garbled, semantically corrupted, or noisy STT with no recoverable meaning, kind=NOISE and/or uncertainty includes NOISE, AMBIGUOUS, or LOW_CONFIDENCE_SOURCE. Do not invent a fluent fact from corrupted audio.
- If the topic has no supported event, return events=[] and noEventReason.
- Surrounding filler, backchannel, or silence does not erase a decision, issue, requirement, or fact. Extract those from content micro-blocks even when most of the topic is low-information.

Examples:

"S3 is failing. We will fix it tomorrow." → ISSUE (memory, actionStrength=NONE) + COMMITMENT (EXPLICIT, object=S3 issue via local coreference, deadline=tomorrow). Do not emit only the COMMITMENT.

"The master prompt should contain elevation, frontend view, images and appearance. Please document these requirements." → REQUIREMENT/FACT (memory of the required contents) + REQUEST (document the requirements). Both events.

"Connection is insecure." / "connection insecure hai" → STATE, isActionable=false, actionStrength=NONE. Not a task.

"connection kal fix kar dena" → COMMITMENT or REQUEST, EXPLICIT, Task-eligible, object from local context or "connection".

"server ID create karna hai" → REQUEST or REQUIREMENT, EXPLICIT, verb=create, object=server ID, objectGroundingType=EXPLICIT.

"We can discuss pricing around 200" / "pricing around 200 rakh sakte hain" / "free plan pe usage limit discuss karte hain" → PROPOSAL, actionStrength=POSSIBLE or NONE, isActionable=false. Not a Task.

"pricing kal final kar lena" / "Monday meeting mein pricing finalize karenge" → COMMITMENT or REQUEST, EXPLICIT, Task-eligible if someone is actually to finalize it.

"GPT coordinate extraction ke liye use karna hai" / "Use GPT for coordinate action" → REQUIREMENT, EXPLICIT INSTRUCTION, Task.

"GPT use kar sakte hain" → PROPOSAL, POSSIBLE, isActionable=false unless context commits to it.

"kal kar denge" / "kal isko fix kar denge" with no single local antecedent → EXPLICIT action possible, object=null, objectGroundingType=UNRESOLVED. Do not emit a generic action.

"X banana hai" / "X karna hai" / "we will build X" / "we'll implement X" with a grounded X → COMMITMENT, EXPLICIT, isActionable=true, role=COMMITMENT. Not a mere FACT with no action.

"we need to investigate X" / "we should check X" / "isko dekhna hai ki …" → FOLLOW_UP or OPEN_QUESTION with EXPLICIT action if someone is to look, and never a confirmed FACT that the investigated thing already holds.

Return only output matching the required schema.
