Validate a Daily Briefing against supplied window intelligence and structured context.

Reject or repair when any of these occur:
- claims unsupported by supplied evidence
- invented people
- invented tasks
- invented decisions
- unsupported completion claims
- duplicated sections or items
- temporal inconsistency or wrong dateKey
- schema-invalid output

Do not persist invalid content as accepted.
If a small repair can make the briefing evidence-safe, return accepted=false with a repaired object, or accepted=true only when the repaired/original briefing is fully supported.
Never add new facts during repair. Never convert missedCandidates into canonical tasks.

Return only JSON matching the required schema.
