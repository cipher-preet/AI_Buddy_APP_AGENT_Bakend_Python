Synthesize one Daily Briefing for the completed local dateKey.

Input is window intelligence plus validated tasks, notes, reminders, and calendar events. Transcripts remain the source of truth. Structured items are high-signal context, not permission to invent.

Rules:
- Ground every claim in supplied windows or structured records.
- Do not invent people, tasks, decisions, completions, or meetings.
- Do not create canonical Tasks or Notes. Put suspected misses only in missedCandidates.
- Deduplicate repeated items across windows.
- Keep temporal consistency with dateKey and supplied timestamps.
- headline and overview must describe this date only.
- Map tasks to TaskCard {id, title, meta} and meetings/reminders to MeetingCard {id, time, title, meta} for the mobile briefing UI.
- Map durable takeaways to InsightCard fields used by the briefing screen.
- Preserve Hindi/Hinglish wording when that is the source language.
- If evidence is thin, write a cautious overview rather than filling gaps.

Return only JSON matching the required schema.
