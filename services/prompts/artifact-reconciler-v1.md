Reconcile each incoming semantic unit with the small set of existing session artifacts.

You compare meaning, not wording. semanticHint is an indexing hint only. It is never identity. Two units with different hints may be the same artifact. Two units with the same hint may still be distinct.

For every incoming unit, choose exactly one action:

- CREATE_NEW: this is a new distinct artifact
- UPDATE_EXISTING: the same artifact gained detail, owner, deadline, assignment, evidence, or a modified plan
- COMPLETE_EXISTING: the earlier artifact is now done
- CANCEL_EXISTING: the earlier artifact is abandoned
- SUPERSEDE_EXISTING: a later meaning replaces an earlier one; keep the new artifact and retire the old
- RELATED_BUT_DISTINCT: connected but not the same action or fact; keep both

When the action modifies an existing artifact (UPDATE_EXISTING, COMPLETE_EXISTING, CANCEL_EXISTING, SUPERSEDE_EXISTING) you MUST return that artifact's artifactId as targetArtifactId. The ID must be one of the provided validTargetArtifactIds.

If you cannot identify a valid target, do not emit a modifying action without an ID. Use CREATE_NEW or RELATED_BUT_DISTINCT instead. Orchestration will not create a second artifact from a modifying action that lacks a valid target.

RELATED_BUT_DISTINCT should also return the related artifactId when one exists.

Do not merge distinct actions that merely share a topic, person, object, or vocabulary.
Do not split paraphrases of the same commitment into two artifacts.
Do not treat matching or mismatching semanticHint as proof of sameness.

Copy evidence spans exactly from the incoming unit. Do not paraphrase, translate, or invent evidence text. Include those exact spans on the decision.

Return only output matching the required schema.
