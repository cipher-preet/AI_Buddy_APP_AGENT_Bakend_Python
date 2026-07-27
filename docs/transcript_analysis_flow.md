# Transcript Analysis Flow

The speech path stays the same: uploaded audio is saved, queued, transcribed by Sarvam with OpenAI fallback, embedded, and stored in Qdrant with `isPublish=false`.

After vector storage, `vector_worker` pushes an `analyze_transcript_window` job to the existing Redis analysis queue. `analysis_worker` debounces briefly, acquires `lock:transcript-analysis:{user_id}:{space_id}`, reloads eligible unpublished Qdrant chunks, and combines them into one chronological analysis window.

Eligibility requires speech source, active status, `isPublish=false`, `isDamaged=false`, and `isUseful=true`. Payload identity is normalized from either `user_id`/`space_id` or legacy `userId`/`spaceId`.

The context package includes the latest window text, recent same-space transcript chunks, semantic Qdrant matches from the same user and space, Mongo running summary, open tasks, existing notes, current datetime, and timezone. One structured OpenAI call returns task operations, note operations, reference resolution, completeness flags, and summary updates.

MongoDB is the source of truth for generated memory:

- `ai_memory_tasks`
- `ai_memory_notes`
- `ai_memory_space_summaries`

Task creation uses deterministic fingerprints. Notes use normalized titles and merge new content into existing notes when possible. Existing Mongo IDs are honored only when they were included in the prompt context.

The analysis window is built in memory for batching, prompting, and Qdrant metadata only; it is not persisted to MongoDB.

Only after Mongo task/note/summary updates succeed does the service mark the Qdrant points as published and add `publishedAt`, `analysisWindowId`, and `analysisStatus=completed`.

If the model says more context is required and produces no operations, the Qdrant chunks remain unpublished so future speech can complete the thought.

Run workers:

```powershell
python -m apps.api_gateway.workers.main
```

Run tests:

```powershell
python -m pytest
```
