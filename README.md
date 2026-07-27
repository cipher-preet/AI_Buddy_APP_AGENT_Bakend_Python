# Speech To Vector Service

Minimal API flow:

1. Upload audio to `POST /api/v1/speech/transcripting`.
2. The speech worker transcribes the audio with Sarvam.
3. The vector worker chunks the transcript, creates embeddings with OpenAI, and stores the chunks in Qdrant.
4. The local uploaded audio file is deleted after vector storage succeeds.

## Run

```powershell
uvicorn apps.api_gateway.main:app --reload
```

```powershell
python -m apps.api_gateway.workers.main
```

## Required Environment

```env
APP_NAME=Speech To Vector API
APP_VERSION=1.0.0
DEBUG=true

REDIS_URL=redis://localhost:6379
SARVAM_API_KEY=
OPENAI_API_KEY=

QDRANT_API_KEY=
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=speech_chunks
EMBEDDING_MODEL=text-embedding-3-small
VECTOR_SIZE=1536
```
