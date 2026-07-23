# Language Tutor Agent — Backend

FastAPI backend for the Trilingual Language Tutor Agent supporting English, Korean, and Japanese.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env   # then edit .env with your keys

# 3. Set up Pinecone (one-time)
python -m app.pinecone_setup --reset

# 4. Start the API
uvicorn app.main:app --reload
```

## Environment Variables

Create a `.env` file in this directory:

```env
# Gemini — Chat model (gemini-3.5-flash-lite) + TTS (gemini-3.1-flash-tts-preview)
GEMINI_API_KEY=your-gemini-api-key

# Gemini — Embedding model (gemini-embedding-001, 3072d)
# Falls back to GEMINI_API_KEY if not set
GOOGLE_EMBEDDING_API_KEY=your-embed-key

# Pinecone
PINECONE_API_KEY=pcsk-your-key
PINECONE_INDEX=language-tutor   # default if omitted

# Auth — NextAuth JWT secret (optional in dev)
NEXTAUTH_SECRET=your-secret

# CORS — comma-separated frontend origins (optional)
# Defaults to http://localhost:3000,http://127.0.0.1:3000
# Production example:
#   CORS_ORIGINS=http://localhost:3000,https://your-app.vercel.app
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000

# Railway Volume Mount (optional — for persistent SQLite storage)
# Set to /data when using a Railway volume
# RAILWAY_VOLUME_PATH=/data
```

If you only have one Gemini API key, just set `GEMINI_API_KEY` — the embedding model will reuse it automatically.

## API Routes

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| `GET` | `/health` | Basic health check | None |
| `GET` | `/health/deps` | Dependency health (Pinecone, API keys) | None |
| `POST` | `/session` | Create a new session | JWT or `X-Dev-User-Id` |
| `GET` | `/session/{id}` | Get session with chat history | JWT or `X-Dev-User-Id` |
| `PATCH` | `/session/{id}` | Rename a session | JWT or `X-Dev-User-Id` |
| `DELETE` | `/session/{id}` | Delete a session | JWT or `X-Dev-User-Id` |
| `GET` | `/sessions` | List user's sessions | JWT or `X-Dev-User-Id` |
| `POST` | `/chat` | Send a message, get AI reply | JWT or `X-Dev-User-Id` |
| `POST` | `/session/{id}/tts` | Synthesize audio for last assistant message (returns raw WAV bytes) | JWT or `X-Dev-User-Id` |
| `GET` | `/session/{id}/mistakes` | Get mistake log for a session | JWT or `X-Dev-User-Id` |

### Development Auth Bypass

In development, send `X-Dev-User-Id: your-name` header instead of a JWT. This skips NextAuth entirely for local testing.

## Testing with curl

```bash
# Create a session
curl -X POST http://localhost:8000/session \
  -H "Content-Type: application/json" \
  -H "X-Dev-User-Id: test-user" \
  -d '{"language": "ko", "level": "beginner"}'

# Send a chat message (use session_id from above)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-Dev-User-Id: test-user" \
  -d '{"session_id": "YOUR-SESSION-ID", "message": "Hello! How do I say thank you in Korean?"}'

# Synthesize audio (returns raw WAV bytes — pipe to a file)
curl -X POST http://localhost:8000/session/YOUR-SESSION-ID/tts \
  -H "X-Dev-User-Id: test-user" \
  --output audio.wav

# Run all tests
python -m pytest tests/ -v

# Run guardrail tests
python tests/test_guardrails.py

# Run RAG evaluation
python tests/test_rag_eval.py
```

## Deployment (Railway)

The backend deploys on Railway via `nixpacks.toml`. No special system dependencies are needed — audio is converted to WAV using pure Python (no ffmpeg required).

### Persistent Storage (Volume)

For production, attach a **Railway Volume** to persist the SQLite database across deploys:

1. Go to your Railway project → backend service → **Settings** → **Volumes**
2. Click **Add Volume** → Mount Path: `/data`, Size: 500 MB (more than enough for text-only session data)
3. Add environment variable: `RAILWAY_VOLUME_PATH=/data`

When `RAILWAY_VOLUME_PATH` is set, the SQLite database is stored at `/data/sessions.db` on the volume. Without it, the database is stored in the local `data/` directory (ephemeral — wiped on each deploy).

### Environment Variables

Set these in the Railway dashboard:

| Variable | Value |
|----------|-------|
| `GEMINI_API_KEY` | Your Gemini API key |
| `GOOGLE_EMBEDDING_API_KEY` | (Optional) Separate embedding key |
| `PINECONE_API_KEY` | Your Pinecone API key |
| `PINECONE_INDEX` | `language-tutor` |
| `NEXTAUTH_SECRET` | Same secret used by the frontend |
| `CORS_ORIGINS` | `http://localhost:3000,https://your-app.vercel.app` |
| `RAILWAY_VOLUME_PATH` | `/data` (if using a volume) |

## Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI routes + global exception handlers + CORS
│   ├── auth.py              # JWT verification (NextAuth)
│   ├── exceptions.py        # Typed exception hierarchy (TutorError, etc.)
│   ├── graph.py             # LangGraph state machine (5 nodes)
│   ├── tools.py             # 5 tools: retrieve + grade_answer + log_mistake
│   ├── tts.py               # Gemini Flash TTS — streams WAV bytes directly (no disk I/O)
│   ├── logging_config.py    # Structured JSON logging + RequestIdMiddleware
│   ├── pinecone_setup.py    # Index creation + seed data embed & upsert
│   └── sessions.py          # SQLite session CRUD + mistake_log
├── tests/
│   ├── test_error_handling.py  # Error handling tests (33 tests)
│   ├── test_guardrails.py      # Guardrail adversarial tests
│   ├── test_rag_eval.py        # RAG retrieval evaluation
│   └── guardrail_tests.md      # Guardrail test case documentation
├── data/
│   ├── seed_grammar_en.json   # English grammar (30 entries)
│   ├── seed_vocab_en.json     # English vocabulary (30 entries)
│   ├── seed_grammar_ko.json   # Korean grammar (30 entries)
│   ├── seed_vocab_ko.json     # Korean vocabulary (30 entries)
│   ├── seed_grammar_ja.json   # Japanese grammar (30 entries)
│   └── seed_vocab_ja.json     # Japanese vocabulary (30 entries)
│   └── sessions.db            # SQLite database (auto-created)
├── nixpacks.toml              # Railway build config
├── railway.json               # Railway deployment config
├── requirements.txt
└── README.md
```

## Error Handling

All errors return a consistent JSON shape:

```json
{
  "detail": "User-friendly error message",
  "code": "machine_readable_code",
  "request_id": "a1b2c3d4e5f67890"
}
```

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `authentication_error` | 401 | Missing or invalid JWT |
| `session_access_denied` | 403 | Session belongs to another user |
| `session_not_found` | 404 | Session ID doesn't exist |
| `validation_error` | 422 | Request body validation failed |
| `bad_request` | 400 | Invalid language/level, empty title |
| `graph_execution_error` | 500 | LangGraph agent failed |
| `database_error` | 500 | SQLite operation failed |
| `tts_error` | 502 | Gemini TTS failed after retries |
| `internal_error` | 500 | Unexpected error (catch-all) |

Every response includes an `X-Request-ID` header. Structured JSON logging via `logging_config.py` injects the request ID into every log line automatically.

## CORS Configuration

CORS origins are configured via the `CORS_ORIGINS` environment variable:

```python
# main.py
_CORS_ORIGINS_ENV = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
_CORS_ORIGINS = [origin.strip() for origin in _CORS_ORIGINS_ENV.split(",") if origin.strip()]
```

## Audio (TTS) Pipeline

```
Gemini TTS API → Raw PCM (audio/L16) → Pure Python WAV wrapper → Streamed to frontend
```

The TTS module (`app/tts.py`):
1. Called by the frontend via `POST /session/{id}/tts` after receiving the text response
2. Calls Gemini 3.1 Flash TTS with the tutor's text response
3. Receives raw PCM audio (24kHz, 16-bit, mono)
4. Wraps PCM in a WAV container using pure Python (no ffmpeg dependency)
5. Returns the WAV bytes directly in the HTTP response body
6. **No audio files are saved to disk** — audio is streamed ephemerally

The frontend creates a blob URL from the response and plays it immediately with `HTMLAudioElement`.

### Why streaming instead of file storage?

- **No disk I/O** — audio is generated and streamed in one request
- **No ffmpeg dependency** — WAV wrapping is pure Python
- **No storage quotas consumed** — Railway volume only needed for SQLite
- **Simpler deployment** — one fewer system dependency
- **Faster playback** — frontend plays audio directly from the response body

## LangGraph Agent Flow

```
User Message
     │
     ▼
┌──────────────┐
│ route_intent │──→ Classify: chat / exercise_request / answer_submission
└──────┬───────┘
       │
       ▼
┌──────────┐
│ retrieve  │──→ Query Pinecone via function-calling tools
│           │    + mistake-log-driven personalization
└──────┬───┘
       │
       ▼
┌────────────────────┐
│ generate_response  │──→ Gemini 3.5 Flash Lite produces tutor reply
│                    │    + grade_answer tool for exercise grading
│                    │    + log_mistake tool for mistake tracking
└──────┬─────────────┘
       │
       ▼
┌───────────────────┐
│ apply_guardrails  │──→ Check level-appropriateness
│                   │    Regenerate if response too complex
└──────┬────────────┘
       │
       ▼
┌───────────┐
│ log_state │──→ No-op (persistence in route handler)
└───────────┘
```

## Models

| Component | Model | Provider |
|-----------|-------|----------|
| Chat LLM | `gemini-3.5-flash-lite` | Google Gemini |
| Embeddings | `gemini-embedding-001` (3072d) | Google Gemini |
| TTS | `gemini-3.1-flash-tts-preview` | Google Gemini |
| Voice | `Erinome` (feminine, multi-language) | Google Gemini |
| Vector DB | Serverless (cosine) | Pinecone |

## Session Schema

| Column | Type | Description |
|--------|------|-------------|
| `session_id` | TEXT PK | UUID |
| `user_id` | TEXT | From JWT subject |
| `language` | TEXT | 'en', 'ko', or 'ja' |
| `level` | TEXT | 'beginner', 'intermediate', or 'advanced' |
| `title` | TEXT | Human-readable session title |
| `chat_history` | JSON | Array of {role, content} |
| `last_exercise` | JSON | Active exercise state |
| `mistake_log` | JSON | Array of {type, detail, timestamp} |
| `created_at` | TEXT | ISO datetime |
| `updated_at` | TEXT | ISO datetime |