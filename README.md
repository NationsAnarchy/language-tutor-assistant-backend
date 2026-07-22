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
# Gemini — Chat model (gemini-2.5-flash) + TTS
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
| `DELETE` | `/session/{id}` | Delete a session (+ audio files) | JWT or `X-Dev-User-Id` |
| `GET` | `/sessions` | List user's sessions | JWT or `X-Dev-User-Id` |
| `POST` | `/chat` | Send a message, get AI reply | JWT or `X-Dev-User-Id` |
| `POST` | `/session/{id}/tts` | Synthesize audio for last assistant message | JWT or `X-Dev-User-Id` |
| `GET` | `/session/{id}/mistakes` | Get mistake log for a session | JWT or `X-Dev-User-Id` |
| `GET` | `/audio/{path}` | Serve synthesized audio file (MP3 or WAV) | None |

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

# Run all tests
python -m pytest tests/ -v

# Run guardrail tests
python tests/test_guardrails.py

# Run RAG evaluation
python tests/test_rag_eval.py
```

## Deployment (Railway)

The backend deploys on Railway via `nixpacks.toml`:

```toml
# nixpacks.toml — installs ffmpeg for MP3 audio conversion
[phases.setup]
nixPkgs = ["ffmpeg"]
```

Set these environment variables in the Railway dashboard:

| Variable | Value |
|----------|-------|
| `GEMINI_API_KEY` | Your Gemini API key |
| `GOOGLE_EMBEDDING_API_KEY` | (Optional) Separate embedding key |
| `PINECONE_API_KEY` | Your Pinecone API key |
| `PINECONE_INDEX` | `language-tutor` |
| `NEXTAUTH_SECRET` | Same secret used by the frontend |
| `CORS_ORIGINS` | `http://localhost:3000,https://your-app.vercel.app` |

> **Note:** Railway uses NixPacks builder. The `nixpacks.toml` file installs `ffmpeg` at build time, which is required by the TTS module to convert raw PCM audio to MP3. PCM→MP3 reduces audio file sizes by ~10x (e.g., 1.5 MB WAV → 120 KB MP3 for a 30s clip), which is critical for bandwidth-limited starter hosting plans.

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
│   ├── tts.py               # Gemini Flash TTS with PCM→MP3 conversion
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
├── audio/                     # Generated TTS audio files (MP3/WAV)
├── nixpacks.toml              # Railway build config (ffmpeg)
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

The audio serving endpoint (`/audio/{path}`) includes proper headers for streaming:

- `Accept-Ranges: bytes` — enables partial content (206) responses for seekable playback
- `Cache-Control: public, max-age=86400` — caches audio files for 24 hours

## Audio (TTS) Pipeline

```
Gemini TTS API → Raw PCM (audio/L16) → ffmpeg → MP3 file (saved to disk)
                                                          ↓
                                              Fallback: WAV (if ffmpeg unavailable)
```

The TTS module (`app/tts.py`):
1. Calls Gemini 2.5 Flash TTS with the tutor's text response
2. Receives raw PCM audio (24kHz, 16-bit, mono)
3. Converts to MP3 via ffmpeg (32 kbps — good quality for speech)
4. If ffmpeg is unavailable, falls back to WAV format
5. Saves to `audio/{user_hash}/{session_id}/{uuid}.mp3`
6. Returns a relative path for the frontend to construct the URL

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
│ generate_response  │──→ Gemini 2.5 Flash produces tutor reply
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
| Chat LLM | `gemini-2.5-flash` | Google Gemini |
| Embeddings | `gemini-embedding-001` (3072d) | Google Gemini |
| TTS | `gemini-2.5-flash-preview-tts` | Google Gemini |
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
| `chat_history` | JSON | Array of {role, content, audio_url} |
| `last_exercise` | JSON | Active exercise state |
| `mistake_log` | JSON | Array of {type, detail, timestamp} |
| `created_at` | TEXT | ISO datetime |
| `updated_at` | TEXT | ISO datetime |