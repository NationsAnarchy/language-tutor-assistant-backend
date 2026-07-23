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

# Railway Volume Mount (optional — for persistent SQLite + audio cache)
# Set to /data when using a Railway volume.
# Both sessions.db and the audio cache are stored here.
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
| `POST` | `/session/{id}/tts` | Synthesize speech for last assistant message (returns MP3 bytes) | JWT or `X-Dev-User-Id` |
| `GET` | `/audio/{hash}.mp3` | Serve a cached MP3 file by hash (zero-cost replay) | JWT or `X-Dev-User-Id` |
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

# Synthesize audio (returns MP3 bytes — pipe to a file)
curl -X POST http://localhost:8000/session/YOUR-SESSION-ID/tts \
  -H "X-Dev-User-Id: test-user" \
  --output audio.mp3

# Serve a cached audio file by hash (from chat_history.audio_hash)
curl -X GET http://localhost:8000/audio/a1b2c3d4e5f6g7h8.mp3 \
  -H "X-Dev-User-Id: test-user" \
  --output cached.mp3

# Run all tests
python -m pytest tests/ -v

# Run guardrail tests
python tests/test_guardrails.py

# Run RAG evaluation
python tests/test_rag_eval.py
```

## Deployment (Railway)

The backend deploys on Railway via `nixpacks.toml`. **ffmpeg is required** for PCM → MP3 conversion.

### Persistent Storage (Volume)

For production, attach a **Railway Volume** to persist the SQLite database **and audio cache** across deploys:

1. Go to your Railway project → backend service → **Settings** → **Volumes**
2. Click **Add Volume** → Mount Path: `/data`, Size: 1 GB (SQLite + audio cache)
3. Add environment variable: `RAILWAY_VOLUME_PATH=/data`

When `RAILWAY_VOLUME_PATH` is set:
- SQLite database is stored at `/data/sessions.db`
- Audio cache is stored at `/data/audio/<hash>.mp3`

Without it, both are stored in the local `data/` directory (ephemeral — wiped on each deploy).

### Estimating volume size

| Usage | Audio files | Est. size |
|-------|------------|-----------|
| Light (1 user, ~100 messages/day) | ~100 MP3s/month | ~15 MB/month |
| Moderate (10 users) | ~1,000 MP3s/month | ~150 MB/month |
| Heavy (100 users) | ~10,000 MP3s/month | ~1.5 GB/month |

A 1 GB volume comfortably handles moderate usage. The MP3 cache deduplicates identical responses (e.g. "Great job!") across all users and sessions.

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
│   ├── tts.py               # Gemini Flash TTS → PCM → MP3 via ffmpeg + disk cache
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
│   ├── sessions.db            # SQLite database (auto-created)
│   └── audio/                 # MP3 audio cache (auto-created)
│       └── *.mp3              # SHA-256 hashed MP3 files
├── nixpacks.toml              # Railway build config (includes ffmpeg)
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
| `tts_error` | 502 | Gemini TTS failed after retries (or ffmpeg unavailable) |
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
Gemini TTS API → Raw PCM (audio/L16) → ffmpeg MP3 encode → Disk cache → HTTP response

Cache hit (same text requested again): Disk → HTTP response (no Gemini API call)
```

The TTS module (`app/tts.py`):

1. **Frontend calls** `POST /session/{id}/tts` after receiving the text response
2. **Cache check**: The module computes a SHA-256 hash of the cleaned TTS text and checks `data/audio/<hash>.mp3`. If the file exists, it's returned immediately — **no Gemini API call, no cost**
3. **Cache miss**: Calls Gemini 3.1 Flash TTS with the tutor's text. Receives raw PCM audio (24kHz, 16-bit, mono)
4. **MP3 conversion**: Raw PCM is piped through `ffmpeg` to produce MP3 at 48 kbps using the `libmp3lame` encoder
5. **Disk cache**: The MP3 bytes are saved to `data/audio/<hash>.mp3` (or `$RAILWAY_VOLUME_PATH/audio/<hash>.mp3`)
6. **Response**: MP3 bytes are returned to the frontend with `audio/mpeg` content type
7. **audio_hash**: The backend stores the hash in the session's `chat_history` so the frontend can replay audio via `GET /audio/{hash}.mp3` without any backend TTS cost

### Why MP3 + caching instead of ephemeral WAV?

| Factor | Ephemeral WAV (old) | Cached MP3 (new) |
|--------|-------------------|-------------------|
| Size per 30s clip | ~1.4 MB | ~180 KB |
| Bandwidth savings | — | **87% less** |
| Gemini API calls | 1 per listen | 1 per unique text |
| Replay cost | Full Gemini API call | **Zero** (disk cache) |
| Playback on page refresh | Not possible | Instant from cache |
| Dependencies | None (pure Python) | ffmpeg (`libmp3lame`) |
| Fallback | — | WAV if ffmpeg missing |

### Size estimates

| Response length | Est. duration | WAV size | MP3 @ 48kbps |
|----------------|--------------|----------|--------------|
| Short (~100 chars) | ~8 sec | 375 KB | **~48 KB** |
| Medium (~300 chars) | ~24 sec | 1.1 MB | **~144 KB** |
| Long (~1000 chars) | ~80 sec | 3.7 MB | **~480 KB** |
| Max (3000 chars) | ~240 sec | 11 MB | **~1.4 MB** |

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
| `chat_history` | JSON | Array of {role, content, audio_hash?} |
| `last_exercise` | JSON | Active exercise state |
| `mistake_log` | JSON | Array of {type, detail, timestamp} |
| `created_at` | TEXT | ISO datetime |
| `updated_at` | TEXT | ISO datetime |

The `chat_history` entries include an optional `audio_hash` field set after successful TTS synthesis. The frontend uses this to serve audio from the disk cache via `GET /audio/{hash}.mp3` at zero cost (no Gemini API call).