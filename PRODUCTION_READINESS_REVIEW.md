# Production Readiness Review — Backend

**Repository:** `language-tutor-assistant-backend`
**Scope:** `app/`, `tests/`, deployment config (`railway.json`, `nixpacks.toml`,
`requirements.txt`), and the compatibility contract in `BACKEND_REVIEW_IMPROVEMENTS.md`.
**Method:** static review of the full application source plus a dependency-install
and test-run attempt in a clean environment.

This document lists **what to change to run this safely and economically in
production**. It deliberately does not repeat items already completed in
`BACKEND_REVIEW_IMPROVEMENTS.md` (schema hardening, error envelope, request IDs,
CORS allow-list, structured logging) — those are done and should be preserved.

---

## 1. Baseline: what is already production-grade

Worth keeping intact while working through the list below:

- JWT verification pins `algorithms=["HS256"]` and requires a configured secret;
  it never accepts unsigned/`none` tokens.
- Session ownership is enforced before read/mutate on every `/session/*` route.
- A single typed error envelope (`detail`, `code`, `request_id`) plus
  `X-Request-ID` propagation, with a catch-all that does not leak tracebacks.
- SQLite uses WAL, `busy_timeout=10000`, and foreign keys; writes to
  `chat_history` are serialized with `BEGIN IMMEDIATE` so concurrent TTS and chat
  writes cannot lose an `audio_hash`.
- CORS uses an explicit origin list (no wildcard) with a bounded method/header set.
- Structured JSON logging with `contextvars`-based request correlation — correct
  choice, avoids the global-logger-filter race.
- Guardrails degrade open (a failed guardrail check does not drop the reply).

---

## 2. Priority summary

| # | Priority | Finding | Primary location |
|---|---|---|---|
| 1 | **P0** | No rate limiting or per-user quota on LLM/TTS routes | `app/main.py` |
| 2 | **P0** | `/audio/{hash}.mp3` is public and content-addressed by guessable text | `app/main.py:566`, `app/tts.py:118` |
| 3 | **P0** | `chat_history` grows without bound and is fully replayed to the LLM each turn | `app/main.py:418`, `app/sessions.py` |
| 4 | **P0** | Blocking graph work can exhaust the default thread pool | `app/main.py:458` |
| 5 | **P0** | Dependencies are unpinned; no lockfile | `requirements.txt` |
| 6 | **P0** | No CI pipeline gates merges | (no `.github/workflows`) |
| 7 | **P1** | SSE is simulated, not streamed; no heartbeat; TTFB = full generation time | `app/main.py:477` |
| 8 | **P1** | `asyncio.wait_for` cannot cancel the worker thread it timed out on | `app/main.py:458` |
| 9 | **P1** | Audio cache has no eviction; `_cache_locks` grows forever | `app/tts.py:62`, `app/tts.py:213` |
| 10 | **P1** | Startup does not fail fast; `/health` reports `ok` while degraded | `app/main.py:112`, `app/main.py:642` |
| 11 | **P1** | Validation errors are logged with raw learner input | `app/main.py:200` |
| 12 | **P1** | Auth accepts tokens with no `sub`; no `aud`/`iss` check; legacy secret fallback | `app/auth.py`, `app/main.py:258` |
| 13 | **P1** | SQLite is a single-writer ceiling and unsafe across replicas | `app/sessions.py:36` |
| 14 | **P1** | 3–6 LLM calls per user message, each building a new client | `app/graph.py`, `app/tools.py:85` |
| 15 | **P1** | No metrics, tracing, or error tracking | `LOGGING.md:50` |
| 16 | **P1** | `/health/deps` is unauthenticated and performs a live Pinecone call | `app/main.py:647` |
| 17 | **P2** | No API versioning or contract tests | `app/main.py` |
| 18 | **P2** | OpenAPI metadata is incomplete (2 routes documented) | `app/main.py:504`, `:566` |
| 19 | **P2** | `/sessions` is unpaginated and reads the whole `chat_history` column | `app/sessions.py:169` |
| 20 | **P2** | No request body size limit | `app/main.py` |
| 21 | **P2** | Client-supplied `X-Request-ID` is echoed unvalidated | `app/logging_config.py:117` |
| 22 | **P2** | Session deletion leaves audio behind; stale comment says otherwise | `app/sessions.py:457` |
| 23 | **P2** | Rename/delete are TOCTOU; `conn.total_changes` is misused | `app/main.py:614`, `app/sessions.py:502` |
| 24 | **P2** | Dead code: unused schema, exception, import, and `app = None` placeholder | `app/schemas.py:110`, `app/main.py:76` |
| 25 | **P2** | `PINECONE_INDEX` is read before `load_dotenv()` | `app/pinecone_setup.py:22` |
| 26 | **P2** | Import-time side effects (DB path resolution, `mkdir`) | `app/sessions.py:29`, `app/tts.py:62` |
| 27 | **P2** | CORS allows credentials; a wildcard origin would be unsafe | `app/config.py:19` |
| 28 | **P2** | No Dockerfile; Nixpacks leaves the Python version unpinned | `nixpacks.toml` |
| 29 | **P2** | Test suite cannot run in a clean environment; integration tests unmarked | `requirements.txt` |
| 30 | **P3** | Proxy-header trust and worker/process tuning unverified | `railway.json:8` |


---

## 3. P0 — fix before opening to real traffic

### 3.1 No rate limiting or spend cap

**Evidence.** Nothing in `app/` imports or configures a limiter. The only trace of
rate limiting is a status-code mapping in the error handler (`app/main.py:184`,
`429: "rate_limit"`). `/chat`, `/session/{id}/tts`, and `/health/deps` are all
unmetered for an authenticated user.

**Impact.** A single valid token — or a scripted loop on one account — can drive
unbounded Gemini LLM + TTS spend and saturate the process. This is the most likely
way the project goes down or generates an unexpected bill. The frontend's 1-hour
token lifetime does not limit request volume within that hour.

**Suggested improvement.**
- Add per-user and per-IP token-bucket limits (`slowapi`, or `limits` backed by
  Redis) keyed on the JWT `sub`.
- Apply tighter budgets to the expensive routes: `/chat` (LLM) and
  `/session/{id}/tts` (TTS + ffmpeg).
- Add a daily request/spend ceiling per user and emit a structured log event when
  it trips so it can be alerted on.
- Return the existing envelope with `code: "rate_limit"` and `429` so the frontend
  needs no change.

### 3.2 Public, content-addressed audio endpoint

**Evidence.** `GET /audio/{audio_hash}.mp3` (`app/main.py:566`) takes no auth by
design (an `<audio>` tag cannot attach a bearer token). The filename is
`sha256(cleaned_tts_text)[:16]` (`app/tts.py:118`) — derived from the tutor's reply
text — and the cache is shared across **all users** (`app/tts.py:62`).

**Impact.** Access control is effectively "know or guess the reply text". Generic
replies ("Correct! Nice work.") are trivially guessable, and any reply an attacker
has seen or can reconstruct is retrievable by anyone, with no user binding, no
revocation, and no rate limit. Because the cache is global and content-addressed,
audio is not tied to the session that produced it, and `delete_session`
(`app/sessions.py:457`) never removes the file — so there is no way to honour a
deletion request for audio.

**Suggested improvement.**
- Key cached audio by a **random opaque id** (e.g. `uuid4().hex`) recorded in the
  session's `chat_history` entry, rather than a hash of the reply text. Keep the
  content hash only as an internal dedup key if cost savings matter.
- Gate `/audio/{id}.mp3` on the session cookie or a short-lived signed URL (HMAC
  with `AUTH_SECRET` plus expiry), so possession of the session is required.
- If the endpoint must stay public, use a non-guessable path, add rate limiting,
  and record the id on the session so deletion can clean it up.

### 3.3 Unbounded conversation history replayed to the model

**Evidence.** Every `/chat` turn loads the full history and sends all of it to the
model (`app/main.py:418` → `_dicts_to_messages`, `app/main.py:293`). Each turn
appends two messages and persists the entire list (`save_turn`, `app/main.py:465`).
Nothing truncates, windows, or summarizes. `mistake_log` is likewise append-only
and fully loaded each turn.

**Impact.** Two compounding failures:
1. **Cost** grows super-linearly per session — turn *n* re-sends turns 1..n-1 as
   input tokens, so a long session pays roughly O(n²).
2. **Hard failure** once history exceeds the model's context window: the request
   errors out and the user can no longer use that session at all.

**Suggested improvement.** Cap the replayed window (last N turns or last M tokens)
and add a rolling summary for older turns, persisted alongside `chat_history` so
the schema change stays additive. Bound `mistake_log` length as well.

### 3.4 Thread-pool exhaustion under concurrency

**Evidence.** `/chat` runs the entire synchronous graph in a worker thread:
`await asyncio.wait_for(asyncio.to_thread(graph_no_tts.invoke, state), timeout=50.0)`
(`app/main.py:458`). `asyncio.to_thread` uses the loop's default executor, sized
`min(32, cpu_count + 4)`. TTS (`app/main.py:527`) and every SQLite call use it too,
and TTS retries `time.sleep` inside the thread (`app/tts.py:293`, `:357`).

**Impact.** Each in-flight chat holds a thread for up to 50 s. On a small container
(1–2 vCPU ⇒ ~5–6 workers) roughly six concurrent chats saturate the pool, after
which *all* work queues behind it — including session loads and TTS — producing
latency spikes and timeouts that look like an outage.

**Suggested improvement.** Bound concurrency explicitly instead of relying on the
implicit pool size: a dedicated sized `ThreadPoolExecutor` (or an
`asyncio.Semaphore`) around graph execution, with backpressure (`503`/`429`) when
saturated. Prefer async-native LLM/TTS clients so the event loop is never blocked.

### 3.5 Unpinned dependencies

**Evidence.** `requirements.txt` uses only `>=` floors (`langgraph>=0.2.0`,
`langchain>=0.3.0`, `fastapi>=0.115.0`, …) with no lockfile, no upper bounds, and
no hashes.

**Impact.** Builds are not reproducible. A new minor release of
`langchain`/`langgraph` can change tool-calling behaviour and break the graph with
no code change on your side, and there is no known-good set to roll back to.

**Suggested improvement.** Add a lockfile (`pip-compile`/`uv pip compile`) and
install from it in the build. Pin exact versions, add upper bounds for the
fast-moving LangChain family, and keep dev-only tooling in `requirements-dev.txt`.

### 3.6 No CI

**Evidence.** There is no `.github/` directory, so nothing runs the test suite,
lints, type-checks, or scans dependencies on a pull request. Separately confirmed:
`pytest` and `httpx` are absent from `requirements.txt`, so a fresh clone cannot
even run the tests without extra installs.

**Impact.** Regressions reach production silently, and the verification plan in
`BACKEND_REVIEW_IMPROVEMENTS.md` depends on a developer remembering to run it by hand.

**Suggested improvement.** Add a CI workflow that installs from the lockfile and
runs `pytest`, `ruff check`, `ruff format --check`, `mypy` (or `pyright`), and

---

## 4. P1 — fix before scaling beyond a handful of users

### 4.1 SSE is simulated, not streamed

**Evidence.** The graph runs to completion (blocking) before any byte is sent
(`app/main.py:458`). Only then is the reply split on spaces and emitted with a
synthetic delay (`app/main.py:477-485`: `if i % 3 == 0: await asyncio.sleep(0.01)`).
No keep-alive/comment frames are emitted. Headers do set `X-Accel-Buffering: no`
(`app/main.py:499`), which is correct but insufficient.

**Impact.** Time-to-first-token equals full graph latency — up to the 50 s timeout,
and realistically 3–6 sequential LLM calls (§4.8). A proxy or load balancer with a
30–60 s idle timeout can drop the connection during that silence, and the client
sees a hang rather than the documented `token`/`done` events. The `asyncio.sleep`
typing effect also adds avoidable latency to every reply.

**Suggested improvement.** Stream from the model itself (`graph.astream_events` or
`llm.astream`) so first tokens leave immediately, and emit periodic SSE comment
heartbeats (`: ping\n\n`) while waiting. Preserve the existing event names and
framing (`token`, `done`, `error`) so the frontend contract is unchanged.

### 4.2 Timeouts do not cancel the underlying work

**Evidence.** `asyncio.wait_for(asyncio.to_thread(...), timeout=50.0)`
(`app/main.py:458`).

**Impact.** When the timeout fires, `wait_for` cancels the *awaitable*, but the
thread keeps running the Gemini calls to completion and its result is discarded.
The worker is occupied for the full duration anyway, so the timeout does not
actually reclaim capacity — it only changes what the user sees. This compounds
§3.4.

**Suggested improvement.** Pass real timeouts down into the SDK calls
(`make_llm` already accepts `request_timeout`, but `_get_retriever`/embeddings and
the TTS client have none) so work stops at the source, and run the graph in an
executor you can bound and drain.

### 4.3 Audio cache has no eviction; lock registry leaks

**Evidence.** `AUDIO_CACHE_DIR` is created once (`app/tts.py:62`) and files are only
ever written (`app/tts.py:323`, `:334`). Nothing deletes, expires, or caps them.
`_cache_locks` (`app/tts.py:65`) is a plain dict populated via
`setdefault(cache_path.name, threading.Lock())` (`app/tts.py:213`) and never pruned.

**Impact.** The README's own estimate is ~1.5 GB/month at 100 users against a
1 GB volume, so the volume fills. When it does, cache writes fail (logged as
non-fatal, `app/tts.py:325`) and `/audio` starts 404-ing for previously cached
replies, silently degrading the replay path. Separately, every distinct reply text
ever requested leaves a `threading.Lock` in memory forever — an unbounded leak
proportional to unique replies.

**Suggested improvement.** Add an LRU/TTL sweep (size cap plus last-access mtime)
run on a background task, and drop locks once the corresponding write finishes
(e.g. reference-counted entries, or a fixed-size lock pool). Longer term, move
audio to object storage (S3/R2) behind a CDN so the volume holds only
`sessions.db`.

### 4.4 Startup does not fail fast; health is always green

**Evidence.** Missing keys only log a warning (`app/main.py:91-96`); a Pinecone
connection failure is caught and swallowed (`app/main.py:106-108`). `GET /health`
returns a hardcoded `{"status": "ok"}` (`app/main.py:642`), and `railway.json`
uses `/health` as `healthcheckPath`.

**Impact.** A deploy with a bad `AUTH_SECRET` or an unreachable Pinecone passes its
health check, receives traffic, and fails per-request (401 for every user, or
degraded retrieval) instead of being rolled back. The `service_ready` log event
already carries `pinecone_initialized` — the signal exists but nothing acts on it.

**Suggested improvement.** Validate required configuration at startup and raise on
missing/blank values in production. Split liveness (`/health`, static) from
readiness (DB reachable, dependencies configured) and point the platform's health
check at readiness. Consider gating startup on Pinecone or making retrieval failure
an explicit, monitored degraded mode.

### 4.5 Validation errors log raw learner input

**Evidence.** `logger.info("Validation error: %s", exc.errors())`
(`app/main.py:200`). Pydantic's `errors()` includes an `input` key holding the
offending value — which for `/chat` is the learner's message and for `/tts` the
reply content.

**Impact.** This directly contradicts the privacy rule in `LOGGING.md:48`
("Do not log … full request bodies, raw learner messages"). Any user who trips a
length limit writes their message text into the log sink, and it is trivially
triggerable at `INFO` level, so it also becomes log-spam.

**Suggested improvement.** Log only `type` and `loc` (field path) from
`exc.errors()` — never `input`/`ctx`. Keep returning the detail to the caller in
the 422 body (that is the client's own data) but strip it from logs.


### 4.6 Auth: missing `sub`, no audience check, legacy secret

**Evidence.** `verify_token` decodes with the secret and `verify_exp` only
(`app/auth.py:31-36`) — no `aud`, no `iss`, no required claim. `_user_id` falls back
to `user.get("sub") or user.get("email")` (`app/main.py:258`). `auth_secret()` also
accepts `NEXTAUTH_SECRET` (`app/config.py:16`).

**Impact.** A token minted for any other service that happens to share the secret is
accepted. A token with neither `sub` nor `email` yields `None`, which flows into
`create_session`/`list_sessions` where `ValueError` is caught by a broad
`except Exception` and surfaced as a 500 `database_error` instead of a 401. And the
legacy-secret fallback widens the set of accepted signing keys indefinitely, with
nothing tracking when migration completes.

**Suggested improvement.** Require a non-empty `sub` at the auth layer and return
401 when absent; validate `aud`/`iss` against configured values; treat
`NEXTAUTH_SECRET` as an explicit, dated deprecation (log a warning when it is used)
and remove it once the frontend is confirmed migrated. Also map `ValueError` from
session helpers to a 400 rather than letting it become a 500.

### 4.7 SQLite is a single-writer ceiling and replica-unsafe

**Evidence.** A new connection is opened per call with `PRAGMA journal_mode=WAL`
and `busy_timeout` re-issued every time (`app/sessions.py:36-55`). `sessions.db`
lives on the mounted volume; `railway.json` starts a single Uvicorn process.

**Impact.** Fine for one replica, but horizontal scaling is unsafe: two containers
writing the same SQLite file on a shared volume can produce lock contention and, on
some volume backends, corruption. Re-issuing `journal_mode` per connection is also
wasted work (it is a persistent DB property and can require a lock). This is the
hard ceiling on scaling the API.

**Suggested improvement.** Decide and document the constraint now: either stay
single-replica (and say so explicitly, with a readiness check that fails if a second
replica is detected), or migrate to Postgres (e.g. `asyncpg` + SQLAlchemy) before
scaling out. Set WAL/busy-timeout once at `init_db` rather than per connection, and
add a connection pool if load grows.

### 4.8 Three to six LLM calls per user message

**Evidence.** Per turn: `retrieve` makes one tool-bound LLM call (`app/graph.py:171`);
`generate_response` makes one (`app/graph.py:232`), possibly a no-tools retry
(`app/graph.py:277`) and a final post-tool call (`app/graph.py:319`);
`apply_guardrails` always makes one more (`app/graph.py:383`) and possibly a
regeneration (`app/graph.py:413`). Each call builds a fresh
`ChatGoogleGenerativeAI` client (`app/tools.py:85`), so no connection is reused.

**Impact.** Minimum three paid LLM round-trips per message, worst case ~6, all
serialized — this drives both the latency (§4.1) and the per-message cost that
§3.1's missing quota would amplify. Creating a new client per call also discards
TLS/connection pooling on every hop.

**Suggested improvement.** Cache the LLM clients at module scope (one per
temperature/timeout profile). Consider folding the retrieval decision into the
response call, and sample or make the guardrail check asynchronous/offline rather
than paying for it on every turn. Track token usage per request in the structured
logs so cost is observable.

### 4.9 No metrics, tracing, or error tracking

**Evidence.** `LOGGING.md:50-55` lists Sentry and OpenTelemetry as recommendations;
neither is wired up. There is no `/metrics` endpoint, no counters, no histograms.

**Impact.** You can read individual requests but cannot answer "what is p95 chat
latency this hour", "what fraction of turns hit the graph timeout", or "how many TTS
failures in the last day" — exactly the questions that matter when the service
degrades. The `LOGGING.md` alerting advice assumes a log sink that is not configured
in the repo.

**Suggested improvement.** Add Sentry (or equivalent) for exception grouping and
release tracking, and a `/metrics` endpoint (`prometheus-client`) exposing request
counts/latency by route, graph-timeout count, TTS failure count, and cache hit
ratio. Keep the JSON logs as the correlation backbone.

### 4.10 `/health/deps` is unauthenticated and calls out to Pinecone

**Evidence.** `GET /health/deps` (`app/main.py:647`) requires no auth and performs
`_pinecone_index.describe_index_stats()` on every request (`app/main.py:664`). It
also reports whether `GEMINI_API_KEY` / `GOOGLE_EMBEDDING_API_KEY` are configured
(`app/main.py:657-658`).

**Impact.** An unauthenticated, cheap-to-call endpoint that triggers an external API
call is an amplification and cost vector, and it discloses which secrets are set.
Combined with §3.1 there is nothing limiting how often it is hit.

**Suggested improvement.** Make the detailed dependency view internal-only (auth or a
shared header), cache the probe result for a few seconds, and keep the public route
to a coarse status. Drop the `configured`/`missing` key reporting from the public
surface.


---

## 5. P2 — correctness, hygiene, and API longevity

### 5.1 No API versioning or contract tests

**Evidence.** All routes are unversioned (`/session`, `/chat`, `/sessions`). The
"compatibility contract" in `BACKEND_REVIEW_IMPROVEMENTS.md` §3 is a prose list, and
frontend verification was a one-time manual step.

**Impact.** Every future change is potentially breaking for the Next.js BFF, and
nothing detects it automatically. The frontend's expectations live only in
documentation.

**Suggested improvement.** Either mount the app under `/v1` (keeping the current paths
as aliases during a transition) or commit a snapshot of `openapi.json` and add a CI job
that fails when it changes unexpectedly. Add contract tests asserting the exact
payloads the frontend consumes.

### 5.2 Session deletion leaves audio behind, and the comment is wrong

**Evidence.** `delete_session` (`app/sessions.py:457-481`) deletes the row only, with a
comment claiming "Audio files are no longer stored on disk (Issue #43), so no audio
cleanup is needed here." That is false — `app/tts.py` writes MP3s to `AUDIO_CACHE_DIR`
and serves them from `/audio/{hash}.mp3`.

**Impact.** Orphaned audio accumulates with no owner (feeds §4.3), and a user who
deletes a conversation cannot have its audio removed — a data-retention/erasure gap.
The incorrect comment will mislead the next reader into preserving the behaviour.

**Suggested improvement.** Fix the comment, and either delete the referenced audio on
session deletion (after §3.2 gives audio a per-session identity) or document audio as
deliberately shared content-addressed data with a retention policy.

### 5.3 `/sessions` is unpaginated and reads the whole history column

**Evidence.** `list_sessions` runs `SELECT * FROM sessions WHERE user_id = ?`
(`app/sessions.py:169`) — pulling the full `chat_history` JSON blob for every session —
while the route only uses id/language/level/title/mistake_log/timestamps
(`app/main.py:382-393`). No `LIMIT`/`OFFSET`. `mistake_count` re-parses `mistake_log`
per row in the route.

**Impact.** Listing sessions gets slower as conversations grow, because the largest
column is transferred and discarded for every row.

**Suggested improvement.** Select only the needed columns, compute `mistake_count` with
`json_array_length(mistake_log)` in SQLite (or store a denormalized counter), and add
`limit`/`offset` with a sane default.

### 5.4 No request body size limit

**Evidence.** Pydantic caps individual strings (4000 chars for `message`, 20000 for TTS
`content`, `app/schemas.py:11-14`), but FastAPI/Uvicorn read and buffer the full body
before validation. No ASGI-level limit is configured.

**Impact.** A large body is fully buffered in memory before being rejected, so a few
concurrent oversized requests can inflate memory.

**Suggested improvement.** Enforce a `Content-Length` cap at the proxy and/or a small
ASGI middleware that rejects oversized bodies early with the standard error envelope.

### 5.5 Client-supplied `X-Request-ID` is trusted verbatim

**Evidence.** `request.headers.get("x-request-id") or uuid4().hex[:16]`
(`app/logging_config.py:117`), then echoed into the response header and every log
record.

**Impact.** A client can inject arbitrary-length content into your logs and response
headers. JSON encoding prevents newline injection, but unbounded length enables log
bloat and makes correlation spoofable.

**Suggested improvement.** Accept the header only if it matches a bounded pattern (e.g.
`^[A-Za-z0-9_-]{1,64}$`); otherwise generate a fresh ID.

### 5.6 Rename/delete races and `total_changes` misuse

**Evidence.** `rename`/`remove_session` call `_load_owned_session` and then a separate
`rename_session`/`delete_session` (`app/main.py:614-639`), so the ownership check and
the mutation are not atomic. `rename_session` reports success via `conn.total_changes`
(`app/sessions.py:502`) — a connection-lifetime counter (always 1 here when a row
matched, even if the title was unchanged) — and the route ignores the return value
anyway, returning `{"ok": True}` even if the row vanished between the two calls.

**Impact.** Low severity, but a delete can silently no-op while reporting success, and
`total_changes` is the wrong primitive for this check.

**Suggested improvement.** Use `cursor.rowcount` and fold ownership into the mutation
itself (`DELETE ... WHERE session_id = ? AND user_id = ?`) so it is one atomic
statement. Return `404` when nothing was affected.


### 5.7 Dead code and stale scaffolding

**Evidence.** `ChatResponse` (`app/schemas.py:110`) is never used. `ToolExecutionError`
(`app/exceptions.py:59`) is never raised. `GraphExecutionError` is imported into
`app/main.py:33` but unused. `app/main.py:71-76` contains a stale middleware comment
block and `app = None  # placeholder`, immediately overwritten at `app/main.py:132`.

**Impact.** Misleading to future contributors (the comment at `:71-76` describes a
middleware ordering that no longer matches the code) and it hides genuinely unused
error paths from coverage analysis.

**Suggested improvement.** Remove the unused schema/exception/import and the
placeholder, and let `ruff`/`vulture` in CI keep it clean.

### 5.8 `PINECONE_INDEX` is read before `.env` is loaded

**Evidence.** `app/pinecone_setup.py:22` evaluates `os.getenv("PINECONE_INDEX", ...)` at
module import, while `load_dotenv()` runs later inside `main()`
(`app/pinecone_setup.py:124`).

**Impact.** A `PINECONE_INDEX` set only in `.env` is ignored by the setup script, so it
silently targets the default `language-tutor` index — a real footgun if you provision a
second environment.

**Suggested improvement.** Call `load_dotenv()` before module-level constants, or read
the value inside `main()` after loading.

### 5.9 Import-time side effects

**Evidence.** `app/sessions.py:29-33` resolves `data_dir()` and `mkdir`s at import;
`app/tts.py:62-63` does the same for the audio cache.

**Impact.** Configuration must be fully present before import, importing the package
touches the filesystem, and tests must monkeypatch module attributes *after* import (as
`tests/test_auth_and_turns.py` does) — fragile and order-dependent.

**Suggested improvement.** Resolve paths lazily inside functions (or in the lifespan
handler) so imports stay pure and configuration is read at a defined time.

### 5.10 CORS allows credentials — guard against a wildcard origin

**Evidence.** `allow_credentials=True` with `allow_origins=cors_origins()`
(`app/main.py:136-142`). `cors_origins()` splits `CORS_ORIGINS` with no validation
(`app/config.py:19-24`).

**Impact.** Today the default is a safe explicit list. But setting `CORS_ORIGINS=*`
would make Starlette echo the request origin while allowing credentials — an
origin-reflection vulnerability introduced purely by an env-var change.

**Suggested improvement.** Fail startup if `*` appears in `CORS_ORIGINS` while
credentials are enabled, and validate that each entry parses as a scheme+host origin.

### 5.11 No Dockerfile; Python version unpinned

**Evidence.** Deployment relies on Nixpacks (`railway.json`, `nixpacks.toml`), which
declares `nixPkgs = ["python3", "ffmpeg"]` — no Python version, no ffmpeg version.

**Impact.** Rebuilds can silently pick up a different Python or ffmpeg, and the setup is
not portable to another host. Combined with §3.5 there is no reproducible artifact.

**Suggested improvement.** Add a `Dockerfile` from a pinned `python:3.11-slim` base that
installs `ffmpeg`, and pin the Nixpkgs channel/Python version for the Nixpacks path.

### 5.12 The test suite cannot run from a clean checkout

**Evidence.** `requirements.txt` has no `pytest`, `httpx`, or `anyio`, so
`python -m pytest tests/ -v` fails on a fresh install (confirmed in this review).
`tests/test_guardrails.py` and `tests/test_rag_eval.py` are script-style and require
live credentials but are not marked as integration tests. There is no `conftest.py`;
each test file manipulates `sys.path` itself.

**Impact.** The documented verification step does not work as written, and a CI job
(§3.6) would need ad-hoc installs. Live-credential scripts would fail in CI rather than
being skipped.

**Suggested improvement.** Add `requirements-dev.txt`, a `conftest.py` with shared
fixtures and path setup, and mark credential-dependent tests with
`@pytest.mark.integration` so they are deselected by default.


---

## 6. P3 — polish and verify

### 6.1 Verify proxy-header trust

`railway.json:8` starts Uvicorn with `--proxy-headers` but no `--forwarded-allow-ips`.
Uvicorn only honours `X-Forwarded-*` from trusted peers, so confirm the setting matches
Railway's proxy. Nothing currently depends on client IP, but this becomes load-bearing
the moment rate limiting (§3.1) keys on it.

### 6.2 Process/worker tuning is undocumented

The start command runs a single Uvicorn process with no `--workers`. That is the right
call given §4.7, but it should be an explicit, documented decision (with the expected
throughput ceiling) rather than a default.

### 6.3 Version metadata drift

`FastAPI(version="0.1.0")` (`app/main.py:132`) is hardcoded and unrelated to
`SERVICE_VERSION` in the logs, so the OpenAPI version and the deployed release can
disagree. Source both from one value.

### 6.4 Graceful shutdown

The lifespan logs `service_stopping` but does not drain in-flight `/chat` streams or
close the Pinecone/DB handles. Add an explicit drain window so deploys do not cut active
SSE streams mid-reply.

### 6.5 Security headers

`X-Content-Type-Options: nosniff` is set on audio responses only. Consider
`Strict-Transport-Security` and a restrictive `Content-Security-Policy` at the edge for
the JSON API, and verify the frontend proxy terminates TLS with HSTS.

---

## 7. Suggested implementation order

Sequenced so each step is independently shippable and the highest-risk items land first.
None of these require changing the browser-facing contract.

1. **Guard the wallet and the process** — §3.1 rate limiting/quotas, §3.4 bounded
   concurrency, §4.10 internal-only dependency health. Pure additive middleware.
2. **Make builds and merges safe** — §3.5 lockfile, §3.6 CI, §5.12 dev requirements and
   test markers. This is what lets everything after it be verified automatically.
3. **Stop unbounded growth** — §3.3 history windowing/summarization, §4.3 audio cache
   eviction and lock cleanup, §5.3 `/sessions` projection + pagination.
4. **Close the data-exposure and privacy gaps** — §3.2 audio identity and access,
   §5.2 deletion semantics, §4.5 validation logging, §4.6 auth claim hardening.
5. **Decide the scaling story** — §4.7 single-replica vs Postgres, §6.2 worker tuning,
   §5.11 Dockerfile with pinned runtime.
6. **Make degradation observable** — §4.4 fail-fast startup and readiness split,
   §4.9 metrics/Sentry, §6.4 graceful shutdown.
7. **Reduce latency and cost per turn** — §4.1 real token streaming, §4.2 real
   cancellation, §4.8 client reuse and guardrail cost.
8. **Hygiene** — §5.1 versioning/contract tests, §5.4–§5.10, §5.7 dead code, §6.1,
   §6.3, §6.5.

---

## 8. Verification for each change

The existing suite (`pytest tests/ -v`, 67 tests) is the regression baseline. For the
items above, add focused tests rather than relying on manual checks:

- **Rate limiting (§3.1):** assert `429` plus the `detail`/`code`/`request_id` envelope
  after the budget is exceeded, and that a second user is unaffected.
- **History windowing (§3.3):** a long synthetic session must send a bounded prompt
  while `chat_history` in SQLite still retains the full transcript.
- **Concurrency (§3.4):** fire N concurrent `/chat` requests against a stubbed graph and
  assert bounded in-flight work plus `503`/`429` backpressure rather than pool queuing.
- **Audio access (§3.2):** a request without the session credential must be rejected;
  audio for a deleted session must be unreachable; `/audio` must not accept a hash
  derived from another session's text.
- **Privacy (§4.5):** capture log records during a 422 and assert the learner's message
  text does not appear.
- **Auth (§4.6):** tokens with no `sub`, wrong `aud`, or wrong `iss` must return `401`.
- **Fail-fast (§4.4):** a missing `AUTH_SECRET` must fail startup in production mode, and
  readiness must report not-ready while retrieval is unconfigured.
- **Cache eviction (§4.3):** after exceeding the cap the oldest files are gone and
  `_cache_locks` does not retain entries for completed writes.
- **Contract (§5.1):** snapshot `openapi.json` and assert the documented payload shapes
  for `/sessions`, `/session/{id}`, and the SSE event sequence.

`pip-audit`, with a coverage floor.
