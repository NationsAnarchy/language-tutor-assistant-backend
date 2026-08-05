# Backend Review Findings and Improvement Plan

**Repository:** `language-tutor-assistant-backend`  
**Reviewed:** 2026-08-05  
**Related frontend:** `../language-tutor-assistant-frontend`

## 1. Review scope

This review covers the FastAPI application, JWT authentication, SQLite session
storage, text-to-speech and cached audio delivery, CORS, health checks,
Pydantic schemas, tests, and the API contract consumed by the corresponding
Next.js frontend.

The objective is to make targeted production-readiness improvements without
changing established browser-facing routes, payloads, SSE events, or error
response shapes.

## 2. Current baseline

### Backend strengths

- Authentication verifies expiring HS256 JWTs and requires a shared secret.
  `AUTH_SECRET` is canonical, with `NEXTAUTH_SECRET` retained as a migration
  fallback.
- Protected session routes enforce authenticated ownership before exposing or
  mutating session data.
- Typed application errors use a consistent JSON envelope containing `detail`,
  `code`, and `request_id`.
- Blocking SQLite, Pinecone health-probe, cached-file, and main graph work is
  mostly moved off the async event loop using `asyncio.to_thread`.
- SQLite uses WAL mode and foreign-key enforcement.
- CORS uses configured explicit origins rather than an allow-all policy.
- Session, mutation, and health routes already have Pydantic response models.
- The test suite covers authentication, ownership, validation, request IDs,
  guardrails, practice modes, text cleanup, and persistence helpers.

### Frontend integration baseline

- Browser API calls consistently target the Next.js same-origin proxy at
  `/api/proxy/*`.
- The frontend token route signs `AUTH_SECRET` JWTs with HS256 and a one-hour
  expiry, matching backend verification.
- Chat is consumed as an SSE stream through the proxy.
- Historical cached audio is requested as
  `/api/proxy/audio/{audio_hash}.mp3`; the backend audio route is intentionally
  public because an HTML `<audio>` element cannot attach a bearer token.

## 3. Compatibility contract to preserve

The following contracts are mandatory during implementation:

### Routes

- `POST /session`
- `GET /session/{session_id}`
- `GET /sessions`
- `POST /chat` (existing SSE event format and terminal behavior)
- `POST /session/{session_id}/tts`
- `GET /audio/{audio_hash}.mp3`
- `GET /session/{session_id}/mistakes`
- `PATCH /session/{session_id}`
- `DELETE /session/{session_id}`
- `GET /health`
- `GET /health/deps`

### Authentication and error handling

- Continue accepting bearer tokens issued by the frontend with `AUTH_SECRET`,
  HS256, a `sub`, and an expiry.
- Keep `NEXTAUTH_SECRET` as a compatibility fallback unless and until deployment
  configuration has been explicitly migrated.
- Keep the `detail`, `code`, and `request_id` fields on all non-SSE error
  responses.
- Preserve `X-Request-ID` response propagation.

### Payloads and responses

- Do not rename existing request fields, response fields, enum values, audio
  URLs, or session persistence fields.
- Valid existing frontend payloads must remain valid.
- Validation improvements may reject malformed, empty, oversized, or unsafe
  values with the existing `422` error envelope.

## 4. Findings and planned improvements

| Priority | Finding | Improvement | Compatibility approach |
| --- | --- | --- | --- |
| High | String validation is inconsistent across request models. Identifiers and TTS inputs lack uniform whitespace, emptiness, and size constraints. | Harden Pydantic v2 request schemas with `model_config` and `field_validator` rules for identifiers, titles, text, language, and speed inputs. | Preserve field names and all valid current payloads. Invalid requests continue to return the existing structured 422 response. |
| High | `GET /health/deps` includes raw Pinecone exception text in its public response. | Log detailed dependency failures server-side and expose stable sanitized status values externally. | Maintain the `status` and `dependencies` fields; remove only internal diagnostic leakage. |
| High | Concurrent TTS requests for identical uncached text can duplicate Gemini calls and write the same cache file concurrently. | Add a per-cache-key synchronization boundary around cache population. | Preserve TTS output format, cached hash/URL behavior, and route shape. |
| Medium | API OpenAPI metadata is sparse. | Add tags, summaries, descriptions, explicit response documentation, and SSE media-type documentation. | Documentation-only; no route or runtime payload changes. |
| Medium | API schemas are embedded in `app/main.py`, making the HTTP layer harder to maintain and test as it grows. | Move schemas into a dedicated `app/schemas.py` module using Pydantic v2 syntax and typed models. | Internal refactor only; preserve schema names/fields and generated API surface. |
| Medium | Route tests use synchronous `TestClient` for an asynchronous API and do not comprehensively test CORS/OpenAPI-facing behavior. | Add focused async tests with `httpx.AsyncClient` and ASGI transport for public API behavior. | Test-only improvement that protects the frontend contract. |
| Medium | SQLite uses WAL but can still surface avoidable lock errors during concurrent writes. | Set an explicit SQLite busy timeout and review transactional write boundaries without altering the schema. | Keep existing `sessions.db`, data location, and persistence format compatible with deployed volumes. |
| Low | Cached audio is read fully into process memory before response creation. | Use Starlette file streaming after current hash/path safety checks. | Preserve public access, MP3 media type, cache headers, and the `/audio/{hash}.mp3` URL. |
| Low | Environment configuration is represented by scattered helper functions. | Consolidate settings validation with Pydantic v2 while retaining existing environment-variable names and defaults. | Existing `.env`, Railway, and frontend deployment configuration remains valid. |

## 5. Implementation sequence

1. **Document and establish regression coverage**
   - Keep this document updated as implementation decisions are completed.
   - Add tests that lock down current frontend-facing routes, status codes,
     error envelopes, CORS preflight, request IDs, OpenAPI, SSE headers, and
     cached-audio headers.

2. **Schema and configuration hardening**
   - Introduce dedicated Pydantic v2 schema definitions.
   - Normalize and bound external string fields while retaining valid existing
     payloads.
   - Centralize environment parsing/validation without hardcoded secrets or
     deployment-only values.

3. **Operational and security improvements**
   - Sanitize public dependency-health failures.
   - Improve SQLite busy handling while retaining WAL and persisted data.
   - Add concurrency-safe TTS cache population.
   - Stream cached audio files efficiently after path safety checks.

4. **OpenAPI and route polish**
   - Add endpoint tags, descriptions, summaries, success statuses, and
     error-response metadata.
   - Explicitly document chat as an SSE response while retaining its exact
     streaming implementation and events.

5. **Frontend compatibility verification**
   - Compare frontend API client payloads and response expectations with the
     FastAPI OpenAPI document.
   - Verify Next.js proxy behavior for JSON errors, SSE streams, request IDs,
     TTS generation, and persisted cached-audio playback.

## 6. Non-goals

- Replacing SQLite with SQLAlchemy or another database in this improvement pass.
- Changing OAuth/NextAuth providers, JWT claim names, signing algorithm, or
  token lifetime.
- Versioning or renaming existing routes.
- Changing chat SSE event names, framing, ordering, or timeout behavior.
- Changing existing session data formats or deleting persisted sessions/audio.
- Removing the legacy `NEXTAUTH_SECRET` fallback before deployment migration is
  explicitly complete.

## 7. Verification plan

### Backend

Run from `/home/incarnation/Development/language-tutor-assistant-backend`:

```bash
.venv/bin/python -m pytest tests/ -v
```

External guardrail and RAG scripts require explicitly configured service
credentials/tokens and should not be treated as local unit-test failures when
those prerequisites are unavailable.

Verify additionally:

- `/openapi.json` lists the existing routes and intended request/response
  schemas.
- CORS allows only configured frontend origins, expected methods, and expected
  headers.
- Errors retain `detail`, `code`, `request_id`, and an `X-Request-ID` header.
- Chat remains SSE and is not buffered by application changes.
- Cached audio retains its media type and cache-control headers.

### Frontend

Run from `/home/incarnation/Development/language-tutor-assistant-frontend`:

```bash
npm test
npm run lint
npx tsc --noEmit
npm run build -- --webpack
```

For local end-to-end verification, configure matching `AUTH_SECRET` values in
both applications, start FastAPI with an explicit local frontend origin in
`CORS_ORIGINS`, and confirm session creation, streaming chat, TTS, reloadable
cached audio, and structured error presentation through `/api/proxy`.

## 8. Implementation status

- [x] Review completed and plan documented.
- [x] Add/adjust regression tests for selected improvements.
- [x] Implement schema/configuration hardening.
- [x] Implement operational and security improvements.
- [x] Improve targeted OpenAPI route metadata.
- [x] Run backend verification (`pytest tests/ -v`: 67 passed; OpenAPI contract checked).
- [x] Run frontend compatibility verification (`npm test`: 34 passed; `npm run lint` and `npx tsc --noEmit` passed).