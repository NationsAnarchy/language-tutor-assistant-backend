# AGENTS.md

## Project overview

This repository contains the FastAPI backend for the Trilingual Language Tutor.
It supports English, Korean, and Japanese tutoring through LangGraph, Gemini,
Pinecone retrieval, SQLite session storage, and Gemini-based text-to-speech.

## Working conventions

- Keep changes focused on the requested behavior; do not overwrite unrelated
  work in a dirty worktree.
- Read nearby code and existing tests before changing an API contract or graph
  behavior.
- Keep secrets in environment variables. Never commit `.env`, API keys, JWTs,
  database files, or generated audio.
- Preserve the error response shape (`detail`, `code`, and `request_id`) and
  the existing authentication and session-ownership checks.
- Treat persisted session data and audio-cache paths as user data. Do not
  delete or reset them unless explicitly requested.

## Code layout

- `app/main.py`: FastAPI application, routes, CORS, lifespan, and error handling.
- `app/graph.py`: LangGraph tutor flow and guardrails.
- `app/agent_state.py`: typed graph state.
- `app/agent_execution.py`: tool execution and private tool-result context.
- `app/tools.py`: retrieval, exercise, grading, and mistake-tracking tools.
- `app/sessions.py`: SQLite session and mistake-log persistence.
- `app/auth.py`: NextAuth JWT verification.
- `app/tts.py`: Gemini TTS, ffmpeg conversion, and audio caching.
- `tests/`: API, authentication, graph/turn, guardrail, and RAG evaluation tests.

## Development and verification

- Install dependencies with `pip install -r requirements.txt`.
- Run the API with `uvicorn app.main:app --reload`.
- Run the regular suite with `python -m pytest tests/ -v`.
- Run guardrail tests only with an explicitly supplied local token:
  `TEST_AUTH_TOKEN=... python tests/test_guardrails.py`.
- Run RAG evaluation with `python tests/test_rag_eval.py`; it may require
  configured external services and credentials.

## Change guidance

- Add or update focused tests for behavior changes.
- Keep streaming/SSE response behavior compatible with existing clients.
- For database changes, consider existing deployments that persist `sessions.db`
  under `RAILWAY_VOLUME_PATH`.
- Do not log JWTs, API keys, full authorization headers, or private tool context.
