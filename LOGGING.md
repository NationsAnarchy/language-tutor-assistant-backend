# Backend Logging and Observability

## Strategy

The backend uses Python's standard `logging` library and writes one JSON object per line to stdout. This is intentional for local containers and Railway: the runtime collects stdout, so application code does not manage persistent log files or rotation.

No logging, monitoring, or tracing dependency has been added. The current design is compatible with a log sink such as Railway logs, Better Stack, Datadog, Grafana Loki, Axiom, or CloudWatch.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `LOG_LEVEL` | `INFO` | Minimum level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL`. Invalid values safely use `INFO`. |
| `APP_ENV` | `development` | Environment label included in every log event. |
| `SERVICE_NAME` | `language-tutor-assistant-backend` | Service label included in every log event. |
| `SERVICE_VERSION` | unset | Optional release or Git SHA included when set. |

Use `LOG_LEVEL=DEBUG` only temporarily when diagnosing a problem. Production should normally use `INFO` or `WARNING` based on desired request-log volume.

## Common event types

All records include `timestamp` (UTC), `level`, `logger`, `module`, `lineno`, `message`, `service`, and `environment`. Exception records also include an `exception` traceback.

| Event | When emitted | Useful fields |
|---|---|---|
| `service_starting` | FastAPI lifespan begins | service/environment metadata |
| `database_ready` | SQLite initialization completes | `database` |
| `service_ready` | Startup checks complete | `startup_duration_ms`, `pinecone_initialized`, `gemini_configured` |
| `service_stopping` | Lifespan shutdown begins | service/environment metadata |
| `http_request_completed` | A request produces a response or fails before one is made | `request_id`, `method`, `path`, `status_code`, `duration_ms` |

Completed HTTP requests log at `INFO` for 2xx/3xx, `WARNING` for 4xx, and `ERROR` for 5xx. Uvicorn's unstructured access logger remains suppressed at `WARNING` because `http_request_completed` replaces it with a consistent JSON event.

`duration_ms` measures middleware processing until the response object is created. For streaming `/chat` responses, it does **not** represent the time until the final SSE token is sent. Add a dedicated stream-completion event if end-to-end SSE duration is needed.

## Request correlation

The `RequestIdMiddleware` accepts a client-provided `X-Request-ID` or creates a 16-character hex ID. It:

1. Returns the value in the response `X-Request-ID` header.
2. Includes it in the existing JSON error body (`detail`, `code`, `request_id`).
3. Stores it in a Python `contextvars.ContextVar`, which the structured log handler adds to records produced in that async request.

`contextvars` avoids the race caused by adding/removing filters on the global root logger while concurrent requests are running.

## Privacy and security

Do not log API keys, JWTs, authorization headers, cookies, full request bodies, private tool context, raw learner messages, or full model responses. The request-completion event deliberately contains only method, path, status, duration, and correlation ID. Session/user identifiers should be added only after a privacy review and should be pseudonymized when possible.

## Monitoring recommendations

1. Configure a platform log sink and retain/search `ERROR` logs by `request_id`.
2. Alert on sustained `http_request_completed` 5xx rate, repeated graph timeouts, and `service_ready` events where `pinecone_initialized` is false unexpectedly.
3. Add Sentry next if exception grouping, deployment tracking, and error notifications are needed.
4. Add OpenTelemetry traces and metrics only when cross-service latency analysis (Next.js BFF → FastAPI → Gemini/Pinecone) becomes necessary. They complement these JSON logs rather than replace them.