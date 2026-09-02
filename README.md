# AI Yoga Pose Coach — Production-Grade Reference Project

Yoga pose correction agent. Upload a clip → MediaPipe extracts skeleton
landmarks → joint angles are computed → a Keras classifier (with a
transparent rule-engine fallback/backbone) scores correctness → a **Google
ADK agent** running on a **local Ollama model** (via LiteLlm) turns that
into natural-language coaching feedback. Streamlit UI, Prometheus metrics,
structured logs, and guardrails included. **Fully free, fully local — no
API keys, no external calls.**

## Why these choices (kept free / lightweight)
- **MediaPipe Pose**, not a full live-webrtc pipeline — CPU-only, no GPU
  needed, works great on sampled video frames.
- **Streamlit + file upload**, not live webcam streaming — avoids the heavy
  `streamlit-webrtc`/`aiortc` stack; frames are sampled from an uploaded clip
  (stride configurable in the sidebar).
- **Rule engine is the source of truth** for corrections (deterministic,
  explainable, zero cost); **Keras MLP** is a thin classifier on top of angle
  features that can be retrained (`scripts/train_model.py`, synthetic data
  generator included so it works with zero setup).
- **Ollama via Google ADK's `LiteLlm` wrapper** — a small local model
  (`llama3.2:1b` by default) turns structured analysis into encouraging
  natural language, entirely on your machine. If ADK, Ollama, or the model
  is unavailable, the agent **falls back to templated feedback**
  automatically — the app never breaks.

## Architecture

```
Streamlit UI (app/main.py)
   │  upload video, sample every Nth frame
   ▼
PoseEstimator (app/cv/pose_estimator.py)         — MediaPipe, on-device
   │  33 landmarks → 6 joint angles
   ▼
classify() (app/cv/pose_classifier.py)           — Keras MLP (or rule fallback)
   │  {asana, score, corrections, confidence, source}
   ▼
Guardrails (app/guardrails)                       — upload validation,
   │                                                 rate limiting, output filter
   ▼
ADK Agent (app/agent/adk_agent.py)                — local Ollama LLM via LiteLlm,
   │                                                 timeout + fallback
   ▼
Streamlit renders: skeleton overlay, score, corrections, AI feedback,
live latency + trace ID

Cross-cutting: app/telemetry — structlog JSON logs, Prometheus /metrics
(frame counters, score histogram, CV + agent latency histograms, error/
guardrail counters), exposed on :8000, scrapeable by the bundled
docker/prometheus.yml.
```

## Project layout
```
app/
  main.py                 Streamlit entrypoint
  config.py                12-factor settings (env-driven)
  cv/
    pose_estimator.py      MediaPipe wrapper
    angle_utils.py         joint-angle geometry
    pose_classifier.py     Keras MLP + rule fallback
  agent/
    pose_rules.py          deterministic per-asana rules
    adk_agent.py            Google ADK + Ollama(LiteLlm) agent, with fallback
  guardrails/               input/output validation, rate limiting
  telemetry/                logging, Prometheus metrics, timing helpers
  utils/video_utils.py      frame sampling + skeleton drawing
  models/                   trained pose_classifier.keras (gitignored)
scripts/train_model.py     trains the Keras model on synthetic angle data
tests/                      pytest unit tests (guardrails + rules)
docker/                     Dockerfile, docker-compose (+ Ollama + Prometheus)
```

## Quickstart

### 1. Install & run Ollama (once)
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &                  # starts the local server on :11434
ollama pull llama3.2:1b         # small, fast model — swap for any model you have
```

### 2. Run the app
```bash
cp .env.example .env            # defaults already point at local Ollama
pip install -r requirements.txt
python scripts/train_model.py   # optional: trains Keras classifier (~10s, CPU)
streamlit run app/main.py
# UI:      http://localhost:8501
# metrics: http://localhost:8000/metrics
```

### Or fully containerized (spins up Ollama + the app + Prometheus)
```bash
docker compose -f docker/docker-compose.yml up --build
docker exec -it $(docker ps -qf name=ollama) ollama pull llama3.2:1b   # first run only
```

## Supported poses (MVP set, extend in `app/agent/pose_rules.py`)
Tree Pose, Warrior II, Downward Dog — add new entries to `ASANA_RULES` and
retrain (`scripts/train_model.py` auto-picks up new classes).

## Configuration (`.env`)
| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_MODEL` | `llama3.2:1b` | Model tag pulled in Ollama |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address |
| `USE_LLM_FEEDBACK` | `true` | Set `false` to use rule-engine-only feedback (zero LLM latency) |
| `AGENT_TIMEOUT_SECONDS` | `20` | Max wait before falling back to templated feedback |
| `MAX_UPLOAD_MB` | `50` | Guardrail: max upload size |
| `RATE_LIMIT_PER_MIN` | `30` | Guardrail: requests per minute |
| `METRICS_PORT` | `8000` | Prometheus `/metrics` port |

## Observability
- **Logs**: structured JSON via `structlog`, includes `trace_id` per request.
- **Metrics** (`prometheus_client`, port `METRICS_PORT`): `frames_processed_total`,
  `pose_correctness_score` (histogram), `cv_inference_latency_seconds`,
  `agent_latency_seconds`, `agent_errors_total`, `guardrail_blocks_total`,
  `requests_total`.
- **Latency**: shown live in the UI per request (CV + agent stages timed
  independently via the `timed()` context manager).

## Guardrails
- Upload validation (size cap, allowed extensions).
- Per-process sliding-window rate limiter.
- LLM output sanitization: banned-term redaction (no medical/guarantee claims),
  max length cap.
- Agent call timeout with automatic fallback to templated feedback.

## Tests
```bash
make test   # or: pytest -q
```

## Notes / next steps for real production
- Swap the synthetic training set in `scripts/train_model.py` for real
  labeled landmark data once available.
- Add OpenTelemetry exporter (SDK already in requirements) to ship traces to
  Jaeger/Tempo if you need distributed tracing beyond single-process metrics.
- Swap the per-process `RateLimiter` for Redis-backed limiting if you deploy
  multi-instance.
- For a bigger/better model, swap `OLLAMA_MODEL` (e.g. `llama3.1:8b`,
  `mistral`, `qwen2.5:7b`) — no code changes needed, just re-pull and update `.env`.
