"""Observability: structured JSON logs, Prometheus metrics, latency timing."""
import logging
import sys
import time
import uuid
from contextlib import contextmanager

import structlog
from prometheus_client import Counter, Histogram, start_http_server

from app.config import settings

# ---------- structured logging ----------
logging.basicConfig(
    format="%(message)s", stream=sys.stdout, level=settings.LOG_LEVEL,
)
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(settings.LOG_LEVEL)),
    logger_factory=structlog.PrintLoggerFactory(),
)
log = structlog.get_logger(app=settings.APP_NAME, env=settings.ENVIRONMENT)

# ---------- prometheus metrics ----------
FRAME_COUNTER = Counter("frames_processed_total", "Total video frames analyzed")
POSE_SCORE_HIST = Histogram("pose_correctness_score", "Distribution of pose correctness scores",
                             buckets=(0, .2, .4, .5, .6, .7, .8, .9, 1.0))
CV_LATENCY = Histogram("cv_inference_latency_seconds", "MediaPipe+Keras inference latency")
AGENT_LATENCY = Histogram("agent_latency_seconds", "Ollama/ADK agent response latency")
AGENT_ERRORS = Counter("agent_errors_total", "Agent call failures", ["reason"])
GUARDRAIL_BLOCKS = Counter("guardrail_blocks_total", "Requests blocked by guardrails", ["rule"])
REQUESTS = Counter("requests_total", "Total analysis requests", ["status"])

_metrics_started = False


def start_metrics_server():
    """Idempotently expose /metrics on METRICS_PORT for Prometheus scraping."""
    global _metrics_started
    if not _metrics_started:
        try:
            start_http_server(settings.METRICS_PORT)
            _metrics_started = True
            log.info("metrics_server_started", port=settings.METRICS_PORT)
        except OSError:
            # already running (e.g. Streamlit re-run) — safe to ignore
            _metrics_started = True


def new_trace_id() -> str:
    return uuid.uuid4().hex[:12]


@contextmanager
def timed(histogram: Histogram, **log_fields):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        histogram.observe(elapsed)
        log.debug("timed_block", latency_ms=round(elapsed * 1000, 1), **log_fields)
