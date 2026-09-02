"""Input/output guardrails: validation, rate limiting, LLM output safety."""
import time
from collections import deque

from app.config import settings
from app.telemetry import log, GUARDRAIL_BLOCKS


class GuardrailError(Exception):
    def __init__(self, rule: str, message: str):
        self.rule = rule
        self.message = message
        super().__init__(message)


def validate_upload(file_bytes: bytes, filename: str):
    """Reject oversized or wrong-type uploads before they hit the CV pipeline."""
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > settings.MAX_UPLOAD_MB:
        GUARDRAIL_BLOCKS.labels(rule="upload_size").inc()
        raise GuardrailError("upload_size", f"File too large ({size_mb:.1f}MB > {settings.MAX_UPLOAD_MB}MB)")
    if not filename.lower().endswith((".mp4", ".mov", ".avi", ".jpg", ".jpeg", ".png")):
        GUARDRAIL_BLOCKS.labels(rule="file_type").inc()
        raise GuardrailError("file_type", "Unsupported file type")


def sanitize_agent_output(text: str) -> str:
    """Cap length, strip banned/unsafe claims from LLM feedback before showing to user."""
    if not text:
        return "No feedback available."
    lowered = text.lower()
    for term in settings.BANNED_TERMS:
        if term in lowered:
            GUARDRAIL_BLOCKS.labels(rule="banned_term").inc()
            log.warning("guardrail_banned_term_stripped", term=term)
            text = text.replace(term, "[redacted]")
    if len(text) > settings.MAX_LLM_OUTPUT_CHARS:
        text = text[: settings.MAX_LLM_OUTPUT_CHARS].rsplit(".", 1)[0] + "."
    return text.strip()


class RateLimiter:
    """Simple sliding-window rate limiter, per-process (fine for single-user Streamlit)."""

    def __init__(self, max_per_min: int = None):
        self.max_per_min = max_per_min or settings.RATE_LIMIT_PER_MIN
        self._hits = deque()

    def allow(self) -> bool:
        now = time.time()
        while self._hits and now - self._hits[0] > 60:
            self._hits.popleft()
        if len(self._hits) >= self.max_per_min:
            GUARDRAIL_BLOCKS.labels(rule="rate_limit").inc()
            return False
        self._hits.append(now)
        return True


rate_limiter = RateLimiter()
