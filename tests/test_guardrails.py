import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
from app.guardrails import validate_upload, sanitize_agent_output, GuardrailError, RateLimiter


def test_validate_upload_rejects_bad_extension():
    with pytest.raises(GuardrailError):
        validate_upload(b"data", "malware.exe")


def test_validate_upload_accepts_mp4():
    validate_upload(b"small file", "clip.mp4")  # should not raise


def test_sanitize_strips_banned_terms():
    out = sanitize_agent_output("This is a guarantee of a cure.")
    assert "guarantee" not in out.lower() or "[redacted]" in out


def test_sanitize_caps_length():
    long_text = "word " * 500
    out = sanitize_agent_output(long_text)
    assert len(out) <= 620


def test_rate_limiter_blocks_after_limit():
    rl = RateLimiter(max_per_min=2)
    assert rl.allow() is True
    assert rl.allow() is True
    assert rl.allow() is False
