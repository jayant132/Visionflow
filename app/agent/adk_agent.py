"""Google ADK agent that turns structured pose-analysis data into natural
coaching feedback, using a local Ollama model (via LiteLlm) as the backend —
no API key, no external network call, fully free/offline.

Resilient by design: if google-adk/litellm/the local Ollama server is
unavailable, falls back to deterministic templated feedback built from the
rule engine output, so the product never breaks for the user.
"""
import asyncio
import time

from app.config import settings
from app.guardrails import sanitize_agent_output
from app.telemetry import log, AGENT_LATENCY, AGENT_ERRORS

_AGENT_INSTRUCTION = """You are a certified yoga instructor AI. You receive a JSON
object describing a student's current asana, a correctness score (0-1), and a list
of geometric corrections. Give concise (2-3 sentence), encouraging, actionable
feedback in plain language. Never give medical advice or claims. If score >= 0.85,
praise the form briefly."""

_runner = None
_agent_init_failed = False


def _build_runner():
    """Lazily construct the ADK Agent + Runner, pointed at local Ollama."""
    global _runner, _agent_init_failed
    if _runner is not None or _agent_init_failed:
        return _runner
    try:
        from google.adk.agents import Agent
        from google.adk.models.lite_llm import LiteLlm
        from google.adk.runners import InMemoryRunner

        agent = Agent(
            name="yoga_coach",
            model=LiteLlm(
                model=f"ollama_chat/{settings.OLLAMA_MODEL}",
                api_base=settings.OLLAMA_BASE_URL,
            ),
            instruction=_AGENT_INSTRUCTION,
        )
        _runner = InMemoryRunner(agent=agent, app_name=settings.APP_NAME)
        log.info("adk_agent_initialized", provider="ollama", model=settings.OLLAMA_MODEL,
                  base_url=settings.OLLAMA_BASE_URL)
    except Exception as e:
        log.error("adk_agent_init_failed", error=str(e))
        _agent_init_failed = True
        _runner = None
    return _runner


def _fallback_feedback(analysis: dict) -> str:
    score = analysis.get("score", 0)
    corrections = analysis.get("corrections", [])
    if score >= 0.85:
        return f"Great {analysis.get('asana', 'pose')}! Your alignment looks solid — hold steady and breathe."
    if corrections:
        return f"Almost there: {', '.join(corrections)}. Adjust gradually and re-check your alignment."
    return "Keep practicing — hold the pose and try to match the reference alignment."


async def _run_agent_async(analysis: dict) -> str:
    from google.genai import types

    runner = _build_runner()
    if runner is None:
        return _fallback_feedback(analysis)

    session = await runner.session_service.create_session(
        app_name=settings.APP_NAME, user_id="streamlit_user"
    )
    prompt = f"Pose analysis JSON: {analysis}"
    content = types.Content(role="user", parts=[types.Part(text=prompt)])

    final_text = None
    async for event in runner.run_async(
        user_id="streamlit_user", session_id=session.id, new_message=content
    ):
        if event.is_final_response() and event.content and event.content.parts:
            final_text = event.content.parts[0].text
    return final_text or _fallback_feedback(analysis)


def get_feedback(analysis: dict) -> dict:
    """Sync entrypoint for Streamlit. Returns {text, latency_ms, source}."""
    start = time.perf_counter()
    if not settings.USE_LLM_FEEDBACK:
        text = _fallback_feedback(analysis)
        return {"text": text, "latency_ms": 0, "source": "rules_only"}

    try:
        text = asyncio.run(
            asyncio.wait_for(_run_agent_async(analysis), timeout=settings.AGENT_TIMEOUT_SECONDS)
        )
        source = "ollama_llm" if _runner is not None else "fallback"
    except asyncio.TimeoutError:
        AGENT_ERRORS.labels(reason="timeout").inc()
        log.warning("agent_timeout", timeout=settings.AGENT_TIMEOUT_SECONDS)
        text, source = _fallback_feedback(analysis), "fallback_timeout"
    except Exception as e:
        AGENT_ERRORS.labels(reason="exception").inc()
        log.error("agent_call_failed", error=str(e))
        text, source = _fallback_feedback(analysis), "fallback_error"

    elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
    AGENT_LATENCY.observe(elapsed_ms / 1000)
    return {"text": sanitize_agent_output(text), "latency_ms": elapsed_ms, "source": source}
