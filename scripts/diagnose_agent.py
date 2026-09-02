"""Standalone diagnostic: isolates exactly where the ADK+Ollama agent
build/call is failing, with a full traceback (the app itself swallows
this exception on purpose so the UI never crashes).

Run from the project root:
    PYTHONPATH=. python3 scripts/diagnose_agent.py
"""
import asyncio
import sys
import traceback

sys.path.insert(0, ".")

from app.config import settings

print(f"OLLAMA_MODEL     = {settings.OLLAMA_MODEL}")
print(f"OLLAMA_BASE_URL  = {settings.OLLAMA_BASE_URL}")
print(f"USE_LLM_FEEDBACK = {settings.USE_LLM_FEEDBACK}")
print("-" * 60)

print("[1/3] Importing google-adk + litellm...")
try:
    from google.adk.agents import Agent
    from google.adk.models.lite_llm import LiteLlm
    from google.adk.runners import InMemoryRunner
    print("      OK")
except Exception:
    print("      FAILED on import:")
    traceback.print_exc()
    sys.exit(1)

print("[2/3] Building Agent + Runner...")
try:
    agent = Agent(
        name="yoga_coach",
        model=LiteLlm(model=f"ollama_chat/{settings.OLLAMA_MODEL}", api_base=settings.OLLAMA_BASE_URL),
        instruction="You are a yoga coach. Reply in one short sentence.",
    )
    runner = InMemoryRunner(agent=agent, app_name="diagnose")
    print("      OK")
except Exception:
    print("      FAILED building agent/runner:")
    traceback.print_exc()
    sys.exit(1)

print("[3/3] Running a real test call through Ollama...")


async def _run():
    from google.genai import types
    session = runner.session_service.create_session(app_name="diagnose", user_id="tester")
    if hasattr(session, "__await__"):
        session = await session
    content = types.Content(role="user", parts=[types.Part(text="Say hello in 5 words.")])
    async for event in runner.run_async(user_id="tester", session_id=session.id, new_message=content):
        if event.is_final_response() and event.content and event.content.parts:
            print("      RESPONSE:", event.content.parts[0].text)
            return
    print("      No final response event received.")


try:
    asyncio.run(_run())
    print("-" * 60)
    print("SUCCESS — the agent pipeline works end-to-end.")
except Exception:
    print("      FAILED during run_async:")
    traceback.print_exc()
    sys.exit(1)
