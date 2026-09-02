"""Centralized config. All values overridable via env / .env (12-factor)."""
import os


class Settings:
    APP_NAME = "yoga-pose-agent"

    # --- LLM (local Ollama via ADK's LiteLlm wrapper, no API key needed) ---
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    USE_LLM_FEEDBACK = os.getenv("USE_LLM_FEEDBACK", "true").lower() == "true"
    AGENT_TIMEOUT_SECONDS = float(os.getenv("AGENT_TIMEOUT_SECONDS", "20"))

    # --- CV pipeline ---
    MIN_LANDMARK_CONFIDENCE = float(os.getenv("MIN_LANDMARK_CONFIDENCE", "0.5"))
    MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))
    FRAME_SAMPLE_STRIDE = int(os.getenv("FRAME_SAMPLE_STRIDE", "5"))  # analyze every Nth frame

    # --- Guardrails ---
    RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "30"))
    MAX_LLM_OUTPUT_CHARS = 600
    BANNED_TERMS = {"medical diagnosis", "guarantee", "cure"}

    # --- Observability ---
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    METRICS_PORT = int(os.getenv("METRICS_PORT", "8000"))
    ENVIRONMENT = os.getenv("ENVIRONMENT", "production")


settings = Settings()
