"""
AgroSage configuration helpers.
"""

from __future__ import annotations

import os
from dotenv import load_dotenv


load_dotenv()


def get_env(name: str, default: str | None = None) -> str | None:
    """Return environment variable value or a default."""
    value = os.getenv(name)
    return value if value is not None and value != "" else default


def get_llm_model(default: str = "llama-3.1-70b-versatile") -> str:
    """Return the configured LLM model name."""
    return get_env("AGROSAGE_LLM_MODEL", default) or default
