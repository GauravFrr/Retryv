"""
Rotating Generation Client for Retryv.

A module-level singleton that rotates across all configured Gemini API keys
for generation (generateContent) calls.  Each key belongs to a different
Google account / GCP project, so each key has its own independent RPM quota.
With N keys we get N×10 effective RPM on the free tier.

Usage:
    from app.core.generation_client import get_generation_client, rotate_generation_key

    client = get_generation_client()          # current key's genai.Client
    rotate_generation_key()                   # advance to next key on 429
"""
import logging
import threading
from google import genai
from app.core.config import settings

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_clients: dict[str, genai.Client] = {}
_current_index: int = 0


def get_generation_client() -> genai.Client:
    """Return the genai.Client for the currently active API key."""
    global _current_index
    with _lock:
        api_keys = settings.gemini_api_keys
        idx = _current_index % len(api_keys)
        key = api_keys[idx]
        if key not in _clients:
            _clients[key] = genai.Client(api_key=key)
        return _clients[key]


def rotate_generation_key() -> bool:
    """Rotate to the next API key.  Returns True if a rotation occurred."""
    global _current_index
    api_keys = settings.gemini_api_keys
    if len(api_keys) <= 1:
        return False
    with _lock:
        _current_index = (_current_index + 1) % len(api_keys)
        masked = api_keys[_current_index][:12]
        logger.warning(
            "GenerationClient: rotated to key index %d (masked: %s...)",
            _current_index,
            masked,
        )
    return True
