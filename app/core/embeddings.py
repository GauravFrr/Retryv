import time
import logging
from typing import List, Union
from google import genai
from google.genai import types
from google.genai.errors import APIError
from app.core.config import settings

logger = logging.getLogger(__name__)


class GeminiEmbedder:
    """Service to generate embeddings using Gemini's configured embedding model."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.GEMINI_EMBED_MODEL
        self._clients = {}
        self._current_key_idx = 0

    @property
    def client(self):
        """Lazily initialize the GenAI client using the current active key."""
        api_keys = settings.gemini_api_keys
        self._current_key_idx = getattr(self, "_current_key_idx", 0) % len(api_keys)
        active_key = api_keys[self._current_key_idx]
        
        if active_key not in self._clients:
            self._clients[active_key] = genai.Client(api_key=active_key)
        return self._clients[active_key]

    def rotate_key(self) -> bool:
        """Rotates to the next Gemini API key if multiple are configured."""
        api_keys = settings.gemini_api_keys
        if len(api_keys) <= 1:
            return False
        
        self._current_key_idx = (self._current_key_idx + 1) % len(api_keys)
        logger.warning(
            f"Rotating API key. Switched to key index {self._current_key_idx} (masked: {api_keys[self._current_key_idx][:12]}...)"
        )
        return True


    def embed_text(self, text: str) -> List[float]:
        """Generates embedding for a single text string."""
        embeddings = self.embed_batch([text])
        return embeddings[0]

    def embed_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Generates embeddings for a batch of text strings with rate-limiting retries."""
        if not texts:
            return []

        all_embeddings: List[List[float]] = []

        # Process in chunks of batch_size
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            
            # Estimate token count (1 token is roughly 4 characters in English)
            char_count = sum(len(text) for text in batch)
            estimated_tokens = max(100, int(char_count / 4.0))

            content_objects = [
                types.Content(parts=[types.Part.from_text(text=t)])
                for t in batch
            ]
            
            # Robust retry loop: Try all available keys sequentially.
            # If all keys are rate-limited or return errors, perform exponential backoff and retry the pool.
            max_attempts = 5
            backoff_factor = 3.0
            success = False
            last_error = None

            for attempt in range(max_attempts):
                api_keys = settings.gemini_api_keys
                keys_to_try = len(api_keys)
                
                for key_try in range(keys_to_try):
                    try:
                        response = self.client.models.embed_content(
                            model=self.model_name,
                            contents=content_objects,
                        )
                        
                        if not response.embeddings:
                            raise ValueError("Gemini API returned empty embeddings list.")
                            
                        batch_embeddings = [emb.values for emb in response.embeddings]
                        all_embeddings.extend(batch_embeddings)
                        
                        # Adaptive sleep: 1M TPM limit (target 300K TPM for safety)
                        sleep_time = max(2.0, (estimated_tokens / 300000.0) * 60.0)
                        logger.info(
                            f"Embedded batch of {len(batch)} items (~{estimated_tokens} tokens). "
                            f"Sleeping {sleep_time:.1f}s to respect TPM limit..."
                        )
                        time.sleep(sleep_time)
                        success = True
                        break
                    except APIError as e:
                        last_error = e
                        # If we have multiple keys, rotate to the next key on any APIError (rate limit, revoked key, etc.)
                        if len(api_keys) > 1 and key_try < keys_to_try - 1:
                            logger.warning(
                                f"API key failed (masked: {api_keys[self._current_key_idx][:12]}...) with error: {e}. Rotating to next key..."
                            )
                            self.rotate_key()
                            time.sleep(1.0)
                            continue
                        else:
                            # If it's a single key or the last key in the pool, bubble up to the backoff handler
                            break
                    except Exception as e:
                        logger.error(f"Unexpected error during embedding generation: {e}")
                        raise e

                if success:
                    break

                if attempt == max_attempts - 1:
                    logger.error(f"Gemini embedding failed after trying all keys in {max_attempts} pool attempts.")
                    raise last_error or RuntimeError("Gemini embedding generation failed.")

                # If all keys in the pool failed, perform backoff sleep before retrying the pool again
                sleep_time = min(60.0, backoff_factor ** attempt + 5.0)
                logger.warning(
                    f"All Gemini API keys in the pool returned errors. Sleeping {sleep_time:.1f}s before retrying the pool..."
                )
                time.sleep(sleep_time)

        return all_embeddings
