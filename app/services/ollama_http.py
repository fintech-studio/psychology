import asyncio
import random
import logging
from typing import Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    import httpx

logger = logging.getLogger(__name__)


class OllamaHttpClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        # Async client is created lazily to avoid heavy imports at module
        # import time (httpx may not be installed in all environments).
        self._client: Optional[Any] = None

    def _ensure_client(self, timeout: int = 10) -> None:
        if self._client is None:
            # import lazily to avoid module-level dependency
            import httpx

            self._client = httpx.AsyncClient(timeout=timeout)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def post_json(
        self,
        url: str,
        payload: dict,
        retries: int = 3,
        timeout: int = 10,
    ) -> Optional["httpx.Response"]:
        """POST JSON with retries and exponential backoff.

        Returns httpx.Response on success or None on failure.
        """
        self._ensure_client(timeout=timeout)
        backoff = 0.5
        max_backoff = 5.0
        for attempt in range(retries):
            try:
                resp = await self._client.post(url, json=payload)
                return resp
            except Exception as e:
                # Catch broad exceptions to allow retries even when httpx
                # classes are unavailable in this environment.
                logger.debug("Attempt %s failed: %s", attempt + 1, e)
                if attempt < retries - 1:
                    delay = min(max_backoff, backoff * (2 ** attempt))
                    delay = delay * (0.8 + random.random() * 0.4)
                    await asyncio.sleep(delay)
                    continue
                return None

    async def health_check_model(
            self, model: str, base_path: str = "/api/generate") -> bool:
        payload = {"model": model, "prompt": "ping", "max_tokens": 1}
        r = await self.post_json(f"{self.base_url}{base_path}", payload)
        if r is None:
            logger.debug(
                "Model health check failed for %s: no response", model)
            return False
        logger.debug("Model %s health check status=%s", model, r.status_code)
        return r.status_code == 200
