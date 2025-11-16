"""Service for making LLM API calls."""

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class LLMService:
    """Service for making LLM API calls with model caching."""

    _models_cache: dict[str, dict[str, Any]] = {}

    @classmethod
    def _get_model_key(cls, url: str, model: str) -> str:
        """Generate a unique key for a model configuration."""
        return f"{url}:{model}"

    @classmethod
    def _add_model_to_cache(cls, url: str, model: str, config: dict[str, Any]) -> None:
        """Add model configuration to cache."""
        key = cls._get_model_key(url, model)
        cls._models_cache[key] = config
        logger.debug(f"Added model to cache: {key}")

    @classmethod
    def _get_model_from_cache(cls, url: str, model: str) -> dict[str, Any] | None:
        """Get model configuration from cache."""
        key = cls._get_model_key(url, model)
        return cls._models_cache.get(key)

    @classmethod
    async def call_llm(
        cls,
        url: str,
        api_key: str,
        model: str,
        system_prompt: str | None,
        prompt: str,
        temperature: float | None = None,
    ) -> str:
        """Make a call to LLM API and return the response.

        Args:
            url: Base URL of the LLM API
            api_key: API key for authentication
            model: Model name to use
            system_prompt: Optional system prompt
            prompt: User prompt
            temperature: Optional temperature parameter

        Returns:
            Response text from LLM

        Raises:
            httpx.HTTPError: If the API request fails
        """
        # Check if model is in cache, if not add it
        cached_model = cls._get_model_from_cache(url, model)
        if cached_model is None:
            cls._add_model_to_cache(
                url,
                model,
                {
                    "url": url,
                    "model": model,
                    "api_key": api_key,
                },
            )
            logger.info(f"First call to model '{model}' at '{url}', added to cache")

        # Prepare messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Prepare request payload
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if temperature is not None:
            payload["temperature"] = temperature

        # Make API request
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        logger.info(f"Calling LLM: {url}, model: {model}")

        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(
                    f"{url.rstrip('/')}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                result = response.json()

                # Extract response text
                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0].get("message", {}).get("content", "")
                    logger.debug(f"LLM response received: {len(content)} characters")
                    return content
                else:
                    error_msg = "No choices in LLM response"
                    logger.error(error_msg)
                    raise ValueError(error_msg)

            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error calling LLM: {e.response.status_code} - {e.response.text}")
                raise
            except httpx.RequestError as e:
                logger.error(f"Request error calling LLM: {e}")
                raise

