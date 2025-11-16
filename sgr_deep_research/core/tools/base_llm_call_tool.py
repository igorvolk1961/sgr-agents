"""Tool for making LLM API calls."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic import Field

from sgr_deep_research.core.base_tool import BaseTool
from sgr_deep_research.core.services.llm_service import LLMService

if TYPE_CHECKING:
    from sgr_deep_research.core.models import ResearchContext

logger = logging.getLogger(__name__)


class BaseLLMCallTool(BaseTool):
    """Tool for making calls to external LLM APIs.

    This tool allows you to send requests to any LLM API endpoint
    and receive responses. The tool maintains a cache of models
    that have been used, adding them on first access.
    """

    url: str = Field(description="Base URL of the LLM API endpoint")
    api_key: str = Field(description="API key for authentication")
    model: str = Field(description="Model name to use for the request")
    system_prompt: str | None = Field(default=None, description="Optional system prompt")
    prompt: str = Field(description="User prompt to send to the LLM")
    temperature: float | None = Field(default=None, description="Optional temperature parameter", ge=0.0, le=2.0)

    async def __call__(self, context: ResearchContext) -> str:
        """Execute LLM call using LLMService."""
        logger.info(f"Calling LLM: {self.model} at {self.url}")

        try:
            response = await LLMService.call_llm(
                url=self.url,
                api_key=self.api_key,
                model=self.model,
                system_prompt=self.system_prompt,
                prompt=self.prompt,
                temperature=self.temperature,
            )
            logger.debug(f"LLM call successful, response length: {len(response)}")
            return response
        except Exception as e:
            error_msg = f"Error calling LLM: {e}"
            logger.error(error_msg)
            return error_msg

