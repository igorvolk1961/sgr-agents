"""Tool for calling LLM with specified parameters."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from pydantic import Field

from sgr_deep_research.core.base_tool import BaseTool

if TYPE_CHECKING:
    from sgr_deep_research.core.models import ResearchContext

# Import service from model_choise_agent/services directory
_services_path = Path(__file__).parent.parent / "services"
if str(_services_path) not in sys.path:
    sys.path.insert(0, str(_services_path))

import_error: Exception | None = None
try:
    from llm_service import LLMService
except ImportError as e:
    LLMService = None  # type: ignore
    import_error = e

logger = logging.getLogger(__name__)

# Global service instance (singleton)
_llm_service: LLMService | None = None


def get_llm_service() -> LLMService:
    """Get or create global LLMService instance."""
    global _llm_service
    if _llm_service is None:
        if LLMService is None:
            raise ImportError(f"LLMService is not available: {import_error}")
        _llm_service = LLMService()
    return _llm_service


class LLMCallTool(BaseTool):
    """Call LLM with specified parameters to get the final response.

    ⚠️ FINAL MODEL CALL: This tool actually calls the selected LLM model and returns the response.
    This is the step that produces the final answer - use it after getting configuration from ConfigModelTool.

    This tool calls an LLM model using OpenAI SDK with the provided configuration.
    All parameters are passed directly to the service.

    WHEN TO USE:
    - Use ONLY when you have ALL configuration parameters from ConfigModelTool result:
      * base_url, model, api_key, max_tokens, temperature, system_prompt
    - After getting configuration, remaining_steps[0] should be "Call model with LLMCallTool"
    - Pass the configuration parameters AND the original user prompt
    - After getting response, remaining_steps[0] should be "Present final answer with FinalAnswerTool"

    REQUIRED PARAMETERS:
    - prompt: Original user request text (from the INITIAL user message, NOT from ConfigModelTool result)
    - base_url: From ConfigModelTool JSON result
    - model: From ConfigModelTool JSON result
    - api_key: From ConfigModelTool JSON result
    - max_tokens: From ConfigModelTool JSON result
    - temperature: From ConfigModelTool JSON result
    - system_prompt: From ConfigModelTool JSON result
    - proxy: Optional, can be null or omitted

    WORKFLOW:
    1. Get configuration JSON from ConfigModelTool
    2. Parse JSON to extract: base_url, model, api_key, max_tokens, temperature, system_prompt
    3. Get original user prompt from the INITIAL user request (check conversation history)
    4. Use this tool with ALL extracted parameters + original user prompt
    5. Get the model response
    6. IMMEDIATELY use FinalAnswerTool with the response
    7. Do NOT repeat this tool - model has already been called
    """

    tool_name: ClassVar[str] = "LLMCallTool"

    prompt: str = Field(description="User prompt to send to LLM")
    system_prompt: str = Field(description="System prompt for LLM")
    base_url: str = Field(description="Base URL for API")
    model: str = Field(description="Model name")
    api_key: str = Field(description="API key")
    max_tokens: int = Field(description="Maximum output tokens")
    temperature: float = Field(description="Temperature for inference", ge=0.0, le=2.0)
    proxy: str | None = Field(default=None, description="Optional proxy URL")

    async def __call__(self, context: ResearchContext) -> str:
        """Call LLM with specified parameters.

        Args:
            context: Research context (not used in this tool)

        Returns:
            JSON string with LLM response or error
        """
        try:
            service = get_llm_service()
        except Exception as e:
            error_msg = f"Failed to get LLMService: {e}"
            logger.error(error_msg)
            return json.dumps(
                {
                    "error": error_msg,
                    "response": "",
                },
                ensure_ascii=False,
                indent=2,
            )

        try:
            # Get streaming_generator from context (added by BaseAgent)
            streaming_generator = getattr(context, "streaming_generator", None)
            
            response = await service.call_llm(
                prompt=self.prompt,
                system_prompt=self.system_prompt,
                base_url=self.base_url,
                api_key=self.api_key,
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                proxy=self.proxy,
                streaming_generator=streaming_generator,
            )

            result = {
                "response": response,
                "error": None,
            }

            return json.dumps(result, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            return json.dumps(
                {
                    "error": str(e),
                    "response": "",
                },
                ensure_ascii=False,
                indent=2,
            )

