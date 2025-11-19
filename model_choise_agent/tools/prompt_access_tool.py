"""Tool for analyzing prompt complexity and determining required model type."""

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

# Import service and config from model_choise_agent/services directory
# Add services directory to path for import
_services_path = Path(__file__).parent.parent / "services"
if str(_services_path) not in sys.path:
    sys.path.insert(0, str(_services_path))

import_error: Exception | None = None
try:
    from llm_service import LLMConfig, LLMService, PromptAnalysisResult
except ImportError as e:
    LLMService = None  # type: ignore
    LLMConfig = None  # type: ignore
    PromptAnalysisResult = None  # type: ignore
    import_error = e

logger = logging.getLogger(__name__)


class TaskAccessTool(BaseTool):
    """FIRST STEP - Analyze prompt complexity and determine required model type and temperature.

    ⚠️ MANDATORY: This tool MUST be used FIRST before any other model selection tools.
    DO NOT skip this step or proceed to ConfigModelTool without using this tool first.

    This tool evaluates the input prompt to determine:
    1. Model type needed (model_kind):
       - light: Models under 10B parameters without tools
       - heavy: Models over 10B parameters without tools
       - agent: Models with connected tools

    2. Temperature setting (result_kind):
       - strict: Low temperature for factual, precise answers
       - creative: High temperature for creative, varied responses

    Returns JSON with reasoning, model_kind, and result_kind.

    WORKFLOW - After using this tool, you MUST:
    1. Extract model_kind and result_kind from the JSON response
    2. Use ConfigModelTool with these values to get model configuration
    3. Use LLMCallTool with the configuration and original prompt to get the final response
    4. Use FinalAnswerTool to present the response
    
    DO NOT skip this tool - it is the required first step.
    """

    tool_name: ClassVar[str] = "TaskAccessTool"

    prompt: str = Field(description="Input prompt text to analyze")

    def __init__(self, **data):
        """Initialize the tool with LLM service."""
        super().__init__(**data)
        # Initialize service
        if LLMService is None:
            logger.error(f"Failed to import LLMService: {import_error}")
            self._service = None
        else:
            self._service = LLMService()
        
        # Load instrumental model config
        self._config_path = Path(__file__).parent.parent / "config.yaml"
        self._instrumental_config: LLMConfig | None = None

    def _get_instrumental_config(self) -> LLMConfig:
        """Get or load instrumental model configuration."""
        if LLMConfig is None:
            raise ValueError(f"LLMConfig is not available: {import_error}")
        if self._instrumental_config is None:
            self._instrumental_config = LLMConfig.from_yaml(self._config_path)
        return self._instrumental_config

    async def __call__(self, context: ResearchContext) -> str:
        """Analyze the prompt and return JSON result.

        Args:
            context: Research context (not used in this tool)

        Returns:
            JSON string with reasoning, model_kind, and result_kind
        """
        if self._service is None:
            # Try to initialize service
            if LLMService is None:
                error_msg = f"LLMService is not available: {import_error if import_error else 'Unknown error'}"
                logger.error(error_msg)
                return json.dumps(
                    {
                        "error": error_msg,
                        "reasoning": "Service not available",
                        "model_kind": "heavy",  # Default fallback
                        "result_kind": "strict",  # Default fallback
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            try:
                self._service = LLMService()
            except Exception as e:
                error_msg = f"Failed to initialize LLMService: {e}"
                logger.error(error_msg)
                return json.dumps(
                    {
                        "error": error_msg,
                        "reasoning": "Service initialization failed",
                        "model_kind": "heavy",  # Default fallback
                        "result_kind": "strict",  # Default fallback
                    },
                    ensure_ascii=False,
                    indent=2,
                )

        try:
            # Load instrumental model configuration
            try:
                config = self._get_instrumental_config()
            except Exception as e:
                error_msg = f"Failed to load instrumental model config: {e}"
                logger.error(error_msg)
                return json.dumps(
                    {
                        "error": error_msg,
                        "reasoning": "Configuration loading failed",
                        "model_kind": "heavy",  # Default fallback
                        "result_kind": "strict",  # Default fallback
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            
            # Load system prompt from file
            system_prompt_path = Path(__file__).parent.parent / "prompts" / "prompt_analysis_system.txt"
            if system_prompt_path.exists():
                system_prompt = system_prompt_path.read_text(encoding="utf-8")
            else:
                # Fallback to default prompt if file not found
                logger.warning(f"System prompt file not found: {system_prompt_path}, using default")
                system_prompt = """You are an expert in analyzing LLM prompts to determine their complexity and requirements.

Analyze the given prompt and determine:
1. What type of LLM model is needed:
   - "light": Simple prompts that can be handled by models under 10B parameters without tools (simple Q&A, basic text generation)
   - "heavy": Complex prompts requiring models over 10B parameters without tools (complex reasoning, multi-step analysis, deep understanding)
   - "agent": Prompts that require external tools or data access (web search, API calls, real-time information, tool usage)

2. What temperature setting is appropriate:
   - "strict": Low temperature for factual, precise, deterministic answers (data analysis, factual questions, calculations)
   - "creative": High temperature for creative, varied, exploratory responses (creative writing, brainstorming, idea generation)

Respond ONLY with valid JSON in this exact format:
{
  "reasoning": "Your detailed explanation of why you chose these options",
  "model_kind": "light|heavy|agent",
  "result_kind": "strict|creative"
}"""

            logger.info(f"🔍 Analyzing prompt complexity (length: {len(self.prompt)} chars)")

            # Use LLMService to call the instrumental model with Pydantic model as response_format
            if PromptAnalysisResult is None:
                raise ValueError("PromptAnalysisResult is not available")
            
            # Get streaming_generator from context (added by BaseAgent)
            streaming_generator = getattr(context, "streaming_generator", None)
            
            result_json = await self._service.call_llm(
                prompt=f"Analyze this prompt:\n\n{self.prompt}",
                system_prompt=system_prompt,
                base_url=config.base_url,
                api_key=config.api_key,
                model=config.model,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                proxy=config.proxy,
                response_format=PromptAnalysisResult,
                streaming_generator=streaming_generator,
            )

            # Parse the JSON response into PromptAnalysisResult
            result = PromptAnalysisResult.model_validate_json(result_json)

            logger.info(
                f"✅ Analysis complete: model_kind={result.model_kind}, "
                f"result_kind={result.result_kind}"
            )

            return json.dumps(
                {
                    "reasoning": result.reasoning,
                    "model_kind": result.model_kind,
                    "result_kind": result.result_kind,
                },
                ensure_ascii=False,
                indent=2,
            )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return json.dumps(
                {
                    "error": f"Invalid JSON response from instrumental model: {e}",
                    "reasoning": f"JSON parsing failed: {e}",
                    "model_kind": "heavy",  # Default fallback
                    "result_kind": "strict",  # Default fallback
                },
                ensure_ascii=False,
                indent=2,
            )
        except Exception as e:
            logger.error(f"Error analyzing prompt: {e}")
            return json.dumps(
                {
                    "error": str(e),
                    "reasoning": f"Analysis failed: {e}",
                    "model_kind": "heavy",  # Default fallback
                    "result_kind": "strict",  # Default fallback
                },
                ensure_ascii=False,
                indent=2,
            )

