"""Tool for getting LLM configuration based on model_kind and result_kind."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

import yaml
from pydantic import Field

from sgr_deep_research.core.base_tool import BaseTool

if TYPE_CHECKING:
    from sgr_deep_research.core.models import ResearchContext

logger = logging.getLogger(__name__)


class ConfigModelTool(BaseTool):
    """Get LLM configuration based on model type and result type.

    ⚠️ INTERMEDIATE STEP: This tool retrieves configuration but does NOT call the model.
    After getting configuration, you MUST use LLMCallTool to actually call the model.

    This tool extracts configuration from config.yaml based on:
    - model_kind: "light", "heavy", or "agent" (from TaskAccessTool result)
    - result_kind: "strict" or "creative" (from TaskAccessTool result)

    Returns JSON with: base_url, model, api_key, max_tokens, temperature, system_prompt.

    WHEN TO USE:
    - Use ONLY when you have model_kind and result_kind from TaskAccessTool result
    - After getting configuration, remaining_steps[0] should be "Call selected model with LLMCallTool"
    - Do NOT use this tool if you already have configuration parameters

    WORKFLOW:
    1. Get model_kind and result_kind from TaskAccessTool result
    2. Use this tool to get configuration
    3. IMMEDIATELY use LLMCallTool with the returned configuration
    4. Do NOT repeat this tool - configuration is already obtained
    """

    tool_name: ClassVar[str] = "ConfigModelTool"

    model_kind: Literal["light", "heavy", "agent"] = Field(
        description="Type of model: light, heavy, or agent"
    )
    result_kind: Literal["strict", "creative"] = Field(
        description="Type of result: strict (low temp) or creative (high temp)"
    )

    def __init__(self, **data):
        """Initialize the tool."""
        super().__init__(**data)
        self._config_path = Path(__file__).parent.parent / "config.yaml"

    async def __call__(self, context: ResearchContext) -> str:
        """Get LLM configuration based on model_kind and result_kind.

        Args:
            context: Research context (not used in this tool)

        Returns:
            JSON string with LLM configuration
        """
        try:
            if not self._config_path.exists():
                error_msg = f"Configuration file not found: {self._config_path}"
                logger.error(error_msg)
                return json.dumps(
                    {
                        "error": error_msg,
                        "base_url": "",
                        "model": "",
                        "api_key": "",
                        "max_tokens": 0,
                        "temperature": 0.0,
                        "system_prompt": "",
                    },
                    ensure_ascii=False,
                    indent=2,
                )

            with self._config_path.open(encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

            # Get the appropriate config section
            config_key = f"llm_{self.model_kind}"
            if config_key not in config_data:
                error_msg = f"Configuration section '{config_key}' not found in config.yaml"
                logger.error(error_msg)
                return json.dumps(
                    {
                        "error": error_msg,
                        "base_url": "",
                        "model": "",
                        "api_key": "",
                        "max_tokens": 0,
                        "temperature": 0.0,
                        "system_prompt": "",
                    },
                    ensure_ascii=False,
                    indent=2,
                )

            model_config = config_data[config_key]

            # Get temperature based on result_kind
            temperature_key = f"temperature_{self.result_kind}"
            if temperature_key not in model_config:
                error_msg = f"Temperature '{temperature_key}' not found in {config_key}"
                logger.error(error_msg)
                return json.dumps(
                    {
                        "error": error_msg,
                        "base_url": "",
                        "model": "",
                        "api_key": "",
                        "max_tokens": 0,
                        "temperature": 0.0,
                        "system_prompt": "",
                    },
                    ensure_ascii=False,
                    indent=2,
                )

            temperature = model_config[temperature_key]

            # Load system prompt from file
            system_prompt_path = model_config.get("system_prompt_path", "")
            system_prompt = ""
            if system_prompt_path:
                prompt_file = Path(__file__).parent.parent / system_prompt_path
                if prompt_file.exists():
                    system_prompt = prompt_file.read_text(encoding="utf-8").strip()
                else:
                    logger.warning(f"System prompt file not found: {prompt_file}")

            result = {
                "base_url": model_config.get("base_url", ""),
                "model": model_config.get("model", ""),
                "api_key": model_config.get("api_key", ""),
                "max_tokens": model_config.get("max_tokens", 0),
                "temperature": temperature,
                "system_prompt": system_prompt,
            }

            logger.info(
                f"✅ Configuration loaded: model_kind={self.model_kind}, "
                f"result_kind={self.result_kind}, model={result['model']}, "
                f"temperature={temperature}"
            )

            return json.dumps(result, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"Error loading LLM configuration: {e}")
            return json.dumps(
                {
                    "error": str(e),
                    "base_url": "",
                    "model": "",
                    "api_key": "",
                    "max_tokens": 0,
                    "temperature": 0.0,
                    "system_prompt": "",
                },
                ensure_ascii=False,
                indent=2,
            )

