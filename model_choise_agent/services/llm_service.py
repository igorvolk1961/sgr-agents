"""Service for managing LLM clients and calls with caching."""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Type

import httpx
import yaml
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from sgr_deep_research.core.stream import OpenAIStreamingGenerator

logger = logging.getLogger(__name__)


class LLMConfig(BaseModel):
    """Configuration for instrumental LLM model."""

    api_key: str = Field(description="API key for instrumental model")
    base_url: str = Field(default="https://api.openai.com/v1", description="Base URL for API")
    model: str = Field(default="gpt-4o-mini", description="Model name for analysis")
    max_tokens: int = Field(default=1000, description="Maximum output tokens")
    temperature: float = Field(default=0.3, ge=0.0, le=1.0, description="Temperature for inference")
    proxy: str | None = Field(default=None, description="Proxy URL (optional)")

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "LLMConfig":
        """Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            LLMConfig instance

        Raises:
            FileNotFoundError: If YAML file not found
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with yaml_path.open(encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        # Extract llm section
        llm_config = config_data.get("llm", {})
        return cls(**llm_config)


class PromptAnalysisResult(BaseModel):
    """Result of prompt analysis."""

    reasoning: str = Field(description="Explanation of the analysis")
    model_kind: Literal["light", "heavy", "agent"] = Field(description="Required model type")
    result_kind: Literal["strict", "creative"] = Field(description="Required result type (temperature)")


class LLMService:
    """Service for managing LLM clients with caching."""

    def __init__(self):
        """Initialize the service with empty clients cache."""
        self._clients: dict[str, AsyncOpenAI] = {}

    def get_client(
        self,
        base_url: str,
        api_key: str,
        proxy: str | None = None,
    ) -> AsyncOpenAI:
        """Get or create OpenAI client for given configuration.

        Args:
            base_url: Base URL for API
            api_key: API key
            proxy: Optional proxy URL

        Returns:
            AsyncOpenAI client (cached if same base_url)
        """
        # Create cache key from base_url only
        cache_key = base_url

        if cache_key not in self._clients:
            client_kwargs = {
                "base_url": base_url,
                "api_key": api_key,
            }

            if proxy:
                client_kwargs["http_client"] = httpx.AsyncClient(proxy=proxy)

            self._clients[cache_key] = AsyncOpenAI(**client_kwargs)
            logger.info(f"Created new LLM client for {base_url}")

        return self._clients[cache_key]

    async def call_llm(
        self,
        prompt: str,
        system_prompt: str,
        base_url: str,
        api_key: str,
        model: str,
        max_tokens: int,
        temperature: float,
        proxy: str | None = None,
        response_format: Type[BaseModel] | None = None,
        streaming_generator: "OpenAIStreamingGenerator | None" = None,
    ) -> str:
        """Call LLM with given parameters.

        Args:
            prompt: User prompt
            system_prompt: System prompt
            base_url: Base URL for API
            api_key: API key
            model: Model name
            max_tokens: Maximum output tokens
            temperature: Temperature for inference
            proxy: Optional proxy URL
            response_format: Optional Pydantic BaseModel class for structured output
            streaming_generator: Optional streaming generator for forwarding chunks

        Returns:
            Response text from LLM. If response_format is a BaseModel,
            returns JSON string that can be parsed into that model.
        """
        client = self.get_client(base_url, api_key, proxy)

        logger.info(f"🤖 Calling LLM: {model} (temperature={temperature})")

        try:
            request_kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            
            # If response_format is a BaseModel class, use parse() method
            if response_format is not None:
                request_kwargs["response_format"] = response_format
                async with client.beta.chat.completions.stream(**request_kwargs) as stream:
                    async for event in stream:
                        if event.type == "chunk":
                            if streaming_generator:
                                streaming_generator.add_chunk(event.chunk)
                    response = await stream.get_final_completion()
                
                # Get parsed response
                parsed = response.choices[0].message.parsed
                if parsed is not None:
                    # Return JSON string of the parsed model
                    logger.info(f"✅ LLM response received (parsed into {response_format.__name__})")
                    return parsed.model_dump_json()
                else:
                    raise ValueError("Empty parsed response from LLM")
            else:
                # For None response_format, use stream() method (same pattern as SGR project)
                async with client.chat.completions.stream(**request_kwargs) as stream:
                    async for event in stream:
                        if event.type == "chunk":
                            if streaming_generator:
                                streaming_generator.add_chunk(event.chunk)
                    response = await stream.get_final_completion()
                
                result_text = response.choices[0].message.content
                
                # If response is from an agent, it may contain structured JSON
                # Try to extract final answer from FinalAnswerTool if present
                if result_text:
                    try:
                        # Try to parse as JSON to check if it's structured data
                        parsed = json.loads(result_text)
                        
                        # Check if this is a FinalAnswerTool response
                        if isinstance(parsed, dict):
                            # Look for FinalAnswerTool structure
                            if "tool_name_discriminator" in parsed and parsed["tool_name_discriminator"] == "FinalAnswerTool":
                                result_text = parsed.get("answer", result_text)
                            elif "answer" in parsed:
                                # Direct answer field
                                result_text = parsed["answer"]
                            elif "response" in parsed:
                                # Nested response field (may contain FinalAnswerTool)
                                nested_response = parsed["response"]
                                if isinstance(nested_response, str):
                                    try:
                                        nested_parsed = json.loads(nested_response)
                                        if isinstance(nested_parsed, dict) and "answer" in nested_parsed:
                                            result_text = nested_parsed["answer"]
                                    except (json.JSONDecodeError, TypeError):
                                        pass
                    except (json.JSONDecodeError, TypeError, AttributeError):
                        # Not JSON, use as-is
                        pass
                
                if not result_text:
                    raise ValueError("Empty response from LLM")

                logger.info(f"✅ LLM response received ({len(result_text)} chars)")
                return result_text

        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            raise


