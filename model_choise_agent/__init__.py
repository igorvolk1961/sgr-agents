"""Model Choice Agent - Tools and services for intelligent model selection."""

# Import all tools to ensure they are registered in ToolRegistry
# This must be imported before agents.yaml is loaded
try:
    from model_choise_agent.tools import (
        LLMCallTool,
        ConfigModelTool,
        TaskAccessTool,
    )
except ImportError as e:
    # If import fails, tools won't be registered but won't crash the app
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Failed to import model_choise_agent tools: {e}")

__all__ = [
    "LLMCallTool",
    "ConfigModelTool",
    "TaskAccessTool",
]
