"""Register model_choise_agent tools in ToolRegistry.

This module must be imported before agents.yaml is loaded to ensure
all tools are registered in the ToolRegistry.

Usage:
    import model_choise_agent.register_tools  # noqa: F401
    # Now tools are registered and can be used in agents.yaml
"""

# Import all tools to trigger registration via ToolRegistryMixin
from model_choise_agent.tools import (
    LLMCallTool,  # noqa: F401
    ConfigModelTool,  # noqa: F401
    TaskAccessTool,  # noqa: F401
)

# Tools will be registered with names:
# - "TaskAccessTool" (TaskAccessTool)
# - "ConfigModelTool" (ConfigModelTool)
# - "LLMCallTool" (LLMCallTool)

