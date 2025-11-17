from typing import Iterable

from openai import pydantic_function_tool
from openai.types.chat import ChatCompletionFunctionToolParam

from sgr_deep_research import AgentFactory
from sgr_deep_research.core import FinalAnswerTool, ReasoningTool
from sgr_deep_research.core.agent_config import GlobalConfig
from sgr_deep_research.core.agents.sgr_tool_calling_agent import SGRToolCallingAgent
from sgr_deep_research.core.tools import ExtractPageContentTool, WebSearchTool
from sgr_deep_research.default_definitions import get_default_agents_definitions


DEFAULT_TOOLKIT = [
    ReasoningTool,
    WebSearchTool,
    ExtractPageContentTool,
    FinalAnswerTool,
]


class BenchmarkAgent(SGRToolCallingAgent):
    """Agent for benchmarking with automatic tool selection."""

    name: str = "benchmark_agent"

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.toolkit = list(DEFAULT_TOOLKIT)

    async def _prepare_tools(self) -> list[ChatCompletionFunctionToolParam]:
        """Prepare available tools for current agent state and progress."""
        tools = set(self.toolkit)
        if self._context.iteration >= self.max_iterations:
            tools = {
                ReasoningTool,
                FinalAnswerTool,
            }
        if self._context.searches_used >= self.max_searches:
            tools -= {
                WebSearchTool,
            }

        return [pydantic_function_tool(tool, name=tool.tool_name, description="") for tool in tools]


async def create_benchmark_agent(
    task: str,
    *,
    config: GlobalConfig | None = None,
    max_iterations: int | None = None,
    toolkit: Iterable[type] | None = None,
):
    config = config or GlobalConfig()

    agent_key = SGRToolCallingAgent.name
    if agent_key not in config.agents:
        config.agents.update(get_default_agents_definitions())

    base_definition = config.agents[agent_key].model_copy(deep=True)
    base_definition.name = BenchmarkAgent.name
    base_definition.base_class = BenchmarkAgent
    base_definition.tools = list(toolkit or DEFAULT_TOOLKIT)

    if max_iterations is not None:
        base_definition.execution = base_definition.execution.model_copy(
            update={"max_iterations": max_iterations}
        )

    return await AgentFactory.create(base_definition, task=task)
