from typing import Type

from openai import AsyncOpenAI
import logging
import httpx
import uuid
from datetime import datetime
from sgr_deep_research.core.models import ResearchContext
from sgr_deep_research.core.stream import OpenAIStreamingGenerator

from sgr_deep_research.core.agent_definition import ExecutionConfig, LLMConfig, PromptsConfig
from sgr_deep_research.core.agents import SGRAgent
from sgr_deep_research.core.tools import (
    BaseTool,
    GeneratePlanTool,
    AdaptPlanTool,
    ClarificationTool,
    CreateReportTool,
    FinalAnswerTool,
    NextStepToolsBuilder,
    NextStepToolStub,
    WebSearchTool,
    ExtractPageContentTool,
)

class SGRYandexAgent(SGRAgent):
    """Agent for deep research tasks using SGR framework."""

    name: str = "sgr_yandex_agent"

    def __init__(
        self,
        task: str,
        openai_client: AsyncOpenAI,
        llm_config: LLMConfig,
        prompts_config: PromptsConfig,
        execution_config: ExecutionConfig,
        toolkit: list[Type[BaseTool]] | None = None,
    ):
        del openai_client

        yandex_cloud_folder = extractCloudFolder(llm_config.model)
        client_kwargs = {"base_url": llm_config.base_url, "api_key": llm_config.api_key,
                         "project": yandex_cloud_folder}
        if llm_config.proxy:
            client_kwargs["http_client"] = httpx.AsyncClient(proxy=llm_config.proxy)

        openai_client = AsyncOpenAI(**client_kwargs)

        super().__init__(
            task=task,
            openai_client=openai_client,
            llm_config=llm_config,
            prompts_config=prompts_config,
            execution_config=execution_config,
            toolkit=toolkit,
        )
        self.max_searches = execution_config.max_searches

    async def _prepare_tools(self) -> Type[NextStepToolStub]:
        """Prepare tool classes with current context limits."""
        tools = set(self.toolkit)
        if self._context.iteration >= self.max_iterations:
            if CreateReportTool in tools:
              tools += {
                CreateReportTool,
                FinalAnswerTool,
              }
            else:
             tools += {
                FinalAnswerTool,
            }
        if self._context.clarifications_used >= self.max_clarifications:
            tools -= {
                ClarificationTool,
            }
        if self._context.searches_used >= self.max_searches:
            tools -= {
                WebSearchTool,
            }
        if self._context.plan_generations_used == 0:
            if GeneratePlanTool in tools:
              tools = {
                  GeneratePlanTool,
              }
        else:
            tools -= {
                GeneratePlanTool
            }
        if  CreateReportTool in  tools:
          if self._context.report_creations_used > 0:
              tools = {
                  FinalAnswerTool,
              }
          else:
              tools -= {
                  FinalAnswerTool,
              }
        if self._context.searches_used == 0:
            tools -= {
                AdaptPlanTool,
                ExtractPageContentTool,
                CreateReportTool,
                FinalAnswerTool,
            }
        if self._context.page_extractions_used == 0:
            tools -= {
                AdaptPlanTool,
            }
        if (self._context.page_extractions_used >= self.max_searches) or \
           (self._context.page_extractions_used >= self._context.searches_used):
            tools -= {
                ExtractPageContentTool,
            }

        if self._context.page_extractions_used < self._context.searches_used: ###??? Возможны ли поиски страниц без их чтения
            tools -= {
                   WebSearchTool,
            }

        if self._context.plan_adaptations_used >= self._context.searches_used:
            tools -= {
                AdaptPlanTool,
            }
        return NextStepToolsBuilder.build_NextStepTools(list(tools))

def extractCloudFolder(base_url):
    """
    Extracts the cloud folder from a base_url of form
    "gpt://<cloud_folder>/..." (returns <cloud_folder>).
    Returns None if not matched.
    """
    prefix = "gpt://"
    if base_url.startswith(prefix):
        after_prefix = base_url[len(prefix):]
        return after_prefix.split("/", 1)[0]
    return None