"""Точка входа для официального агента управления строительными проектами."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from sgr_deep_research import AgentFactory, __version__
from sgr_deep_research.api.endpoints import router
from sgr_deep_research.core import AgentRegistry, ToolRegistry
from sgr_deep_research.core.agent_config import GlobalConfig
from sgr_deep_research.default_definitions import get_default_agents_definitions
from sgr_deep_research.settings import ServerConfig, setup_logging

# Классы агентов будут импортированы автоматически при загрузке YAML конфигурации

# Загружаем переменные окружения из .env файла
# Ищем .env в корне проекта (на уровень выше official-agent)
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    # Пробуем загрузить из текущей директории
    load_dotenv()

setup_logging()
logger = logging.getLogger(__name__)

# Логируем загрузку .env после инициализации логирования
if env_path.exists():
    logger.info(f"Loaded environment variables from: {env_path}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения."""
    for tool in ToolRegistry.list_items():
        logger.info(f"Tool registered: {tool.__name__}")
    for agent in AgentRegistry.list_items():
        logger.info(f"Agent registered: {agent.__name__}")
    for defn in AgentFactory.get_definitions_list():
        logger.info(f"Agent definition loaded: {defn}")
    yield


def main():
    """Запуск FastAPI сервера для официального агента."""
    args = ServerConfig()
    
    # Загружаем базовую конфигурацию
    config = GlobalConfig.from_yaml(args.config_file)
    
    # Добавляем дефолтные агенты из sgr_deep_research
    config.agents.update(get_default_agents_definitions())
    
    # Загружаем определения агентов из YAML
    # Если файл не указан, используем дефолтный путь (в корне official-agent)
    agents_file = args.agents_file or str(Path(__file__).parent / "agents.yaml")
    if Path(agents_file).exists():
        config.definitions_from_yaml(agents_file)
    else:
        logger.warning(f"Agents file not found: {agents_file}. Using default agents only.")
    
    app = FastAPI(
        title="Official Agent API - Управление строительными проектами",
        version=__version__,
        lifespan=lifespan,
    )
    
    # CORS настройки (для разработки)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.include_router(router)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

