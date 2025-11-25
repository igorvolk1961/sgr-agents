"""Скрипт для синхронизации TODO с JIRA."""

import logging
import re
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Добавляем путь к корню проекта для импорта модулей
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Импорт модуля интеграции с JIRA
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.jira_integration import JiraIntegration

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

# Загрузка конфигурации
config_path = Path(__file__).parent.parent / "config.yaml"
if config_path.exists():
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
else:
    config = {}

# Параметры JIRA из конфигурации и переменных окружения
import os

jira_config = config.get("jira", {})
jira_server = jira_config.get("server") or os.getenv("JIRA_SERVER", "http://172.16.254.6:8080")
jira_username = jira_config.get("username") or os.getenv("JIRA_USERNAME", "igorvolk")
jira_password = jira_config.get("password") or os.getenv("JIRA_PASSWORD")
jira_project_key = jira_config.get("project_key") or os.getenv("JIRA_PROJECT_KEY", "PFM")
jira_parent_issue = jira_config.get("parent_issue") or os.getenv("JIRA_PARENT_ISSUE", "PFM-33")

if not jira_password:
    logger.error("Пароль JIRA не найден. Укажите JIRA_PASSWORD в .env или config.yaml")
    exit(1)


def parse_todo_file(todo_path: Path) -> list[dict]:
    """Парсинг TODO файла и извлечение пунктов.

    Args:
        todo_path: Путь к файлу TODO_cursor_plan.md

    Returns:
        Список словарей с информацией о пунктах TODO
    """
    items = []
    current_section = None
    current_subsection = None

    with open(todo_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Регулярное выражение для поиска пунктов TODO
    # Формат: - 1.1 [DONE] Описание
    pattern = r"^\s*-\s+(\d+(?:\.\d+)*)\s+\[(\w+)\]\s+(.+)$"

    for line in content.split("\n"):
        # Определение основных разделов (1., 2., 3. и т.д.)
        main_section_match = re.match(r"^(\d+)\.\s+(.+)$", line.strip())
        if main_section_match:
            current_section = {
                "number": main_section_match.group(1),
                "title": main_section_match.group(2).strip(),
            }
            continue

        # Поиск подпунктов
        match = re.match(pattern, line)
        if match:
            item_number = match.group(1)
            status = match.group(2)
            description = match.group(3).strip()

            # Определение уровня вложенности
            parts = item_number.split(".")
            level = len(parts)

            items.append(
                {
                    "number": item_number,
                    "status": status,
                    "description": description,
                    "level": level,
                    "section": current_section["number"] if current_section else parts[0],
                    "section_title": current_section["title"] if current_section else "",
                }
            )

    return items


def sync_todo_to_jira():
    """Основная функция синхронизации TODO с JIRA."""
    # Инициализация JIRA
    jira = JiraIntegration(
        server=jira_server,
        username=jira_username,
        password=jira_password,
        project_key=jira_project_key,
    )

    # Проверка существования родительской задачи
    try:
        parent_issue = jira.get_issue(jira_parent_issue)
        logger.info(f"Родительская задача найдена: {jira_parent_issue} - {parent_issue.fields.summary}")
    except Exception as e:
        logger.error(f"Ошибка получения родительской задачи {jira_parent_issue}: {e}")
        return

    # Парсинг TODO файла
    todo_path = Path(__file__).parent.parent / "TODO_cursor_plan.md"
    if not todo_path.exists():
        logger.error(f"Файл TODO не найден: {todo_path}")
        return

    todo_items = parse_todo_file(todo_path)
    logger.info(f"Найдено {len(todo_items)} пунктов в TODO")

    # Создание/обновление подзадач
    created_count = 0
    updated_count = 0

    for item in todo_items:
        # Формирование summary для JIRA
        summary = f"{item['number']}: {item['description'][:100]}"
        if len(item["description"]) > 100:
            summary += "..."

        # Формирование description
        description = f"""Пункт {item['number']} из TODO плана разработки.

Статус: {item['status']}
Раздел: {item['section']}. {item['section_title']}

{item['description']}
"""

        try:
            # Поиск или создание подзадачи
            # Тип задачи будет определен автоматически
            is_new_issue = False
            try:
                # Пробуем найти существующую задачу
                issue_key = jira.find_subtask_by_number(jira_parent_issue, item["number"])
                if not issue_key:
                    # Создаем новую задачу
                    issue_key = jira.create_subtask(
                        parent_key=jira_parent_issue,
                        summary=summary,
                        description=description,
                        issue_type=None,  # Автоматическое определение типа
                    )
                    is_new_issue = True
                    created_count += 1
            except Exception as e:
                logger.error(f"Ошибка поиска/создания подзадачи для пункта {item['number']}: {e}")
                continue

            # Обновление Assignee для всех задач
            parent_issue = jira.get_issue(jira_parent_issue)
            reporter = parent_issue.fields.reporter
            jira.update_assignee(issue_key, reporter.name if hasattr(reporter, "name") else None)

            # Обновление статуса согласно TODO
            current_status = jira.get_current_status(issue_key)
            
            if item["status"] == "DONE":
                # Для задач со статусом DONE в TODO:
                # Workflow: "В работе" -> "На ревью" (финальный статус для подзадач)
                success = False
                
                # Получаем доступные переходы
                transitions = jira.get_available_transitions(issue_key)
                
                # Проверяем, не в финальном статусе ли уже ("На ревью" - финальный статус)
                if "ревью" in current_status.lower() or "review" in current_status.lower():
                    success = True
                else:
                    # Если задача в статусе "В работе", переводим в "На ревью" (финальный статус)
                    status_lower = current_status.lower().strip()
                    # Проверяем различные варианты статуса "В работе"
                    has_work = (
                        status_lower == "в работе" or
                        status_lower.startswith("в работе") or
                        "в работе" in status_lower or
                        status_lower == "in progress" or
                        "in progress" in status_lower or
                        status_lower == "work" or
                        status_lower.startswith("work")
                    )
                    
                    if has_work:
                        # Переводим в "На ревью" (финальный статус)
                        # Ищем переход, который ведет в статус "На ревью"
                        found_transition = None
                        for transition in transitions:
                            target_status = transition.get("to", "").lower()
                            if "ревью" in target_status or "review" in target_status:
                                found_transition = transition
                                break
                        
                        if found_transition:
                            if jira.execute_transition(issue_key, found_transition["id"]):
                                updated_count += 1
                                success = True
                                new_status = jira.get_current_status(issue_key)
                                logger.info(f"Задача {issue_key} переведена из '{current_status}' в '{new_status}'")
                            else:
                                logger.error(f"Не удалось выполнить переход для задачи {issue_key}")
                        else:
                            logger.warning(f"Переход в 'На ревью' не найден для задачи {issue_key}. Доступные переходы: {[t['from'] + ' -> ' + t['to'] for t in transitions]}")
                    
                    # Если задача в начальном статусе, сначала переводим в "Взять в работу", затем в "На ревью"
                    else:
                        if jira.transition_to_status(issue_key, "Взять в работу"):
                            import time
                            time.sleep(0.5)  # Небольшая задержка между переходами
                            current_status = jira.get_current_status(issue_key)
                            transitions = jira.get_available_transitions(issue_key)
                            
                            # Теперь переводим в "На ревью" (финальный статус)
                            for transition in transitions:
                                if "ревью" in transition["to"].lower() or "review" in transition["to"].lower():
                                    if jira.execute_transition(issue_key, transition["id"]):
                                        updated_count += 1
                                        success = True
                                        logger.info(f"Задача {issue_key} переведена в финальный статус '{transition['to']}'")
                                        break
                            
                            if not success:
                                logger.warning(f"Задача {issue_key} переведена в 'В работе', но не удалось перевести в 'На ревью'. Доступные переходы: {[t['from'] + ' -> ' + t['to'] for t in transitions]}")
                        else:
                            logger.warning(f"Задача {issue_key} не переведена в 'Взять в работу'. Текущий статус: {current_status}")
                    
            elif item["status"] == "IN_PROGRESS":
                # Пробуем разные варианты названий статуса "In Progress"
                # Но только если задача не уже в этом статусе
                if current_status.lower() not in ["в работе", "in progress", "в процессе"]:
                    for status_variant in ["В РАБОТЕ", "In Progress", "В работе", "В процессе"]:
                        if jira.transition_to_status(issue_key, status_variant):
                            updated_count += 1
                            logger.info(f"Задача {issue_key} переведена в статус '{status_variant}'")
                            break
                        
            elif item["status"] == "PENDING":
                # Для задач со статусом PENDING в TODO:
                # Статус в JIRA не меняем (ни для новых, ни для существующих задач)
                logger.debug(f"Задача {issue_key}: статус PENDING в TODO - статус в JIRA не изменяется (текущий: {current_status})")

            created_count += 1
            logger.info(f"Обработан пункт {item['number']}: {issue_key}")

        except Exception as e:
            logger.error(f"Ошибка обработки пункта {item['number']}: {e}")

    logger.info(f"Синхронизация завершена. Создано/найдено: {created_count}, Обновлено статусов: {updated_count}")


if __name__ == "__main__":
    sync_todo_to_jira()

