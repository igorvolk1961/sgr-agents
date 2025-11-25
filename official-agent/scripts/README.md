# Скрипты для официального агента

## Синхронизация TODO с JIRA

Скрипт `sync_todo_to_jira.py` синхронизирует пункты из `TODO_cursor_plan.md` с подзадачами в JIRA.

### Установка зависимостей

```bash
pip install -r requirements.txt
```

Или установить только необходимые пакеты:

```bash
pip install jira python-dotenv PyYAML
```

### Настройка

1. Убедитесь, что в `.env` файле (в корне проекта `sgr-agents`) указаны учетные данные JIRA:

```env
JIRA_SERVER=http://172.16.254.6:8080
JIRA_USERNAME=igorvolk
JIRA_PASSWORD=your-password
JIRA_PROJECT_KEY=PFM
JIRA_PARENT_ISSUE=PFM-33
```

2. Или настройте в `config.yaml`:

```yaml
jira:
  server: "http://172.16.254.6:8080"
  project_key: "PFM"
  parent_issue: "PFM-33"
```

### Использование

Запустите скрипт из корня проекта:

```bash
python official-agent/scripts/sync_todo_to_jira.py
```

Или из директории `official-agent`:

```bash
cd official-agent
python scripts/sync_todo_to_jira.py
```

### Что делает скрипт

1. Парсит файл `TODO_cursor_plan.md` и извлекает все пункты с их статусами
2. Для каждого пункта создает подзадачу в JIRA (если её еще нет)
3. Обновляет статус подзадачи в соответствии со статусом в TODO:
   - `[DONE]` → `Done` в JIRA
   - `[PENDING]` → `To Do` в JIRA
   - `[IN_PROGRESS]` → `In Progress` в JIRA

### Формат подзадач

Каждая подзадача создается как дочерняя задача для `PFM-33` с:
- **Summary**: номер пункта и краткое описание (например, "1.1: Изучить текущий orchestrator...")
- **Description**: полное описание пункта из TODO
- **Type**: Sub-task

### Примечания

- Скрипт ищет существующие подзадачи по summary, чтобы не создавать дубликаты
- При повторном запуске скрипт обновит статусы существующих подзадач
- Все операции логируются в консоль

