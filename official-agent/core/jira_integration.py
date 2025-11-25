"""Модуль для интеграции с JIRA API."""

import logging
from typing import Any

from jira import JIRA
from jira.exceptions import JIRAError

logger = logging.getLogger(__name__)


class JiraIntegration:
    """Класс для работы с JIRA API."""

    def __init__(
        self,
        server: str,
        username: str,
        password: str,
        project_key: str,
    ):
        """Инициализация интеграции с JIRA.

        Args:
            server: URL JIRA сервера (например, http://172.16.254.6:8080)
            username: Имя пользователя JIRA
            password: Пароль или API token
            project_key: Ключ проекта (например, PFM)
        """
        self.server = server
        self.username = username
        self.password = password
        self.project_key = project_key
        self._jira: JIRA | None = None

    @property
    def jira(self) -> JIRA:
        """Получить экземпляр JIRA клиента (ленивая инициализация)."""
        if self._jira is None:
            try:
                self._jira = JIRA(
                    server=self.server,
                    basic_auth=(self.username, self.password),
                )
                logger.info(f"Успешное подключение к JIRA: {self.server}")
            except JIRAError as e:
                logger.error(f"Ошибка подключения к JIRA: {e}")
                raise
        return self._jira

    def get_issue(self, issue_key: str) -> Any:
        """Получить задачу по ключу.

        Args:
            issue_key: Ключ задачи (например, PFM-33)

        Returns:
            Объект задачи JIRA
        """
        try:
            return self.jira.issue(issue_key)
        except JIRAError as e:
            logger.error(f"Ошибка получения задачи {issue_key}: {e}")
            raise

    def get_available_issue_types(self) -> list[dict]:
        """Получить список доступных типов задач для проекта.

        Returns:
            Список словарей с информацией о типах задач
        """
        try:
            issue_types = self.jira.issue_types()
            project = self.jira.project(self.project_key)
            # Получаем типы задач, доступные для проекта
            available_types = []
            for issue_type in issue_types:
                available_types.append(
                    {
                        "id": issue_type.id,
                        "name": issue_type.name,
                        "description": getattr(issue_type, "description", ""),
                    }
                )
            logger.info(f"Доступные типы задач: {[t['name'] for t in available_types]}")
            return available_types
        except JIRAError as e:
            logger.error(f"Ошибка получения типов задач: {e}")
            return []

    def find_subtask_type(self) -> str | None:
        """Найти тип задачи для подзадачи.

        Returns:
            Название типа задачи или None, если не найден
        """
        available_types = self.get_available_issue_types()
        # Ищем подходящий тип (Sub-task, Подзадача, Task и т.д.)
        subtask_keywords = ["sub", "подзадача", "subtask", "sub-task"]
        for issue_type in available_types:
            type_name_lower = issue_type["name"].lower()
            if any(keyword in type_name_lower for keyword in subtask_keywords):
                logger.info(f"Найден тип для подзадачи: {issue_type['name']}")
                return issue_type["name"]
        # Если не нашли, пробуем "Task"
        for issue_type in available_types:
            if "task" in issue_type["name"].lower():
                logger.info(f"Используется тип задачи: {issue_type['name']}")
                return issue_type["name"]
        # Возвращаем первый доступный тип
        if available_types:
            logger.warning(f"Используется первый доступный тип: {available_types[0]['name']}")
            return available_types[0]["name"]
        return None

    def create_subtask(
        self,
        parent_key: str,
        summary: str,
        description: str = "",
        issue_type: str | None = None,
    ) -> str:
        """Создать подзадачу.

        Args:
            parent_key: Ключ родительской задачи (например, PFM-33)
            summary: Краткое описание задачи
            description: Подробное описание задачи
            issue_type: Тип задачи (если None, будет автоматически определен)

        Returns:
            Ключ созданной задачи (например, PFM-37)
        """
        try:
            # Если тип не указан, пытаемся найти подходящий
            if issue_type is None:
                issue_type = self.find_subtask_type()
                if issue_type is None:
                    raise ValueError("Не удалось определить тип задачи для подзадачи")

            # Получаем родительскую задачу для определения Reporter (для установки Assignee)
            parent_issue = self.get_issue(parent_key)
            reporter = parent_issue.fields.reporter

            issue_dict = {
                "project": {"key": self.project_key},
                "summary": summary,
                "description": description,
                "issuetype": {"name": issue_type},
                "parent": {"key": parent_key},
                # Reporter устанавливается автоматически JIRA, его нельзя задавать при создании
                # Устанавливаем только Assignee равным Reporter родительской задачи
                "assignee": {"accountId": reporter.accountId} if hasattr(reporter, "accountId") else {"name": reporter.name},
            }
            new_issue = self.jira.create_issue(fields=issue_dict)
            logger.info(f"Создана подзадача: {new_issue.key} - {summary} (Assignee: {reporter.displayName})")

            # Не переводим статус автоматически при создании - это будет сделано в скрипте синхронизации
            # на основе статуса из TODO

            return new_issue.key
        except JIRAError as e:
            logger.error(f"Ошибка создания подзадачи: {e}")
            raise

    def get_available_transitions(self, issue_key: str) -> list[dict]:
        """Получить список доступных переходов для задачи.

        Args:
            issue_key: Ключ задачи

        Returns:
            Список словарей с информацией о переходах, включая текущий статус
        """
        try:
            issue = self.jira.issue(issue_key)
            current_status = issue.fields.status.name
            transitions = self.jira.transitions(issue)
            available = []
            for transition in transitions:
                available.append(
                    {
                        "id": transition["id"],
                        "name": transition["name"],  # Название перехода (например, "На ревью")
                        "from": current_status,  # Текущий статус задачи
                        "to": transition["to"]["name"],  # Целевой статус
                    }
                )
            logger.debug(f"Доступные переходы для {issue_key}: {[t['name'] for t in available]}")
            return available
        except JIRAError as e:
            logger.error(f"Ошибка получения переходов для задачи {issue_key}: {e}")
            return []

    def get_current_status(self, issue_key: str) -> str | None:
        """Получить текущий статус задачи.

        Args:
            issue_key: Ключ задачи

        Returns:
            Название текущего статуса или None
        """
        try:
            issue = self.jira.issue(issue_key)
            return issue.fields.status.name
        except JIRAError as e:
            logger.error(f"Ошибка получения статуса задачи {issue_key}: {e}")
            return None

    def execute_transition(self, issue_key: str, transition_id: int | str) -> bool:
        """Выполнить переход по ID.

        Args:
            issue_key: Ключ задачи
            transition_id: ID перехода (может быть int или str)

        Returns:
            True если успешно, False в противном случае
        """
        try:
            issue = self.jira.issue(issue_key)
            self.jira.transition_issue(issue, str(transition_id))
            return True
        except JIRAError as e:
            logger.error(f"Ошибка выполнения перехода {transition_id} для задачи {issue_key}: {e}")
            return False
        except Exception as e:
            logger.error(f"Неожиданная ошибка при выполнении перехода {transition_id} для задачи {issue_key}: {e}")
            return False

    def transition_to_status(self, issue_key: str, target_status: str) -> bool:
        """Перевести задачу в указанный статус через переход.

        Args:
            issue_key: Ключ задачи
            target_status: Целевой статус (например, "В РАБОТЕ", "НА РЕВЬЮ", "Done")

        Returns:
            True если успешно, False в противном случае
        """
        try:
            transitions = self.get_available_transitions(issue_key)
            issue = self.jira.issue(issue_key)
            current_status = issue.fields.status.name

            # Если уже в целевом статусе, ничего не делаем
            if current_status.lower() == target_status.lower():
                logger.debug(f"Задача {issue_key} уже в статусе '{target_status}'")
                return True

            # Если нет доступных переходов, ничего не делаем
            if not transitions:
                logger.debug(f"Нет доступных переходов для задачи {issue_key} (текущий статус: '{current_status}')")
                return False

            # Ищем переход, который ведет к целевому статусу (точное совпадение)
            for transition in transitions:
                if transition["to"].lower() == target_status.lower():
                    self.jira.transition_issue(issue, transition["id"])
                    logger.info(f"Задача {issue_key} переведена в статус '{target_status}'")
                    return True

            # Если не нашли прямой переход, пробуем найти по названию перехода
            status_keywords = {
                "В РАБОТЕ": ["взять в работу", "в работе", "в работу", "начать", "start", "begin", "work"],
                "НА РЕВЬЮ": ["на ревью", "ревью", "review", "на проверку", "проверка"],
                "DONE": ["done", "готово", "завершено", "выполнено"],
                "TO DO": ["to do", "к выполнению", "открыт"],
            }

            target_lower = target_status.lower()
            keywords = status_keywords.get(target_status.upper(), [target_lower])

            for transition in transitions:
                transition_name_lower = transition["name"].lower()
                transition_to_lower = transition["to"].lower()
                
                # Проверяем по названию перехода
                if any(keyword in transition_name_lower for keyword in keywords):
                    self.jira.transition_issue(issue, transition["id"])
                    logger.info(f"Задача {issue_key} переведена в статус '{transition['to']}' через переход '{transition['name']}'")
                    return True
                
                # Проверяем по целевому статусу перехода
                if any(keyword in transition_to_lower for keyword in keywords):
                    self.jira.transition_issue(issue, transition["id"])
                    logger.info(f"Задача {issue_key} переведена в статус '{transition['to']}'")
                    return True

            # Логируем только на уровне DEBUG, чтобы не засорять логи
            logger.debug(f"Переход к статусу '{target_status}' не найден для задачи {issue_key}. Текущий статус: '{current_status}', доступные переходы: {[t['name'] + ' -> ' + t['to'] for t in transitions]}")
            return False
        except JIRAError as e:
            logger.error(f"Ошибка перевода задачи {issue_key} в статус '{target_status}': {e}")
            return False

    def update_issue_status(self, issue_key: str, status_name: str) -> bool:
        """Обновить статус задачи (устаревший метод, используйте transition_to_status).

        Args:
            issue_key: Ключ задачи
            status_name: Название статуса (например, "In Progress", "Done")

        Returns:
            True если успешно, False в противном случае
        """
        return self.transition_to_status(issue_key, status_name)

    def add_comment(self, issue_key: str, comment: str) -> bool:
        """Добавить комментарий к задаче.

        Args:
            issue_key: Ключ задачи
            comment: Текст комментария

        Returns:
            True если успешно, False в противном случае
        """
        try:
            self.jira.add_comment(issue_key, comment)
            logger.info(f"Добавлен комментарий к задаче {issue_key}")
            return True
        except JIRAError as e:
            logger.error(f"Ошибка добавления комментария к задаче {issue_key}: {e}")
            return False

    def update_assignee(self, issue_key: str, assignee_name: str | None = None) -> bool:
        """Обновить Assignee задачи.

        Args:
            issue_key: Ключ задачи
            assignee_name: Имя пользователя для назначения (если None, используется Reporter)

        Returns:
            True если успешно, False в противном случае
        """
        try:
            issue = self.jira.issue(issue_key)
            if assignee_name is None:
                # Используем Reporter как Assignee
                reporter = issue.fields.reporter
                assignee_name = reporter.name if hasattr(reporter, "name") else reporter.accountId

            # Обновляем Assignee
            self.jira.assign_issue(issue_key, assignee_name)
            logger.info(f"Assignee задачи {issue_key} установлен: {assignee_name}")
            return True
        except JIRAError as e:
            logger.error(f"Ошибка обновления Assignee задачи {issue_key}: {e}")
            return False

    def find_subtask_by_number(self, parent_key: str, item_number: str) -> str | None:
        """Найти подзадачу по номеру пункта из TODO.

        Args:
            parent_key: Ключ родительской задачи
            item_number: Номер пункта (например, "1.1", "2.3")

        Returns:
            Ключ задачи или None, если не найдена
        """
        try:
            # Ищем все подзадачи родительской задачи и фильтруем по номеру
            # Используем простой поиск без экранирования точки
            jql = f'project = {self.project_key} AND parent = {parent_key}'
            issues = self.jira.search_issues(jql, maxResults=200)
            
            # Ищем точное совпадение номера в начале summary
            for issue in issues:
                if issue.fields.summary.startswith(f"{item_number}:"):
                    return issue.key
            return None
        except JIRAError as e:
            logger.warning(f"Ошибка поиска подзадачи по номеру {item_number}: {e}")
            return None

    def find_or_create_subtask(
        self,
        parent_key: str,
        summary: str,
        description: str = "",
        issue_type: str | None = None,
        item_number: str | None = None,
    ) -> str:
        """Найти существующую подзадачу или создать новую.

        Args:
            parent_key: Ключ родительской задачи
            summary: Краткое описание задачи
            description: Подробное описание задачи
            issue_type: Тип задачи (если None, будет автоматически определен)
            item_number: Номер пункта из TODO (например, "1.1") для поиска существующей задачи

        Returns:
            Ключ задачи (существующей или новой)
        """
        try:
            # Поиск существующих подзадач по номеру пункта
            if item_number:
                issue_key = self.find_subtask_by_number(parent_key, item_number)
                if issue_key:
                    logger.info(f"Найдена существующая подзадача: {issue_key}")
                    # Обновляем Assignee для существующей задачи
                    parent_issue = self.get_issue(parent_key)
                    reporter = parent_issue.fields.reporter
                    self.update_assignee(issue_key, reporter.name if hasattr(reporter, "name") else None)
                    return issue_key

            # Если не нашли по номеру, пробуем поиск по summary (для обратной совместимости)
            try:
                # Берем только начало summary до первого спецсимвола для безопасного поиска
                safe_summary = summary.split(":")[0] + ":" if ":" in summary else summary[:50]
                jql = f'project = {self.project_key} AND parent = {parent_key} AND summary ~ "{safe_summary}"'
                issues = self.jira.search_issues(jql, maxResults=1)
                if issues:
                    issue_key = issues[0].key
                    logger.info(f"Найдена существующая подзадача: {issue_key}")
                    parent_issue = self.get_issue(parent_key)
                    reporter = parent_issue.fields.reporter
                    self.update_assignee(issue_key, reporter.name if hasattr(reporter, "name") else None)
                    return issue_key
            except JIRAError:
                # Если поиск по summary не удался, просто создаем новую задачу
                pass

            # Создать новую подзадачу (Assignee уже устанавливается при создании)
            return self.create_subtask(parent_key, summary, description, issue_type)
        except JIRAError as e:
            logger.error(f"Ошибка поиска/создания подзадачи: {e}")
            raise

    def map_todo_status_to_jira(self, todo_status: str) -> str:
        """Преобразовать статус из TODO в статус JIRA.

        Args:
            todo_status: Статус из TODO (DONE, PENDING, IN_PROGRESS)

        Returns:
            Название статуса JIRA (может быть на русском или английском)
        """
        # Пробуем разные варианты названий статусов
        status_mapping = {
            "DONE": ["Done", "Готово", "Завершено", "Выполнено"],
            "PENDING": ["To Do", "К выполнению", "Открыт", "Ожидает"],
            "IN_PROGRESS": ["In Progress", "В работе", "В процессе"],
        }
        
        variants = status_mapping.get(todo_status.upper(), ["To Do"])
        # Возвращаем первый вариант, реальное название будет определено при переходе
        return variants[0]

