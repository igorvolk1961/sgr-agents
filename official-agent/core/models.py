"""Модели данных для официального агента управления строительными проектами."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    """Типы документов, которые может создавать официальный агент."""

    ORDER = "order"  # Приказ
    INSTRUCTION = "instruction"  # Инструкция
    PLAN = "plan"  # План
    REPORT = "report"  # Отчет
    ESCALATION = "escalation"  # Документ эскалации
    # Дополнительные типы будут добавлены по мере необходимости


class DocumentStatus(str, Enum):
    """Статусы обработки документа."""

    PENDING = "pending"  # Ожидает обработки
    IN_PROGRESS = "in_progress"  # В процессе
    DATA_COLLECTING = "data_collecting"  # Сбор данных
    DATA_VERIFYING = "data_verifying"  # Верификация данных
    GENERATING = "generating"  # Генерация документа
    COMPLETED = "completed"  # Завершен
    FAILED = "failed"  # Ошибка
    ESCALATED = "escalated"  # Эскалирован


class CollectionStatus(str, Enum):
    """Статусы сбора данных."""

    PENDING = "pending"  # Ожидает сбора
    IN_PROGRESS = "in_progress"  # В процессе сбора
    COLLECTED = "collected"  # Собрано
    FAILED = "failed"  # Ошибка сбора


class VerificationStatus(str, Enum):
    """Статусы верификации данных."""

    VERIFIED = "verified"  # Верифицировано
    HAS_CONFLICTS = "has_conflicts"  # Есть конфликты
    MISSING_DATA = "missing_data"  # Отсутствуют данные
    FAILED = "failed"  # Ошибка верификации


class ComplianceStatus(str, Enum):
    """Статусы проверки соответствия."""

    PASSED = "passed"  # Пройдено
    FAILED = "failed"  # Не пройдено
    NEEDS_REVIEW = "needs_review"  # Требуется проверка
    WARNING = "warning"  # Предупреждение


class UserPosition(BaseModel):
    """Модель должности пользователя."""

    position_name: str = Field(description="Название должности")
    position_id: str | None = Field(default=None, description="Идентификатор должности")
    authority_level: int = Field(default=0, description="Уровень полномочий (числовой)")
    allowed_document_types: list[DocumentType] = Field(
        default_factory=list, description="Список допустимых типов документов"
    )
    job_responsibilities: list[str] = Field(
        default_factory=list, description="Список должностных обязанностей"
    )
    supervisor_position: str | None = Field(
        default=None, description="Должность непосредственного руководителя"
    )
    subordinate_positions: list[str] = Field(
        default_factory=list, description="Список подчиненных должностей"
    )


class DataCollectionItem(BaseModel):
    """Элемент плана сбора данных."""

    priority: int = Field(
        ge=1, le=14, description="Приоритет сбора (1-14 согласно PRIORITY_HIERARCHY)"
    )
    category: str = Field(description="Категория данных")
    description: str = Field(description="Описание что нужно собрать")
    sources: list[str] = Field(
        default_factory=list, description="Список источников данных"
    )
    status: CollectionStatus = Field(
        default=CollectionStatus.PENDING, description="Статус сбора данных"
    )
    collected_data: dict[str, Any] = Field(
        default_factory=dict, description="Собранные данные"
    )
    required_for_documents: list[str] = Field(
        default_factory=list,
        description="Список идентификаторов документов, для которых нужны эти данные",
    )


class DataCollectionPlan(BaseModel):
    """План сбора данных с приоритетами."""

    items: list[DataCollectionItem] = Field(
        default_factory=list,
        description="Список элементов плана, отсортированный по приоритету",
    )


class Conflict(BaseModel):
    """Модель конфликта данных между источниками."""

    field: str = Field(description="Поле с конфликтом")
    value1: Any = Field(description="Значение из первого источника")
    value2: Any = Field(description="Значение из второго источника")
    source1: str = Field(description="Первый источник")
    source2: str = Field(description="Второй источник")
    resolution: str | None = Field(
        default=None, description="Разрешение конфликта (если разрешен)"
    )


class MissingData(BaseModel):
    """Модель отсутствующих данных."""

    field: str = Field(description="Поле с отсутствующими данными")
    description: str = Field(description="Описание отсутствующих данных")
    priority: int = Field(description="Приоритет важности данных")
    source: str | None = Field(
        default=None, description="Ожидаемый источник данных"
    )


class VerificationResult(BaseModel):
    """Результат верификации данных для одного документа."""

    document_id: str = Field(description="Идентификатор документа")
    verified_data: dict[str, Any] = Field(
        default_factory=dict, description="Верифицированные данные"
    )
    conflicts: list[Conflict] = Field(
        default_factory=list, description="Список конфликтов данных"
    )
    missing_data: list[MissingData] = Field(
        default_factory=list, description="Список отсутствующих данных"
    )
    verification_status: VerificationStatus = Field(
        default=VerificationStatus.VERIFIED, description="Статус верификации"
    )


class VerificationResults(BaseModel):
    """Результаты верификации данных для всех документов."""

    results: list[VerificationResult] = Field(
        default_factory=list,
        description="Список результатов верификации для каждого документа",
    )


class ComplianceCheckItem(BaseModel):
    """Элемент проверки соответствия."""

    check_id: str = Field(description="Идентификатор проверки")
    category: str = Field(
        description="Категория проверки: workflow, data, language, formatting, authority"
    )
    description: str = Field(description="Описание проверки")
    status: ComplianceStatus = Field(description="Результат проверки")
    details: str | None = Field(
        default=None, description="Детали проверки"
    )
    required: bool = Field(
        default=True, description="Обязательная проверка или предупреждение"
    )


class ComplianceReport(BaseModel):
    """Отчет о соответствии требованиям для одного документа."""

    document_id: str = Field(description="Идентификатор документа")
    checks: list[ComplianceCheckItem] = Field(
        default_factory=list, description="Список проверок"
    )
    overall_status: ComplianceStatus = Field(
        description="Общий статус проверки"
    )
    issues: list[str] = Field(
        default_factory=list, description="Список проблем"
    )
    is_ready: bool = Field(
        default=False, description="Флаг готовности документа к отправке"
    )


class ComplianceReports(BaseModel):
    """Отчеты о соответствии для всех документов."""

    reports: list[ComplianceReport] = Field(
        default_factory=list,
        description="Список отчетов для каждого документа",
    )


class DocumentTask(BaseModel):
    """Модель задачи на разработку документа."""

    document_id: str = Field(description="Уникальный идентификатор документа")
    document_type: DocumentType = Field(description="Тип документа")
    template_id: str | None = Field(default=None, description="Идентификатор шаблона документа")
    is_within_authority: bool = Field(
        default=True, description="Находится ли документ в пределах полномочий должности"
    )
    status: DocumentStatus = Field(
        default=DocumentStatus.PENDING, description="Статус обработки документа"
    )
    source: str = Field(
        description="Источник задачи: исходный запрос пользователя или эскалация"
    )
    final_document: str | None = Field(
        default=None, description="Финальный текст документа (на русском языке)"
    )
    compliance_report: ComplianceReport | None = Field(
        default=None, description="Отчет о соответствии требованиям"
    )


class OfficialAgentContext(BaseModel):
    """Основной контекст официального агента для хранения в ResearchContext.custom_context."""

    user_position: UserPosition | None = Field(
        default=None, description="Должность пользователя"
    )
    documents: list[DocumentTask] = Field(
        default_factory=list,
        description="Список документов для разработки (может включать документы эскалации)",
    )
    data_collection_plan: DataCollectionPlan = Field(
        default_factory=DataCollectionPlan,
        description="План сбора данных с приоритетами",
    )
    verification_results: VerificationResults = Field(
        default_factory=VerificationResults,
        description="Результаты верификации данных для каждого документа",
    )


