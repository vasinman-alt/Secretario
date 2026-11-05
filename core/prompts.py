"""Сервис загрузки промптов для LLM. Один источник истины."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Дефолтные промпты (используются, если файлы не найдены)
# ----------------------------------------------------------------------
DEFAULT_CONVERSATION_ANALYSIS = """\
Ты — аналитик телефонных переговоров. Анализируешь стенограммы
звонков между владельцем приложения ("Оператор") и его собеседником
("Собеседник"). Некоторые реплики могут быть помечены как
"Неопределённо" — используй контекст диалога, чтобы присвоить
реплику одному из участников.

Верни ответ строго в формате:

# Краткое резюме
2–5 предложений.

# Основные темы
Тема 1
- ...

# Договорённости и следующие действия
К чему пришли и что каждый планирует делать дальше.
Если ничего не определено: "Явных договорённостей не достигнуто."

# Важные факты
Цифры, сроки, оборудование, объекты, даты, люди.

# Неясные места
Если их нет — "Нет".
"""

DEFAULT_STITCHING = """\
Ты — редактор телефонных стенограмм. У тебя есть сырые фрагменты
речи (от Silero VAD) с метками говорящего (Оператор/Собеседник/
Неопределённо) и временными метками.

Задачи:
1. Объедини соседние реплики одного говорящего, если пауза < 1.5 с
   и по смыслу это одна фраза.
2. Исправь ошибки ASR по контексту.
3. Восстанови пунктуацию.
4. Для "Неопределённо" — определи говорящего по контексту, или
   оставь метку.
5. Сохрани таймкоды в формате "MM:SS".

Верни результат в формате:

MM:SS
Оператор:
Текст.

MM:SS
Собеседник:
Текст.
"""

DEFAULT_USER_TEMPLATE = """\
Стенограмма телефонного разговора:

{transcript}"""


# ----------------------------------------------------------------------
# Сервис
# ----------------------------------------------------------------------
class PromptService:
    """
    Загружает промпты из файловой системы. Все три файла опциональны —
    при отсутствии используются встроенные дефолты.
    """

    def __init__(
        self,
        conversation_analysis_path: str | Path,
        stitching_path: str | Path,
        user_template_path: str | Path,
    ):
        self._analysis_path = Path(conversation_analysis_path)
        self._stitching_path = Path(stitching_path)
        self._user_path = Path(user_template_path)

        self._analysis_cache: Optional[str] = None
        self._stitching_cache: Optional[str] = None
        self._user_cache: Optional[str] = None

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------
    def get_conversation_analysis(self) -> str:
        if self._analysis_cache is None:
            self._analysis_cache = self._read_or_default(
                self._analysis_path,
                DEFAULT_CONVERSATION_ANALYSIS,
                "conversation_analysis",
            )
        return self._analysis_cache

    def get_stitching(self) -> str:
        if self._stitching_cache is None:
            self._stitching_cache = self._read_or_default(
                self._stitching_path,
                DEFAULT_STITCHING,
                "stitching",
            )
        return self._stitching_cache

    def get_user_template(self) -> str:
        if self._user_cache is None:
            self._user_cache = self._read_or_default(
                self._user_path,
                DEFAULT_USER_TEMPLATE,
                "user_template",
            )
        return self._user_cache

    def render_user(self, transcript: str) -> str:
        return self.get_user_template().replace("{transcript}", transcript)

    def render_stitching(self, raw_segments: str) -> str:
        template = self.get_stitching()
        return template.replace("{transcript}", raw_segments)

    def reload(self) -> None:
        self._analysis_cache = None
        self._stitching_cache = None
        self._user_cache = None

    # ------------------------------------------------------------------
    # Служебные
    # ------------------------------------------------------------------
    @staticmethod
    def _read_or_default(path: Path, default: str, label: str) -> str:
        if not path.exists():
            logger.warning(
                "Prompt file %s not found, using default", path
            )
            return default
        try:
            return path.read_text(encoding="utf-8").strip()
        except Exception as e:
            logger.error(
                "Failed to read %s prompt %s: %s", label, path, e
            )
            return default
