# core/progress.py
"""Абстракция прогресса. В UI будет Qt-реализация, в тестах — заглушка."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class ProgressState:
    stage: str
    value: float  # 0.0 .. 1.0
    message: str = ""


class ProgressReporter:
    """Базовый класс. UI наследуется и эмитит сигналы."""

    def stage(self, name: str, value: float, message: str = "") -> None:
        self.on_update(ProgressState(stage=name, value=value, message=message))

    def on_update(self, state: ProgressState) -> None:
        raise NotImplementedError

    def log(self, message: str) -> None:
        """Неструктурированное сообщение (например, имя файла)."""
        self.on_update(ProgressState(stage="log", value=-1.0, message=message))


class CallbackProgress(ProgressReporter):
    """Прогресс через callback. Удобно для тестов и CLI."""

    def __init__(self, cb: Callable[[ProgressState], None]):
        self._cb = cb

    def on_update(self, state: ProgressState) -> None:
        self._cb(state)


class NullProgress(ProgressReporter):
    """Ничего не делает. Для batch-режима или тестов."""

    def on_update(self, state: ProgressState) -> None:
        pass
