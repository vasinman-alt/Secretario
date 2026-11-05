"""Qt-адаптер: запускает ConversationService в QThread, эмитит сигналы."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, QThread, Signal

from core.progress import CallbackProgress, ProgressState
from core.services.conversation_service import ConversationService
from core.types import PipelineResult

logger = logging.getLogger(__name__)


class PipelineWorker(QObject):
    """
    Создаётся в UI-потоке, перемещается в QThread.
    UI вызывает worker.start() — выполнение идёт в дочернем потоке.

    Конвейер этапов (архитектура 2.1):
        enrollment (lazy, при необходимости) →
        vad → speaker_id → asr → stitching → llm_analysis
    """

    # stage: vad|speaker_id|asr|stitch|llm|enroll|log
    # value: 0..1 (для log — отрицательное)
    # message: str
    progress = Signal(str, float, str)
    finished = Signal(object)         # PipelineResult
    error = Signal(str)

    def __init__(
        self,
        service: ConversationService,
        audio_path: Path | str,
        run_analysis: bool = True,
    ):
        super().__init__()
        self._service = service
        self._audio_path = Path(audio_path)
        self._run_analysis = run_analysis
        self._thread: Optional[QThread] = None

    def start(self) -> None:
        """Запуск в отдельном QThread. Возвращает управление немедленно."""
        self._thread = QThread()
        self.moveToThread(self._thread)
        self._thread.started.connect(self._run)
        self.finished.connect(self._thread.quit)
        self.error.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _run(self) -> None:
        try:
            progress = CallbackProgress(self._emit_state)
            result = self._service.process(
                self._audio_path, progress, self._run_analysis,
            )
            self.finished.emit(result)
        except Exception as e:
            logger.exception("Pipeline worker error")
            self.error.emit(f"{type(e).__name__}: {e}")

    def _emit_state(self, state: ProgressState) -> None:
        self.progress.emit(state.stage, state.value, state.message)
