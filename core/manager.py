"""ModelManager — единая точка управления жизненным циклом моделей Secretario."""
from __future__ import annotations

import contextlib
import logging
from typing import Optional

from core.config import AppConfig
from core.prompts import PromptService

# Новые импорты — старые сервисы (Whisper, Diarization, LLM) удалены
from core.asr.gigaam import GigaAMService
from core.speaker_id.ecapa import EcapaService
from core.speaker_id.enrollment import EnrollmentService
from core.vad.silero import SileroVadService
from core.llm.qwen_llm import QwenLLMService

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Ленивая инициализация сервисов + политика 'одна тяжёлая модель в RAM'.

    ASR и LLM не должны находиться в памяти одновременно —
    для этого используется контекстный менеджер `llm_session()`.
    """

    def __init__(self, config: AppConfig):
        self._config = config
        self._vad: Optional[SileroVadService] = None
        self._speaker_id: Optional[EcapaService] = None
        self._enrollment: Optional[EnrollmentService] = None
        self._asr: Optional[GigaAMService] = None
        self._llm: Optional[QwenLLMService] = None
        self._prompts = PromptService(
            config.system_prompt_file,
            config.stitching_prompt_file,
            config.user_prompt_file,
        )

    # ------------------------------------------------------------------
    # Доступ к сервисам (ленивая загрузка)
    # ------------------------------------------------------------------
    def vad(self) -> SileroVadService:
        if self._vad is None:
            self._vad = SileroVadService(self._config)
        return self._vad

    def speaker_id(self) -> EcapaService:
        if self._speaker_id is None:
            self._speaker_id = EcapaService(self._config)
        return self._speaker_id

    def enrollment(self) -> EnrollmentService:
        if self._enrollment is None:
            self._enrollment = EnrollmentService(self._config)
        return self._enrollment

    def asr(self) -> GigaAMService:
        if self._asr is None:
            self._asr = GigaAMService(self._config)
        return self._asr

    def llm(self) -> QwenLLMService:
        if self._llm is None:
            self._llm = QwenLLMService(self._config, self._prompts)
        return self._llm

    def prompts(self) -> PromptService:
        return self._prompts

    # ------------------------------------------------------------------
    # Контекстный менеджер: гарантирует, что ASR выгружен, пока работает LLM
    # ------------------------------------------------------------------
    @contextlib.contextmanager
    def llm_session(self):
        """
        Перед входом выгружает ASR из RAM (если загружена).
        После выхода — оставляет LLM загруженным (для UI-диалогов) либо
        выгружает, если config.llm.auto_unload_after=True.
        """
        if self._asr is not None:
            logger.info("Unloading ASR before LLM session")
            self._asr.unload()
        try:
            yield self.llm()
        finally:
            if self._config.llm.auto_unload_after:
                logger.info("Auto-unloading LLM after session")
                self.llm().unload()

    @contextlib.contextmanager
    def asr_session(self):
        """Перед входом выгружает LLM из RAM."""
        if self._llm is not None:
            logger.info("Unloading LLM before ASR session")
            self._llm.unload()
        try:
            yield self.asr()
        finally:
            logger.debug("ASR session ended")

    # ------------------------------------------------------------------
    # Полная очистка
    # ------------------------------------------------------------------
    def unload_all(self) -> None:
        logger.info("Unloading all models")
        for svc in (
            self._vad,
            self._speaker_id,
            self._asr,
            self._llm,
        ):
            if svc is not None:
                try:
                    svc.unload()
                except Exception as e:
                    logger.warning("Unload failed: %s", e)

    def stats(self) -> dict:
        return {
            "vad_loaded": self._vad is not None,
            "speaker_id_loaded": self._speaker_id is not None,
            "asr_loaded": self._asr is not None,
            "llm_loaded": self._llm is not None,
            "llm_model": (
                str(self._config.llm.model_path) if self._llm else None
            ),
        }
