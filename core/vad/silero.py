"""Silero VAD — детекция речевых сегментов."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from core.config import AppConfig
from core.progress import ProgressReporter
from core.types import VadSegment

logger = logging.getLogger(__name__)


class SileroVadService:
    """
    Обёртка над Silero VAD v5.
    Модель: ~2 МБ, скачивается через torch.hub при первом вызове.
    """

    def __init__(self, config: AppConfig):
        self._config = config
        self._model = None
        self._utils = None

    def _load(self) -> None:
        if self._model is not None:
            return
        logger.info("Loading Silero VAD via torch.hub...")
        self._model, self._utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )
        logger.info("Silero VAD loaded")

    def segment(
        self,
        audio: np.ndarray,
        sample_rate: int,
        progress: Optional[ProgressReporter] = None,
    ) -> List[VadSegment]:
        """
        Возвращает список речевых сегментов.

        audio: float32 mono, длина любая
        sample_rate: 16000 (только!)
        """
        prog = progress or ProgressReporter()
        self._load()

        if sample_rate != 16000:
            raise ValueError(
                f"Silero VAD supports only 16 kHz, got {sample_rate}"
            )

        cfg = self._config.vad
        # get_speech_timestamps поддерживает длинные записи (>30 сек) — бьёт сам
        speech_timestamps = self._get_speech_ts(
            audio,
            threshold=cfg.threshold,
            min_speech_duration_ms=cfg.min_speech_duration_ms,
            min_silence_duration_ms=cfg.min_silence_duration_ms,
            speech_pad_ms=cfg.speech_pad_ms,
        )

        segments: List[VadSegment] = []
        for ts in speech_timestamps:
            segments.append(
                VadSegment(
                    start=ts["start"] / sample_rate,
                    end=ts["end"] / sample_rate,
                    audio_path=Path(""),  # in-memory
                    raw_audio=None,
                )
            )
            if progress:
                progress.log(
                    f"VAD: {ts['start']/sample_rate:.2f}-"
                    f"{ts['end']/sample_rate:.2f} сек"
                )

        return segments

    def _get_speech_ts(
        self,
        audio: np.ndarray,
        threshold: float,
        min_speech_duration_ms: int,
        min_silence_duration_ms: int,
        speech_pad_ms: int,
    ):
        """Обёртка: get_speech_timestamps с делением на чанки."""
        get_speech_ts, _, _, _, _ = self._utils
        # Для длинных аудио (>30 сек) нужно резать вручную
        chunk_size = 16000 * 30  # 30 секунд
        all_ts = []
        for start in range(0, len(audio), chunk_size):
            end = min(start + chunk_size, len(audio))
            chunk = audio[start:end]
            ts = get_speech_ts(
                chunk,
                self._model,
                sampling_rate=16000,
                threshold=threshold,
                min_speech_duration_ms=min_speech_duration_ms,
                min_silence_duration_ms=min_silence_duration_ms,
                speech_pad_ms=speech_pad_ms,
                return_seconds=False,
            )
            # Сдвигаем таймкоды на offset
            for t in ts:
                t["start"] += start
                t["end"] += start
            all_ts.extend(ts)
        return all_ts

    def extract_segment_audio(
        self, audio: np.ndarray, sample_rate: int, seg: VadSegment,
    ) -> np.ndarray:
        """Вырезает кусок аудио для одного VAD-сегмента."""
        a = int(seg.start * sample_rate)
        b = int(seg.end * sample_rate)
        return audio[a:b]

    def unload(self) -> None:
        self._model = None
        self._utils = None
        logger.info("Silero VAD unloaded")
