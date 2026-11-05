"""ECAPA-TDNN — идентификация владельца по голосу (Speaker ID)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from core.config import AppConfig
from core.progress import ProgressReporter
from core.types import SpeakerIdResult, VadSegment

logger = logging.getLogger(__name__)


class EcapaService:
    """
    Обёртка над ECAPA-TDNN из speechbrain.
    Сравнивает каждый VAD-сегмент с эталоном владельца.
    Возвращает метку: "Оператор" / "Собеседник" / "Неопределённо".
    """

    def __init__(self, config: AppConfig):
        self._config = config
        self._model = None
        self._enrollment_embedding: Optional[np.ndarray] = None
        self._enrollment_name: Optional[str] = None

    def _load(self) -> None:
        if self._model is not None:
            return
        logger.info("Loading ECAPA-TDNN from speechbrain...")
        from speechbrain.inference.speaker import EncoderClassifier

        self._model = EncoderClassifier.from_hparams(
            source=self._config.speaker_id.model_source,
            savedir=str(Path(self._config.speaker_id_model_path)),
            run_opts={"device": "cpu"},
        )
        logger.info("ECAPA-TDNN loaded")

    def set_enrollment(self, embedding: np.ndarray, name: str) -> None:
        """Установить эталон владельца (вызывается из EnrollmentService)."""
        self._enrollment_embedding = embedding
        self._enrollment_name = name
        logger.info("Enrollment set: %s (dim=%d)", name, len(embedding))

    def has_enrollment(self) -> bool:
        return self._enrollment_embedding is not None

    def identify(
        self,
        audio: np.ndarray,
        sample_rate: int,
        vad_segments: List[VadSegment],
        progress: Optional[ProgressReporter] = None,
    ) -> List[SpeakerIdResult]:
        """
        Для каждого VAD-сегмента возвращает метку говорящего.
        Если эталона нет — все результаты "Неопределённо".
        """
        prog = progress or ProgressReporter()
        self._load()

        # Если нет эталона — нечего идентифицировать
        if not self.has_enrollment():
            logger.warning("No enrollment set, all segments marked uncertain")
            return [
                SpeakerIdResult(
                    label="Неопределённо", confidence=0.0, is_uncertain=True,
                )
                for _ in vad_segments
            ]

        cfg = self._config.speaker_id
        min_dur = cfg.min_segment_duration_s
        match_th = cfg.match_threshold
        uncertain_th = cfg.uncertain_threshold

        import torch
        import torchaudio

        results: List[SpeakerIdResult] = []
        ref_emb = torch.from_numpy(self._enrollment_embedding).unsqueeze(0)

        for i, seg in enumerate(vad_segments):
            duration = seg.duration
            # Правило v2.1: короткие сегменты -> сразу "Неопределённо"
            if duration < min_dur:
                results.append(
                    SpeakerIdResult(
                        label="Неопределённо",
                        confidence=0.0,
                        is_uncertain=True,
                    )
                )
                prog.log(
                    f"Speaker ID [{i+1}/{len(vad_segments)}]: "
                    f"короткий ({duration:.2f}с) -> неопределённо"
                )
                continue

            # Вырезаем аудио сегмента
            a = int(seg.start * sample_rate)
            b = int(seg.end * sample_rate)
            seg_audio = audio[a:b]
            seg_tensor = torch.from_numpy(seg_audio).unsqueeze(0)

            # Считаем эмбеддинг
            with torch.no_grad():
                emb = self._model.encode_batch(seg_tensor)
            emb = emb.squeeze().numpy()

            # Косинусное сходство с эталоном
            sim = float(
                np.dot(emb, self._enrollment_embedding)
                / (
                    np.linalg.norm(emb)
                    * np.linalg.norm(self._enrollment_embedding)
                )
            )

            # Классификация по порогам
            if sim >= match_th:
                label = "Оператор"
                is_uncertain = False
            elif sim < uncertain_th:
                label = "Собеседник"
                is_uncertain = False
            else:
                label = "Неопределённо"
                is_uncertain = True

            results.append(
                SpeakerIdResult(
                    label=label,
                    confidence=sim,
                    is_uncertain=is_uncertain,
                )
            )
            prog.log(
                f"Speaker ID [{i+1}/{len(vad_segments)}]: "
                f"{label} (sim={sim:.3f})"
            )

        return results

    def compute_embedding(
        self, audio: np.ndarray, sample_rate: int,
    ) -> np.ndarray:
        """Считает эмбеддинг для всего фрагмента (для enrollment)."""
        self._load()
        import torch

        tensor = torch.from_numpy(audio).unsqueeze(0)
        with torch.no_grad():
            emb = self._model.encode_batch(tensor)
        return emb.squeeze().numpy()

    def unload(self) -> None:
        self._model = None
        self._enrollment_embedding = None
        logger.info("ECAPA unloaded")
