"""GigaAM-v3-e2e-ctc — распознавание русской речи.

GigaAM: ~240M параметров, лицензия MIT, обучен на русском,
включая телефонию. CTC-декодер (не авторегрессивный) — быстрый на CPU.

Использует .transcribe() с word_timestamps=True (лимит 25 сек на вызов).
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from core.config import AppConfig
from core.progress import ProgressReporter
from core.types import (
    AsrSegment,
    SpeakerIdResult,
    VadSegment,
    WordTimestamp,
)

logger = logging.getLogger(__name__)


class GigaAMService:
    """Обёртка над GigaAM. Модель грузится лениво."""

    def __init__(self, config: AppConfig):
        self._config = config
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        logger.info("Loading GigaAM model: %s", self._config.asr.model_name)
        try:
            import gigaam  # type: ignore
        except ImportError as e:
            raise ImportError(
                "gigaam не установлен. Установите:\n"
                "  pip install git+https://github.com/salute-developers/GigaAM.git"
            ) from e

        model_name = self._config.asr.model_name
        self._model = gigaam.load_model(
            model_name,
            device=self._config.asr.device,
            use_flash_attention=False,  # у нас CPU
        )
        logger.info("GigaAM loaded: %s", model_name)

    def transcribe_segments(
        self,
        audio: np.ndarray,
        sample_rate: int,
        vad_segments: List[VadSegment],
        speaker_results: List[SpeakerIdResult],
        progress: Optional[ProgressReporter] = None,
    ) -> List[AsrSegment]:
        """
        Распознаёт каждый VAD-сегмент отдельно (GigaAM .transcribe()
        имеет лимит ~25 сек; отдельные реплики всегда короче).
        """
        prog = progress or ProgressReporter()
        self._load()

        cfg = self._config.asr
        max_dur = cfg.max_segment_duration_s
        asr_segments: List[AsrSegment] = []

        for i, (vad_seg, spk_res) in enumerate(
            zip(vad_segments, speaker_results)
        ):
            if vad_seg.duration > max_dur:
                logger.warning(
                    "Segment %d too long (%.1f s), splitting",
                    i, vad_seg.duration,
                )
                # Для очень длинных сегментов — режем по max_dur
                sub_segments = self._split_long_segment(
                    audio, sample_rate, vad_seg, max_dur,
                )
                for sub_vad in sub_segments:
                    sub_audio = self._extract(
                        audio, sample_rate, sub_vad,
                    )
                    seg = self._transcribe_one(
                        sub_audio, sample_rate, sub_vad, spk_res, prog,
                    )
                    asr_segments.append(seg)
            else:
                seg_audio = self._extract(audio, sample_rate, vad_seg)
                seg = self._transcribe_one(
                    seg_audio, sample_rate, vad_seg, spk_res, prog,
                )
                asr_segments.append(seg)

            if prog:
                prog.stage(
                    "asr", (i + 1) / len(vad_segments),
                    f"ASR: {i+1}/{len(vad_segments)}",
                )

        return asr_segments

    def _transcribe_one(
        self,
        seg_audio: np.ndarray,
        sample_rate: int,
        vad_seg: VadSegment,
        spk_res: SpeakerIdResult,
        progress: Optional[ProgressReporter] = None,
    ) -> AsrSegment:
        """Распознаёт один сегмент."""
        try:
            # gigaam принимает numpy array напрямую
            result = self._model.transcribe(
                seg_audio,
                language="ru",
                word_timestamps=self._config.asr.word_timestamps,
                temperature=self._config.asr.decode_temperature,
            )
        except Exception as e:
            logger.error("GigaAM transcribe failed: %s", e)
            return AsrSegment(
                start=vad_seg.start,
                end=vad_seg.end,
                text="",
                speaker=spk_res.label,
                speaker_confidence=spk_res.confidence,
                is_uncertain=spk_res.is_uncertain,
            )

        # result — dict-like: {"text": ..., "words": [...]}
        # Зависит от версии GigaAM; нормализуем
        text = self._extract_text(result)
        words = self._extract_words(result, vad_seg.start, sample_rate)

        return AsrSegment(
            start=vad_seg.start,
            end=vad_seg.end,
            text=text,
            speaker=spk_res.label,
            speaker_confidence=spk_res.confidence,
            is_uncertain=spk_res.is_uncertain,
            words=words,
        )

    def _extract_text(self, result) -> str:
        """Извлекает текст из результата GigaAM."""
        if isinstance(result, dict):
            return (result.get("text") or "").strip()
        # Если вернулся объект с атрибутом text
        return (getattr(result, "text", "") or "").strip()

    def _extract_words(
        self, result, offset: float, sample_rate: int,
    ) -> List[WordTimestamp]:
        """Извлекает пословные таймкоды."""
        words_data = []
        if isinstance(result, dict):
            words_data = result.get("words") or result.get("word_timestamps") or []
        else:
            words_data = getattr(result, "words", []) or []

        out: List[WordTimestamp] = []
        for w in words_data:
            if isinstance(w, dict):
                word = w.get("word") or w.get("text") or ""
                start = w.get("start", 0.0)
                end = w.get("end", 0.0)
            else:
                word = getattr(w, "word", "") or getattr(w, "text", "")
                start = getattr(w, "start", 0.0)
                end = getattr(w, "end", 0.0)

            if not word:
                continue
            out.append(
                WordTimestamp(
                    word=word,
                    start=offset + float(start),
                    end=offset + float(end),
                )
            )
        return out

    def _extract(
        self, audio: np.ndarray, sample_rate: int, seg: VadSegment,
    ) -> np.ndarray:
        a = int(seg.start * sample_rate)
        b = int(seg.end * sample_rate)
        return audio[a:b]

    def _split_long_segment(
        self, audio: np.ndarray, sample_rate: int,
        seg: VadSegment, max_dur: float,
    ) -> List[VadSegment]:
        """Режет длинный VAD-сегмент на части <= max_dur."""
        subs = []
        t = seg.start
        while t < seg.end:
            sub_end = min(t + max_dur, seg.end)
            subs.append(
                VadSegment(
                    start=t,
                    end=sub_end,
                    audio_path=seg.audio_path,
                    raw_audio=None,
                )
            )
            t = sub_end
        return subs

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            logger.info("GigaAM unloaded")
