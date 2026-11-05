"""Главный оркестратор конвейера Secretario 2.1.

Этапы:
    1. Чтение аудио
    2. VAD (Silero) -> список речевых сегментов
    3. Speaker ID (ECAPA) -> метка "Оператор"/"Собеседник"/"Неопределённо"
    4. ASR (GigaAM) -> текст + таймкоды для каждого сегмента
    5. Stitching (LLM) -> склейка, исправление ASR, разрешение неопределённости
    6. Analysis (LLM) -> резюме, договорённости, факты

Политика памяти: ASR и LLM не находятся в памяти одновременно —
stitching+analysis идут внутри llm_session() (ASR выгружается автоматически).
"""
from __future__ import annotations

import contextlib
import logging
import time
from pathlib import Path
from typing import Optional

from core.config import AppConfig
from core.filename_parser import parse_call_metadata
from core.manager import ModelManager
from core.progress import ProgressReporter
from core.types import (
    AnalysisResult,
    AsrSegment,
    PipelineResult,
    Segment,
    SpeakerIdResult,
    TranscriptionResult,
    VadSegment,
)

logger = logging.getLogger(__name__)


class ConversationService:
    """
    Склеивает все модули конвейера в единый process().
    Не хранит состояния между вызовами (всё в ModelManager).
    """

    def __init__(self, manager: ModelManager, config: AppConfig):
        self._manager = manager
        self._config = config

    # ------------------------------------------------------------------
    # Главный метод
    # ------------------------------------------------------------------
    def process(
        self,
        audio_path: Path | str,
        progress: Optional[ProgressReporter] = None,
        run_analysis: bool = True,
    ) -> PipelineResult:
        """
        Обработать один аудиофайл. Возвращает PipelineResult
        со всеми промежуточными и финальными данными.
        """
        prog = progress or ProgressReporter()  # базовый — без callback'а
        audio_path = Path(audio_path)

        logger.info("ConversationService.process(%s)", audio_path)
        t0 = time.time()

        # --- 1. Загрузка аудио и метаданных ---
        prog.stage("vad", 0.0, f"Загрузка {audio_path.name}")
        metadata = parse_call_metadata(audio_path)
        audio, sample_rate = self._load_audio(audio_path)
        duration = len(audio) / sample_rate
        prog.log(
            f"Аудио: {duration:.1f} сек, {sample_rate} Гц, "
            f"метаданные: {'есть' if metadata else 'нет'}"
        )

        # --- 2. VAD (Silero) ---
        prog.stage("vad", 0.1, "Silero VAD...")
        vad_segments = self._manager.vad().segment(audio, sample_rate, prog)
        prog.log(f"VAD: {len(vad_segments)} речевых сегментов")

        # --- 3. Speaker ID (ECAPA) ---
        prog.stage("speaker_id", 0.0, "ECAPA идентификация...")
        speaker_results = self._manager.speaker_id().identify(
            audio, sample_rate, vad_segments, prog,
        )
        confident = sum(
            1 for r in speaker_results if not r.is_uncertain
        )
        prog.log(
            f"Speaker ID: {confident}/{len(speaker_results)} "
            f"уверенных, "
            f"{len(speaker_results) - confident} неопределённых"
        )

        # --- 4. ASR (GigaAM) ---
        prog.stage("asr", 0.0, "GigaAM распознавание...")
        asr_segments = self._manager.asr().transcribe_segments(
            audio, sample_rate, vad_segments, speaker_results, prog,
        )
        prog.log(f"ASR: {len(asr_segments)} сегментов распознано")

        # --- Конвертация в старый формат Segment для обратной совместимости ---
        legacy_segments = [
            Segment(
                start=s.start,
                end=s.end,
                text=s.text,
                speaker=s.speaker,
            )
            for s in asr_segments
        ]

        transcription = TranscriptionResult(
            audio_path=audio_path,
            language="ru",
            duration=duration,
            segments=legacy_segments,
            transcript_path=None,        # будет заполнено ниже
            metadata=metadata,
            asr_segments=asr_segments,
        )

        # --- Сохранение сырой расшифровки ---
        transcript_path = self._save_raw_transcript(transcription)
        transcription.transcript_path = transcript_path
        prog.log(f"Сохранено: {transcript_path}")

        # --- 5-6. Stitching + Analysis (LLM, с выгрузкой ASR) ---
        analysis: Optional[AnalysisResult] = None
        if run_analysis:
            with self._manager.llm_session() as llm:
                # 5. Склейка + разрешение неопределённости
                prog.stage("stitch", 0.0, "Склейка сегментов (LLM)...")
                stitched_text, llm_segments = llm.stitch_segments(
                    asr_segments, prog,
                )
                prog.log(
                    f"Сшитая стенограмма: {len(llm_segments)} реплик, "
                    f"{len(stitched_text)} символов"
                )

                # Сохраняем стенограмму в читаемом виде
                stitched_path = self._save_stitched(
                    transcript_path, stitched_text,
                )

                # 6. Анализ (резюме, договорённости, факты)
                prog.stage("llm", 0.0, "Анализ разговора (LLM)...")
                analysis = llm.analyze(
                    stitched_text, prog,
                    transcript_path=stitched_path,
                    metadata=metadata,
                )
                analysis.transcript_path = stitched_path
                prog.log(f"Анализ: {len(analysis.raw)} символов")
        else:
            prog.log("Анализ пропущен (run_analysis=False)")

        elapsed = time.time() - t0
        logger.info("Pipeline done in %.1f sec", elapsed)
        prog.log(f"Готово за {elapsed:.1f} сек")

        return PipelineResult(
            audio_path=audio_path,
            transcription=transcription,
            analysis=analysis,
            vad_segments=vad_segments,
            speaker_id_results=speaker_results,
        )

    # ------------------------------------------------------------------
    # Утилиты
    # ------------------------------------------------------------------
    def _load_audio(self, path: Path) -> tuple:
        """Загружает аудио в mono float32 с указанной частотой."""
        import numpy as np
        import soundfile as sf

        audio, sr = sf.read(str(path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        target_sr = self._config.vad.sample_rate
        if sr != target_sr:
            import torchaudio.transforms as T

            audio_t = T.Resample(sr, target_sr)(
                __import__("torch").from_numpy(audio)
            ).numpy()
            audio = audio_t
            sr = target_sr

        return audio, sr

    def _save_raw_transcript(self, tr: TranscriptionResult) -> Path:
        """Сохраняет сырую расшифровку (формат старого проекта)."""
        out_dir = Path(self._config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Имя файла: <metadata>_<timestamp>_transcript.txt
        ts = time.strftime("%Y%m%d%H%M%S")
        if tr.metadata:
            base = (
                f"{tr.metadata.client_name}"
                f"({tr.metadata.client_phone_raw})_"
                f"{tr.metadata.call_datetime.strftime('%Y%m%d%H%M%S')}"
            )
        else:
            base = tr.audio_path.stem

        path = out_dir / f"{base}_{ts}_transcript.txt"
        lines = []
        for s in tr.segments:
            speaker = s.speaker or "Неопределённо"
            mm = int(s.start // 60)
            ss = int(s.start % 60)
            lines.append(f"{mm:02d}:{ss:02d}\n{speaker}:\n{s.text}\n")
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def _save_stitched(
        self, transcript_path: Path, stitched: str,
    ) -> Path:
        """Сохраняет сшитую стенограмму рядом с сырой."""
        stitched_path = transcript_path.with_name(
            transcript_path.stem.replace("_transcript", "_stitched") + ".txt"
        )
        stitched_path.write_text(stitched, encoding="utf-8")
        return stitched_path
