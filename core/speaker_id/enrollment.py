"""Enrollment — запись и импорт голосового эталона владельца.

Три способа:
    1. record_from_microphone() — запись через микрофон
    2. import_from_file(path) — импорт готового аудиофайла (предпочтительно)
    3. bootstrap_from_segments(segments) — бутстрап из расшифровки

Для импорта рекомендуются записи, сделанные ТЕМ ЖЕ устройством/кодеком,
что и реальные звонки — это обеспечивает совпадение акустического домена.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np

from core.config import AppConfig
from core.types import EnrollmentProfile, VadSegment
from core.vad.silero import SileroVadService

logger = logging.getLogger(__name__)


class EnrollmentService:
    """
    Управляет голосовым профилем владельца.
    Профиль = (embedding, метаданные), хранится в enrollment/<name>.emb + .meta.json
    """

    def __init__(self, config: AppConfig):
        self._config = config
        self._vad = SileroVadService(config)
        # Не храним ссылку на EcapaService — получаем по требованию

    # ------------------------------------------------------------------
    # Способ 1: запись с микрофона
    # ------------------------------------------------------------------
    def record_from_microphone(
        self,
        duration_s: float = 25.0,
        sample_rate: int = 16000,
    ) -> Path:
        """
        Записывает аудио с микрофона. Возвращает путь к .wav файлу.
        Пользователь должен прочитать рекомендованный текст.
        """
        import sounddevice as sd  # type: ignore

        out_path = (
            Path(self._config.enrollment.enrollment_dir)
            / f"recording_{int(time.time())}.wav"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Recording from microphone: %.1f sec...", duration_s)
        audio = sd.rec(
            int(duration_s * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        audio = audio.flatten()

        import soundfile as sf
        sf.write(str(out_path), audio, sample_rate)
        logger.info("Recording saved: %s", out_path)
        return out_path

    # ------------------------------------------------------------------
    # Способ 2: импорт из файла (рекомендуемый)
    # ------------------------------------------------------------------
    def import_from_file(self, audio_path: Path | str) -> Path:
        """
        Принимает готовый аудиофайл (например, диктофонную запись
        с того же телефона), копирует в enrollment/ для последующей
        обработки. Возвращает путь к скопированному файлу.
        """
        src = Path(audio_path)
        if not src.exists():
            raise FileNotFoundError(f"Source file not found: {src}")

        dst_dir = Path(self._config.enrollment.enrollment_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"imported_{int(time.time())}_{src.name}"

        import shutil
        shutil.copy2(src, dst)
        logger.info("Imported audio: %s -> %s", src, dst)
        return dst

    # ------------------------------------------------------------------
    # Главный метод: построить эталон из аудиофайла
    # ------------------------------------------------------------------
    def build_profile(
        self,
        audio_path: Path | str,
        profile_name: Optional[str] = None,
    ) -> EnrollmentProfile:
        """
        Из аудиофайла строит голосовой профиль:
        1. VAD -> речевые сегменты
        2. ECAPA -> эмбеддинги для каждого сегмента
        3. Средний эмбеддинг -> финальный эталон
        4. Сохранение .emb + .meta.json
        """
        from core.manager import ModelManager  # поздний импорт

        cfg = self._config.enrollment
        profile_name = profile_name or cfg.default_profile
        audio_path = Path(audio_path)

        # 1. Загрузка аудио
        import soundfile as sf
        audio, sr = sf.read(str(audio_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != cfg.sample_rate:
            import torchaudio.transforms as T
            import torch
            audio = T.Resample(sr, cfg.sample_rate)(
                torch.from_numpy(audio)
            ).numpy()
            sr = cfg.sample_rate

        duration = len(audio) / sr
        if duration < cfg.min_duration_s:
            raise ValueError(
                f"Audio too short: {duration:.1f} sec "
                f"(min {cfg.min_duration_s:.0f} required)"
            )

        logger.info("Building enrollment profile '%s' (%.1f sec)", profile_name, duration)

        # 2. VAD — оставляем только речь
        vad_segments = self._vad.segment(audio, sr)
        speech_audio_parts = [
            audio[int(s.start * sr):int(s.end * sr)]
            for s in vad_segments
        ]
        if not speech_audio_parts:
            raise ValueError("No speech detected in enrollment audio")

        # 3. ECAPA — считаем эмбеддинги
        manager = ModelManager(self._config)
        ecapa = manager.speaker_id()
        ecapa._load()

        import torch
        embeddings = []
        for chunk in speech_audio_parts:
            tensor = torch.from_numpy(chunk).unsqueeze(0)
            with torch.no_grad():
                emb = ecapa._model.encode_batch(tensor)
            embeddings.append(emb.squeeze().numpy())

        # Средний вектор с L2-нормализацией
        mean_emb = np.mean(embeddings, axis=0)
        mean_emb = mean_emb / np.linalg.norm(mean_emb)

        # 4. Сохранение
        emb_path = (
            Path(cfg.enrollment_dir) / f"{profile_name}.emb"
        )
        meta_path = emb_path.with_suffix(".meta.json")
        emb_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(emb_path, mean_emb)

        meta = {
            "name": profile_name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "sample_rate": sr,
            "duration_s": duration,
            "source": str(audio_path),
            "num_speech_segments": len(vad_segments),
            "embedding_dim": len(mean_emb),
        }
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # 5. Устанавливаем эталон в ECAPA-сервис
        ecapa.set_enrollment(mean_emb, profile_name)

        profile = EnrollmentProfile(
            name=profile_name,
            embedding_path=emb_path,
            meta_path=meta_path,
            created_at=meta["created_at"],
            sample_rate=sr,
            duration_s=duration,
            source=str(audio_path),
        )
        logger.info("Profile built: %s", profile)
        return profile

    # ------------------------------------------------------------------
    # Загрузка существующего профиля
    # ------------------------------------------------------------------
    def load_profile(
        self, profile_name: Optional[str] = None,
    ) -> Optional[EnrollmentProfile]:
        """Загружает ранее сохранённый профиль."""
        cfg = self._config.enrollment
        profile_name = profile_name or cfg.default_profile

        emb_path = Path(cfg.enrollment_dir) / f"{profile_name}.emb"
        meta_path = emb_path.with_suffix(".meta.json")

        if not emb_path.exists() or not meta_path.exists():
            return None

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        embedding = np.load(emb_path)

        # Устанавливаем в ECAPA
        from core.manager import ModelManager
        manager = ModelManager(self._config)
        manager.speaker_id().set_enrollment(embedding, profile_name)

        return EnrollmentProfile(
            name=meta["name"],
            embedding_path=emb_path,
            meta_path=meta_path,
            created_at=meta["created_at"],
            sample_rate=meta["sample_rate"],
            duration_s=meta["duration_s"],
            source=meta.get("source", ""),
        )

    def get_default_profile(self) -> Optional[EnrollmentProfile]:
        """Удобный метод для UI — загружает дефолтный профиль."""
        return self.load_profile(self._config.enrollment.default_profile)

    # ------------------------------------------------------------------
    # Способ 3: бутстрап из расшифровки (после первой расшифровки)
    # ------------------------------------------------------------------
    def bootstrap_from_segments(
        self,
        audio: np.ndarray,
        sample_rate: int,
        vad_segments: List[VadSegment],
        speaker_results: list,    # List[SpeakerIdResult]
        profile_name: Optional[str] = None,
    ) -> Optional[EnrollmentProfile]:
        """
        Строит профиль из реплик одного спикера (предположительно владельца)
        после первой расшифровки. Требует подтверждения пользователя.
        """
        # Берём все реплики, помеченные как "Оператор" с высокой уверенностью
        confident_ops = [
            (seg, res)
            for seg, res in zip(vad_segments, speaker_results)
            if res.label == "Оператор" and res.confidence > 0.7
        ]
        if len(confident_ops) < 3:
            logger.warning(
                "Not enough confident 'Operator' segments for bootstrap: %d",
                len(confident_ops),
            )
            return None

        from core.manager import ModelManager
        manager = ModelManager(self._config)
        ecapa = manager.speaker_id()
        ecapa._load()

        import torch
        embeddings = []
        for seg, _ in confident_ops:
            a = int(seg.start * sample_rate)
            b = int(seg.end * sample_rate)
            tensor = torch.from_numpy(audio[a:b]).unsqueeze(0)
            with torch.no_grad():
                emb = ecapa._model.encode_batch(tensor)
            embeddings.append(emb.squeeze().numpy())

        mean_emb = np.mean(embeddings, axis=0)
        mean_emb = mean_emb / np.linalg.norm(mean_emb)

        # Сохранение
        cfg = self._config.enrollment
        profile_name = profile_name or cfg.default_profile
        emb_path = Path(cfg.enrollment_dir) / f"{profile_name}.emb"
        meta_path = emb_path.with_suffix(".meta.json")
        emb_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(emb_path, mean_emb)

        meta = {
            "name": profile_name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "sample_rate": sample_rate,
            "duration_s": sum(
                seg.duration for seg, _ in confident_ops
            ),
            "source": "bootstrap_from_call",
            "num_speech_segments": len(confident_ops),
            "embedding_dim": len(mean_emb),
        }
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        ecapa.set_enrollment(mean_emb, profile_name)
        logger.info("Bootstrap profile built from %d segments", len(confident_ops))

        return EnrollmentProfile(
            name=profile_name,
            embedding_path=emb_path,
            meta_path=meta_path,
            created_at=meta["created_at"],
            sample_rate=sample_rate,
            duration_s=meta["duration_s"],
            source="bootstrap_from_call",
        )

    def unload(self) -> None:
        self._vad.unload()
