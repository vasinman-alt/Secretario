"""Типизированные структуры данных для Secretario."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Literal

from core.filename_parser import CallMetadata


# ----------------------------------------------------------------------
# Метки спикеров
# ----------------------------------------------------------------------
SpeakerLabel = Literal["Оператор", "Собеседник", "Неопределённо"]


# ----------------------------------------------------------------------
# VAD
# ----------------------------------------------------------------------
@dataclass
class VadSegment:
    """Сегмент речи, выделенный Silero VAD."""
    start: float                  # секунды от начала аудио
    end: float
    audio_path: Path              # путь к вырезанному фрагменту (если был сохранён)
    raw_audio: Optional[bytes] = None   # сырые PCM-байты (альтернатива файлу)

    @property
    def duration(self) -> float:
        return self.end - self.start


# ----------------------------------------------------------------------
# Speaker ID
# ----------------------------------------------------------------------
@dataclass
class SpeakerIdResult:
    """Результат идентификации говорящего для одного сегмента."""
    label: SpeakerLabel
    confidence: float             # косинусное сходство с эталоном (0..1)
    is_uncertain: bool = False    # True -> "Неопределённо"


# ----------------------------------------------------------------------
# ASR
# ----------------------------------------------------------------------
@dataclass
class WordTimestamp:
    word: str
    start: float
    end: float


@dataclass
class AsrSegment:
    """Один речевой сегмент после VAD + ECAPA + GigaAM."""
    start: float                  # секунды от начала аудио
    end: float
    text: str                     # распознанный текст с пунктуацией
    speaker: SpeakerLabel = "Неопределённо"
    speaker_confidence: float = 0.0
    is_uncertain: bool = False
    words: List[WordTimestamp] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "speaker": self.speaker,
            "speaker_confidence": self.speaker_confidence,
            "is_uncertain": self.is_uncertain,
            "words": [
                {"word": w.word, "start": w.start, "end": w.end}
                for w in self.words
            ],
        }


# ----------------------------------------------------------------------
# Enrollment (голосовой профиль владельца)
# ----------------------------------------------------------------------
@dataclass
class EnrollmentProfile:
    name: str
    embedding_path: Path          # путь к .emb (numpy array)
    meta_path: Path               # путь к .meta.json
    created_at: str
    sample_rate: int
    duration_s: float
    source: str = "manual"        # "manual" | "from_call"


# ----------------------------------------------------------------------
# Финальные результаты
# ----------------------------------------------------------------------
@dataclass
class Segment:
    """Обратная совместимость со старым кодом (используется в UI)."""
    start: float
    end: float
    text: str
    speaker: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "speaker": self.speaker,
        }


@dataclass
class TranscriptionResult:
    audio_path: Path
    language: str
    duration: float
    segments: List[Segment] = field(default_factory=list)
    transcript_path: Optional[Path] = None
    metadata: Optional[CallMetadata] = None
    # Новые поля
    asr_segments: List[AsrSegment] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n".join(s.text for s in self.segments if s.text)


@dataclass
class AnalysisResult:
    raw: str
    transcript_path: Optional[Path] = None
    analysis_path: Optional[Path] = None
    model_name: Optional[str] = None
    metadata: Optional[CallMetadata] = None
    # Новые поля (для структурированного анализа)
    summary: Optional[str] = None
    action_items: List[str] = field(default_factory=list)
    contacts: List[str] = field(default_factory=list)
    dates: List[str] = field(default_factory=list)
    amounts: List[str] = field(default_factory=list)
    reminders: List[str] = field(default_factory=list)
    unclear: List[str] = field(default_factory=list)


@dataclass
class PipelineResult:
    audio_path: Path
    transcription: TranscriptionResult
    analysis: Optional[AnalysisResult] = None
    # Новые поля
    vad_segments: List[VadSegment] = field(default_factory=list)
    speaker_id_results: List[SpeakerIdResult] = field(default_factory=list)
    enrollment_profile: Optional[EnrollmentProfile] = None
