"""Единая точка конфигурации приложения Secretario."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_NAME = "config.json"


# ----------------------------------------------------------------------
# VAD (Silero)
# ----------------------------------------------------------------------
@dataclass
class VADSettings:
    backend: str = "silero"
    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 100
    speech_pad_ms: int = 100
    window_size_samples: int = 512
    sample_rate: int = 16000


# ----------------------------------------------------------------------
# Speaker identification (ECAPA-TDNN)
# ----------------------------------------------------------------------
@dataclass
class SpeakerIdSettings:
    backend: str = "ecapa"
    model_source: str = "speechbrain/spkrec-ecapa-voxceleb"
    embedding_dim: int = 192
    match_threshold: float = 0.65        # >= match_threshold -> "Оператор"
    uncertain_threshold: float = 0.50     # < uncertain_threshold -> "Собеседник"
                                           # иначе -> "Неопределённо"
    min_segment_duration_s: float = 1.5   # короче -> "Неопределённо" (правило v2.1)
    enrollment_required: bool = True


# ----------------------------------------------------------------------
# Enrollment (голосовой профиль владельца)
# ----------------------------------------------------------------------
@dataclass
class EnrollmentSettings:
    enrollment_dir: str = "enrollment"
    default_profile: str = "default_user"
    sample_rate: int = 16000
    min_duration_s: float = 10.0          # минимум для надёжного эталона
    recommend_duration_s: float = 30.0


# ----------------------------------------------------------------------
# ASR (GigaAM-v3-e2e-ctc)
# ----------------------------------------------------------------------
@dataclass
class ASRSettings:
    backend: str = "gigaam"
    model_name: str = "v3_e2e_ctc"        # альтернативы: v3_e2e_rnnt
    device: str = "cpu"
    sample_rate: int = 16000
    word_timestamps: bool = True
    punctuation: bool = True
    normalization: bool = True
    decode_temperature: float = 1.0
    max_segment_duration_s: float = 25.0  # лимит GigaAM .transcribe() на один вызов


# ----------------------------------------------------------------------
# LLM (Qwen3 via llama-cpp)
# ----------------------------------------------------------------------
@dataclass
class LLMSettings:
    backend: str = "qwen3-4b"
    model_path: str = "models/llm/qwen3-4b-instruct-q4_k_m.gguf"
    n_ctx: int = 8192
    n_threads: int = 4
    max_tokens: int = 1500
    temperature: float = 0.3
    top_p: float = 0.9
    repeat_penalty: float = 1.1
    auto_unload_after: bool = True        # выгружать LLM из RAM после использования


# ----------------------------------------------------------------------
# Структурированный анализ разговора
# ----------------------------------------------------------------------
@dataclass
class AnalysisSettings:
    extract_summary: bool = True
    extract_action_items: bool = True
    extract_contacts: bool = True
    extract_dates: bool = True
    extract_amounts: bool = True
    extract_reminders: bool = True


# ----------------------------------------------------------------------
# Поведение пайплайна
# ----------------------------------------------------------------------
@dataclass
class PipelineSettings:
    keep_intermediate_audio: bool = False
    save_debug_segments: bool = False
    uncertain_segments_to_llm: bool = True   # правило v2.1


# ----------------------------------------------------------------------
# Главный конфиг
# ----------------------------------------------------------------------
@dataclass
class AppConfig:
    models_dir: str = "models"
    output_dir: str = "output"
    logs_dir: str = "logs"
    enrollment_dir: str = "enrollment"

    # Промпты для LLM
    system_prompt_file: str = "core/llm/prompts/conversation_analysis.txt"
    stitching_prompt_file: str = "core/llm/prompts/stitching.txt"
    user_prompt_file: str = "core/llm/prompts/user_template.txt"

    # Идентификация владельца
    self_user_name: str = "Оператор"
    other_user_name: str = "Собеседник"

    vad: VADSettings = field(default_factory=VADSettings)
    speaker_id: SpeakerIdSettings = field(default_factory=SpeakerIdSettings)
    enrollment: EnrollmentSettings = field(default_factory=EnrollmentSettings)
    asr: ASRSettings = field(default_factory=ASRSettings)
    llm: LLMSettings = field(default_factory=LLMSettings)
    analysis: AnalysisSettings = field(default_factory=AnalysisSettings)
    pipeline: PipelineSettings = field(default_factory=PipelineSettings)

    # Удалённые поля — заглушки для обратной совместимости
    hf_token_env: str = ""               # больше не используется
    whisper: dict = field(default_factory=dict)   # для безопасной сериализации

    # ------------------------------------------------------------------
    # Сериализация
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "AppConfig":
        return cls(
            models_dir=data.get("models_dir", "models"),
            output_dir=data.get("output_dir", "output"),
            logs_dir=data.get("logs_dir", "logs"),
            enrollment_dir=data.get("enrollment_dir", "enrollment"),
            system_prompt_file=data.get(
                "system_prompt_file",
                "core/llm/prompts/conversation_analysis.txt",
            ),
            stitching_prompt_file=data.get(
                "stitching_prompt_file",
                "core/llm/prompts/stitching.txt",
            ),
            user_prompt_file=data.get(
                "user_prompt_file",
                "core/llm/prompts/user_template.txt",
            ),
            self_user_name=data.get("self_user_name", "Оператор"),
            other_user_name=data.get("other_user_name", "Собеседник"),
            vad=VADSettings(**(data.get("vad") or {})),
            speaker_id=SpeakerIdSettings(**(data.get("speaker_id") or {})),
            enrollment=EnrollmentSettings(**(data.get("enrollment") or {})),
            asr=ASRSettings(**(data.get("asr") or {})),
            llm=LLMSettings(**(data.get("llm") or {})),
            analysis=AnalysisSettings(**(data.get("analysis") or {})),
            pipeline=PipelineSettings(**(data.get("pipeline") or {})),
        )

    def save(self, path: Path | str = DEFAULT_CONFIG_NAME) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Config saved to %s", path)

    @classmethod
    def load(cls, path: Path | str = DEFAULT_CONFIG_NAME) -> "AppConfig":
        p = Path(path)
        if not p.exists():
            logger.info("Config %s not found, using defaults", path)
            return cls()
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(
                "Failed to read config %s: %s. Using defaults.", path, e
            )
            return cls()
        return cls.from_dict(data)

    # ------------------------------------------------------------------
    # Удобные свойства
    # ------------------------------------------------------------------
    @property
    def vad_model_path(self) -> Path:
        return Path(self.models_dir) / "vad"

    @property
    def speaker_id_model_path(self) -> Path:
        return Path(self.models_dir) / "speaker_id"

    @property
    def asr_model_path(self) -> Path:
        return Path(self.models_dir) / "asr"

    @property
    def llm_model_path(self) -> Path:
        return Path(self.models_dir) / "llm" / Path(self.llm.model_path).name \
            if not Path(self.llm.model_path).is_absolute() \
            else Path(self.llm.model_path)
