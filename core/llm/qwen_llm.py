"""Qwen3-4B через llama-cpp-python.

Два режима:
    1. stitch_segments() — склейка VAD-сегментов, исправление ASR,
       разрешение неопределённых меток спикеров.
    2. analyze() — финальный анализ (резюме, договорённости, факты).
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from core.config import AppConfig
from core.filename_parser import CallMetadata
from core.progress import ProgressReporter
from core.prompts import PromptService
from core.types import (
    AnalysisResult,
    AsrSegment,
)

logger = logging.getLogger(__name__)


@dataclass
class LLMSegment:
    """Структура реплики после склейки."""
    speaker: str
    text: str
    timestamp: str    # "MM:SS"


class QwenLLMService:
    """Обёртка над llama-cpp-python. Модель грузится лениво."""

    def __init__(self, config: AppConfig, prompts: PromptService):
        self._config = config
        self._prompts = prompts
        self._llm = None

    def _load(self) -> None:
        if self._llm is not None:
            return
        from llama_cpp import Llama

        model_path = self._config.llm.model_path
        if not Path(model_path).is_absolute():
            model_path = str(
                Path(self._config.models_dir) / "llm" / Path(model_path).name
            )
        logger.info("Loading LLM: %s", model_path)

        self._llm = Llama(
            model_path=model_path,
            n_ctx=self._config.llm.n_ctx,
            n_threads=self._config.llm.n_threads,
            verbose=False,
        )
        logger.info("LLM loaded: %s", model_path)

    # ------------------------------------------------------------------
    # Склейка сегментов (правило v2.1)
    # ------------------------------------------------------------------
    def stitch_segments(
        self,
        asr_segments: List[AsrSegment],
        progress: Optional[ProgressReporter] = None,
    ) -> Tuple[str, List[LLMSegment]]:
        """
        Склеивает VAD-сегменты в читаемую стенограмму.
        Возвращает (полный_текст, список_LLMSegment).
        """
        prog = progress or ProgressReporter()
        self._load()

        if not asr_segments:
            return "", []

        # Формируем входные данные для LLM
        raw_input = self._format_segments_for_stitching(asr_segments)
        sys_prompt = self._prompts.get_stitching()

        logger.info(
            "Stitching %d segments (%d chars input)",
            len(asr_segments), len(raw_input),
        )
        prog.stage("stitch", 0.0, f"Склейка {len(asr_segments)} сегментов...")

        t0 = time.time()
        response = self._llm(
            sys_prompt,
            raw_input,
            max_tokens=self._config.llm.max_tokens,
            temperature=self._config.llm.temperature,
            top_p=self._config.llm.top_p,
            repeat_penalty=self._config.llm.repeat_penalty,
            stop=["</s>", "<|im_end|>"],
        )
        elapsed = time.time() - t0
        logger.info("Stitching done in %.1f sec", elapsed)

        text = response["choices"][0]["text"].strip()
        segments = self._parse_stitched(text)
        return text, segments

    def _format_segments_for_stitching(
        self, asr_segments: List[AsrSegment],
    ) -> str:
        lines = []
        for s in asr_segments:
            mm = int(s.start // 60)
            ss = int(s.start % 60)
            speaker = s.speaker
            if s.is_uncertain:
                speaker = "Неопределённо"
            lines.append(
                f"[{mm:02d}:{ss:02d}] {speaker}: {s.text}"
            )
        return "\n".join(lines)

    def _parse_stitched(self, text: str) -> List[LLMSegment]:
        """Парсит ответ LLM в структурированный список реплик."""
        segments = []
        current_speaker = None
        current_text_lines = []
        current_ts = ""

        for line in text.splitlines():
            line = line.rstrip()
            # Ищем метку реплики: "MM:SS" на отдельной строке + "Speaker:" ниже
            if line and line[0].isdigit() and ":" in line and len(line) <= 6:
                # Сохраняем предыдущую реплику
                if current_speaker and current_text_lines:
                    segments.append(
                        LLMSegment(
                            speaker=current_speaker,
                            text=" ".join(current_text_lines).strip(),
                            timestamp=current_ts,
                        )
                    )
                current_ts = line
                current_text_lines = []
                continue

            if line.endswith(":") and line[:-1] in (
                "Оператор", "Собеседник", "Неопределённо",
            ):
                current_speaker = line[:-1]
                continue

            if line:
                current_text_lines.append(line)

        # Финальная реплика
        if current_speaker and current_text_lines:
            segments.append(
                LLMSegment(
                    speaker=current_speaker,
                    text=" ".join(current_text_lines).strip(),
                    timestamp=current_ts,
                )
            )

        return segments

    # ------------------------------------------------------------------
    # Финальный анализ
    # ------------------------------------------------------------------
    def analyze(
        self,
        transcript_text: str,
        progress: Optional[ProgressReporter] = None,
        transcript_path: Optional[Path] = None,
        metadata: Optional[CallMetadata] = None,
    ) -> AnalysisResult:
        """
        Анализирует стенограмму: резюме, договорённости, факты.
        """
        prog = progress or ProgressReporter()
        self._load()

        sys_prompt = self._prompts.get_conversation_analysis()
        user_msg = self._prompts.render_user(transcript_text)

        # Дополняем системный промпт контекстом, если есть
        if metadata:
            sys_prompt += (
                f"\n\nКонтекст звонка: "
                f"Клиент — {metadata.client_name}, "
                f"телефон {metadata.client_phone_formatted}, "
                f"дата {metadata.call_date_str} {metadata.call_time_str}."
            )

        logger.info("LLM analysis (%d chars)", len(user_msg))
        prog.stage("llm", 0.0, "Анализ разговора...")

        t0 = time.time()
        response = self._llm(
            sys_prompt,
            user_msg,
            max_tokens=self._config.llm.max_tokens,
            temperature=self._config.llm.temperature,
            top_p=self._config.llm.top_p,
            repeat_penalty=self._config.llm.repeat_penalty,
            stop=["</s>", "<|im_end|>"],
        )
        elapsed = time.time() - t0
        logger.info("Analysis done in %.1f sec", elapsed)

        raw = response["choices"][0]["text"].strip()
        prog.stage("llm", 1.0, f"Анализ готов ({len(raw)} символов)")

        # Парсим структурированные поля
        summary, action_items, contacts, dates, amounts, reminders, unclear = (
            self._parse_analysis(raw)
        )

        return AnalysisResult(
            raw=raw,
            transcript_path=transcript_path,
            analysis_path=None,
            model_name=Path(self._config.llm.model_path).name,
            metadata=metadata,
            summary=summary,
            action_items=action_items,
            contacts=contacts,
            dates=dates,
            amounts=amounts,
            reminders=reminders,
            unclear=unclear,
        )

    def _parse_analysis(self, raw: str) -> tuple:
        """Извлекает структурированные поля из markdown-ответа LLM."""
        summary = None
        action_items: List[str] = []
        contacts: List[str] = []
        dates: List[str] = []
        amounts: List[str] = []
        reminders: List[str] = []
        unclear: List[str] = []

        current = None
        for line in raw.splitlines():
            stripped = line.strip()
            lower = stripped.lower()
            if lower.startswith("# краткое резюме"):
                current = "summary"
                continue
            if lower.startswith("# основные темы"):
                current = "topics"
                continue
            if lower.startswith("# договорённости"):
                current = "actions"
                continue
            if lower.startswith("# важные факты"):
                current = "facts"
                continue
            if lower.startswith("# неясные места"):
                current = "unclear"
                continue

            if current == "summary" and stripped and not stripped.startswith("#"):
                summary = (summary or "") + " " + stripped
            elif current == "actions" and stripped.startswith("-"):
                action_items.append(stripped[1:].strip())
            elif current == "facts" and stripped.startswith("-"):
                # Простая эвристика: ищем телефоны/даты/суммы в строке
                import re
                if re.search(r"\+?\d[\d\-\s\(\)]{7,}", stripped):
                    contacts.append(stripped[1:].strip())
                if re.search(r"\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4}", stripped):
                    dates.append(stripped[1:].strip())
                if re.search(
                    r"\d+\s*(руб|₽|тыс|млн|долл|евро|\$|€)",
                    stripped, re.IGNORECASE,
                ):
                    amounts.append(stripped[1:].strip())
            elif current == "unclear" and stripped.startswith("-"):
                unclear.append(stripped[1:].strip())

        if summary:
            summary = summary.strip()
        if not unclear:
            unclear = ["Нет"] if "нет" in raw.lower().split("# неясные места")[-1].lower()[:50] else []

        return summary, action_items, contacts, dates, amounts, reminders, unclear

    # ------------------------------------------------------------------
    # Служебные
    # ------------------------------------------------------------------
    def list_available_models(self) -> List:
        """Сканирует models/llm/ и возвращает список GGUF-моделей."""
        llm_dir = Path(self._config.models_dir) / "llm"
        if not llm_dir.exists():
            return []
        return sorted(llm_dir.glob("*.gguf"))

    def unload(self) -> None:
        if self._llm is not None:
            del self._llm
            self._llm = None
            logger.info("LLM unloaded")
