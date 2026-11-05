"""Извлечение структурированных данных из текста разговора.

На текущем этапе парсинг уже встроен в QwenLLMService._parse_analysis().
Этот модуль зарезервирован для более продвинутых правил и пост-обработки
(например, извлечение телефонов/дат регулярками из свободного текста).
"""
from __future__ import annotations

import re
from typing import List


def extract_phones(text: str) -> List[str]:
    """Извлекает все телефонные номера из произвольного текста."""
    pattern = re.compile(
        r"\+?\d[\d\-\s\(\)]{8,}\d", re.MULTILINE,
    )
    found = set()
    for m in pattern.finditer(text):
        raw = re.sub(r"[^\d+]", "", m.group())
        if 10 <= len(re.sub(r"\D", "", raw)) <= 15:
            found.add(raw)
    return sorted(found)


def extract_dates(text: str) -> List[str]:
    """Извлекает даты в распространённых форматах."""
    patterns = [
        r"\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4}",
        r"\d{1,2}\s+(января|февраля|марта|апреля|мая|июня|июля|"
        r"августа|сентября|октября|ноября|декабря)\s+\d{4}",
    ]
    found = set()
    for p in patterns:
        for m in re.finditer(p, text, re.IGNORECASE):
            found.add(m.group())
    return sorted(found)


def extract_amounts(text: str) -> List[str]:
    """Извлекает суммы (рубли, доллары, евро)."""
    pattern = re.compile(
        r"\d[\d\s,]*\s*(руб|₽|тыс\.?\s*руб|млн\.?\s*руб|"
        r"долл|\$|евро|€|тыс\.|млн\.)",
        re.IGNORECASE,
    )
    return sorted({m.group().strip() for m in pattern.finditer(text)})
