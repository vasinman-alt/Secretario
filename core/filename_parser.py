# core/filename_parser.py
"""Парсинг имени файла, записанного на телефоне Redmi.

Ожидаемый формат:
    Имя Фамилия(<телефон>)_YYYYMMDDhhmmss.<ext>

Примеры:
    Артём Горелов(0079313830282)_20251028175404.mp3
    Ольга Чернова(0079277481232)_20260731145721.m4a
    Эдуард Кряров(0079183126481)_20260730102341.wav

Телефон обычно в международном формате с ведущим 00 (Россия/СНГ).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Паттерн: имя (с пробелами) + (телефон) + _ + 14 цифр + . + расширение
# Имя — не жадное, до первой открывающей скобки.
# Расширения — только аудио.
FILENAME_PATTERN = re.compile(
    r"^(?P<name>.+?)\((?P<phone>\d+)\)_(?P<dt>\d{14})\.(?P<ext>mp3|m4a|wav|ogg|flac)$",
    re.IGNORECASE,
)


@dataclass
class CallMetadata:
    """Метаданные телефонного звонка, извлечённые из имени файла."""
    client_name: str           # "Артём Горелов"
    client_phone_raw: str      # "0079313830282" (как в имени)
    client_phone_formatted: str  # "+7 (931) 383-02-82"
    call_datetime: datetime    # datetime(2025, 10, 28, 17, 54, 4)
    call_date_str: str         # "28-10-2025"
    call_time_str: str         # "17:54"
    source_filename: str       # исходное имя файла
    source_extension: str      # "mp3"

    def to_dict(self) -> dict:
        return {
            "client_name": self.client_name,
            "client_phone_raw": self.client_phone_raw,
            "client_phone_formatted": self.client_phone_formatted,
            "call_datetime": self.call_datetime.isoformat(),
            "call_date": self.call_date_str,
            "call_time": self.call_time_str,
            "source_filename": self.source_filename,
            "source_extension": self.source_extension,
        }


def parse_call_metadata(audio_path: Path | str) -> Optional[CallMetadata]:
    """
    Пытается извлечь метаданные звонка из имени файла.
    Возвращает None, если имя не соответствует ожидаемому формату.
    """
    p = Path(audio_path)
    # Берём stem без учёта возможных промежуточных суффиксов
    # (например, .wav.copy или подобного).
    name = p.name
    m = FILENAME_PATTERN.match(name)
    if not m:
        logger.debug("Filename %s doesn't match call pattern", name)
        return None

    raw_phone = m.group("phone")
    try:
        dt = datetime.strptime(m.group("dt"), "%Y%m%d%H%M%S")
    except ValueError as e:
        logger.warning("Bad datetime in %s: %s", name, e)
        return None

    return CallMetadata(
        client_name=m.group("name").strip(),
        client_phone_raw=raw_phone,
        client_phone_formatted=_format_phone(raw_phone),
        call_datetime=dt,
        call_date_str=dt.strftime("%d-%m-%Y"),
        call_time_str=dt.strftime("%H:%M"),
        source_filename=name,
        source_extension=m.group("ext").lower(),
    )


def _format_phone(raw: str) -> str:
    """
    Приводит телефон к формату +X (XXX) XXX-XX-XX.

    Логика:
    - 00XXXXXXXXXX (13 символов) — международный формат, 00 -> +,
      остальное форматируем как российский номер.
    - 7XXXXXXXXXX (11 символов, начинается с 7) — российский.
    - 8XXXXXXXXXX (11 символов, начинается с 8) — российский, 8 -> 7.
    - Иначе — выводим как есть с префиксом +.
    """
    digits = re.sub(r"\D", "", raw)

    if digits.startswith("00") and len(digits) == 13:
        # 00 7XX XXX XX XX -> +7 (XXX) XXX-XX-XX
        digits = digits[2:]
    elif digits.startswith("8") and len(digits) == 11:
        # 8XXX XXX XX XX -> +7 (XXX) XXX-XX-XX
        digits = "7" + digits[1:]

    if len(digits) == 11 and digits.startswith("7"):
        # Российский: +7 (XXX) XXX-XX-XX
        return (
            f"+7 ({digits[1:4]}) "
            f"{digits[4:7]}-{digits[7:9]}-{digits[9:11]}"
        )

    # Фолбэк: общий международный
    return f"+{digits}"
