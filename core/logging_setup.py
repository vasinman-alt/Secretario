"""Логирование Secretario: пишет в файл и в консоль."""
from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

APP_LOGGER_NAME = "secretario"
LOG_FILE_NAME = "secretario.log"


def setup_logging(
    logs_dir: str = "logs",
    level: int = logging.INFO,
) -> None:
    """
    Настраивает корневой логгер.
    Безопасно вызывать несколько раз — старые хэндлеры снимаются.
    """
    Path(logs_dir).mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)

    # Снимаем старые хэндлеры
    for h in list(root.handlers):
        root.removeHandler(h)

    formatter = logging.Formatter(_LOG_FORMAT, _DATE_FORMAT)

    # Файл с ротацией
    file_handler = logging.handlers.RotatingFileHandler(
        filename=Path(logs_dir) / LOG_FILE_NAME,
        maxBytes=2 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    root.addHandler(file_handler)

    # Консоль (только WARNING+ чтобы не спамить)
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    console.setFormatter(formatter)
    root.addHandler(console)

    # Приглушаем болтливые сторонние логгеры
    QUIET_LOGGERS = [
        APP_LOGGER_NAME,                      # основной логгер Secretario
        "faster_whisper",                     # на случай остатков
        "pyannote",                           # на случай остатков
        "speechbrain",                        # ECAPA
        "llama_cpp",                          # Qwen3
        "ctranslate2",
        "nemo_logger",                        # GigaAM / NeMo
        "torch",                              # torch-сообщения уровня INFO
        "torchaudio",
        "transformers",
        "huggingface_hub",
        "urllib3",
    ]
    for name in QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Вспомогательная функция: создаёт логгер с префиксом 'secretario'."""
    if not name.startswith(APP_LOGGER_NAME):
        name = f"{APP_LOGGER_NAME}.{name}"
    return logging.getLogger(name)
