"""Secretario — entry point."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# Load .env BEFORE any other project imports
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from PySide6.QtWidgets import QApplication  # noqa: E402

from core import AppConfig  # noqa: E402
from core.logging_setup import setup_logging  # noqa: E402
from ui import MainWindow  # noqa: E402


def main() -> None:
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("=== secretario starting ===")

    config = AppConfig.load()

    app = QApplication(sys.argv)
    app.setApplicationName("Secretario")
    app.setOrganizationName("Secretario")

    window = MainWindow(config)
    window.show()

    exit_code = app.exec()
    logger.info("=== exit code %s ===", exit_code)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
