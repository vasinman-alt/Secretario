"""Диалог записи голосового эталона владельца.

Подробная реализация появится вместе с модулем core/speaker_id/enrollment.py.
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QMessageBox,
)


class EnrollmentDialog(QDialog):
    def __init__(self, manager, parent=None):
        super().__init__(parent)
        self._manager = manager
        self.setWindowTitle("Запись голосового эталона")
        self.setMinimumSize(500, 300)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel(
            "<h3>Запись голосового эталона</h3>"
            "<p>Прочитайте текст ниже вслух в нормальном темпе. "
            "Рекомендуемая длительность: 20–30 секунд.</p>"
        ))

        text = QLabel(
            "«Здравствуйте, меня зовут владелец приложения. "
            "Я звоню вам по поводу нашего оборудования — центробежных "
            "насосов и сепараторов. Сегодня обсудим условия поставки, "
            "сроки и стоимость. Пожалуйста, уточните адрес доставки "
            "и контактное лицо на вашей стороне.»"
        )
        text.setWordWrap(True)
        text.setStyleSheet("padding: 12px; background: #f0f0f0;")
        layout.addWidget(text)

        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("🎤 Начать запись")
        self.start_btn.clicked.connect(self._on_start)
        btn_row.addWidget(self.start_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.status = QLabel("Готов к записи")
        layout.addWidget(self.status)

    def _on_start(self) -> None:
        QMessageBox.information(
            self, "Запись эталона",
            "Запись голосового эталона пока не реализована.\n\n"
            "Полная версия появится вместе с core/speaker_id/enrollment.py.",
        )
        self.reject()
