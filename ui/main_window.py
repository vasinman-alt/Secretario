"""Главное окно Secretario.

Делегирует работу PipelineWorker, сам только UI.
Этапы конвейера (2.1): Enrollment → VAD → ECAPA → GigaAM → Qwen3.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core import AppConfig, ModelManager
from core.progress import CallbackProgress
from core.services.conversation_service import ConversationService
from core.ui_adapters import PipelineWorker
from ui.enrollment_dialog import EnrollmentDialog

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self, config: Optional[AppConfig] = None):
        super().__init__()
        self.setWindowTitle("Secretario")
        self.setMinimumSize(900, 950)

        # ---- Состояние ----
        self._config = config or AppConfig.load()
        self._manager = ModelManager(self._config)
        self._service = ConversationService(self._manager, self._config)
        self._worker: Optional[PipelineWorker] = None
        self._a_worker: Optional[QObject] = None
        self._a_thread: Optional[QThread] = None
        self._last_transcript_path: Optional[Path] = None

        self._build_ui()
        self._refresh_llm_models()
        self._refresh_enrollment_status()

    # ===================== UI =====================
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # --- Файл ---
        file_row = QHBoxLayout()
        file_row.addWidget(QLabel("Файл:"))
        self.file_edit = QLineEdit()
        self.file_edit.setPlaceholderText(
            "Выберите аудиофайл (.m4a, .mp3, .wav)"
        )
        file_row.addWidget(self.file_edit)
        browse = QPushButton("📁")
        browse.clicked.connect(self._browse_file)
        file_row.addWidget(browse)
        layout.addLayout(file_row)

        # --- Enrollment (голосовой профиль владельца) ---
        enr_row = QHBoxLayout()
        enr_row.addWidget(QLabel("Голосовой профиль:"))
        self.enrollment_status = QLabel("не записан")
        self.enrollment_status.setStyleSheet("color: #a33;")
        enr_row.addWidget(self.enrollment_status)
        self.enrollment_btn = QPushButton("🎤 Записать эталон")
        self.enrollment_btn.clicked.connect(self._start_enrollment)
        enr_row.addWidget(self.enrollment_btn)
        enr_row.addStretch()
        layout.addLayout(enr_row)

        # --- ASR (GigaAM) ---
        asr_row = QHBoxLayout()
        asr_row.addWidget(QLabel("ASR:"))
        self.asr_combo = QComboBox()
        self.asr_combo.addItems([
            "v3_e2e_ctc (быстрый, рекомендован)",
            "v3_e2e_rnnt (точнее, медленнее на CPU)",
        ])
        asr_row.addWidget(self.asr_combo)
        asr_row.addStretch()
        layout.addLayout(asr_row)

        # --- Действие ---
        action_row = QHBoxLayout()
        self.transcribe_btn = QPushButton("▶ Расшифровать")
        self.transcribe_btn.clicked.connect(self._start_pipeline)
        action_row.addWidget(self.transcribe_btn)

        self.cancel_btn = QPushButton("⏹ Отмена")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel)
        action_row.addWidget(self.cancel_btn)

        self.analyze_only_btn = QPushButton("Только анализ TXT")
        self.analyze_only_btn.clicked.connect(self._start_analysis_only)
        action_row.addWidget(self.analyze_only_btn)
        layout.addLayout(action_row)

        # --- Прогресс ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.stage_label = QLabel("Статус: готов")
        layout.addWidget(self.stage_label)

        # --- Лог ---
        layout.addWidget(QLabel("Лог:"))
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(180)
        layout.addWidget(self.log)

        # --- LLM ---
        llm_row = QHBoxLayout()
        llm_row.addWidget(QLabel("LLM модель:"))
        self.llm_combo = QComboBox()
        llm_row.addWidget(self.llm_combo)
        self.reload_prompt_btn = QPushButton("🔄 Промпты")
        self.reload_prompt_btn.clicked.connect(self._reload_prompt)
        llm_row.addWidget(self.reload_prompt_btn)

        self.save_config_btn = QPushButton("💾 Сохранить настройки")
        self.save_config_btn.clicked.connect(self._save_config)
        llm_row.addWidget(self.save_config_btn)
        llm_row.addStretch()
        layout.addLayout(llm_row)

        # --- Результат анализа ---
        layout.addWidget(QLabel("Результат анализа:"))
        self.analysis = QTextEdit()
        self.analysis.setReadOnly(True)
        layout.addWidget(self.analysis)

    # ===================== Helpers =====================
    def _browse_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите аудиофайл", "",
            "Аудиофайлы (*.m4a *.mp3 *.wav);;Все файлы (*)",
        )
        if path:
            self.file_edit.setText(path)

    def _refresh_enrollment_status(self) -> None:
        """Проверяет, есть ли записанный эталон голоса."""
        try:
            profile = self._manager.enrollment().get_default_profile()
            if profile is not None:
                self.enrollment_status.setText(
                    f"✓ {profile.name} (от {profile.created_at[:10]})"
                )
                self.enrollment_status.setStyleSheet("color: #383;")
                self.transcribe_btn.setEnabled(True)
            else:
                self.enrollment_status.setText("не записан — запись эталона")
                self.enrollment_status.setStyleSheet("color: #a33;")
                self.transcribe_btn.setEnabled(False)
        except Exception as e:
            self.enrollment_status.setText(f"ошибка: {e}")
            self.enrollment_status.setStyleSheet("color: #a33;")
            self.transcribe_btn.setEnabled(False)

    def _start_enrollment(self) -> None:
        dialog = EnrollmentDialog(self._manager, self)
        if dialog.exec():
            self._refresh_enrollment_status()
            self._log("Голосовой профиль записан")

    def _refresh_llm_models(self) -> None:
        self.llm_combo.blockSignals(True)
        try:
            self.llm_combo.clear()
            models = self._manager.llm().list_available_models()
            if not models:
                self.llm_combo.addItem("Нет GGUF моделей", None)
                return
            for m in models:
                self.llm_combo.addItem(m.name, str(m))
            if self._config.llm.model_path:
                idx = self.llm_combo.findData(self._config.llm.model_path)
                if idx >= 0:
                    self.llm_combo.setCurrentIndex(idx)
        finally:
            self.llm_combo.blockSignals(False)
        self.llm_combo.currentIndexChanged.connect(self._on_llm_change)

    def _on_llm_change(self, _idx: int) -> None:
        self._config.llm.model_path = self.llm_combo.currentData()
        self._manager.llm().unload()
        name = self.llm_combo.currentText()
        self._log(f"LLM модель: {name}")

    def _reload_prompt(self) -> None:
        self._manager.prompts().reload()
        self._log("Промпты перезагружены (анализ, склейка, шаблон)")

    def _save_config(self) -> None:
        try:
            self._config.save()
            self._log("Настройки сохранены в config.json")
        except Exception as e:
            QMessageBox.warning(self, "Ошибка", f"Не удалось сохранить: {e}")

    def _log(self, msg: str) -> None:
        self.log.append(msg)
        logger.info(msg)

    def _set_running(self, running: bool) -> None:
        self.transcribe_btn.setEnabled(not running and self._is_enrolled())
        self.analyze_only_btn.setEnabled(not running)
        self.cancel_btn.setEnabled(running)
        self.save_config_btn.setEnabled(not running)
        self.enrollment_btn.setEnabled(not running)

    def _is_enrolled(self) -> bool:
        try:
            return self._manager.enrollment().get_default_profile() is not None
        except Exception:
            return False

    def _cancel(self) -> None:
        self._log("Отмена пока не реализована")

    # ===================== Действия =====================
    def _start_pipeline(self) -> None:
        path = self.file_edit.text().strip()
        if not path or not Path(path).exists():
            QMessageBox.warning(self, "Ошибка", "Выберите существующий аудиофайл")
            return

        # Конвертируем выбор ASR в config
        asr_idx = self.asr_combo.currentIndex()
        self._config.asr.model_name = "v3_e2e_ctc" if asr_idx == 0 else "v3_e2e_rnnt"

        self._set_running(True)
        self.progress_bar.setValue(0)
        self.stage_label.setText("Запуск...")
        self.log.clear()
        self.analysis.clear()

        self._worker = PipelineWorker(
            service=self._service,
            audio_path=path,
            run_analysis=True,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _start_analysis_only(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл расшифровки", "",
            "Текстовые файлы (*.txt);;JSON (*.json)",
        )
        if not path:
            return
        try:
            text = Path(path).read_text(encoding="utf-8").strip()
        except Exception as e:
            QMessageBox.warning(self, "Ошибка", f"Не удалось прочитать: {e}")
            return
        if not text:
            QMessageBox.warning(self, "Ошибка", "Файл пуст")
            return

        self._set_running(True)
        self.progress_bar.setValue(0)

        if self._a_thread is not None:
            try:
                self._a_thread.quit()
                self._a_thread.wait(2000)
            except Exception:
                pass

        class AnalyzeWorker(QObject):
            progress = Signal(str, float, str)
            finished = Signal(str)
            error = Signal(str)

            def __init__(self, manager, text, transcript_path):
                super().__init__()
                self._manager = manager
                self._text = text
                self._transcript_path = transcript_path

            def run(self):
                try:
                    progress = CallbackProgress(
                        lambda s: self.progress.emit(s.stage, s.value, s.message)
                    )
                    progress.stage("llm", 0.0, "Старт")
                    with self._manager.llm_session() as llm:
                        result = llm.analyze(
                            self._text, progress,
                            transcript_path=self._transcript_path,
                        )
                    self.finished.emit(result.raw)
                except Exception as e:
                    self.error.emit(str(e))

        self._a_worker = AnalyzeWorker(self._manager, text, Path(path))
        self._a_thread = QThread()
        self._a_worker.moveToThread(self._a_thread)
        self._a_thread.started.connect(self._a_worker.run)
        self._a_worker.progress.connect(self._on_progress)
        self._a_worker.finished.connect(self._on_analysis_done)
        self._a_worker.error.connect(self._on_error)
        self._a_worker.finished.connect(self._a_thread.quit)
        self._a_worker.error.connect(self._a_thread.quit)
        self._a_thread.start()

    # ===================== Сигналы =====================
    def _on_progress(self, stage: str, value: float, message: str) -> None:
        if value < 0:
            if message:
                self._log(message)
            return

        # Этапы конвейера 2.1
        stage_to_range = {
            "vad":         (0,  15),
            "speaker_id":  (15, 35),
            "asr":         (35, 75),
            "stitch":      (75, 85),
            "llm":         (85, 100),
        }
        a, b = stage_to_range.get(stage, (0, 100))
        v = a + (b - a) * max(0.0, min(1.0, value))
        self.progress_bar.setValue(int(v))

        if message:
            self.stage_label.setText(f"{stage}: {message}")
            self._log(message)

    def _on_finished(self, result) -> None:
        self._set_running(False)
        self.progress_bar.setValue(100)
        self.stage_label.setText("Готово")
        if result.transcription.transcript_path:
            self._last_transcript_path = result.transcription.transcript_path
            self._log(f"Транскрипт: {result.transcription.transcript_path}")
        if result.analysis and result.analysis.raw:
            self.analysis.setPlainText(result.analysis.raw)
            if result.analysis.analysis_path:
                self._log(f"Анализ: {result.analysis.analysis_path}")

    def _on_analysis_done(self, text: str) -> None:
        self._set_running(False)
        self.progress_bar.setValue(100)
        self.analysis.setPlainText(text)
        self._log("Анализ завершён")

    def _on_error(self, message: str) -> None:
        self._set_running(False)
        self.stage_label.setText("Ошибка")
        self._log(f"ОШИБКА: {message}")
        QMessageBox.critical(self, "Ошибка пайплайна", message)
