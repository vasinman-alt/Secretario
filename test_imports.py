@'
"""Проверка импортов модулей Secretario."""
import sys
import traceback

modules = [
    "core",
    "core.config",
    "core.types",
    "core.prompts",
    "core.filename_parser",
    "core.progress",
    "core.logging_setup",
    "core.manager",
    "core.services.conversation_service",
    "core.vad.silero",
    "core.speaker_id.ecapa",
    "core.speaker_id.enrollment",
    "core.asr.gigaam",
    "core.llm.qwen_llm",
    "core.analysis.structured_data",
    "core.analysis.summary_extractor",
    "core.ui_adapters.pipeline_worker",
    "ui.main_window",
    "ui.enrollment_dialog",
]

ok = 0
fail = 0
for name in modules:
    try:
        __import__(name)
        print(f"[OK]   {name}")
        ok += 1
    except Exception as e:
        print(f"[FAIL] {name}: {e}")
        fail += 1

print(f"\n{ok} OK, {fail} FAIL")
sys.exit(0 if fail == 0 else 1)
'@ | Out-File -FilePath "test_imports.py" -Encoding UTF8
