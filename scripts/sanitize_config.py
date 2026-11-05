#!/usr/bin/env python3
"""
Очищает локальный config.json от известных проблемных ключей.
"""

import json
from pathlib import Path

CONFIG_PATH = Path("config.json")


def prompt_path_exists(value: str) -> bool:
    if not value:
        return False

    p = Path(value).expanduser()
    if p.exists():
        return True

    candidates = [
        Path("core") / "llm" / "prompts" / p.name,
        Path("core") / "llm" / "prompts" / Path(value).name,
    ]

    return any(c.exists() for c in candidates)


def sanitize(data):
    if not isinstance(data, dict):
        return data

    def walk(node):
        if isinstance(node, dict):
            for key, value in list(node.items()):
                if (
                    isinstance(value, str)
                    and value.lower().endswith((".txt", ".md"))
                    and "prompt" in key.lower()
                ):
                    if not prompt_path_exists(value):
                        node.pop(key, None)
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(data)

    llm = data.get("llm")
    if isinstance(llm, dict):
        llm.pop("quantization_recommendation", None)

    return data


def main():
    if not CONFIG_PATH.exists():
        print("config.json не найден")
        return

    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    cleaned = sanitize(raw)

    CONFIG_PATH.write_text(
        json.dumps(cleaned, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("config.json очищен от известных битых ключей")


if __name__ == "__main__":
    main()