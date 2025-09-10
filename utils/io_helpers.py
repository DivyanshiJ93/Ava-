"""
utils/io_helpers.py

Small helpers for saving outputs and safe JSON parsing.
"""

import json
from datetime import datetime
from typing import Any
import os

def timestamped_filename(prefix: str, ext: str) -> str:
    """Return a safe timestamped filename like prefix_YYYYMMDD_HHMMSS.ext"""
    t = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = f"{prefix}_{t}.{ext}"
    return safe

def save_text(text: str, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def save_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def try_parse_json(text: str):
    try:
        return json.loads(text)
    except Exception:
        return None

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
