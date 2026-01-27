from __future__ import annotations

import json
from typing import Any


_DEBUG_ENABLED: bool = False


def configure_logging(*, debug: bool = False) -> None:
    """Enable/disable debug logging globally for this process."""
    global _DEBUG_ENABLED
    _DEBUG_ENABLED = bool(debug)


def is_debug() -> bool:
    return _DEBUG_ENABLED


def log_stage(stage: str, detail: str | None = None, data: Any | None = None) -> None:
    """Debug-only structured logging."""
    if not _DEBUG_ENABLED:
        return

    print(f"\n=== {stage} ===")
    if detail:
        print(detail)
    if data is None:
        return

    if isinstance(data, str):
        print(data)
        return

    try:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    except Exception:
        print(data)
