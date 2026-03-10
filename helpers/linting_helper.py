from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from typing import Any, Dict, List, Tuple


def run_ruff_outputs(code: str) -> Tuple[List[Dict[str, Any]], str]:
    """
    Runs:
      - ruff check --output-format=json
      - ruff format --diff

    Returns:
      (ruff_json, ruff_format_diff)
    """
    ruff_json: List[Dict[str, Any]] = []
    ruff_format_diff: str = ""

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "solution.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)

        def run(cmd: List[str], *, timeout: int = 20) -> Tuple[int, str, str]:
            try:
                p = subprocess.run(cmd, cwd=td, capture_output=True, text=True, timeout=timeout, check=False)
                return p.returncode, (p.stdout or ""), (p.stderr or "")
            except Exception:
                return 999, "", ""

        _, out, _ = run(["ruff", "check", "--output-format=json", "--no-cache", "solution.py"])
        if out.strip():
            try:
                parsed = json.loads(out)
                if isinstance(parsed, list):
                    ruff_json = parsed
            except Exception:
                ruff_json = []

        _, out, _ = run(["ruff", "format", "--diff", "solution.py"])
        ruff_format_diff = (out or "").strip()

    return ruff_json, ruff_format_diff


def run_mypy_stdout(
    code: str,
    *,
    python_version: str = "3.11",
    strict: bool = False,
    ignore_missing_imports: bool = False,
) -> str:
    """
    Runs mypy and returns stdout+stderr as a single string.
    """
    mypy_out = ""

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "solution.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)

        def run(cmd: List[str], *, timeout: int = 40) -> Tuple[int, str, str]:
            try:
                p = subprocess.run(cmd, cwd=td, capture_output=True, text=True, timeout=timeout, check=False)
                return p.returncode, (p.stdout or ""), (p.stderr or "")
            except Exception:
                return 999, "", ""

        cmd = [
            "mypy",
            "--python-version", python_version,
            "--show-error-codes",
            "--no-error-summary",
            "--pretty",
            "--no-incremental",
        ]
        if strict:
            cmd.append("--strict")
        if ignore_missing_imports:
            cmd.append("--ignore-missing-imports")

        cmd.append("solution.py")

        _, out, err = run(cmd)
        combined = "\n".join([s for s in [(out or "").strip(), (err or "").strip()] if s]).strip()
        mypy_out = combined

    return mypy_out