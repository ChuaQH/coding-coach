from __future__ import annotations

from typing import Any, Dict

from state_access import session_state, style_state
from utils.logging import log_stage
from utils.style_bundle import compute_style_bundle


def pep_ruff_mypy_node(state: Dict[str, Any]) -> Dict[str, Any]:
    sess = session_state(state)  # type: ignore[arg-type]
    sty = style_state(state)     # type: ignore[arg-type]

    latest_attempt = (sess.attempts or [])[-1] if sess.attempts else {}
    if not latest_attempt:
        log_stage("pep_ruff_mypy", "no attempts found; skipping")
        return {}

    code = (latest_attempt.get("code") or "").strip()
    if not code:
        log_stage("pep_ruff_mypy", "no code in latest attempt; skipping")
        return {}

    ruff_json, ruff_format_diff, mypy_stdout, pep_context = compute_style_bundle(
        code,
        python_version=sess.python_version,
        strict=sess.mypy_strict,
        ignore_missing_imports=sess.mypy_ignore_missing_imports,
    )

    log_stage(
        "pep_ruff_mypy",
        "ruff results",
        {
            "issue_count": len(ruff_json) if isinstance(ruff_json, list) else 0,
            "has_format_diff": bool(ruff_format_diff.strip()),
            "format_diff_preview": ruff_format_diff[:400],
        },
    )
    log_stage(
        "pep_ruff_mypy",
        "mypy results",
        {"has_output": bool(mypy_stdout.strip()), "output_preview": mypy_stdout[:400]},
    )
    log_stage("pep_ruff_mypy", "pep context", {"count": len(pep_context), "preview": pep_context[:2]})

    sty.ruff_json = ruff_json
    sty.ruff_format_diff = ruff_format_diff
    sty.mypy_stdout = mypy_stdout
    sty.pep_context = pep_context

    return {"style": sty.model_dump()}
