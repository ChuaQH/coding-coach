from __future__ import annotations

from typing import Any, Dict, List, Tuple
from helpers.linting_helper import run_mypy_stdout, run_ruff_outputs
from helpers.pep_helper import build_pep_queries_from_findings, query_pep_kb
from kb.pep_kb import get_in_memory_vector_store

# Compute style bundle: Ruff JSON + formatter diff, MyPy stdout, PEP context
def compute_style_bundle(
    code: str,
    *,
    python_version: str,
    strict: bool,
    ignore_missing_imports: bool,
) -> Tuple[Any, str, str, List[Dict[str, Any]]]:
    """
    Single source of truth:
      - Ruff JSON + formatter diff
      - mypy stdout
      - pep_context from in-memory PEP KB
    Returns: (ruff_json, ruff_format_diff, mypy_stdout, pep_context)
    """
    ruff_json, ruff_format_diff = run_ruff_outputs(code)
    mypy_stdout = run_mypy_stdout(
        code,
        python_version=python_version,
        strict=strict,
        ignore_missing_imports=ignore_missing_imports,
    )

    vs = get_in_memory_vector_store()
    queries = build_pep_queries_from_findings(code, ruff_json, mypy_stdout)
    pep_context = query_pep_kb(vs, queries, k=2, max_queries=16)
    return ruff_json, ruff_format_diff, mypy_stdout, pep_context
