from __future__ import annotations

import json
from typing import Any, Dict, List


# ---------- helpers for PEP retrieval ----------

def _extract_context(code: str, line_1based: int, radius: int = 2) -> str:
    lines = code.splitlines()
    i = max(0, line_1based - 1)
    start = max(0, i - radius)
    end = min(len(lines), i + radius + 1)
    out: List[str] = []
    for idx in range(start, end):
        prefix = ">> " if idx == i else "   "
        out.append(f"{prefix}{idx+1:4d}: {lines[idx]}")
    return "\n".join(out)


def build_pep_queries_from_findings(code: str, ruff_json: Any, mypy_stdout: str) -> List[str]:
    """
    Build high-signal queries from Ruff + mypy findings, with small code-context snippets.
    This is the single canonical query builder (avoid duplicating this logic elsewhere).
    """
    queries: List[str] = []

    # Ruff -> specific queries
    if isinstance(ruff_json, list):
        for item in ruff_json[:40]:
            rule = item.get("code") or item.get("rule")
            msg = (item.get("message") or "").strip()
            loc = item.get("location") or {}
            row = loc.get("row")
            snippet = _extract_context(code, int(row), radius=2) if isinstance(row, int) else ""

            base = f"Ruff {rule}: {msg}".strip()
            if snippet:
                base = f"{base}\n{snippet}"
            queries.append(base)

            # “PEP routing” nudges
            if rule:
                if str(rule).startswith(("E", "W")):
                    queries.append(f"PEP 8 guidance relevant to {rule}: {msg}".strip())
                if str(rule).startswith("D"):
                    queries.append(f"PEP 257 docstring guidance relevant to {rule}: {msg}".strip())

    # mypy -> queries
    if mypy_stdout:
        for line in mypy_stdout.splitlines()[:40]:
            s = line.strip()
            # keep it simple: use mypy’s text
            if "error:" in s or "note:" in s:
                queries.append(f"mypy typing issue: {s}")

    # de-dupe preserving order
    seen: set[str] = set()
    uniq: List[str] = []
    for q in queries:
        qq = q.strip()
        if not qq or qq in seen:
            continue
        seen.add(qq)
        uniq.append(qq)
    return uniq


def query_pep_kb(vs: Any, queries: List[str], *, k: int = 2, max_queries: int = 16) -> List[Dict[str, Any]]:
    """
    Queries the PEP vector store and returns compact 'context cards' for prompting.

    Returns:
      {
        "query": "...",
        "source_url": "https://peps.python.org/pep-0008/",
        "type": "base_best_practices",
        "excerpt": "..."
      }
    """
    out: List[Dict[str, Any]] = []
    seen = set()

    for q in (queries or [])[:max_queries]:
        try:
            docs = vs.similarity_search(q, k=k)
        except Exception:
            docs = []

        for d in docs:
            meta = getattr(d, "metadata", {}) or {}
            src = meta.get("source_url", "unknown")
            excerpt = (getattr(d, "page_content", "") or "").strip()
            if not excerpt:
                continue

            key = (src, hash(excerpt[:400]))
            if key in seen:
                continue
            seen.add(key)

            out.append(
                {
                    "query": q,
                    "source_url": src,
                    "type": meta.get("type"),
                    "excerpt": excerpt[:900],
                }
            )

    return out


def compact_pep_context(pep_context: List[Dict[str, Any]], max_items: int = 6, max_chars: int = 450) -> str:
    cards = []
    for c in (pep_context or [])[:max_items]:
        cards.append(
            {
                "source_url": c.get("source_url"),
                "query": c.get("query"),
                "excerpt": (c.get("excerpt") or "")[:max_chars],
            }
        )
    return json.dumps(cards, ensure_ascii=False)


def summarize_ruff(ruff_json: Any, ruff_format_diff: str, max_items: int = 8) -> Dict[str, Any]:
    findings = []
    if isinstance(ruff_json, list):
        for item in ruff_json[:max_items]:
            loc = item.get("location") or {}
            findings.append(
                {
                    "code": item.get("code") or item.get("rule"),
                    "message": item.get("message"),
                    "row": loc.get("row"),
                    "col": loc.get("column"),
                }
            )
    return {
        "ruff_issue_count": len(ruff_json) if isinstance(ruff_json, list) else None,
        "ruff_has_format_diff": bool((ruff_format_diff or "").strip()),
        "ruff_top_findings": findings,
    }


def render_ruff_report_md(ruff_json: Any, ruff_format_diff: str) -> str:
    parts: List[str] = []
    if isinstance(ruff_json, list) and ruff_json:
        parts.append(
            "### ruff check (json)\n```json\n"
            + json.dumps(ruff_json, ensure_ascii=False, indent=2)[:12000]
            + "\n```"
        )
    else:
        parts.append("### ruff check\n(no findings or no JSON output)")

    if (ruff_format_diff or "").strip():
        parts.append("### ruff format --diff\n```diff\n" + ruff_format_diff[:12000] + "\n```")
    else:
        parts.append("### ruff format --diff\n(no diff; already formatted)")

    return "\n\n".join(parts).strip()
