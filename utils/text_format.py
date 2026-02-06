from __future__ import annotations

import json
from typing import Any, Dict, List

# Format CLI feedback into structured sections
def format_cli_feedback(text: str) -> str:
    if not text:
        return ""
    raw = text.strip()
    lines = [ln.rstrip() for ln in raw.splitlines()]
    sections = {
        "TRIAGE": [],
        "1) CORRECTNESS": [],
        "2) EFFICIENCY / ALGORITHMS": [],
        "3) STYLE / BEST PRACTICES": [],
        "4) DELIVERABLE": [],
        "NEXT STEP": [],
        "QUESTIONS": [],
        "ANSWER": [],
        "ONE IMPROVEMENT": [],
        "PSEUDOCODE": [],
        "PATCH / SKELETON": [],
        "OPTIONAL SOLUTION": [],
        "CHECKLIST": [],
    }

    def normalize_heading(heading: str) -> str | None:
        h = heading.strip()
        if h.startswith("NEXT STEP"):
            return "NEXT STEP"
        if h.startswith("QUESTIONS"):
            return "QUESTIONS"
        if h.startswith("4) DELIVERABLE"):
            return "4) DELIVERABLE"
        if h.startswith("PSEUDOCODE"):
            return "PSEUDOCODE"
        if h.startswith("PATCH / SKELETON"):
            return "PATCH / SKELETON"
        if h.startswith("OPTIONAL SOLUTION"):
            return "OPTIONAL SOLUTION"
        return h if h in sections else None

    current = None
    for ln in lines:
        stripped = ln.strip()
        if stripped == "---":
            continue
        if stripped.startswith("## "):
            heading = stripped.replace("## ", "", 1).strip()
            current = normalize_heading(heading)
            continue
        if current:
            sections[current].append(ln)

    def render_block(title: str, items: List[str]) -> List[str]:
        if not items:
            return []
        out = [f"\n{title}"]
        for item in items:
            item = item.rstrip()
            if not item:
                continue
            out.append(f"  {item}")
        return out

    ordered = [
        ("TRIAGE", "TRIAGE"),
        ("1) CORRECTNESS", "1) CORRECTNESS"),
        ("2) EFFICIENCY / ALGORITHMS", "2) EFFICIENCY / ALGORITHMS"),
        ("3) STYLE / BEST PRACTICES", "3) STYLE / BEST PRACTICES"),
        ("4) DELIVERABLE", "4) DELIVERABLE"),
        ("NEXT STEP", "NEXT STEP"),
        ("QUESTIONS", "QUESTIONS"),
        ("ANSWER", "ANSWER"),
        ("ONE IMPROVEMENT", "ONE IMPROVEMENT"),
        ("PSEUDOCODE", "PSEUDOCODE"),
        ("PATCH / SKELETON", "PATCH / SKELETON"),
        ("OPTIONAL SOLUTION", "OPTIONAL SOLUTION"),
        ("CHECKLIST", "CHECKLIST"),
    ]

    rendered: List[str] = []
    has_sections = any(sections[k] for k in sections)
    if has_sections:
        for key, title in ordered:
            rendered.extend(render_block(title, sections.get(key, [])))
        return "\n".join(rendered).strip()

    # Fallback for unstructured feedback
    import textwrap
    paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]
    out: List[str] = ["FEEDBACK"]
    for p in paragraphs:
        for w in textwrap.wrap(p, width=100):
            out.append(f"  - {w}")
    return "\n".join(out).strip()

# Compact knowledge base cards into JSON string
def compact_kb_cards(retrieval_results: List[Dict[str, Any]], max_chars: int = 600) -> str:
    cards = []
    for r in (retrieval_results or [])[:4]:
        md = r.get("metadata", {}) or {}
        title = md.get("title") or md.get("source_url") or "KB chunk"
        tags = md.get("tags_csv") or md.get("tags") or ""
        content = (r.get("content") or "").strip()
        snippet = content[:max_chars]
        cards.append({"title": title, "tags": tags, "snippet": snippet})
    return json.dumps(cards, ensure_ascii=False)

# Truncate text to a maximum number of characters
def truncate_text(s: str, max_chars: int = 1400) -> str:
    if not s:
        return ""
    s = s.strip()
    return s if len(s) <= max_chars else (s[:max_chars] + "\n... (truncated) ...")