from __future__ import annotations

import json
import re
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
def compact_kb_cards(retrieval_results: List[Dict[str, Any]], max_chars: int = 900) -> str:
    cards: List[Dict[str, Any]] = []

    for r in (retrieval_results or [])[:4]:
        md = r.get("metadata", {}) or {}
        content = (r.get("content") or "").strip()

        # Normalize lines
        lines = [ln.strip() for ln in content.splitlines() if ln.strip()]

        # Get title
        title = md.get("title")
        if not title:
            if lines:
                title = lines[0]
            else:
                title = md.get("source_url") or "KB chunk"

        # Get summary
        summary = ""
        if lines:
            for i, ln in enumerate(lines[1:] if len(lines) > 1 else []):
                low = ln.lower()
                if (
                    low.startswith("key ideas:")
                    or low.startswith("pitfalls:")
                    or low.startswith("use when:")
                    or low.startswith("avoid when:")
                    or low.startswith("tags:")
                ):
                    break
                summary = ln
                break

        key_ideas: List[str] = []
        pitfalls: List[str] = []
        use_when = ""
        avoid_when = ""
        tags: List[str] = []

        # Parse lines for labeled sections
        for ln in lines:
            low = ln.lower()
            if low.startswith("key ideas:"):
                s = ln.split(":", 1)[1].strip()
                parts = [p.strip(" -\t") for p in re.split(r"[;]+", s) if p.strip()]
                key_ideas = parts
            elif low.startswith("pitfalls:"):
                s = ln.split(":", 1)[1].strip()
                parts = [p.strip(" -\t") for p in re.split(r"[;]+", s) if p.strip()]
                pitfalls = parts
            elif low.startswith("use when:"):
                use_when = ln.split(":", 1)[1].strip()
            elif low.startswith("avoid when:"):
                avoid_when = ln.split(":", 1)[1].strip()
            elif low.startswith("tags:"):
                s = ln.split(":", 1)[1].strip()
                tags = [t.strip() for t in s.split(",") if t.strip()]

        # Fallback tags from metadata if content didn't include them
        if not tags:
            mt = md.get("tags_csv") or md.get("tags") or ""
            if isinstance(mt, str):
                tags = [t.strip() for t in mt.split(",") if t.strip()]
            elif isinstance(mt, list):
                tags = [str(t).strip() for t in mt if str(t).strip()]

        # Cap for each field
        key_ideas = (key_ideas or [])[:4]
        pitfalls = (pitfalls or [])[:3]
        tags = (tags or [])[:8]

        # Light truncation of long fields
        if summary and len(summary) > 220:
            summary = summary[:219] + "…"
        if use_when and len(use_when) > 220:
            use_when = use_when[:219] + "…"
        if avoid_when and len(avoid_when) > 220:
            avoid_when = avoid_when[:219] + "…"
        if key_ideas:
            key_ideas = [(x[:160] + "…") if len(x) > 161 else x for x in key_ideas]
        if pitfalls:
            pitfalls = [(x[:160] + "…") if len(x) > 161 else x for x in pitfalls]

        card = {
            "title": title,
            "type": md.get("type"),
            "tags": tags,
            "summary": summary,
            "key_ideas": key_ideas,
            "pitfalls": pitfalls,
            "use_when": use_when,
            "avoid_when": avoid_when,
            "source_url": md.get("source_url"),
        }

        # If still too big, create a snippet instead
        blob = json.dumps(card, ensure_ascii=False)
        if len(blob) > max_chars:
            snippet_parts = []
            if summary:
                snippet_parts.append(summary)
            if key_ideas:
                snippet_parts.append("Key ideas: " + "; ".join(key_ideas))
            if pitfalls:
                snippet_parts.append("Pitfalls: " + "; ".join(pitfalls))
            if use_when:
                snippet_parts.append("Use when: " + use_when)
            if avoid_when:
                snippet_parts.append("Avoid when: " + avoid_when)
            if tags:
                snippet_parts.append("Tags: " + ", ".join(tags))
            snippet = "\n".join(snippet_parts).strip()
            if len(snippet) > max_chars:
                snippet = snippet[: max_chars - 1] + "…"
            card = {"title": title, "type": md.get("type"), "tags": tags, "snippet": snippet}

        cards.append(card)

    return json.dumps(cards, ensure_ascii=False)

# Truncate text to a maximum number of characters
def truncate_text(s: str, max_chars: int = 1400) -> str:
    if not s:
        return ""
    s = s.strip()
    return s if len(s) <= max_chars else (s[:max_chars] + "\n... (truncated) ...")