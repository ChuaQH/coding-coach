from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from resources import llm, vector_store
from state_access import problem_state, session_state
from utils.logging import log_stage
from state_models import CoachState

class SolutionChunkModel(BaseModel):
    title: str
    summary: str
    tags: List[str] = Field(default_factory=list)
    prerequisites: List[str] = Field(default_factory=list)
    complexity: Dict[str, str] = Field(default_factory=dict)
    pitfalls: List[str] = Field(default_factory=list)
    key_ideas: List[str] = Field(default_factory=list)
    pseudocode: str
    python_snippet: str
    when_to_use: str
    when_not_to_use: str
    type: Literal["algorithm", "data_structure", "python_template", "note"]
    source: Literal["solution"] = "solution"

class SolutionChunksOutput(BaseModel):
    chunks: List[SolutionChunkModel] = Field(default_factory=list)

# Create Document from solution chunk
def make_solution_document(state: Dict[str, Any]) -> Document:

    def _norm_str(x: Any) -> str:
        return "" if x is None else str(x).strip()

    def _norm_list(x: Any) -> List[str]:
        if x is None:
            return []
        if isinstance(x, list):
            return [str(i).strip() for i in x if str(i).strip()]
        s = str(x).strip()
        return [p.strip() for p in s.split(",") if p.strip()] if s else []

    def _norm_dict(x: Any) -> Dict[str, str]:
        if isinstance(x, dict):
            out: Dict[str, str] = {}
            for k, v in x.items():
                ks, vs = str(k).strip(), str(v).strip()
                if ks and vs:
                    out[ks] = vs
            return out
        return {}

    def _json(obj: Any) -> str:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True)

    def _slugify_tag(tag: str) -> str:
        t = tag.strip().lower()
        t = re.sub(r"\s+", "_", t)
        t = re.sub(r"[^a-z0-9_]+", "", t)
        return t

    doc_type = _norm_str(state.get("type") or "note")
    title = _norm_str(state.get("title"))
    summary = _norm_str(state.get("summary"))
    when_to_use = _norm_str(state.get("when_to_use"))
    when_not_to_use = _norm_str(state.get("when_not_to_use"))
    source_platform = _norm_str(state.get("source_platform"))
    source_url = _norm_str(state.get("source_url"))
    source = _norm_str(state.get("source") or "solution")

    tags = _norm_list(state.get("tags"))
    prerequisites = _norm_list(state.get("prerequisites"))
    pitfalls = _norm_list(state.get("pitfalls"))
    key_ideas = _norm_list(state.get("key_ideas"))

    pseudocode = _norm_str(state.get("pseudocode"))
    python_snippet = _norm_str(state.get("python_snippet"))
    complexity = _norm_dict(state.get("complexity"))

    parts: List[str] = []
    if title:
        parts.append(f"Title: {title}")
    parts.append(f"Type: {doc_type}")
    parts.append(f"Source: {source}")

    if summary:
        parts += ["", "Summary:", summary]
    if when_to_use:
        parts += ["", "When to use:", when_to_use]
    if when_not_to_use:
        parts += ["", "When NOT to use:", when_not_to_use]

    if key_ideas:
        parts += ["", "Key ideas:"] + [f"- {x}" for x in key_ideas]
    if prerequisites:
        parts += ["", "Prerequisites:"] + [f"- {x}" for x in prerequisites]

    if complexity:
        parts += ["", "Complexity:"] + [f"- {k}: {v}" for k, v in complexity.items()]

    if pitfalls:
        parts += ["", "Pitfalls:"] + [f"- {x}" for x in pitfalls]

    if tags:
        parts += ["", "Tags:", ", ".join(tags)]

    if pseudocode:
        parts += ["", "Pseudocode:", pseudocode]
    if python_snippet:
        parts += ["", "Python snippet:", python_snippet]

    if source_platform or source_url:
        parts += ["", "Sources:"]
        if source_platform:
            parts.append(f"- platform: {source_platform}")
        if source_url:
            parts.append(f"- {source_url}")

    page_content = "\n".join(parts).strip()

    metadata: Dict[str, Any] = {
        "type": doc_type or None,
        "title": title or None,
        "source": source or None,
        "source_platform": source_platform or None,
        "source_url": source_url or None,
        "tags_csv": ", ".join(tags) if tags else None,
        "prerequisites_csv": ", ".join(prerequisites) if prerequisites else None,
        "complexity_json": _json(complexity) if complexity else None,
        "doc_schema": "SolutionChunk_v1",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    for t in tags:
        slug = _slugify_tag(t)
        if slug:
            metadata[f"tag__{slug}"] = True

    metadata = {k: v for k, v in metadata.items() if v is not None and v != ""}

    return Document(page_content=page_content, metadata=metadata)

# Node to get user's code attempt and notes
def solution_knowledge_ingest_node (state: CoachState) -> Dict[str, Any]:

    prob = problem_state(state)      
    sess = session_state(state)      

    attempt = (sess.attempts or [])[-1] if sess.attempts else {}
    code = (attempt.get("code") or "").strip()
    notes = (attempt.get("notes") or "").strip()

    leetcode_code = prob.leetcode_solution_code or ""

    if not bool(code or leetcode_code.strip()):
        log_stage("solution_knowledge_ingest", "skipped", {"reason": "no_solution_code"})
        return {}

    SYSTEM = """
    You are a knowledge distiller for competitive programming solutions.
    Your goal is to produce reusable, cross-problem knowledge chunks (algorithmic patterns),
    NOT a summary of the specific problem.

    Input: a problem description (optional), a solution analysis (optional), and solution code (optional). Any field may be empty.
    Output: ONLY strict JSON with:

    chunks: list of 1–5 items, each with:
    1) title: short descriptive title
    2) summary: 1–3 sentence summary
    3) tags: list of short tags
    4) prerequisites: list (can be empty)
    5) complexity: {time: "...", space: "..."} (use "unknown" if unclear)
    6) pitfalls: list
    7) key_ideas: list of 3–8 bullets
    8) pseudocode: short pseudocode string
    9) python_snippet: minimal Python template if relevant, else ""
    10) when_to_use: 1–3 sentences
    11) when_not_to_use: 1–3 sentences
    12) type: one of ["algorithm","data_structure","python_template","note"]

    CRITICAL TITLE RULES (must follow):
    - Title MUST be a canonical technique/pattern name, not the problem name.
    - Title MUST NOT contain: the problem title, platform name (LeetCode, Codeforces, etc.), “Solution”, “Approach”, or “Answer”.
    - Title SHOULD be recognizable across problems (e.g., "Monotonic Stack for Next Greater", "Dijkstra with State Compression", "Binary Search on Answer", "DP on Trees (Rerooting)").
    - If you cannot infer the technique, use a neutral pattern title like "Unclear Technique (Insufficient Info)" rather than guessing.

    ABSTRACTION RULES:
    - Do not include problem-specific entities (e.g., “balloons”, “houses”, “ships”) unless they are essential to the technique.
    - Prefer generic terms: array/string/grid/graph/tree/intervals.
    - Do not copy variable names or long code fragments; describe the reusable idea.
    - pseudocode must be generic; python_snippet must be a minimal template (not the full solution).

    CHUNKING RULES:
    - If the solution is missing/empty, return {"chunks": []}.
    - If multiple distinct techniques are used (e.g., preprocessing + DP + segment tree), split into multiple chunks.
    - Keep chunks independent: each chunk should stand alone as a retrievable “note”.

    TAG RULES:
    - Tags should support retrieval across problems: include at least one of:
    algorithm family (dp/greedy/graph), core data structure, and pattern name.
    - Keep tags concise; avoid duplicates; prefer canonical spellings (e.g., “two_pointers”, “monotonic_stack”, “0-1_bfs”).

    ACCURACY RULES:
    - Do NOT invent facts not supported by the input.
    - If complexity is unclear, use "unknown".

    OUTPUT RULES:
    - Output must match schema exactly, no extra keys, no markdown.
    - Ensure valid JSON (double quotes, no trailing commas).
    """.strip()

    # Mask problem titles in problem metadata
    metadata_masked = dict(prob.metadata or {})
    for key in ("normalized_question_title", "existing_question_title", "resolved_title"):
        if key in metadata_masked:
            metadata_masked[key] = "[REDACTED_PROBLEM_TITLE]"

    user_prompt = f"""
    <problem_metadata>
    {json.dumps(metadata_masked, ensure_ascii=False)}
    </problem_metadata>

    <problem_description>
    {json.dumps(prob.description or {}, ensure_ascii=False)}
    </problem_description>

    <user_notes>
    {notes}
    </user_notes>

    <user_code>
    {code}
    </user_code>

    <full_solution_code_if_allowed>
    {leetcode_code}
    </full_solution_code_if_allowed>

    <instructions>
    Return ONLY strict JSON matching the SYSTEM schema.
    Titles must be canonical technique/pattern names (not the problem title; do not include "Solution").
    Do not add extra headings or commentary.
    </instructions>
    """.strip()

    structured_llm = llm.with_structured_output(SolutionChunksOutput)
    resp = structured_llm.invoke([("system", SYSTEM), ("user", user_prompt)])
    parsed = resp.model_dump()

    chunks = parsed.get("chunks") or []
    documents: List[Document] = [make_solution_document(c) for c in chunks]

    log_stage(
        "solution_knowledge_ingest",
        "prepared_documents",
        {"chunk_count": len(chunks), "doc_count": len(documents)},
    )

    if documents:
        vector_store.add_documents(documents)
        log_stage(
            "solution_knowledge_ingest",
            "inserted_documents",
            {"doc_count": len(documents)},
        )

    return {}
