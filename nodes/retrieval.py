from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, Tuple

from langchain_core.documents import Document
from copy import copy
from resources import llm, vector_store
from kb.tags import VOCAB_SET, TAG_RESOLVER, build_must_filter, canonicalize_tags, expand_exact, tag_key
from state_access import problem_state, retrieval_state
from state_models import RetrievalAssessmentOutput, RetrievalQueryOutput
from utils.logging import log_stage

# Function to decide whether to continue retrieval
def should_continue_retrieval(state: Dict[str, Any]) -> bool:
    ret = retrieval_state(state)  
    assessment = ret.assessment or {}
    if assessment.get("sufficient", True):
        return False

    c = ret.counters
    if c.retrieval_round >= c.max_retrieval_rounds:
        return False
    if c.no_progress_rounds >= c.max_no_progress_rounds:
        return False

    followups = assessment.get("recommended_followup_queries", []) or []
    used = c.used_followup_queries or []
    remaining = [q for q in followups if q not in used]
    if not remaining and c.last_retrieval_added == 0:
        return False

    return True

# Node to generate kb retrieval query
def generate_retrieval_query_node(state: Dict[str, Any]) -> Dict[str, Any]:
    prob = problem_state(state)  
    ret = retrieval_state(state)  

    md = prob.metadata or {}
    pd = prob.description or {}

    title = md.get("resolved_title") or md.get("existing_question_title") or ""
    summary = pd.get("summary", "")
    constraints = pd.get("constraints", {})
    problem_tags = pd.get("tags", []) or []

    sol = prob.solution_analysis or {}
    solution_matches = bool(sol.get("solution_matches_problem", False))
    solution_strategy = sol.get("solution_strategy", "") or ""
    solution_tags = sol.get("solution_tags", []) or []
    solution_time = sol.get("solution_time_complexity", "") or ""
    solution_space = sol.get("solution_space_complexity", "") or ""
    solution_invariants = sol.get("solution_invariants", []) or []
    solution_pitfalls = sol.get("solution_pitfalls", []) or []

    use_solution = solution_matches and (len(solution_tags) > 0 or len(solution_invariants) > 0)
    mode = "solution_guided" if use_solution else "problem_only"

    SYSTEM = """
    You generate a retrieval query for a knowledge base of algorithm cards.

    Return ONLY JSON with:
    - query: a concise search string combining high-signal keywords and tags
    - query_parts: {"title_keywords": [...], "summary_keywords": [...], "constraint_keywords": [...], "tags": [...], "key_ideas": [...]}
    - must_tags: tags that must be present
    - should_tags: tags that are helpful
    - avoid_tags: tags that are misleading
    - rationale: 1–2 sentences describing how the query aligns to the stored card fields

    Modes:
    - If mode == "solution_guided":
        * Prefer solution_tags as must_tags (high precision).
        * Use problem title/summary/constraints as supporting keywords only.
        * Use solution_invariants/solution_pitfalls ONLY as key_ideas keywords, not as must_tags.
        * Use solution time/space complexity as constraint_keywords when helpful.
    - If mode == "problem_only":
        * Infer tags from the problem and be conservative with must_tags.
        * Use constraints and summary to choose high-signal keywords.

    General rules:
    - Do NOT propose a solution.
    - Keep must_tags short (0–3 items) and high precision.
    - Output must follow the schema exactly.
    """.strip()

    inv_sample = solution_invariants[:4]
    pit_sample = solution_pitfalls[:4]

    user_prompt = f"""
    Mode: {mode}

    Problem title: {title}
    Problem summary: {summary}
    Problem constraints: {json.dumps(constraints, ensure_ascii=False)}
    Problem-derived candidate tags: {problem_tags}

    {"Solution strategy: " + solution_strategy if use_solution and solution_strategy else ""}
    {"Solution tags: " + ", ".join(solution_tags) if use_solution and solution_tags else ""}
    {"Solution time complexity: " + solution_time if use_solution and solution_time else ""}
    {"Solution space complexity: " + solution_space if use_solution and solution_space else ""}
    {"Solution invariants: " + "; ".join(inv_sample) if use_solution and inv_sample else ""}
    {"Solution pitfalls: " + "; ".join(pit_sample) if use_solution and pit_sample else ""}
    """.strip()

    structured_llm = llm.with_structured_output(RetrievalQueryOutput)
    resp = structured_llm.invoke([("system", SYSTEM), ("user", user_prompt)])
    rq = resp.model_dump()
    rq["must_tags"] = canonicalize_tags(rq.get("must_tags") or [])
    rq["should_tags"] = canonicalize_tags(rq.get("should_tags") or [])
    rq["avoid_tags"] = canonicalize_tags(rq.get("avoid_tags") or [])

    ret.query = rq
    ret.original_query = copy(rq)
    return {"retrieval": ret.model_dump()}

# Node to validate and expand retrieval tags
def validate_retrieval_tags_node(state: Dict[str, Any]) -> Dict[str, Any]:
    ret = retrieval_state(state)  
    rq = ret.query or {}

    must_tags = rq.get("must_tags") or []
    should_tags = rq.get("should_tags") or []
    avoid_tags = rq.get("avoid_tags") or []
    log_stage("validate_retrieval_tags", "input", {"must": must_tags, "should": should_tags, "avoid": avoid_tags})

    # Filter out any missing tags that do not have exact match
    must_exp, must_missing = expand_exact(must_tags)
    should_exp, should_missing = expand_exact(should_tags)
    avoid_exp, avoid_missing = expand_exact(avoid_tags)

    rq["must_tags"] = must_exp
    rq["should_tags"] = [t for t in should_exp if t not in set(must_exp)]
    rq["avoid_tags"] = [t for t in avoid_exp if t not in set(must_exp)]

    missing_any = list(dict.fromkeys(must_missing + should_missing + avoid_missing))
    ret.query = rq
    ret.missing_tags = missing_any
    ret.needs_tag_resolution = bool(missing_any)
    return {"retrieval": ret.model_dump()}

# Node to resolve missing retrieval tags
def resolve_missing_tags_node(state: Dict[str, Any]) -> Dict[str, Any]:
    ret = retrieval_state(state)
    rq = ret.query or {}
    missing = ret.missing_tags or []
    if not missing:
        log_stage("resolve_missing_tags", "no missing tags")
        return {}

    resolved = TAG_RESOLVER.resolve_many(
        missing,
        top_n=5,
        min_score=85.0,
        close_min_score=95.0,
        include_close_delta=1.0,
    )
    log_stage("resolve_missing_tags", "resolved", {"missing": missing, "resolution": resolved.get("resolution")})

    must_set = set(rq.get("must_tags") or [])
    should_set = set(rq.get("should_tags") or [])
    avoid_set = set(rq.get("avoid_tags") or [])

    original = ret.original_query or {}
    orig_must = set(original.get("must_tags", []) or [])
    orig_avoid = set(original.get("avoid_tags", []) or [])

    for raw_tag, info in (resolved.get("resolution") or {}).items():
        chosen = [t for t in (info.get("chosen") or []) if t in VOCAB_SET]
        # Sort tags based on where it originally came from
        if raw_tag in orig_must:
            for t in chosen:
                must_set.add(t)
                should_set.discard(t)
                avoid_set.discard(t)
        elif raw_tag in orig_avoid:
            for t in chosen:
                avoid_set.add(t)
                must_set.discard(t)
        else:
            for t in chosen:
                if t not in must_set:
                    should_set.add(t)

    rq["must_tags"] = list(must_set)
    rq["should_tags"] = [t for t in should_set if t not in must_set]
    rq["avoid_tags"] = [t for t in avoid_set if t not in must_set]

    ret.query = rq
    ret.tag_resolution = resolved
    return {"retrieval": ret.model_dump()}

# Node to retrieve algorithm kb chunks
def retrieve_algo_kb_node(state: Dict[str, Any]) -> Dict[str, Any]:
    ret = retrieval_state(state)  
    rq = ret.query or {}

    base_query = (rq.get("query") or "").strip()
    must_tags = rq.get("must_tags") or []
    should_tags = rq.get("should_tags") or []
    avoid_tags = rq.get("avoid_tags") or []

    final_query = " ".join([base_query, *must_tags, *should_tags]).strip() or base_query
    must_filter = build_must_filter(must_tags)

    best_score = 0.0
    docs: List[Document] = []

    try:
        scored = vector_store.similarity_search_with_relevance_scores(final_query, k=12, filter=must_filter)
        scored = [(d, s) for (d, s) in scored if d is not None]
        if scored:
            best_score = max(s for _, s in scored)
            scored = [(d, s) for (d, s) in scored if s >= 0.35]
            docs = [d for d, _ in scored]
    except Exception:
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 12, "fetch_k": 30, "lambda_mult": 0.5, "filter": must_filter},
        )
        docs = retriever.invoke(final_query)

    if avoid_tags and docs:
        avoid_keys = {tag_key(t) for t in avoid_tags}
        filtered = []
        for d in docs:
            md = d.metadata or {}
            if any(md.get(k) is True for k in avoid_keys):
                continue
            filtered.append(d)
        docs = filtered

    docs = docs[:4]
    new_results = [{"metadata": d.metadata, "content": d.page_content} for d in docs]

    prev = ret.results or []
    merged = prev + new_results

    seen = set()
    deduped = []
    for r in merged:
        md = r.get("metadata", {}) or {}
        key = md.get("title") or md.get("source_url") or json.dumps(md, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)

    added = max(0, len(deduped) - len(prev))
    log_stage(
        "retrieve_algo_kb",
        "results",
        {
            "query": final_query,
            "added": added,
            "total": len(deduped),
            "best_score": best_score,
            "titles": [r.get("metadata", {}).get("title") for r in deduped],
        },
    )

    ret.results = deduped
    ret.counters.last_retrieval_added = added
    ret.counters.last_retrieval_best_score = float(best_score)

    c = ret.counters
    c.retrieval_round += 1
    added = int(c.last_retrieval_added)
    best_score = float(c.last_retrieval_best_score)

    weak = (best_score > 0 and best_score < 0.35)
    if added == 0 or weak:
        c.no_progress_rounds += 1
    else:
        c.no_progress_rounds = 0

    ret.counters = c
    return {"retrieval": ret.model_dump()}

# Node to assess retrieval coverage
def assess_retrieval_coverage_node(state: Dict[str, Any]) -> Dict[str, Any]:
    prob = problem_state(state)  
    ret = retrieval_state(state)  

    md = prob.metadata or {}
    pd = prob.description or {}
    sa = prob.solution_analysis or {}
    retrieval_results = ret.results or []
    leetcode_solution_code = (prob.leetcode_solution_code or "").strip()

    existing_question = (md.get("existing_question") or "").strip()
    solution_matches_problem = bool(sa.get("solution_matches_problem", False))
    has_known_solution = bool(leetcode_solution_code) and (existing_question in ("leetcode", "inferred"))

    log_stage(
        "assess_retrieval_coverage",
        "input",
        {
            "retrieval_count": len(retrieval_results),
            "has_known_solution": has_known_solution,
            "existing_question": existing_question,
            "solution_matches_problem": solution_matches_problem,
        },
    )

    condensed_results = [
        {
            "title": (r.get("metadata", {}) or {}).get("title"),
            "type": (r.get("metadata", {}) or {}).get("type"),
            "tags": (r.get("metadata", {}) or {}).get("tags_csv"),
            "content_snippet": ((r.get("content") or "")[:400]),
            "score": r.get("score"),
        }
        for r in retrieval_results
    ]

    if has_known_solution:
        system_prompt = """
        You are a routing/verification judge.

        Goal: Decide whether we need MORE retrieval to deliver a correct and optimal answer.

        Key rule:
        - If an authoritative solution is present, DO NOT require missing algorithms/KB chunks.
        - Mark sufficient=true unless the solution appears incorrect, incomplete, or non-optimal for the given constraints.

        Return ONLY JSON with:
        - sufficient: true/false
        - missing_concepts: list (ONLY items that block correctness/optimality)
        - missing_algorithms: list (ONLY if necessary to fix the solution)
        - recommended_followup_queries: list (ONLY if sufficient=false)
        - rationale: 1-2 sentences
        """.strip()

        user_prompt = f"""
        Problem metadata: {json.dumps(md, ensure_ascii=False)}
        Problem description: {json.dumps(pd, ensure_ascii=False)}
        Solution analysis: {json.dumps(sa, ensure_ascii=False)}
        LeetCode solution code (truncated): {leetcode_solution_code[:2000]}
        Retrieved chunks (optional): {json.dumps(condensed_results[:5], ensure_ascii=False)}
        """.strip()

    else:
        system_prompt = """
        You are evaluating whether retrieved knowledge chunks contain enough relevant concepts and algorithms
        to solve the problem in an optimal way.

        Return ONLY JSON with:
        - sufficient: true/false
        - missing_concepts: list of missing concepts REQUIRED for a correct/optimal solution
        - missing_algorithms: list of missing algorithms REQUIRED
        - recommended_followup_queries: list of short search queries to retrieve missing items
        - rationale: 1-2 sentences
        """.strip()

        user_prompt = f"""
        Problem metadata: {json.dumps(md, ensure_ascii=False)}
        Problem description: {json.dumps(pd, ensure_ascii=False)}
        Retrieved chunks: {json.dumps(condensed_results, ensure_ascii=False)}
        """.strip()

    structured_llm = llm.with_structured_output(RetrievalAssessmentOutput)
    resp = structured_llm.invoke([("system", system_prompt), ("user", user_prompt)])
    assessment = resp.model_dump()
    assessment["missing_concepts"] = canonicalize_tags(assessment.get("missing_concepts") or [])
    assessment["missing_algorithms"] = canonicalize_tags(assessment.get("missing_algorithms") or [])

    if has_known_solution and assessment.get("sufficient") is False:
        blockers = (assessment.get("missing_concepts") or []) + (assessment.get("missing_algorithms") or [])
        if not blockers and solution_matches_problem:
            assessment["sufficient"] = True
            assessment["recommended_followup_queries"] = []
            assessment["rationale"] = "Known solution is present and matches the problem; additional retrieval is not required."

    log_stage("assess_retrieval_coverage", "output", assessment)
    ret.assessment = assessment
    return {"retrieval": ret.model_dump()}

# Node to plan next retrieval query
def plan_next_retrieval_node(state: Dict[str, Any]) -> Dict[str, Any]:
    ret = retrieval_state(state)  
    assessment = ret.assessment or {}
    rq = ret.query or {}

    followups = assessment.get("recommended_followup_queries", []) or []
    used = ret.counters.used_followup_queries or []

    next_q = next((q for q in followups if q and q not in used), None)

    boosts = []
    boosts += assessment.get("missing_algorithms", []) or []
    boosts += assessment.get("missing_concepts", []) or []
    boosts = [b.strip() for b in boosts if isinstance(b, str) and b.strip()]

    if next_q:
        rq["query"] = (rq.get("query", "") + " " + next_q).strip()
        used.append(next_q)

    if boosts:
        rq["should_tags"] = list(dict.fromkeys((rq.get("should_tags") or []) + boosts))

    ret.query = rq
    ret.counters.used_followup_queries = used
    return {"retrieval": ret.model_dump()}

# Node to set knowledge base sufficiency flag after retrieval is done
def set_kb_sufficiency_flag_node(state: Dict[str, Any]) -> Dict[str, Any]:
    ret = retrieval_state(state)  
    assessment = ret.assessment or {}
    ret.kb_insufficient = not bool(assessment.get("sufficient", True))
    return {"retrieval": ret.model_dump()}
