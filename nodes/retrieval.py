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
from utils.text_format import compact_kb_cards

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

    tag_candidates = canonicalize_tags(problem_tags + (solution_tags if use_solution else []))
    tag_candidate_set = set(tag_candidates)

    SYSTEM = """
    You generate a retrieval query for a knowledge base of algorithm cards.

    IMPORTANT: Stored cards are plain text like:
    <Title line>
    <1-2 line summary>
    Key ideas: <Key ideas separated by semicolons>
    Pitfalls: <Pitfalls separated by semicolons>
    Use when: <Use when description>
    Avoid when: <Avoid when description>
    Tags: <tag1>, <tag2>, ...

    Return ONLY JSON with:
    - query: a concise search string optimized to match those cards
    - query_parts: {"title_keywords": [...], "summary_keywords": [...], "constraint_keywords": [...], "tags": [...], "key_ideas": [...]}
    - must_tags: 0–1 tags (be conservative; prefer [] over a wrong must_tag)
    - should_tags: up to 6 tags
    - avoid_tags: up to 4 tags
    - rationale: 1–2 sentences describing alignment to stored card fields

    Rules:
    1) Tags MUST be chosen ONLY from tag_candidates provided by the user.
    2) query should look like what’s in the cards: technique/pattern words + key idea words + tags.
    3) Avoid problem story nouns; prefer canonical technique/pattern language.
    4) If mode == solution_guided: use solution tags/strategy as hints, but do NOT over-constrain must_tags.

    Output must follow schema exactly. No extra keys. No markdown.
    """.strip()

    inv_sample = solution_invariants[:4]
    pit_sample = solution_pitfalls[:4]

    user_prompt = f"""
    Mode: {mode}

    Problem title: {title}
    Problem summary: {summary}
    Problem constraints: {json.dumps(constraints, ensure_ascii=False)}
    tag_candidates: {tag_candidates}

    {"Solution strategy: " + solution_strategy if use_solution and solution_strategy else ""}
    {"Solution time complexity: " + solution_time if use_solution and solution_time else ""}
    {"Solution space complexity: " + solution_space if use_solution and solution_space else ""}
    {"Solution invariants (keywords only): " + "; ".join(inv_sample) if use_solution and inv_sample else ""}
    {"Solution pitfalls (keywords only): " + "; ".join(pit_sample) if use_solution and pit_sample else ""}
    """.strip()

    structured_llm = llm.with_structured_output(RetrievalQueryOutput)
    resp = structured_llm.invoke([("system", SYSTEM), ("user", user_prompt)])
    rq = resp.model_dump()

    rq["must_tags"] = canonicalize_tags(rq.get("must_tags") or [])
    rq["should_tags"] = canonicalize_tags(rq.get("should_tags") or [])
    rq["avoid_tags"] = canonicalize_tags(rq.get("avoid_tags") or [])

    rq["must_tags"] = [t for t in rq["must_tags"] if t in tag_candidate_set][:1]
    rq["should_tags"] = [t for t in rq["should_tags"] if t in tag_candidate_set][:6]
    rq["avoid_tags"] = [t for t in rq["avoid_tags"] if t in tag_candidate_set][:4]

    chosen_tags = rq["must_tags"] + rq["should_tags"]
    if chosen_tags:
        q = (rq.get("query") or "").strip()
        tag_tail = " Tags: " + ", ".join(chosen_tags)
        if "tags:" not in q.lower():
            rq["query"] = (q + tag_tail).strip()

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

    tag_candidates = canonicalize_tags((pd.get("tags") or []) + (sa.get("solution_tags") or []))
    tag_candidate_set = set(tag_candidates)

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

    condensed_results = compact_kb_cards(retrieval_results)

    if has_known_solution:
        system_prompt = """
        You are a routing/verification judge.

        Goal: Decide whether we need MORE retrieval to deliver a correct and optimal answer.

        Key rule:
        - If an authoritative solution is present, DO NOT require missing algorithms/KB chunks.
        - Mark sufficient=true unless the solution appears incorrect, incomplete, or non-optimal for the given constraints.

        If you recommend follow-up queries, they MUST be short, query-like strings optimized for the KB cards:
        <Title line>
        <1-2 line summary>
        Key ideas: <...>
        Pitfalls: <...>
        Use when: <...>
        Avoid when: <...>
        Tags: <tag1>, <tag2>, ...

        Use only tag candidates provided (if any). If you include tags, add "Tags: ..." to the query.

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
        Retrieved chunks (optional): {condensed_results}
        tag_candidates: {tag_candidates}
        """.strip()

    else:
        system_prompt = """
        You are evaluating whether retrieved knowledge chunks contain enough relevant concepts and algorithms
        to solve the problem in an optimal way.

        IMPORTANT: The KB chunks may contain ONLY algorithms, code implementations, and complexity.
        Do NOT require conceptual exposition (e.g., "optimal substructure") if the algorithm/recurrence/implementation implies it.

        Define sufficient=true if, using (problem metadata + description + retrieved chunks), we can:
        (1) identify at least one applicable standard/optimal algorithm for this problem,
        (2) provide actionable code feedback using a reference implementation or clear algorithm steps,
        (3) benchmark efficiency using stated or safely inferable time/space complexity.

        Only mark missing_concepts/missing_algorithms if they BLOCK algorithm selection or code feedback.
        If multiple algorithms could apply, require enough criteria to choose between them (e.g., constraints, preconditions).
        Return evidence for each requirement from the chunks (quote short phrases or point to titles/tags).

        If you recommend follow-up queries, they MUST be short, query-like strings optimized for the KB cards:
        <Title line>
        <1-2 line summary>
        Key ideas: <...>
        Pitfalls: <...>
        Use when: <...>
        Avoid when: <...>
        Tags: <tag1>, <tag2>, ...

        Use only tag candidates provided (if any). If you include tags, add "Tags: ..." to the query.

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
        Retrieved chunks: {condensed_results}
        tag_candidates: {tag_candidates}
        """.strip()

    structured_llm = llm.with_structured_output(RetrievalAssessmentOutput)
    resp = structured_llm.invoke([("system", system_prompt), ("user", user_prompt)])
    assessment = resp.model_dump()
    assessment["missing_concepts"] = canonicalize_tags(assessment.get("missing_concepts") or [])
    assessment["missing_algorithms"] = canonicalize_tags(assessment.get("missing_algorithms") or [])

    followups = assessment.get("recommended_followup_queries") or []
    filtered_followups = []
    for q in followups:
        if not isinstance(q, str) or not q.strip():
            continue
        if "tags:" in q.lower() and tag_candidate_set:
            tags_part = q.split("Tags:", 1)[1] if "Tags:" in q else q.split("tags:", 1)[1]
            chosen = canonicalize_tags([t.strip() for t in tags_part.split(",") if t.strip()])
            chosen = [t for t in chosen if t in tag_candidate_set]
            base = q.split("Tags:", 1)[0].strip() if "Tags:" in q else q.split("tags:", 1)[0].strip()
            if chosen:
                q = f"{base} Tags: {', '.join(chosen)}".strip()
            else:
                q = base
        filtered_followups.append(q)
    assessment["recommended_followup_queries"] = filtered_followups

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
