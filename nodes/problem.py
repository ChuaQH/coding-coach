from __future__ import annotations

import json
from typing import Any, Dict

from helpers.leetcode_solution_helper import fetch_leetcode_python_solution, get_leetcode_title_from_id

from resources import llm
from state_access import problem_state, session_state
from state_models import ProblemDescriptionModel, ProblemMetadataModel, SolutionAnalysisModel
from utils.logging import log_stage

# Node to resolve problem metadata
def resolve_problem_metadata_node(state: Dict[str, Any]) -> Dict[str, Any]:
    prob = problem_state(state)  
    sess = session_state(state)  

    q = prob.user_question_title.strip()
    log_stage("resolve_problem_metadata", "input", {"user_question_title": q})

    SYSTEM = """
    Task: Process user input to identify the type of coding problem.

    Return JSON with:
    - normalized_question_title (string of all lowercase)
    - existing_question (string value of "leetcode" | "inferred" | "user_provided")
    - existing_question_title (optional string if existing_question is "leetcode")
    - existing_question_id (optional string if existing_question is "leetcode")

    Rules:
    - Remove markdown fences if present.
    - Do not change title logic, only formatting cleanup.
    Return ONLY JSON.
    """.strip()

    structured_llm = llm.with_structured_output(ProblemMetadataModel)
    resp = structured_llm.invoke([("system", SYSTEM), ("user", q)])
    md = resp.model_dump()

    if md.get("existing_question") == "leetcode":
        lc_id = md.get("existing_question_id")
        log_stage("resolve_problem_metadata", f"LeetCode lookup for ID: {lc_id}")
        resolved_title = get_leetcode_title_from_id(lc_id)
        md.update({"platform": "leetcode", "resolved_title": resolved_title})
        log_stage("resolve_problem_metadata", "Resolved title", {"resolved_title": resolved_title})

    prob.metadata = md
    return {"problem": prob.model_dump(), "session": sess.model_dump()}

# Node to resolve problem description
def resolve_problem_description_node(state: Dict[str, Any]) -> Dict[str, Any]:
    prob = problem_state(state)  
    description = prob.user_question_description.strip()
    log_stage("resolve_problem_description", "input", {"description_len": len(description)})

    SYSTEM = """
    Extract a structured understanding of the problem.

    Return ONLY JSON with:
    - summary: one-paragraph restatement of the problem in your own words
    - constraints: {n_max, value_range, time_limit, memory_limit, notes, inferred}
    - tags: list of likely techniques/patterns (e.g., "two pointers", "dp", "graph", "binary search")
    - uncertainties: list of missing/ambiguous details

    Rules:
    - If constraints are not explicit, set inferred=true and estimate n_max as null.
    - Do NOT propose a solution.
    """.strip()

    structured_llm = llm.with_structured_output(ProblemDescriptionModel)
    resp = structured_llm.invoke([("system", SYSTEM), ("user", description)])
    pd = resp.model_dump()

    log_stage(
        "resolve_problem_description",
        "parsed",
        {"summary": pd.get("summary"), "tags": pd.get("tags"), "constraints": pd.get("constraints")},
    )

    prob.description = pd
    return {"problem": prob.model_dump()}

# Node to fetch LeetCode solution code (from LeetCode-Solutions repo by kamyu104)
def fetch_leetcode_solution_node(state: Dict[str, Any]) -> Dict[str, Any]:
    prob = problem_state(state)  
    md = prob.metadata or {}
    platform = md.get("platform")
    resolved_title = md.get("resolved_title") or ""
    existing_question_id = md.get("existing_question_id") or ""
    log_stage("fetch_leetcode_solution", "input", {"platform": platform, "resolved_title": resolved_title})

    if platform != "leetcode" and not resolved_title and not existing_question_id:
        log_stage("fetch_leetcode_solution", "no LeetCode platform or title; skipping")
        return {}

    # Try to find leetcode solution by title first
    try:
        code = fetch_leetcode_python_solution(resolved_title)
        prob.leetcode_solution_code = code
        log_stage("fetch_leetcode_solution", "fetched solution", {"code_len": len(code), "code_preview": code[:200]})
        return {"problem": prob.model_dump()}
    except Exception as e:
        log_stage("fetch_leetcode_solution", "error fetching solution (title)", {"error": str(e)})

    # Fallback on id
    try:
        code = fetch_leetcode_python_solution(existing_question_id)
        prob.leetcode_solution_code = code
        log_stage("fetch_leetcode_solution", "fetched solution", {"code_len": len(code), "code_preview": code[:200]})
        return {"problem": prob.model_dump()}
    except Exception as e:
        log_stage("fetch_leetcode_solution", "error fetching solution (id)", {"error": str(e)})
        return {}

# Node to analyze LeetCode solution code
def resolve_leetcode_solution_node(state: Dict[str, Any]) -> Dict[str, Any]:
    prob = problem_state(state)  
    code = (prob.leetcode_solution_code or "").strip()
    if not code:
        log_stage("resolve_leetcode_solution", "no code to resolve")
        return {}

    problem_description = prob.description or {}
    problem_summary = problem_description.get("summary", "")
    user_problem_description = prob.user_question_description or ""

    SYSTEM = """
    Analyze the provided user given problem and LeetCode solution code.

    Return ONLY JSON with:
    - solution_matches_problem: boolean indicating if the solution addresses the problem
    - solution_strategy: 1–2 sentences describing the overall approach used in the solution
    - solution_tags: list of canonical names of algorithmic techniques used ordered from most central to least central
    - solution_time_complexity: big-O notation string
    - solution_space_complexity: big-O notation string
    - solution_invariants: list of 2-5 concrete, checkable invariants the solution maintains
    - solution_pitfalls: list of 3–6 common mistakes or pitfalls in implementing this solution
    """.strip()

    user_prompt = f"""
    User given problem summary: {problem_summary}
    User given problem description: {user_problem_description}
    Leetcode solution code: {code}
    """.strip()

    structured_llm = llm.with_structured_output(SolutionAnalysisModel)
    resp = structured_llm.invoke([("system", SYSTEM), ("user", user_prompt)])
    sa = resp.model_dump()

    log_stage("resolve_leetcode_solution", "parsed", sa)
    prob.solution_analysis = sa
    return {"problem": prob.model_dump()}