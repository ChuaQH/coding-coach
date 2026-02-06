from __future__ import annotations

from typing import Any
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from state_models import CoachState
from nodes.problem import (
    fetch_leetcode_solution_node,
    resolve_leetcode_solution_node,
    resolve_problem_description_node,
    resolve_problem_metadata_node,
)
from nodes.retrieval import (
    assess_retrieval_coverage_node,
    generate_retrieval_query_node,
    plan_next_retrieval_node,
    retrieve_algo_kb_node,
    should_continue_retrieval,
    set_kb_sufficiency_flag_node,
    validate_retrieval_tags_node,
    resolve_missing_tags_node,
)
from nodes.review import review_node_factory
from nodes.style import pep_ruff_mypy_node
from nodes.flow import decide_node, route_next
from nodes.followup import followup_question_node_factory
from nodes.attempt import get_attempt_node
from nodes.ingest import solution_knowledge_ingest_node
from state_access import problem_state, retrieval_state


def build_graph() -> Any:
    review_node = review_node_factory()
    followup_node = followup_question_node_factory()

    g = StateGraph(CoachState)

    g.add_node("resolve_problem_metadata", resolve_problem_metadata_node)
    g.add_node("resolve_problem_description", resolve_problem_description_node)
    g.add_node("fetch_leetcode_solution", fetch_leetcode_solution_node)
    g.add_node("resolve_leetcode_solution", resolve_leetcode_solution_node)
    g.add_node("generate_retrieval_query", generate_retrieval_query_node)
    g.add_node("validate_retrieval_tags", validate_retrieval_tags_node)
    g.add_node("resolve_missing_tags", resolve_missing_tags_node)
    g.add_node("retrieve_algo_kb", retrieve_algo_kb_node)
    g.add_node("assess_retrieval_coverage", assess_retrieval_coverage_node)
    g.add_node("plan_next_retrieval", plan_next_retrieval_node)
    g.add_node("set_kb_sufficiency_flag", set_kb_sufficiency_flag_node)
    g.add_node("get_attempt", get_attempt_node)
    g.add_node("pep_ruff_mypy", pep_ruff_mypy_node)
    g.add_node("review", review_node)
    g.add_node("decide", decide_node)
    g.add_node("followup_question", followup_node)
    g.add_node("solution_knowledge_ingest_node", solution_knowledge_ingest_node)

    g.add_edge(START, "resolve_problem_metadata")
    g.add_edge("resolve_problem_metadata", "resolve_problem_description")
    g.add_edge("resolve_problem_description", "fetch_leetcode_solution")

    def route_after_solution_fetch(state: dict) -> str:
        prob = problem_state(state)  
        code = (prob.leetcode_solution_code or "").strip()
        return "resolve_leetcode_solution" if code else "generate_retrieval_query"

    g.add_conditional_edges(
        "fetch_leetcode_solution",
        route_after_solution_fetch,
        {"resolve_leetcode_solution": "resolve_leetcode_solution", "generate_retrieval_query": "generate_retrieval_query"},
    )

    g.add_edge("resolve_leetcode_solution", "generate_retrieval_query")
    g.add_edge("generate_retrieval_query", "validate_retrieval_tags")

    def route_after_validate(state: dict) -> str:
        ret = retrieval_state(state)  
        return "resolve_missing_tags" if ret.needs_tag_resolution else "retrieve_algo_kb"

    g.add_conditional_edges(
        "validate_retrieval_tags",
        route_after_validate,
        {"resolve_missing_tags": "resolve_missing_tags", "retrieve_algo_kb": "retrieve_algo_kb"},
    )

    g.add_edge("resolve_missing_tags", "retrieve_algo_kb")
    g.add_edge("retrieve_algo_kb", "assess_retrieval_coverage")

    def route_after_assess(state: dict) -> str:
        return "plan_next_retrieval" if should_continue_retrieval(state) else "set_kb_sufficiency_flag"

    g.add_conditional_edges(
        "assess_retrieval_coverage",
        route_after_assess,
        {"plan_next_retrieval": "plan_next_retrieval", "set_kb_sufficiency_flag": "set_kb_sufficiency_flag"},
    )

    g.add_edge("plan_next_retrieval", "validate_retrieval_tags")
    g.add_edge("set_kb_sufficiency_flag", "get_attempt")
    g.add_edge("get_attempt", "pep_ruff_mypy")
    g.add_edge("pep_ruff_mypy", "review")
    g.add_edge("review", "decide")

    g.add_conditional_edges(
        "decide",
        route_next,
        {
            "continue": "decide",
            "retry": "get_attempt",
            "followup": "followup_question",
            "review": "review",
            "end": "solution_knowledge_ingest_node",
        },
    )
    g.add_edge("followup_question", "decide")
    g.add_edge("solution_knowledge_ingest_node", END)

    checkpointer = InMemorySaver()
    return g.compile(checkpointer=checkpointer)