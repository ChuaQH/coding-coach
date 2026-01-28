from __future__ import annotations

from typing import Any, Dict, Literal

from langgraph.types import interrupt

from state_access import followup_state, problem_state, session_state
from utils.logging import log_stage


def decide_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Loop control:
    - /done ends.
    - /answer bumps hint_level to 5 and forces a review.
    - /retry re-enters solution.
    - /hint N changes the hint_level
    - otherwise treated as follow-up question.
    """
    sess = session_state(state)       # type: ignore[arg-type]
    prob = problem_state(state)       # type: ignore[arg-type]
    fol = followup_state(state)       # type: ignore[arg-type]

    payload = interrupt({
        "message": "Enter a command (/done, /answer, /retry, /hint N) or type a follow-up question:",
        "expected_schema": {"command": "str"},
        "hint_level": sess.hint_level,
        "problem": {"metadata": prob.metadata, "description": prob.description},
    })

    raw = (payload.get("command") or "").strip()
    cmd = raw.lower().strip()
    log_stage("decide", "command", {"command": cmd})

    # defaults for each decide turn
    fol.has_followup_question = False
    fol.followup_question = ""
    sess.force_review = False

    if cmd == "/done":
        sess.done = True
        sess.retry = False
        return {"session": sess.model_dump(), "followup": fol.model_dump()}

    if cmd == "/answer":
        sess.hint_level = 5
        sess.done = False
        sess.retry = False
        sess.force_review = True
        return {"session": sess.model_dump(), "followup": fol.model_dump()}

    if cmd == "/retry":
        sess.done = False
        sess.retry = True
        return {"session": sess.model_dump(), "followup": fol.model_dump()}

    if cmd.startswith("/hint"):
        parts = cmd.split()
        if len(parts) >= 2:
            try:
                sess.hint_level = int(parts[1])
            except ValueError:
                pass
        sess.done = False
        sess.retry = False
        return {"session": sess.model_dump(), "followup": fol.model_dump()}

    # default: follow-up question
    sess.done = False
    sess.retry = False
    fol.has_followup_question = True
    fol.followup_question = raw
    return {"session": sess.model_dump(), "followup": fol.model_dump()}


def route_next(state: Dict[str, Any]) -> Literal["continue", "end", "followup", "review"]:
    sess = session_state(state)       # type: ignore[arg-type]
    fol = followup_state(state)       # type: ignore[arg-type]

    if sess.done:
        return "end"
    if sess.retry:
        return "retry"
    if sess.force_review:
        return "review"
    if fol.has_followup_question:
        return "followup"
    return "continue"
