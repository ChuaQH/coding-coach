from __future__ import annotations

from typing import Any, Dict
from langgraph.types import interrupt
from utils.logging import log_stage
from state_models import CoachState
from state_access import problem_state, session_state

# Node to get user's code attempt and notes
def get_attempt_node(state: CoachState) -> Dict[str, Any]:

    prob = problem_state(state)
    sess = session_state(state)

    payload = interrupt({
        "message": (
            "Paste your Python solution attempt + brief notes. "
        ),
        "expected_schema": {"code": "str (python)", "notes": "str"},
        "hint_level": sess.hint_level,
        "problem": {
            "metadata": prob.metadata,
            "description": prob.description,
        },
    })

    code = (payload.get("code") or "").strip()
    notes = (payload.get("notes") or "").strip()
    log_stage("get_attempt", "received", {"code_len": len(code), "notes": notes})

    attempts = list(sess.attempts or [])
    attempts.append({"code": code, "notes": notes})

    sess.attempts = attempts
    sess.retry = False
    sess.force_review = False
    sess.done = False

    return {
        "session": sess.model_dump(),
    }