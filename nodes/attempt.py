from __future__ import annotations

from typing import Any, Dict
from langgraph.types import interrupt
from utils.logging import log_stage
from state_models import CoachState

# Node to get user's code attempt and notes
def get_attempt_node(state: CoachState) -> Dict[str, Any]:

    payload = interrupt({
        "message": (
            "Paste your Python solution attempt + brief notes. "
        ),
        "expected_schema": {"code": "str (python)", "notes": "str"},
        "hint_level": state.get("hint_level", 0),
        "problem": {
            "metadata": state.get("problem_metadata", {}),
            "description": state.get("problem_description", {}),
        },
    })

    code = (payload.get("code") or "").strip()
    notes = (payload.get("notes") or "").strip()
    log_stage("get_attempt", "received", {"code_len": len(code), "notes": notes})

    attempts = list(state.get("attempts") or [])
    attempts.append({"code": code, "notes": notes})

    return {
        "attempts": attempts,
        "retry": False,
        "has_followup_question": False,
        "followup_question": "",
        "force_review": False,
        "done": False,
    }
