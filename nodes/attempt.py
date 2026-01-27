from __future__ import annotations

from typing import Any, Dict

from langgraph.types import interrupt

from utils.logging import log_stage
from state_models import CoachState


def get_attempt_node(state: CoachState) -> Dict[str, Any]:
    """
    Human-in-the-loop: get a Python attempt.
    Resume payload schema:
      {"code": "...python...", "notes": "..."}
    """

    payload = interrupt(
        {
            "message": (
                "Paste your Python solution attempt + brief notes. "
                "Commands: '/done' to finish, '/answer' for a full solution, '/retry' to re-enter your code. "
                "Use 'stuck' in notes if you want a higher hint level. "
                "Otherwise, write a question about the problem or your solution."
            ),
            "expected_schema": {"code": "str (python)", "notes": "str"},
            "hint_level": state.get("hint_level", 0),
            # Use your existing state fields (your state doesn't define 'problem')
            "problem": {
                "metadata": state.get("problem_metadata", {}),
                "description": state.get("problem_description", {}),
            },
        }
    )

    code = (payload.get("code") or "").strip()
    notes = (payload.get("notes") or "").strip()
    log_stage("get_attempt", "received", {"code_len": len(code), "notes": notes})

    attempts = list(state.get("attempts") or [])
    attempts.append({"code": code, "notes": notes})

    # IMPORTANT: clear these flags so routing doesn't get stuck
    return {
        "attempts": attempts,
        "retry": False,
        "has_followup_question": False,
        "followup_question": "",
        "force_review": False,
        "done": False,
    }
