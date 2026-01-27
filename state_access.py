from __future__ import annotations

from typing import Any

from state_models import (
    CoachState,
    FollowupStateModel,
    ProblemStateModel,
    RetrievalStateModel,
    SessionStateModel,
    StyleStateModel,
)


def problem_state(state: CoachState) -> ProblemStateModel:
    return ProblemStateModel.model_validate(state.get("problem") or {})


def retrieval_state(state: CoachState) -> RetrievalStateModel:
    return RetrievalStateModel.model_validate(state.get("retrieval") or {})


def style_state(state: CoachState) -> StyleStateModel:
    return StyleStateModel.model_validate(state.get("style") or {})


def followup_state(state: CoachState) -> FollowupStateModel:
    return FollowupStateModel.model_validate(state.get("followup") or {})


def session_state(state: CoachState) -> SessionStateModel:
    return SessionStateModel.model_validate(state.get("session") or {})


def shallow_copy_dict(d: dict[str, Any]) -> dict[str, Any]:
    # for "original_query" copies, etc.
    return dict(d) if isinstance(d, dict) else {}
