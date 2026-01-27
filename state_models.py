from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict

from pydantic import BaseModel, Field, field_validator


# -------------------------
# LLM structured outputs
# -------------------------

class ProblemMetadataModel(BaseModel):
    normalized_question_title: str = ""
    existing_question: str = "user_provided"  # "leetcode" | "inferred" | "user_provided"
    existing_question_title: Optional[str] = None
    existing_question_id: Optional[str] = None
    platform: Optional[str] = None
    resolved_title: Optional[str] = None

    @field_validator("existing_question", mode="before")
    @classmethod
    def _normalize_existing_question(cls, v: Any) -> str:
        s = str(v or "").strip().lower()
        return s if s in {"leetcode", "inferred", "user_provided"} else "user_provided"


class ProblemDescriptionModel(BaseModel):
    summary: str
    constraints: Dict[str, str] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    uncertainties: List[str] = Field(default_factory=list)


class SolutionAnalysisModel(BaseModel):
    solution_matches_problem: bool
    solution_strategy: str
    solution_tags: List[str] = Field(default_factory=list)
    solution_time_complexity: str
    solution_space_complexity: str
    solution_invariants: List[str] = Field(default_factory=list)
    solution_pitfalls: List[str] = Field(default_factory=list)


class RetrievalQueryOutput(BaseModel):
    query: str
    query_parts: Dict[str, List[str]] = Field(default_factory=dict)
    must_tags: List[str] = Field(default_factory=list)
    should_tags: List[str] = Field(default_factory=list)
    avoid_tags: List[str] = Field(default_factory=list)
    rationale: str


class RetrievalAssessmentOutput(BaseModel):
    sufficient: bool
    missing_concepts: List[str] = Field(default_factory=list)
    missing_algorithms: List[str] = Field(default_factory=list)
    recommended_followup_queries: List[str] = Field(default_factory=list)
    rationale: str


# -------------------------
# State submodels (stored as dicts in LangGraph state)
# -------------------------

class RetrievalCountersModel(BaseModel):
    retrieval_round: int = 0
    max_retrieval_rounds: int = 4
    no_progress_rounds: int = 0
    max_no_progress_rounds: int = 2
    used_followup_queries: List[str] = Field(default_factory=list)
    last_retrieval_added: int = 0
    last_retrieval_best_score: float = 0.0


class ProblemStateModel(BaseModel):
    user_question_title: str = ""
    user_question_description: str = ""

    # store dumps only
    metadata: Dict[str, Any] = Field(default_factory=dict)
    description: Dict[str, Any] = Field(default_factory=dict)
    leetcode_solution_code: str = ""
    solution_analysis: Dict[str, Any] = Field(default_factory=dict)


class RetrievalStateModel(BaseModel):
    query: Dict[str, Any] = Field(default_factory=dict)
    original_query: Dict[str, Any] = Field(default_factory=dict)
    results: List[Dict[str, Any]] = Field(default_factory=list)
    assessment: Dict[str, Any] = Field(default_factory=dict)

    missing_tags: List[str] = Field(default_factory=list)
    needs_tag_resolution: bool = False
    tag_resolution: Dict[str, Any] = Field(default_factory=dict)

    counters: RetrievalCountersModel = Field(default_factory=RetrievalCountersModel)
    kb_insufficient: bool = False
    max_retrievals: int = 3


class StyleStateModel(BaseModel):
    ruff_json: Any = Field(default_factory=list)
    ruff_format_diff: str = ""
    mypy_stdout: str = ""
    pep_context: List[Dict[str, Any]] = Field(default_factory=list)


class FollowupStateModel(BaseModel):
    has_followup_question: bool = False
    followup_question: str = ""
    followup_questions: List[str] = Field(default_factory=list)
    followup_intent: Optional[Literal["correctness", "efficiency", "style", "meta"]] = None


class SessionStateModel(BaseModel):
    hint_level: int = 0
    attempts: List[Dict[str, str]] = Field(default_factory=list)  # {"code": str, "notes": str}
    last_feedback: str = ""
    done: bool = False
    retry: bool = False
    force_review: bool = False

    # preserved knobs from your old code
    python_version: str = "3.11"
    mypy_strict: bool = False
    mypy_ignore_missing_imports: bool = False


# -------------------------
# LangGraph state (grouped blobs)
# -------------------------

class CoachState(TypedDict, total=False):
    problem: Dict[str, Any]
    retrieval: Dict[str, Any]
    style: Dict[str, Any]
    followup: Dict[str, Any]
    session: Dict[str, Any]

    # internal handles (not serialized across processes)
    _vs: Any
