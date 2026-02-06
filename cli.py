from __future__ import annotations

from typing import List

from langgraph.types import Command

from graph import build_graph
from state_models import CoachState, FollowupStateModel, ProblemStateModel, RetrievalStateModel, SessionStateModel, StyleStateModel
from utils.logging import configure_logging, log_stage
from utils.text_format import format_cli_feedback
import argparse

def main() -> None:
    parser = argparse.ArgumentParser(prog="coding-coach")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    configure_logging(debug=args.debug)

    graph = build_graph()
    thread_id = "python-coach-thread-1"
    config = {"configurable": {"thread_id": thread_id}}

    log_stage("main", "graph compiled", {"thread_id": thread_id})

    print("Enter problem description; finish with a line containing only: ```END```")
    desc_lines: List[str] = []
    while True:
        line = input()
        if line.strip() == "```END```":
            break
        desc_lines.append(line)

    prob = ProblemStateModel(
        user_question_title=input("Enter problem (e.g. 'leetcode 1' or paste prompt): ").strip(),
        user_question_description="\n".join(desc_lines).strip(),
    )

    init: CoachState = {
        "problem": prob.model_dump(),
        "retrieval": RetrievalStateModel().model_dump(),
        "style": StyleStateModel().model_dump(),
        "followup": FollowupStateModel().model_dump(),
        "session": SessionStateModel().model_dump(),
    }

    log_stage("main", "initial state", {"title": prob.user_question_title, "desc_len": len(prob.user_question_description)})

    result = graph.invoke(init, config=config)

    while "__interrupt__" in result:
        intr = result["__interrupt__"][0].value

        message = (intr.get("message") or "").strip()
        if message:
            print(f"\n{message}\n")

        # Debug-only: rich metadata
        log_stage(
            "interrupt",
            "coach request",
            {
                "message": intr.get("message"),
                "hint_level": intr.get("hint_level", 0),
                "expected_schema": intr.get("expected_schema", {}),
            },
        )
        current_hint_level = intr.get("hint_level", 0)

        expected = intr.get("expected_schema", {}) or {}
        if "command" in expected:
            command = input(f"Current hint level: {current_hint_level}\nCommand (/done, /answer, /retry, /hint N, or a question): ").strip()
            payload = {"command": command}
        else:
            print("\nPaste Python code, finish with a line containing only: ```END```")
            lines: List[str] = []
            while True:
                line = input()
                if line.strip() == "```END```":
                    break
                lines.append(line)

            notes = input("Notes (short explanation): ").strip()
            payload = {"code": "\n".join(lines), "notes": notes}

        result = graph.invoke(Command(resume=payload), config=config)

        sess = SessionStateModel.model_validate(result.get("session") or {})
        if sess.last_feedback:
            formatted = format_cli_feedback(sess.last_feedback)

            print("\n" + formatted + "\n")

            # Debug-only trace
            log_stage("feedback_debug", None, {"chars": len(sess.last_feedback)})

    log_stage("main", "session complete")


if __name__ == "__main__":
    main()
