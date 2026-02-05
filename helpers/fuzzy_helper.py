from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple
from rapidfuzz import fuzz, process

def tag_processor(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s)
    return s

class TagCandidate:
    tag: str
    score: float

class RapidFuzzTagResolver:
    def __init__(self, vocab_tags: List[str]) -> None:
        self.vocab_tags = list(vocab_tags)

        self._proc_to_canonicals: Dict[str, List[str]] = {}
        for t in self.vocab_tags:
            p = tag_processor(t)
            self._proc_to_canonicals.setdefault(p, []).append(t)

    def resolve_one(
        self,
        proposed: str,
        *,
        top_n: int = 5,
        min_score: float = 85.0,
        include_close_delta: float = 1.0,
        close_min_score: float = 95.0,
        scorer=fuzz.WRatio,
    ) -> Dict[str, Any]:
        p = tag_processor(proposed)
        if not p:
            return {"proposed": proposed, "status": "unresolved", "chosen": [], "candidates": []}

        # Exact match by normalized form -> return ALL canonicals with same processed form
        if p in self._proc_to_canonicals:
            canonicals = self._proc_to_canonicals[p]
            # Remove all duplicates
            canonicals = list(dict.fromkeys(canonicals))
            return {
                "proposed": proposed,
                "status": "exact",
                "chosen": canonicals,
                "candidates": [{"tag": t, "score": 100.0} for t in canonicals],
            }

        matches: List[Tuple[str, float, int]] = process.extract(
            proposed,
            self.vocab_tags,
            scorer=scorer,
            processor=tag_processor,
            limit=max(top_n, 10),
        )

        # Filter by min_score
        filtered = [(m[0], float(m[1])) for m in matches if float(m[1]) >= min_score]
        if not filtered:
            return {"proposed": proposed, "status": "unresolved", "chosen": [], "candidates": []}

        # Sort by score desc, stable
        filtered.sort(key=lambda x: x[1], reverse=True)

        top_score = filtered[0][1]
        chosen = []
        candidates = [{"tag": t, "score": s} for t, s in filtered[:top_n]]

        # Always choose the top match
        chosen.append(filtered[0][0])

        # If “extremely close”, include additional near-ties (your boolean algebra case)
        if top_score >= close_min_score:
            for t, s in filtered[1:]:
                if (top_score - s) <= include_close_delta:
                    chosen.append(t)
                else:
                    break

        # De-dupe chosen, preserve order
        chosen = list(dict.fromkeys(chosen))

        return {
            "proposed": proposed,
            "status": "fuzzy",
            "chosen": chosen,
            "candidates": candidates,
        }

    def resolve_many(
        self,
        proposed_tags: List[str],
        *,
        top_n: int = 5,
        min_score: float = 85.0,
        include_close_delta: float = 1.0,
        close_min_score: float = 95.0,
        scorer=fuzz.WRatio,
    ) -> Dict[str, Any]:
        resolution = {}
        for t in proposed_tags or []:
            resolution[t] = self.resolve_one(
                t,
                top_n=top_n,
                min_score=min_score,
                include_close_delta=include_close_delta,
                close_min_score=close_min_score,
                scorer=scorer,
            )

        return {
            "proposed": proposed_tags,
            "resolution": resolution,
            "unresolved": [k for k, v in resolution.items() if not v["chosen"]],
            "ambiguous": [k for k, v in resolution.items() if len(v["chosen"]) > 1],
        }
