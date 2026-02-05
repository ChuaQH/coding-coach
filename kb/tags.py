from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple

from helpers.fuzzy_helper import RapidFuzzTagResolver, tag_processor
from helpers.tags_helper import TagVocabStore


# Load vocab once
VOCAB_TAGS: List[str] = TagVocabStore().list_tags()
VOCAB_SET: Set[str] = set(VOCAB_TAGS)

# Resolver instance (reuse)
TAG_RESOLVER = RapidFuzzTagResolver(VOCAB_TAGS)


def normalize_to_vocab_exact(tag: str) -> List[str]:
    """
    Exact existence check that also supports processor collisions:
    if multiple vocab tags normalize to same form, return all of them.
    """
    _ = tag_processor(tag)  # keep normalization consistent with resolver behavior
    r = TAG_RESOLVER.resolve_one(tag, min_score=100.0)  # exact only
    if r.get("status") == "exact":
        return r.get("chosen") or []
    return [tag] if tag in VOCAB_SET else []


def tag_key(tag: str) -> str:
    return f"tag__{tag}"


def build_must_filter(must_tags: List[str]) -> Dict[str, Any] | None:
    must_tags = [t for t in (must_tags or []) if t]
    if not must_tags:
        return None
    clauses = [{tag_key(t): True} for t in must_tags]
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def expand_exact(tags: List[str]) -> Tuple[List[str], List[str]]:
    expanded: List[str] = []
    missing: List[str] = []
    for t in tags or []:
        exacts = normalize_to_vocab_exact(t)
        if exacts:
            expanded.extend(exacts)
        else:
            missing.append(t)
    expanded = list(dict.fromkeys(expanded))
    return expanded, missing
