from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from helpers.fuzzy_helper import RapidFuzzTagResolver, tag_processor
from helpers.tags_helper import TagVocabStore


# Load vocab once
VOCAB_TAGS: List[str] = TagVocabStore().list_tags()
VOCAB_SET: Set[str] = set(VOCAB_TAGS)

# Resolver instance (reuse)
TAG_RESOLVER = RapidFuzzTagResolver(VOCAB_TAGS)

_TAG_COUNTS: Dict[str, int] | None = None


def load_tag_counts(tag_vocab_path: str | Path = "./chroma_db/tag_vocab.json") -> Dict[str, int]:
    global _TAG_COUNTS
    if _TAG_COUNTS is not None:
        return _TAG_COUNTS

    vocab_path = Path(tag_vocab_path)
    counts: Dict[str, int] = {}
    if vocab_path.exists():
        try:
            with vocab_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            data = {}
        if isinstance(data, dict):
            for key, entry in data.items():
                if not isinstance(key, str):
                    continue
                if isinstance(entry, dict) and isinstance(entry.get("count"), int):
                    counts[key] = entry["count"]

    _TAG_COUNTS = counts
    return counts


def normalize_tag(tag: str) -> str:
    t = tag.strip().lower()
    t = re.sub(r"[\s-]+", "_", t)
    t = re.sub(r"_+", "_", t)
    t = re.sub(r"[^a-z0-9_]+", "", t)
    return t.strip("_")


def singularize_token(token: str) -> str:
    if len(token) <= 3:
        return token
    if token.endswith("ss"):
        return token
    if token.endswith("ies") and len(token) > 3:
        return token[:-3] + "y"
    if token.endswith("ses") and len(token) > 3:
        return token[:-2]
    if token.endswith("s") and not token.endswith(("us", "is")):
        return token[:-1]
    return token


def pluralize_token(token: str) -> str:
    if len(token) <= 3:
        return token
    if token.endswith("y") and len(token) > 1:
        return token[:-1] + "ies"
    if token.endswith(("s", "x", "z", "ch", "sh")):
        return token + "es"
    if token.endswith(("us", "is")):
        return token
    return token + "s"


def canonicalize_tag(tag: str, counts: Dict[str, int] | None = None) -> str:
    normalized = normalize_tag(tag)
    if not normalized:
        return normalized
    tokens = normalized.split("_")
    if not tokens:
        return normalized

    if counts is None:
        counts = load_tag_counts()

    last = singularize_token(tokens[-1])
    singular = "_".join(tokens[:-1] + [last]) if last else normalized
    plural = "_".join(tokens[:-1] + [pluralize_token(last)]) if last else normalized

    candidates = [normalized, singular, plural]
    candidates = [c for c in candidates if c]
    best = normalized
    best_count = counts.get(best, 0)
    for cand in candidates:
        cand_count = counts.get(cand, 0)
        if cand_count > best_count:
            best = cand
            best_count = cand_count

    return best if best in VOCAB_SET else normalized


def canonicalize_tags(tags: List[str]) -> List[str]:
    counts = load_tag_counts()
    seen = set()
    out: List[str] = []
    for tag in tags:
        canonical = canonicalize_tag(tag, counts=counts)
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)
        out.append(canonical)
    return out


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
