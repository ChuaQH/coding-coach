from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List


class TagVocabStore:
    def __init__(self, tag_vocab_path: str | Path = "./chroma_db/tag_vocab.json") -> None:
        self.path = Path(tag_vocab_path)
        self.vocab: Dict[str, Dict[str, Any]] = {}

    def load(self) -> Dict[str, Dict[str, Any]]:
        if not self.path.exists():
            self.vocab = {}
            return self.vocab
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.vocab = data if isinstance(data, dict) else {}
                return self.vocab
        except (json.JSONDecodeError, OSError):
            self.vocab = {}
            return self.vocab

    def list_tags(self) -> List[str]:
        if not self.vocab:
            self.load()
        return sorted(self.vocab.keys())

    def get_tag_examples(self, tag_name: str) -> List[str]:
        if not self.vocab:
            self.load()
        entry = self.vocab.get(tag_name) or {}
        examples = entry.get("examples") or []
        return list(examples)

# vocab_store = TagVocabStore()
# vocab_store.load()
# print(vocab_store.list_tags())