from __future__ import annotations

import os
import pickle
from typing import List

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


IN_MEMORY_STORE_PATH = "./chroma_cache/in_memory_vector_store.pkl"
_in_memory_vector_store: InMemoryVectorStore | None = None


def fetch_url(url: str, timeout: int = 20) -> str:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "python-coach/0.1"})
    r.raise_for_status()
    return r.text


def pep_html_to_section_docs(html: str, base_url: str) -> list[Document]:
    header_tags = {"h1", "h2", "h3"}
    soup = BeautifulSoup(html, "html.parser")
    root = soup.find("article") or soup.body or soup

    docs: list[Document] = []
    current_headers: list[str] = []
    current_anchor: str | None = None
    buf: list[str] = []

    def flush() -> None:
        nonlocal buf, current_anchor
        text = "\n".join([t for t in buf if t.strip()]).strip()
        if not text:
            buf = []
            return

        header_path = " > ".join(current_headers) if current_headers else "PEP"
        page_content = f"{header_path}\n\n{text}"

        source_url = f"{base_url}#{current_anchor}" if current_anchor else base_url

        docs.append(
            Document(
                page_content=page_content,
                metadata={
                    "source_url": source_url,
                    "header_path": header_path,
                    "anchor": current_anchor,
                    "type": "base_best_practices",
                },
            )
        )
        buf = []

    for el in root.find_all(["h1", "h2", "h3", "p", "pre", "li"]):
        if el.name in header_tags:
            flush()
            title = el.get_text(" ", strip=True)

            if el.name == "h1":
                current_headers = [title]
            elif el.name == "h2":
                current_headers = current_headers[:1] + [title] if current_headers else [title]
            else:  # h3
                current_headers = current_headers[:2] + [title] if len(current_headers) >= 2 else current_headers + [title]

            current_anchor = el.get("id") or (el.find("a") and el.find("a").get("id"))
        else:
            t = el.get_text("\n", strip=True)
            if t:
                buf.append(t)

    flush()
    return docs


def _build_in_memory_kb() -> InMemoryVectorStore:
    embed = OllamaEmbeddings(model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"))
    vs = InMemoryVectorStore(embedding=embed)

    base_urls = [
        "https://peps.python.org/pep-0008/",
        "https://peps.python.org/pep-0257/",
    ]

    docs: List[Document] = []
    for url in base_urls:
        try:
            html = fetch_url(url)
            docs.extend(pep_html_to_section_docs(html, url))
        except Exception:
            continue

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    vs.add_documents(chunks)
    return vs


def get_in_memory_vector_store() -> InMemoryVectorStore:
    global _in_memory_vector_store
    if _in_memory_vector_store is not None:
        return _in_memory_vector_store

    os.makedirs(os.path.dirname(IN_MEMORY_STORE_PATH), exist_ok=True)

    if os.path.exists(IN_MEMORY_STORE_PATH):
        try:
            with open(IN_MEMORY_STORE_PATH, "rb") as f:
                _in_memory_vector_store = pickle.load(f)
                return _in_memory_vector_store
        except Exception:
            _in_memory_vector_store = None

    _in_memory_vector_store = _build_in_memory_kb()
    try:
        with open(IN_MEMORY_STORE_PATH, "wb") as f:
            pickle.dump(_in_memory_vector_store, f)
    except Exception:
        pass

    return _in_memory_vector_store
