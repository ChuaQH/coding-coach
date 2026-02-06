from __future__ import annotations

import os

from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings

# Initialize Ollama LLM
llm = ChatOllama(
    model=os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5-coder:14b-instruct-q4_K_S"),
    temperature=0,
)

# Initialize Chroma vector store with Ollama embeddings
vector_store = Chroma(
    collection_name="algo-kb",
    embedding_function=OllamaEmbeddings(model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")),
    persist_directory="./chroma_db",
)
