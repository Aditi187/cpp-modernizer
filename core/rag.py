from __future__ import annotations

import hashlib
import importlib
import os
import threading
from dataclasses import dataclass
from typing import Any


@dataclass
class _StoredDoc:
    doc_id: str
    code: str
    metadata: dict[str, Any]


class CodeRAG:
    """Optional retrieval layer for code snippets.

    Backend priority:
    1) ChromaDB + sentence-transformers embeddings
    2) Lightweight lexical fallback in memory
    """

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = bool(enabled)
        self._lock = threading.Lock()
        self._documents: dict[str, _StoredDoc] = {}

        self._chroma_collection = None
        self._embedder = None
        if not self.enabled:
            return

        model_name = os.environ.get("RAG_EMBED_MODEL", "all-MiniLM-L6-v2").strip() or "all-MiniLM-L6-v2"
        persist_dir = os.environ.get("RAG_PERSIST_DIR", os.path.join(os.getcwd(), "cache", "rag_chroma")).strip()

        try:
            chromadb = importlib.import_module("chromadb")
            sentence_transformers = importlib.import_module("sentence_transformers")
            sentence_transformer_cls = getattr(sentence_transformers, "SentenceTransformer", None)
            if sentence_transformer_cls is None:
                raise RuntimeError("SentenceTransformer class not available")

            self._embedder = sentence_transformer_cls(model_name)
            client = chromadb.PersistentClient(path=persist_dir)
            self._chroma_collection = client.get_or_create_collection("code_chunks")
        except Exception:
            self._embedder = None
            self._chroma_collection = None

    @staticmethod
    def _stable_id(code: str, metadata: dict[str, Any]) -> str:
        payload = f"{code}|{sorted((metadata or {}).items())}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def add_document(self, code: str, metadata: dict[str, Any] | None = None) -> None:
        if not self.enabled:
            return
        text = str(code or "").strip()
        if not text:
            return

        meta = dict(metadata or {})
        doc_id = self._stable_id(text, meta)
        with self._lock:
            self._documents[doc_id] = _StoredDoc(doc_id=doc_id, code=text, metadata=meta)

        if self._chroma_collection is None or self._embedder is None:
            return

        try:
            embedding = self._embedder.encode([text])[0].tolist()
            self._chroma_collection.upsert(
                ids=[doc_id],
                documents=[text],
                metadatas=[meta],
                embeddings=[embedding],
            )
        except Exception:
            return

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        if not self.enabled:
            return []

        top_k = max(1, int(k))
        query_text = str(query or "").strip()
        if not query_text:
            return []

        if self._chroma_collection is not None and self._embedder is not None:
            try:
                query_embedding = self._embedder.encode([query_text])[0].tolist()
                result = self._chroma_collection.query(query_embeddings=[query_embedding], n_results=top_k)
                docs = (result.get("documents") or [[]])[0]
                metas = (result.get("metadatas") or [[]])[0]
                out: list[dict[str, Any]] = []
                for idx, doc in enumerate(docs):
                    out.append(
                        {
                            "code": str(doc or ""),
                            "metadata": metas[idx] if idx < len(metas) and isinstance(metas[idx], dict) else {},
                        }
                    )
                return out
            except Exception:
                pass

        # Lexical fallback when embedding backend is unavailable.
        query_tokens = set(query_text.lower().split())
        scored: list[tuple[float, _StoredDoc]] = []
        with self._lock:
            docs = list(self._documents.values())
        for item in docs:
            code_tokens = set(item.code.lower().split())
            if not code_tokens:
                continue
            overlap = len(query_tokens & code_tokens)
            denom = max(1, len(query_tokens | code_tokens))
            score = overlap / denom
            if score > 0:
                scored.append((score, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)

        return [
            {"code": item.code, "metadata": item.metadata}
            for _score, item in scored[:top_k]
        ]


_GLOBAL_RAG: CodeRAG | None = None
_GLOBAL_RAG_LOCK = threading.Lock()


def get_global_rag(enabled: bool = False) -> CodeRAG | None:
    """Return a process-wide RAG instance when enabled, otherwise None."""
    if not enabled:
        return None
    global _GLOBAL_RAG
    with _GLOBAL_RAG_LOCK:
        if _GLOBAL_RAG is None:
            _GLOBAL_RAG = CodeRAG(enabled=True)
        return _GLOBAL_RAG
