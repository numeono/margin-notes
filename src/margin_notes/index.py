"""SQLite index. Every retrieval branch filters the collection before ranking."""

import hashlib
import json
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol

import numpy as np

from .models import Document, Hit

STOPWORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "to",
    "of",
    "in",
    "on",
    "for",
    "from",
    "and",
    "or",
    "with",
    "by",
    "what",
    "when",
    "where",
    "who",
    "why",
    "how",
    "do",
    "does",
    "did",
    "can",
    "could",
    "would",
    "should",
    "it",
    "its",
    "i",
    "we",
    "our",
    "my",
    "your",
    "you",
    "their",
    "they",
    "this",
    "that",
    "these",
    "those",
    "have",
    "has",
    "about",
    "much",
    "many",
}
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Pin the model weights, not just the model name. Apache-2.0 model.
MODEL_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"


def terms(text: str) -> set[str]:
    return {t for t in re.findall(r"\w+", text.lower()) if t not in STOPWORDS}


def chunks(text: str, size: int = 100, overlap: int = 20):
    """Overlapping word windows, retaining exact character offsets into the original."""
    if size < 1 or not 0 <= overlap < size:
        raise ValueError("Require size > overlap >= 0")
    words = list(re.finditer(r"\S+", text))
    for i in range(0, len(words), size - overlap):
        start, end = words[i].start(), words[min(i + size, len(words)) - 1].end()
        yield start, end, text[start:end]
        if i + size >= len(words):
            break


class Embedder(Protocol):
    identity: str

    def encode(self, texts: list[str]) -> np.ndarray: ...


class MiniLM:
    def __init__(self):
        from sentence_transformers import SentenceTransformer

        self.identity = f"{MODEL}@{MODEL_REVISION}"
        self.model = SentenceTransformer(MODEL, revision=MODEL_REVISION, device="cpu")

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


class Index:
    def __init__(self, path: str | Path, embedder: Embedder | None = None):
        self.path = str(path)
        self.embedder = embedder
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS documents (
                    collection TEXT, id TEXT, fingerprint TEXT NOT NULL,
                    PRIMARY KEY (collection, id)
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY, collection TEXT NOT NULL, document_id TEXT NOT NULL,
                    title TEXT, source TEXT, text TEXT, start INTEGER, end INTEGER,
                    embedding TEXT, model TEXT
                );
                CREATE INDEX IF NOT EXISTS collection_idx ON chunks(collection, document_id);
                CREATE VIRTUAL TABLE IF NOT EXISTS search USING fts5(
                    chunk_id UNINDEXED, text, tokenize='unicode61'
                );
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def ingest(self, doc: Document) -> bool:
        model = self.embedder.identity if self.embedder else "lexical"
        fingerprint = hashlib.sha256((doc.model_dump_json() + model).encode()).hexdigest()
        with self.connect() as db:
            old = db.execute(
                "SELECT fingerprint FROM documents WHERE collection=? AND id=?",
                (doc.collection, doc.id),
            ).fetchone()
            if old and old[0] == fingerprint:
                return False
        windows = list(chunks(doc.text))
        if not windows:
            raise ValueError("Document must contain non-whitespace text")
        vectors = self.embedder.encode([w[2] for w in windows]) if self.embedder else None
        # Compute embeddings before opening the write transaction. A failed encoder leaves
        # the previous version intact; replacement of documents/chunks/FTS is atomic.
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._delete(db, doc.collection, doc.id)
            db.execute(
                "INSERT INTO documents VALUES (?,?,?)", (doc.collection, doc.id, fingerprint)
            )
            for i, (start, end, text) in enumerate(windows):
                cid = hashlib.sha256(
                    f"{doc.collection}:{doc.id}:{fingerprint}:{start}".encode()
                ).hexdigest()[:24]
                vector = json.dumps(vectors[i].tolist()) if vectors is not None else None
                db.execute(
                    "INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (
                        cid,
                        doc.collection,
                        doc.id,
                        doc.title,
                        doc.source,
                        text,
                        start,
                        end,
                        vector,
                        model,
                    ),
                )
                db.execute("INSERT INTO search(chunk_id,text) VALUES (?,?)", (cid, text))
        return True

    @staticmethod
    def _delete(db, collection: str, document_id: str):
        db.execute(
            "DELETE FROM search WHERE chunk_id IN "
            "(SELECT id FROM chunks WHERE collection=? AND document_id=?)",
            (collection, document_id),
        )
        db.execute(
            "DELETE FROM chunks WHERE collection=? AND document_id=?", (collection, document_id)
        )
        db.execute("DELETE FROM documents WHERE collection=? AND id=?", (collection, document_id))

    def delete(self, collection: str, document_id: str):
        with self.connect() as db:
            self._delete(db, collection, document_id)

    def search(
        self, question: str, collection: str, top_k: int = 3, mode: str = "lexical"
    ) -> list[Hit]:
        if mode not in {"lexical", "hybrid"} or not 1 <= top_k <= 20:
            raise ValueError("Invalid search configuration")
        query_terms = terms(question)
        if not query_terms:
            return []
        # Only token strings enter MATCH, never raw user-provided FTS operators.
        match = " OR ".join('"' + token + '"' for token in sorted(query_terms))
        with self.connect() as db:
            lexical = db.execute(
                "SELECT c.*, bm25(search) AS rank FROM search "
                "JOIN chunks c ON c.id=search.chunk_id "
                "WHERE search MATCH ? AND c.collection=? "
                "ORDER BY rank, c.id LIMIT 50",
                (match, collection),
            ).fetchall()
            dense_rows = []
            if mode == "hybrid":
                if self.embedder is None:
                    raise ValueError("Hybrid search requires an embedding model")
                dense_rows = db.execute(
                    "SELECT * FROM chunks WHERE collection=?", (collection,)
                ).fetchall()
                if any(
                    r["model"] != self.embedder.identity or r["embedding"] is None
                    for r in dense_rows
                ):
                    raise ValueError(
                        "Re-ingest this collection with the configured embedding model"
                    )
        rows = {r["id"]: r for r in lexical}
        scores: dict[str, float] = {}
        cosine: dict[str, float] = {}
        for rank, row in enumerate(lexical, 1):
            scores[row["id"]] = 1 / (60 + rank)
        if dense_rows:
            query = np.asarray(self.embedder.encode([question])[0])
            vectors = np.asarray([json.loads(r["embedding"]) for r in dense_rows])
            denom = np.linalg.norm(vectors, axis=1) * np.linalg.norm(query)
            similarities = (vectors @ query) / np.maximum(denom, 1e-12)
            ranked = sorted(zip(dense_rows, similarities), key=lambda x: (-x[1], x[0]["id"]))[:50]
            for rank, (row, sim) in enumerate(ranked, 1):
                cosine[row["id"]] = float(sim)
                if sim < 0.25:
                    continue
                rows[row["id"]] = row
                scores[row["id"]] = scores.get(row["id"], 0) + 1 / (60 + rank)
        hits = []
        for cid in sorted(scores, key=lambda c: (-scores[c], c))[:top_k]:
            row = rows[cid]
            hits.append(
                Hit(
                    chunk_id=cid,
                    document_id=row["document_id"],
                    title=row["title"],
                    source=row["source"],
                    text=row["text"],
                    start=row["start"],
                    end=row["end"],
                    score=scores[cid],
                    lexical_overlap=len(query_terms & terms(row["text"])) / len(query_terms),
                    cosine_similarity=cosine.get(cid),
                )
            )
        return hits

    def stats(self):
        with self.connect() as db:
            return {
                "documents": db.execute("SELECT count(*) FROM documents").fetchone()[0],
                "chunks": db.execute("SELECT count(*) FROM chunks").fetchone()[0],
            }
