import json
import os
from pathlib import Path

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from margin_notes.answer import answer, citation, select_ollama
from margin_notes.api import create_app
from margin_notes.index import Index, MiniLM, chunks
from margin_notes.models import Document, Query

EXAMPLES = Path(__file__).parents[1] / "examples/documents.jsonl"


@pytest.fixture
def index(tmp_path):
    db = Index(tmp_path / "test.sqlite")
    for line in EXAMPLES.read_text().strip().splitlines():
        db.ingest(Document.model_validate_json(line))
    return db


def test_exact_unicode_offsets_and_overlap():
    text = "  Café\n" + "reading naïve notes. " * 110
    windows = list(chunks(text))
    assert len(windows) > 1
    assert all(text[start:end] == chunk for start, end, chunk in windows)
    assert windows[1][0] < windows[0][1]
    assert windows[-1][1] == len(text.rstrip())


def test_ingest_is_idempotent_and_update_removes_stale_fts(index):
    doc = Document(id="test", title="Test", text="Galactic marmalade procedure.", source="test")
    assert index.ingest(doc)
    before = index.stats()
    assert not index.ingest(doc)
    assert before == index.stats()
    assert index.search("marmalade", "demo")[0].document_id == "test"
    assert index.ingest(doc.model_copy(update={"text": "Updated pineapple procedure."}))
    assert not index.search("marmalade", "demo")
    assert index.search("pineapple", "demo")[0].text == "Updated pineapple procedure."


def test_delete_removes_chunks_and_fts(index):
    index.delete("demo", "travel")
    assert not index.search("reimbursements", "demo")
    assert index.stats()["documents"] == 9


def test_collection_isolation_and_same_document_ids(index):
    assert not index.search("987654 payroll", "demo")
    assert index.search("987654 payroll", "restricted")[0].document_id == "private-payroll"
    assert "90 days" in index.search("access requests expire", "demo")[0].text
    assert "seven days" in index.search("access requests expire", "other")[0].text


@pytest.mark.parametrize(
    "question", ['" OR * NOT (', "the a is", "🦄", "x' UNION SELECT * FROM documents;"]
)
def test_query_syntax_is_data(index, question):
    assert isinstance(index.search(question, "demo"), list)
    assert index.stats()["documents"] == 10


def test_answer_has_verifiable_original_document_span(index):
    result = answer(index, Query(question="When must travel reimbursements be submitted?"))
    assert not result.abstained
    assert "30 days" in result.answer
    docs = {d["id"]: d for d in map(json.loads, EXAMPLES.read_text().strip().splitlines())}
    for c in result.citations:
        assert docs[c.document_id]["text"][c.start : c.end] == c.quote
    assert result.answer == result.citations[0].quote


def test_unknown_question_abstains(index):
    result = answer(index, Query(question="What is the unicorn breeding schedule?"))
    assert result.abstained and not result.answer and not result.citations


def test_provider_rejects_fabricated_quotes_and_chunk_ids(index, monkeypatch):
    hits = index.search("travel reimbursements", "demo")
    with pytest.raises(ValueError, match="verbatim"):
        citation(hits[0], "They are due in 900 days.")
    for selection in [
        {"chunk_id": "invented", "quote": "anything"},
        {"chunk_id": hits[0].chunk_id, "quote": "false evidence"},
    ]:

        def post(*args, selection=selection, **kwargs):
            return httpx.Response(
                200,
                json={"response": json.dumps({"selections": [selection]})},
                request=httpx.Request("POST", "http://test/api/generate"),
            )

        monkeypatch.setattr(httpx, "post", post)
        with pytest.raises(ValueError):
            select_ollama("travel", hits, "http://test", "test-model")


def test_provider_accepts_exact_quote(index, monkeypatch):
    hits = index.search("travel reimbursements", "demo")
    quote = "Travel reimbursements must be submitted within 30 days of the trip ending."

    def post(*args, **kwargs):
        assert kwargs["json"]["stream"] is False
        assert "untrusted" in kwargs["json"]["system"]
        return httpx.Response(
            200,
            json={
                "response": json.dumps(
                    {"selections": [{"chunk_id": hits[0].chunk_id, "quote": quote}]}
                )
            },
            request=httpx.Request("POST", "http://test/api/generate"),
        )

    monkeypatch.setattr(httpx, "post", post)
    assert select_ollama("travel", hits, "http://test", "model")[0].quote == quote


def test_api_validation_health_and_provider_error(index, monkeypatch):
    with TestClient(create_app(index)) as client:
        assert client.get("/health").json()["documents"] == 10
        response = client.post("/query", json={"question": "travel reimbursements"})
        assert response.status_code == 200
        assert "Server-Timing" in response.headers
        assert client.post("/query", json={"question": "x", "top_k": 100}).status_code == 422
        assert (
            client.post("/query", json={"question": "x", "url": "http://other"}).status_code == 422
        )
        assert client.post("/query", json={"question": "x", "mode": "hybrid"}).status_code == 422
        monkeypatch.setenv("MARGIN_OLLAMA_MODEL", "test")

        def fail(*args, **kwargs):
            raise httpx.ConnectError("down")

        monkeypatch.setattr(httpx, "post", fail)
        assert (
            client.post(
                "/query", json={"question": "travel reimbursements", "answerer": "ollama"}
            ).status_code
            == 502
        )


class FakeEmbedder:
    identity = "test-vectors-v1"

    def encode(self, texts):
        return np.array([[1.0, 0.0] if "travel" in t.lower() else [0.0, 1.0] for t in texts])


def test_hybrid_filters_before_ranking_and_checks_model(tmp_path):
    index = Index(tmp_path / "vectors.sqlite", FakeEmbedder())
    index.ingest(
        Document(
            id="x", title="Private", text="Travel private facts.", source="x", collection="private"
        )
    )
    index.ingest(Document(id="x", title="Public", text="Other public facts.", source="x"))
    assert not index.search("travel", "demo", mode="hybrid")
    assert index.search("travel", "private", mode="hybrid")[0].cosine_similarity == pytest.approx(1)
    index.embedder.identity = "different-model"
    with pytest.raises(ValueError, match="Re-ingest"):
        index.search("travel", "demo", mode="hybrid")


def test_embedding_failure_preserves_previous_document(tmp_path):
    encoder = FakeEmbedder()
    index = Index(tmp_path / "atomic.sqlite", encoder)
    doc = Document(id="x", title="Travel", text="Travel original.", source="x")
    index.ingest(doc)

    def fail(texts):
        raise RuntimeError("encoder unavailable")

    encoder.encode = fail
    with pytest.raises(RuntimeError):
        index.ingest(doc.model_copy(update={"text": "Travel replacement."}))
    assert index.search("original", "demo")[0].text == "Travel original."


@pytest.mark.semantic
@pytest.mark.skipif(os.getenv("RUN_SEMANTIC") != "1", reason="Opt-in real model download")
def test_real_minilm_retrieves_paraphrase(tmp_path):
    index = Index(tmp_path / "semantic.sqlite", MiniLM())
    for line in EXAMPLES.read_text().strip().splitlines():
        index.ingest(Document.model_validate_json(line))
    hits = index.search("How frequently are copies of the database saved?", "demo", mode="hybrid")
    assert hits[0].document_id == "backups"
    assert hits[0].cosine_similarity > 0.25
    assert all(h.document_id != "private-payroll" for h in hits)
