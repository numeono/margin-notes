from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Document(StrictModel):
    id: str = Field(min_length=1, max_length=160, pattern=r"^[\w.-]+$")
    title: str = Field(min_length=1, max_length=300)
    text: str = Field(min_length=1, max_length=500_000)
    source: str = Field(min_length=1, max_length=500)
    collection: str = Field(default="demo", min_length=1, max_length=100, pattern=r"^[\w.-]+$")


class Query(StrictModel):
    question: str = Field(min_length=1, max_length=2000)
    collection: str = Field(default="demo", min_length=1, max_length=100)
    top_k: int = Field(default=3, ge=1, le=20)
    mode: Literal["lexical", "hybrid"] = "lexical"
    answerer: Literal["extractive", "ollama"] = "extractive"


class Hit(StrictModel):
    chunk_id: str
    document_id: str
    title: str
    source: str
    text: str
    start: int
    end: int
    score: float
    lexical_overlap: float
    cosine_similarity: float | None = None


class Citation(StrictModel):
    document_id: str
    chunk_id: str
    source: str
    quote: str
    start: int
    end: int


class Answer(StrictModel):
    answer: str
    abstained: bool
    citations: list[Citation]
    retrieved: list[Hit]
    latency_ms: float
    mode: str
    answerer: str
