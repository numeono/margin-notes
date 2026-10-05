import logging
import os
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request

from .answer import answer
from .index import Index, MiniLM
from .models import Answer, Query

logger = logging.getLogger("margin_notes")


def create_app(index: Index | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        app.state.index = index or Index(
            os.getenv("MARGIN_DB", "data/notes.sqlite"),
            MiniLM() if os.getenv("MARGIN_SEMANTIC") == "1" else None,
        )
        yield

    app = FastAPI(
        title="Margin Notes",
        version="0.1.0",
        lifespan=lifespan,
        description="Local evidence retrieval prototype. Ingest through the CLI.",
    )

    @app.middleware("http")
    async def timings(request: Request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - started) * 1000
        response.headers["Server-Timing"] = f"app;dur={elapsed:.2f}"
        # No prompts, evidence, request bodies, or arbitrary URL paths in logs.
        logger.info(
            "request method=%s status=%s elapsed_ms=%.2f",
            request.method,
            response.status_code,
            elapsed,
        )
        return response

    @app.get("/health")
    def health():
        return {"status": "ok", **app.state.index.stats()}

    @app.post("/query", response_model=Answer)
    def query(body: Query):
        try:
            return answer(
                app.state.index,
                body,
                os.getenv("MARGIN_OLLAMA_URL", "http://127.0.0.1:11434"),
                os.getenv("MARGIN_OLLAMA_MODEL"),
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except (httpx.HTTPError, KeyError) as exc:
            raise HTTPException(
                status_code=502, detail="Answer provider unavailable or invalid response"
            ) from exc

    return app


app = create_app()
