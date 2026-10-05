FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir . && useradd --create-home appuser && mkdir /data && chown appuser /data
COPY examples ./examples
ENV MARGIN_DB=/data/notes.sqlite
USER appuser
RUN margin-notes --db /data/notes.sqlite ingest examples/documents.jsonl
EXPOSE 8000
CMD ["uvicorn", "margin_notes.api:app", "--host", "0.0.0.0", "--port", "8000"]

