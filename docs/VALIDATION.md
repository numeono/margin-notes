# Validation

Run on October 5, 2026, on an Apple M1 Mac mini using Python 3.13.15.

## Executed checks

- `ruff check src tests`: passed.
- `RUN_SEMANTIC=1 pytest -q`: 16 passed, including a real CPU MiniLM download/inference test.
- Real HTTP ingestion/query workflow with ten synthetic documents across three collections.
- Both lexical and hybrid API modes passed the companion 15-case Evidence Gate smoke suite.
- Updating documents removes stale FTS entries; deletion, exact Unicode offsets, provider
  errors, collection filtering, and fabricated model citations have automated tests.
- The local LLM selector is contract-tested with mocked Ollama responses. No live Ollama
  model was available during this run; live model-selection quality is not verified.

## What the harder checks show

The easy smoke fixture is not a retrieval benchmark. A separate seven-question challenge
set includes paraphrases, a retention exception, access negation, and an unsupported question.
Neither mode passes that suite's quality gates:

| Metric | Lexical | Hybrid |
| --- | ---: | ---: |
| Recall@3 | 0.667 | 0.667 |
| Answer phrase coverage | 0.500 | 0.667 |
| Abstention accuracy | 0.571 | 0.714 |
| Collection integrity | 1.000 | 1.000 |

These are observed fixture results, not held-out or production estimates. Hybrid retrieval
helps one paraphrase reach an answer, but conservative abstention and the simple sentence
selector still miss valid answers. The next useful experiment is separately calibrated
abstention and a reranker, evaluated on newly labeled data. No thresholds were changed after
seeing these challenge results.

Raw predictions and reports are in
[Evidence Gate's results](https://github.com/numeono/evidence-gate/tree/main/docs/results).
Its quote-integrity aggregate gives missing citations zero for an answerable case, so a low
aggregate can mean refusal to answer rather than a fabricated quote. Inspect per-case reports.

The Docker build and runtime smoke test are exercised by the GitHub Actions container job.
See the [latest workflow](https://github.com/numeono/margin-notes/actions/workflows/ci.yml)
for the actual status. There was no local Docker daemon on the development host.

## Reproduction

Follow the README for installation, tests, and API startup. The exact packages used locally
are recorded in `environment-py313.txt`; it is an environment record, not a portable lockfile.
Model weights are pinned to revision `c9745ed1d9f207416be6d2e6f8de32d1f16199bf`.
The default tests have one upstream Starlette/httpx deprecation warning on the recorded
environment; it does not affect the passing assertions.

This prototype has no production deployment, measured user adoption, cloud scaling evidence,
or production accuracy claim. It implements and tests a bounded document-retrieval workflow.

