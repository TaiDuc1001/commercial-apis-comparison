# RAG QA API Evaluation

Standalone `uv` subproject for the 50-task grounded RAG QA evaluation. It does not import the parent RAG app.

## Setup

```bash
uv sync
copy .env.example .env
```

Set `OPENROUTER_API_KEY` in `.env` or in your shell before running network steps.

## Workflow

```bash
uv run python run_test.py
uv run python run_full.py
```

`run_test.py` uses only the cheap GPT model and the first 5 tasks.

`run_full.py` uses all 50 RAG tasks and the six configured models.

Each run writes exactly two output files under `outputs/`:

- `test_results.csv` / `test_results.json`
- `full_results.csv` / `full_results.json`
