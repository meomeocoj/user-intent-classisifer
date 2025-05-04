# Query Router Service

A high-performance HTTP router service that classifies incoming queries into three depth labels: simple (closed-book/FAQ), semantic (needs retrieval), and agent (multi-step research).

---

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Setup & Installation](#setup--installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Specification](#api-specification)
- [Demo Script](#demo-script)
- [Performance & Accuracy](#performance--accuracy)
- [Developer Notes](#developer-notes)
- [License](#license)

---

## Features
- Fast query classification using configurable transformer models (default: BART-MNLI or mDeBERTa)
- Safety checks with Llama Prompt Guard
- LLM-based fallback routing
- High-performance async API with FastAPI
- Comprehensive test coverage
- Docker support for local development and production
- Performance and accuracy evaluation harnesses

## Requirements
- Python 3.11 or higher
- Poetry or pip for dependency management
- Docker (optional)

## Setup & Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd query-router
   ```
2. Install dependencies:
   ```bash
   pip install .
   # or for development
   pip install .[dev]
   ```
3. (Optional) Install extra tools for performance/accuracy testing:
   ```bash
   pip install requests pandas scikit-learn numpy psutil
   ```

## Configuration
- Main config: `config/config.yaml`
- Environment variables can override config values (see `.env` for examples)
- Key options:
  - `models.classifier.name`: Model name for classification (e.g., BART, mDeBERTa)
  - `models.llm_router.model`: LLM fallback model (e.g., gpt-4o)
  - `models.prompt_guard.name`: Safety checker model
  - `logging.*`: Logging format and options
  - `server.*`: Host, port, workers

## Usage
1. Start the server:
   ```bash
   uvicorn src.main:app --reload
   ```
2. Make a request:
   ```bash
   curl -X POST http://localhost:8000/api/v1/route \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the capital of France?", "history": []}'
   ```

## API Specification
### POST /api/v1/route
Request body:
```json
{
  "query": "string",
  "history": ["string"]  // optional
}
```
Response:
```json
{
  "route": "simple | semantic | agent",
  "confidence": 0.0-1.0,
  "trace_id": "UUID",
  "model": "classifier model name"
}
```

## Demo Script
Run the demo script to see example queries and responses:
```bash
python scripts/demo_router.py --api-url http://localhost:8000/api/v1/route
```
This will send one query from each category (simple, semantic, agent) and print the results.

## Performance & Accuracy
- Use `scripts/perf_test.py` to measure latency and throughput:
  ```bash
  python scripts/perf_test.py --api-url http://localhost:8000/api/v1/route --requests 100 --concurrency 1
  ```
- Use `scripts/eval_harness.py` with your eval set to measure accuracy:
  ```bash
  python scripts/eval_harness.py --api-url http://localhost:8000/api/v1/route --csv data/eval_set.csv
  ```
- Target: p95 latency ≤ 250 ms, accuracy ≥ 95% on evaluation set

## Developer Notes
- Extend or modify the router by editing `src/query_router/services/router_service.py` and related model/service files.
- Add new models or fallback logic by updating config and implementing new classes in `src/query_router/models/`.
- Run tests with:
  ```bash
  pytest
  ```
- Lint and type-check:
  ```bash
  black .
  isort .
  mypy .
  ruff check .
  ```
- See `scripts/` for additional utilities and harnesses.

## License
MIT 