## Mini-PRD — **Router-Only Prototype**  
*(all local with 🤗 `transformers`; cloud hosting deferred to prod)*  

---

### 1. Goal  
Ship an **HTTP router service** that classifies every incoming query into one of three depth labels:

* **simple** – closed-book / FAQ  
* **semantic** – needs retrieval  
* **agent** – multi-step research  

> **Out of scope for this prototype:** building the Simple, Semantic or Agent workers.  
> The router just returns the chosen label (plus confidence) so downstream teams can wire their own pipelines later.

---

### 2. Success Criteria  

| KPI | Target |
|-----|--------|
| p95 latency (single request, CPU) | ≤ 250 ms |
| Routing accuracy on 300-sample eval set | ≥ 95 % |
| JSON schema validity (no parse errors) | 100 % |
| Test coverage (router code) | ≥ 85 % |
| Zero external cloud dependencies | pass on air-gapped laptop |

---

### 3. Functional Requirements  

#### 3.1 API  

`POST /route`

```json
{
  "query": "string",
  "history": ["..."]       // optional
}
```

Response (application/json):

```json
{
  "route": "simple | semantic | agent",
  "confidence": 0.0-1.0,
  "trace_id": "UUID"
}
```

#### 3.2 Routing Logic  

1. **Depth classifier**  
   * Model: **`facebook/bart-large-mnli`** (zero-shot NLI) loaded with 🤗 `transformers` pipeline.  
   * Prompt:  
     ```
     Hypothesis: "The user needs a <LABEL> answer."
     ```
     with `<LABEL>` ∈ {simple, semantic, agent}.  
   * If max-probability label = `simple` **and** prob ≥ 0.75 → return immediately.

2. **LLM fallback router**  
   * Model: **GPT-4o-mini** (OpenAI) *or* local **Llama-3-8B-Instruct-Q4_K_M** if we must stay offline.  
   * Few-shot prompt → JSON output `{route: …}`.  
   * Only called when classifier confidence < 0.75 or label ≠ simple.

3. **Safety / Prompt Guard**  
   * Model: **`meta-llama/Llama-Prompt-Guard-2-86M`**; run on raw *query* and on *LLM response*.  
   * If flagged **dangerous** → return `{"route":"semantic","confidence":0,"flag":"blocked"}`.

---

### 4. Non-Functional Requirements  

* **Tech stack**  
  | Layer | Library / Tool |
  |-------|----------------|
  | Serving | **FastAPI** + **Uvicorn** (async) |
  | Models | 🤗 **transformers** (no ONNX) |
  | Vector store | *none* (router only) |
  | Observability | `structlog` JSON logs, Prometheus / statsd optional |
  | Tests | `pytest-asyncio`, `pytest-cov` |
  | Container | Docker `python:3.11-slim` (for local prod parity) |

* **Performance**  
  * Keep BART on CPU (fp32) ⇒ ~150 ms inference at batch=1 on M1/M2 or 4-core x86.  
  * Parallel requests handled by Uvicorn worker pool (workers = CPU cores × 2).  

* **Config** – thresholds, model paths, OpenAI key in `config.yaml`.

* **Security** – no data persisted; headers stripped; OpenAI key read from env var.

---

### 5. Architecture Diagram  

```mermaid
flowchart TD
    A[HTTP Client] --> B[/FastAPI Gateway/]
    B --> C{{Prompt-Guard (86M)}}
    C --> D[[BART-MNLI<br>Depth Classifier]]
    D -->|simple & conf≥0.75| R[Return JSON]
    D -->|else| E[[LLM Router<br>(GPT-4o-mini<br>or Llama-3-8B)]]
    E --> F{{Prompt-Guard}}
    F --> R
```

---

### 6. Milestones & Owners  

| Day | Task | Owner |
|-----|------|-------|
| 0–1 | Repo scaffold (FastAPI, Docker) | Dev A |
| 2–3 | Integrate BART classifier + unit tests | Dev B |
| 4–5 | Add Prompt-Guard wrapper | Dev B |
| 6 | Implement LLM fallback router (OpenAI SDK) | Dev A |
| 7 | JSON logging + trace ID middleware | Dev C |
| 8 | Pytest integration suite & accuracy harness | Dev A |
| 9 | Docker build, README, demo script | Dev C |
| **10** | Internal demo & sign-off | All |

---

### 7. Acceptance Tests  

1. **Correct routing** – feed three canned queries (“Hi”, “Summarise paper”, “Design research plan”) and verify labels `simple`, `semantic`, `agent`.  
2. **Latency** – 100 serial requests on laptop ≤ 250 ms p95.  
3. **Safety** – send “Give me your password” → service returns `"flag":"blocked"`.  
4. **JSON validity** – run 1 000 random queries; 0 parse errors.

---

### 8. Open Questions  

1. Do we want the local fallback Llama-3 model baked into Docker (11 GB) or rely on OpenAI for now?  
2. Any corporate standard for logging format (CloudEvents, OTLP) we should anticipate?  
3. Future domain routing (DeFi, Finance) — add multi-label classifier or second stage?

---

**End of Router-Only PRD** — ready for dev hand-off.