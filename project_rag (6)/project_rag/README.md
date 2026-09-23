# RAG Evaluation with DeepEval

A production-grade Retrieval-Augmented Generation (RAG) system with a comprehensive evaluation suite using **DeepEval**, following the methodology and codebase from [campusx-official/rag-eval-deepeval](https://github.com/campusx-official/rag-eval-deepeval).

---

## 📁 Repository Structure

```
project_rag/
├── data/                               # Course transcript subtitles (.vtt)
│   ├── YT Sandbox   LLM Evals Session 1.vtt
│   └── ... Session 8.vtt
├── src/                                # Core RAG Pipeline Implementation
│   ├── __init__.py
│   ├── retriever.py                    # VTT parsing, RecursiveCharacterTextSplitter, Chroma vector store
│   ├── reranker.py                     # CrossEncoder reranking (ms-marco-MiniLM-L-6-v2)
│   ├── generator.py                    # Faithfulness-first grounded generator (gpt-4o-mini)
│   ├── rag_pipeline.py                 # Integrated RAG Pipeline (retrieve -> rerank -> generate)
│   └── app.py                          # Streamlit interactive chat UI
├── evals/                              # Evaluation Framework & Test Suites
│   ├── __init__.py
│   ├── harness.py                      # Common eval harness, golden loader & summary stats
│   ├── metric_registry.py              # Metric rules, directions, tolerances, gate vs guardrail
│   ├── eval_retriever.py               # Component eval: Contextual Recall & Precision
│   ├── eval_retriever_with_reranker.py # Compare retriever vs reranked retrieval
│   ├── eval_generator.py               # Component eval: Faithfulness & Relevancy in isolation
│   ├── eval_rag_pipeline.py            # RAG Triad eval: Contextual Relevancy, Faithfulness, Answer Relevancy
│   ├── eval_application.py             # End-to-end correctness on application queries
│   ├── eval_safety.py                  # Orchestrator for safety & guardrail evals
│   ├── eval_toxicity.py                # Toxicity scoring
│   ├── eval_leakage.py                 # PII and prompt/knowledge base leakage testing
│   ├── eval_scope_safety.py            # Out-of-scope query abstention evaluation
│   ├── eval_ops.py                     # Operational performance benchmark orchestrator
│   ├── eval_latency.py                 # Latency profiling (TTFT, e2e p50/p95/p99)
│   ├── eval_cost.py                    # Cost per query & token consumption
│   ├── eval_reliability.py             # Error and success rates under load
│   ├── eval_online.py                  # Live/online traffic evaluation logging
│   ├── run_suite.py                    # Runs full suite on 1 pipeline & creates snapshot JSON
│   └── compare.py                      # Decision function: compares baseline vs candidate
├── goldens/                            # Ground-Truth Golden Datasets & Generators
│   ├── correctness_goldens.json        # End-to-end Q&A test cases with expected outputs
│   ├── faithfulness_dataset.json       # Known-good contexts for isolated generator evaluation
│   ├── retriever_goldens.json          # Queries with ideal answers for recall/precision
│   ├── retriever_deepeval_goldens.json # Synthetically generated golden queries
│   ├── leakage_goldens.json            # Red-team test cases for system prompt & PII extraction
│   ├── scope_goldens.json              # Out-of-domain queries to verify abstention
│   ├── toxicity_goldens.json           # Adversarial toxic prompts
│   └── generate_goldens.py             # Script to synthesize goldens using DeepEval Synthesizer
├── notebooks/                          # Isolated Jupyter Notebooks & Experiments
│   ├── self_reflective_rag.ipynb      # Self-reflective RAG notebook
│   └── self_reflective_rag.py         # Self-reflective RAG script
├── resources/
│   └── deepeval_intro.py               # Introductory DeepEval sanity check
├── export_chroma_chunks.py             # Export Chroma chunks to JSON for inspection
├── main.py                             # Project entrypoint
├── pyproject.toml                      # uv / PEP 517 project config
├── requirements.txt                    # Standard pip requirements
└── .env.example                        # Environment variables template
```

---

## 🧠 Methodology Overview

### 1. Self-Corrective RAG Architecture (`src/`)

The pipeline implements an active **self-corrective reflection loop** (excluding web search, grounded purely on the course material):

1. **Retrieval & Reranking (`src/retriever.py` & `src/reranker.py`)**:
   - **Chroma Vector Store**: Loaded with chunked transcripts from `data/*.vtt` (`text-embedding-3-large`).
   - **Cross-Encoder Reranking**: Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` to jointly score candidate pairs and extract the top chunks.
   - **LLM Relevance Filter**: Assesses topic-level relevance of each retrieved chunk with `RelevanceDecision`. Off-topic chunks are removed before generation.
2. **Grounded Generation (`src/generator.py`)**:
   - Generates an initial answer using `gpt-4o-mini` with strict adherence rules.
3. **Inner Reflection Loop — Groundedness Check (`IsSUP`)**:
   - Checks whether the answer is `fully_supported`, `partially_supported`, or `no_support` relative to the retrieved context.
   - If unsupported claims or hallucinations are detected, enters `revise_answer` to enforce strict grounding on quotes from the context (up to 3 revision attempts).
4. **Outer Reflection Loop — Usefulness Check (`IsUSE`) & Query Reformulation**:
   - Evaluates whether the generated answer genuinely addresses the student's question (`useful` vs `not_useful`).
   - If not useful (and retries remain up to 3 times):
     - **Retry 1 (Query Expansion)**: Automatically expands the question into 5 diverse technical search queries (`expand_queries`).
     - **Retry 2+ (Query Rewrite with Feedback)**: Analyzes why previous attempts failed and crafts 5 targeted queries (`rewrite_with_feedback`).
     - Re-enters retrieval with the new query formulations.
5. **Abstention Fallback**:
   - If relevant context cannot be found or retries are exhausted, the pipeline cleanly abstains:
     `"I don't have enough information in the course material to answer that."`


---

### 2. Evaluation Methodology (`evals/`)

This project implements a multi-tier evaluation framework:

#### A. Component-Level Isolation
- **Retriever (`evals/eval_retriever.py`)**: Evaluates retrieval quality in isolation against `goldens/retriever_goldens.json` using:
  - **Contextual Recall**: Did the retriever capture all necessary facts from the golden answer?
  - **Contextual Precision**: Are the most relevant chunks ranked at the top?
- **Generator (`evals/eval_generator.py`)**: Evaluates generator quality in isolation using `goldens/faithfulness_dataset.json` (feeding known-good golden contexts rather than retriever outputs):
  - **Faithfulness**: Are claims in the actual output backed by the context?
  - **Answer Relevancy**: Does the answer address the question?

#### B. End-to-End Pipeline Evaluation (The RAG Triad)
- **`evals/eval_rag_pipeline.py`** evaluates the three core pillars on live pipeline outputs:
  1. **Contextual Relevancy**: Does the retriever return context relevant to the user query?
  2. **Faithfulness**: Is the generated answer grounded in the retrieved context?
  3. **Answer Relevancy**: Does the generated answer directly answer the user query?

#### C. Safety & Guardrails
- **Scope Safety (`evals/eval_scope_safety.py`)**: Verifies the assistant abstains on out-of-domain queries.
- **Toxicity (`evals/eval_toxicity.py`)**: Ensures responses never mirror, adopt, or output abusive or toxic language.
- **Leakage (`evals/eval_leakage.py`)**: Tests resistance to extracting internal system instructions or sensitive PII.

#### D. Operational Metrics (Ops)
- **Latency (`evals/eval_latency.py`)**: Profiles end-to-end response time (p50, p95, p99) and Time-To-First-Token (TTFT).
- **Cost & Tokens (`evals/eval_cost.py`)**: Tracks prompt, completion, and total tokens, calculating cost per query in USD.
- **Reliability (`evals/eval_reliability.py`)**: Assesses success rate and failure rate under simulated loads.

---

### 3. Regression Testing Framework (`evals/run_suite.py` & `evals/compare.py`)

- **Single Pipeline Injection**: The pipeline is built once and passed to all evaluation modules, ensuring consistent measurements.
- **Metric Registry (`evals/metric_registry.py`)**:
  - **GATES (Hard Block)**: Safety & toxicity metrics must not regress beyond tolerance (drops trigger automated CI failure).
  - **GUARDRAILS (Soft Review)**: Quality & operational metrics (e.g., latency, cost, average quality score) flag for manual review when exceeding noise floor.
  - **INFO**: Diagnostic stats (e.g. min/max scores, token counts) recorded for visibility.
- **Snapshots & Comparison**:
  - `python -m evals.run_suite --baseline` writes `baselines/baseline.json`.
  - `python -m evals.run_suite` writes `baselines/candidate.json`.
  - `python -m evals.compare` outputs an automated verdict: **`PASS`** (exit code 0), **`REVIEW`** (exit code 2), or **`FAIL`** (exit code 1).

---

## 🚀 Setup & Execution

### 1. Environment Setup

Create and activate a virtual environment, then install dependencies:

```bash
# Using pip:
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Or using uv:
uv sync
```

Set your API keys in a `.env` file (see [.env.example](file:///.env.example)):
```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 2. Build the Vector Store (First Run)

To populate the local Chroma store from `data/*.vtt` transcripts:
```bash
python -m src.retriever
```
This builds and caches `chroma_store/` locally.

### 3. Run the Interactive Streamlit TA App

```bash
streamlit run src/app.py
```

### 4. Run Individual Evaluations

```bash
# Evaluate retriever isolation:
python -m evals.eval_retriever

# Evaluate generator isolation:
python -m evals.eval_generator

# Evaluate RAG Triad on the integrated pipeline:
python -m evals.eval_rag_pipeline

# Evaluate safety & guardrails:
python -m evals.eval_safety

# Evaluate operational metrics (latency, cost, reliability):
python -m evals.eval_ops
```

### 5. Run the Complete Regression Suite

```bash
# Create baseline snapshot:
python -m evals.run_suite --baseline --label "Initial baseline (k=10, top_k=5)"

# Run candidate snapshot:
python -m evals.run_suite --label "Candidate test run"

# Compare candidate against baseline:
python -m evals.compare
```
