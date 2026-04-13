# 🏗️ System Architecture — Research Corpus Agent

## Table of Contents

- [High Level Overview](#high-level-overview)
- [Layer 1 — Data Platform](#layer-1--data-platform)
- [Layer 2 — Retrieval System](#layer-2--retrieval-system)
- [Layer 3 — Agentic Layer](#layer-3--agentic-layer)
- [Layer 4 — Evaluation](#layer-4--evaluation)
- [Data Flow](#data-flow)
- [Tech Stack Justification](#tech-stack-justification)
- [Design Decisions](#design-decisions)

---

## High Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   USER QUERY                                                    │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              AGENTIC LAYER (CrewAI)                     │   │
│  │                                                         │   │
│  │   Planner → Retriever → Analyst → Critic                │   │
│  └────────────────────┬────────────────────────────────────┘   │
│                       │ search queries                         │
│                       ▼                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              RETRIEVAL LAYER                            │   │
│  │                                                         │   │
│  │   Query Embedding → Vector Similarity Search → Top-K   │   │
│  └────────────────────┬────────────────────────────────────┘   │
│                       │ fetch vectors                          │
│                       ▼                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              DATA PLATFORM LAYER                        │   │
│  │                                                         │   │
│  │   ChromaDB Vector Store (20,000 chunks)                 │   │
│  └─────────────────────────────────────────────────────────┘   │
│                       │                                         │
│                       ▼                                         │
│   FINAL ANSWER + CITATIONS + LOGS                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Layer 1 — Data Platform

### Overview
The data platform layer handles all data operations from raw CSV to searchable vector embeddings stored in ChromaDB.

### Pipeline Flow

```
arxiv_paper.csv (170MB, 136,238 papers)
         │
         ▼
┌─────────────────────────┐
│   CSVLoader             │  LangChain document loader
│   (langchain_community) │  Loads raw CSV rows as Documents
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Text Cleaning         │  Remove \n characters
│                         │  Normalize whitespace
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   RecursiveCharacter    │  chunk_size    = 1500 characters
│   TextSplitter          │  chunk_overlap = 50 characters
│                         │  20,000 chunks total
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   HuggingFace           │  Model: all-MiniLM-L6-v2
│   Embeddings            │  Device: CPU
│                         │  Batch size: 256
│                         │  Normalized embeddings: True
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   ChromaDB              │  Collection: example_collection
│   Vector Store          │  Persist directory: ./chroma_db
│                         │  Batch upsert: 500 docs/batch
│                         │  Total: 20,000 vectors stored
└─────────────────────────┘
```

### Key Files
- `src/ingestion.py` — Complete ingestion pipeline

### Configuration

| Parameter | Value | Reason |
|---|---|---|
| Chunk size | 1500 chars | Captures full abstract context |
| Chunk overlap | 50 chars | Prevents losing context at boundaries |
| Embedding model | all-MiniLM-L6-v2 | Fast, free, good semantic quality |
| Batch size | 500 | Within ChromaDB's 5461 limit |
| Total chunks | 20,000 | Sufficient for diverse retrieval |

---

## Layer 2 — Retrieval System

### Overview
The retrieval system converts user queries into vector embeddings and finds the most semantically similar document chunks from ChromaDB.

### Retrieval Flow

```
User Query (text)
         │
         ▼
┌─────────────────────────┐
│   Query Embedding       │  Same model as ingestion
│   (all-MiniLM-L6-v2)   │  Ensures compatible vector space
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   ChromaDB              │  similarity_search()
│   Similarity Search     │  Cosine similarity
│                         │  Returns top-K documents
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   format_results()      │  Formats docs as readable text
│                         │  for LLM agents to consume
└─────────────────────────┘
```

### Key Files
- `src/retriever.py` — Vector search and formatting
- `src/tools.py` — CrewAI tool wrapper

### Lazy Loading Pattern

```python
# ChromaDB loads only when first search is called
# Prevents conflicts with CrewAI async initialization

_db = None

def get_db():
    global _db
    if _db is None:
        _db = Chroma(...)   # loads on first call only
    return _db
```

---

## Layer 3 — Agentic Layer

### Overview
The agentic layer uses CrewAI to orchestrate 4 specialized agents that collaborate sequentially to answer research queries.

### Agent Architecture

```
User Query
     │
     ▼
┌──────────────────────────────────────────────────────────┐
│                    CREW (Sequential Process)             │
│                                                          │
│  ┌─────────────────────────────────────────────────┐    │
│  │  Agent 1: PLANNER                               │    │
│  │  Role   : Research Query Planner                │    │
│  │  Goal   : Break query into search sub-questions │    │
│  │  Tools  : None                                  │    │
│  │  Output : Retrieval plan with search terms      │    │
│  └───────────────────────┬─────────────────────────┘    │
│                          │ passes plan                   │
│                          ▼                               │
│  ┌─────────────────────────────────────────────────┐    │
│  │  Agent 2: RETRIEVER                             │    │
│  │  Role   : Research Paper Retriever              │    │
│  │  Goal   : Fetch relevant papers from ChromaDB   │    │
│  │  Tools  : vector_search_tool ✅                 │    │
│  │  Output : Top-K relevant paper chunks           │    │
│  └───────────────────────┬─────────────────────────┘    │
│                          │ passes papers                 │
│                          ▼                               │
│  ┌─────────────────────────────────────────────────┐    │
│  │  Agent 3: ANALYST                               │    │
│  │  Goal   : Synthesize comprehensive answer       │    │
│  │  Tools  : None                                  │    │
│  │  Output : Detailed answer with comparisons      │    │
│  └───────────────────────┬─────────────────────────┘    │
│                          │ passes answer                 │
│                          ▼                               │
│  ┌─────────────────────────────────────────────────┐    │
│  │  Agent 4: CRITIC                                │    │
│  │  Role   : Answer Critic                         │    │
│  │  Goal   : Validate accuracy and add citations   │    │
│  │  Tools  : None                                  │    │
│  │  Output : Final validated answer with sources   │    │
│  └─────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
     │
     ▼
Final Answer + Citations
```

### Task Flow

| Task | Agent | Input | Output |
|---|---|---|---|
| Task 1 | Planner | User query | Search plan |
| Task 2 | Retriever | Search plan | Retrieved papers |
| Task 3 | Analyst | Retrieved papers | Synthesized answer |
| Task 4 | Critic | Draft answer | Validated final answer |

### Key Files
- `src/agents.py` — Agent definitions
- `src/tasks.py` — Task definitions
- `src/crew.py` — Crew assembly and execution
- `src/tools.py` — ChromaDB search tool

### LLM Configuration

```
Provider : Google Gemini
Model    : gemini/gemini-2.5-flash
Tier     : Free tier
Rate     : 10 requests/minute
```

---

## Layer 4 — Evaluation

### Overview
The evaluation layer measures both retrieval quality and answer quality using quantitative and qualitative metrics.

### Evaluation Flow

```
25 Test Queries (test_queries.py)
         │
         ├──── 20 Normal Queries ─────────────────────────────┐
         │                                                     │
         │     ┌──────────────────────────────────────┐       │
         │     │  RETRIEVAL EVALUATION                │       │
         │     │  vector_search(query, top_k=5)       │       │
         │     │  Check keyword matches in results    │       │
         │     │  Calculate Precision@5 and Recall@5  │       │
         │     └──────────────────────────────────────┘       │
         │                                                     │
         ├──── 5 Queries for Answer Quality ──────────────────┤
         │                                                     │
         │     ┌──────────────────────────────────────┐       │
         │     │  ANSWER QUALITY EVALUATION           │       │
         │     │  run_research_agent(query)            │       │
         │     │  Measure keyword coverage            │       │
         │     │  Measure word count                  │       │
         │     └──────────────────────────────────────┘       │
         │                                                     │
         └──── 5 Failure Case Queries ────────────────────────┤
                                                               │
               ┌──────────────────────────────────────┐       │
               │  FAILURE CASE ANALYSIS               │       │
               │  Test out-of-domain queries          │       │
               │  Check if system handles gracefully  │       │
               └──────────────────────────────────────┘       │
                                                               │
                              ▼                                │
               evaluation/results.json ◄──────────────────────┘
```

### Metrics

| Metric | Formula | Score |
|---|---|---|
| Precision@5 | matched_keywords / total_keywords | 0.9125 |
| Recall@5 | matched_keywords / total_keywords | 0.9125 |
| Keyword Coverage | matched_in_answer / total_keywords | 0.9167 |
| Success Rate | successful_runs / total_runs | 3/3 |

### Key Files
- `src/evaluate.py` — Evaluation scripts
- `src/test_queries.py` — 25 test queries
- `evaluation/results.json` — Results output

---

## Data Flow

### Complete End-to-End Flow

```
INPUT: "Compare BERT and GPT language models"
         │
         ▼
[PLANNER] Analyzes query
         Creates plan: Search for "BERT model",
         "GPT model", "language model comparison"
         │
         ▼
[RETRIEVER] Calls vector_search_tool("BERT model")
         ChromaDB returns 5 most similar chunks
         Calls vector_search_tool("GPT model")
         ChromaDB returns 5 more chunks
         │
         ▼
[ANALYST] Reads all 10 chunks
         Identifies: BERT = bidirectional, masked LM
         GPT = unidirectional, autoregressive
         Writes comprehensive comparison
         │
         ▼
[CRITIC] Reviews answer
         Checks accuracy of claims
         Adds paper citations
         Returns final validated answer
         │
         ▼
OUTPUT: Detailed comparison of BERT vs GPT
        with citations from retrieved papers
        Logged to logs/crew_runs.json
```

---

## Tech Stack Justification

| Technology | Why Chosen |
|---|---|
| **CrewAI** | Native multi-agent support, built-in verbose logging, sequential process control |
| **ChromaDB** | Easy local setup, LangChain integration, persistent storage, free |
| **all-MiniLM-L6-v2** | Free, runs on CPU, good semantic quality for scientific text |
| **Gemini 2.0 Flash** | Free tier, fast responses, respects tool boundaries |
| **LangChain** | CSVLoader and TextSplitter simplify ingestion pipeline |
| **Python** | Rich ML ecosystem, all required libraries available |

---

## Design Decisions

### 1. Lazy Loading for ChromaDB
ChromaDB is loaded only when first search is called — not at import time. This prevents conflicts with CrewAI's async initialization.

### 2. Sequential Agent Process
Agents run sequentially (Planner → Retriever → Analyst → Critic) rather than in parallel. This ensures each agent has complete context from the previous agent before proceeding.

### 3. Batch Ingestion
Documents are ingested in batches of 500 to stay within ChromaDB's maximum batch size limit of 5461 and avoid memory issues.

### 4. Empty Tools for Non-Retriever Agents
Planner, Analyst, and Critic agents have `tools=[]` explicitly set. This prevents the LLM from hallucinating tool calls to non-existent tools like `brave_search`.

### 5. Chunk Size of 1500
A chunk size of 1500 characters was chosen to capture complete abstract context while keeping embeddings focused enough for precise retrieval.

---

