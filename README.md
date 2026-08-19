# 🔬 Research Corpus Agent(RAG-BASED)

An intelligent multi-agent AI system that answers complex queries over a large corpus of ArXiv research papers using **CrewAI**, **ChromaDB**, and **OpenAI GPT**.

---

## 📌 Table of Contents

- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [System Architecture](#system-architecture)
- [Interactive Web UI](#interactive-web-ui)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation Results](#evaluation-results)
- [Example Queries](#example-queries)
- [Known Limitations](#known-limitations)

---

## 📖 Project Overview

The **Research Corpus Agent** is an AI-powered system designed to answer complex research questions over a large corpus of 136,000+ ArXiv scientific papers. The system uses a **multi-agent architecture** built with CrewAI where specialized agents collaborate to plan, retrieve, analyze, and validate answers. It is supported by a rich, modern web interface for interactive search, custom paper ingestion, and performance visualization.

### Key Capabilities

- ✅ **Summarization** — Summarize key contributions of research papers
- ✅ **Cross-document Reasoning** — Connect insights across multiple papers
- ✅ **Multi-hop Queries** — Answer questions requiring multiple retrieval steps
- ✅ **Comparisons** — Compare methodologies, models, and approaches
- ✅ **Aggregations** — Identify trends and patterns across papers
- ✅ **Interactive Dashboard** — Premium dark-themed UI built with FastAPI and Chart.js
- ✅ **Paper Ingestion Hub** — Dynamically load and index new papers into ChromaDB
- ✅ **Persistent History** — Sidebar showing past searches and response times
- ⚡ **Async Parallel Retrieval** — Concurrently execute vector database searches for sub-queries to optimize system latency and merge/deduplicate results

---

## 📊 Dataset Description

## 📊 Dataset Description

| Property | Details |
|---|---|
| **Source** | [ArXiv Scientific Dataset — Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv) |
| **Full Dataset Size** | 170 MB (136,238 papers) |
| **Sample in Repo** | 10,000 rows (`data/arxiv_sample.csv`) |
| **Total Chunks** | 20,000 chunks stored in ChromaDB |
| **Format** | CSV |
| **Fields** | id, title, category, published_date, authors, summary |

  **Note:** 
  Due to GitHub's 100MB file size limit, only a
  10,000 row sample is included in this repository at
 `data/arxiv_sample.csv`. The full dataset (170MB, 136,238 papers)
 can be downloaded from
 [Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv).
 To run ingestion on the full dataset, replace
 `data/arxiv_sample.csv` with the full file and run
 `python src/ingestion.py`.

### Categories Covered

| Category | Papers |
|---|---|
| Machine Learning | 39,986 |
| Computer Vision and Pattern Recognition | 29,057 |
| Computation and Language (NLP) | 25,202 |
| Artificial Intelligence | 12,969 |
| Machine Learning (Statistics) | 10,447 |
| Neural and Evolutionary Computing | 5,509 |

### Preprocessing Steps

1. **Loading** — CSV loaded using LangChain `CSVLoader`
2. **Cleaning** — Removed newlines and special characters from abstracts
3. **Chunking** — Split into chunks of 1500 characters with 50 character overlap using `RecursiveCharacterTextSplitter`
4. **Embedding** — Generated embeddings using OpenAI's `text-embedding-3-small` API
5. **Storage** — Stored 20,000 chunks in ChromaDB with batch size of 500

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA PLATFORM LAYER                     │
│                                                             │
│   ArXiv CSV (170MB)                                         │
│        │                                                    │
│   CSVLoader → Text Cleaning → RecursiveCharacterSplitter    │
│        │                                                    │
│   OpenAI Embeddings (text-embedding-3-small)                │
│        │                                                    │
│   ChromaDB Vector Store (20,000 chunks)                     │
└─────────────────────────────────────────────────────────────┘
                          ↕ similarity search
┌─────────────────────────────────────────────────────────────┐
│                     RETRIEVAL LAYER                         │
│                                                             │
│   Queries → Embedding → Async Parallel Vector Search        │
└─────────────────────────────────────────────────────────────┘
                          ↕ retrieved docs
┌─────────────────────────────────────────────────────────────┐
│                  AGENTIC LAYER (CrewAI)                     │
│                                                             │
│   🧠 Planner    → Breaks query into search sub-questions    │
│        ↓                                                    │
│   🔍 Retriever  → Executes parallel searches via asyncio     │
│        ↓                                                    │
│   📊 Analyst    → Synthesizes answer from retrieved papers  │
│        ↓                                                    │
│   ✅ Critic     → Validates answer and adds citations       │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                        OUTPUT                               │
│   Final Answer + Paper Citations + Run Logs                 │
└─────────────────────────────────────────────────────────────┘
```

For detailed architecture see [ARCHITECTURE.md](ARCHITECTURE.md)

---

## 🖥️ Interactive Web UI

The project features a premium glassmorphic dark-theme Single Page Application (SPA) that acts as a cockpit for the multi-agent system:

1. **Research Console**: Input research queries, select suggested query chips, and watch the agents execute visually through a status animation (Planner → Retriever → Analyst → Critic).
2. **Paper Ingestion Hub**: Dynamically ingest custom research papers (Title, Authors, Category, Published Date, Abstract) directly into ChromaDB. The paper is parsed, chunked, embedded, and stored instantly.
3. **Evaluation Dashboard**: Visualizes the system's performance using interactive Chart.js graphs, including Precision@K, Recall@K, keyword coverage, and category-wise performance breakdown.
4. **History Sidebar**: Keeps track of previous queries, complete answers, and execution latency. Click any item to load its results instantly.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Language** | Python 3.12 |
| **Agent Framework** | CrewAI |
| **LLM** | OpenAI gpt-4o-mini |
| **Vector Database** | ChromaDB |
| **Embeddings** | OpenAI text-embedding-3-small |
| **Data Loading** | LangChain CSVLoader |
| **Text Splitting** | LangChain RecursiveCharacterTextSplitter |
| **Web Server Framework** | FastAPI (backend) |
| **Web Server Runner** | Uvicorn |
| **Frontend Styling** | Vanilla CSS (Dark Premium Glassmorphic design) |
| **Frontend Graphs** | Chart.js |
| **Environment** | python-dotenv |
| **Progress Tracking** | tqdm |

---

## 📁 Project Structure

```
RESEARCH_CORPUS_AGENT/
├── src/
│   ├── ingestion.py        # Data loading, cleaning, embedding, ChromaDB storage
│   ├── retriever.py        # Vector search and result formatting
│   ├── tools.py            # CrewAI tool wrapping retriever functions
│   ├── agents.py           # 4 CrewAI agents definition
│   ├── tasks.py            # Task definitions for each agent
│   ├── crew.py             # Crew assembly and execution
│   ├── evaluate.py         # Evaluation scripts (Recall@K, Precision@K)
│   ├── test_queries.py     # 25 test queries for evaluation
│   └── static/             # Frontend assets served by FastAPI
│       ├── index.html      # Main SPA Dashboard HTML
│       ├── index.css       # Premium Dark-themed styling
│       └── app.js          # Tab routing, search execution, chart rendering
├── data/
│   ├── arxiv_paper.csv     # ArXiv dataset (not tracked by git)
│   └── history.json        # Local storage for search history
├── evaluation/
│   └── results.json        # Evaluation results
├── chroma_db/              # ChromaDB vector store (not tracked by git)
├── logs/
│   └── crew_runs.json      # Agent run logs for observability
├── .env.example            # Environment variables template
├── .gitignore              # Git ignore rules
├── .dockerignore           # Excluded files for Docker builds
├── Dockerfile              # Docker compilation and runner blueprint
├── requirements.txt        # Python dependencies
├── app.py                  # FastAPI application entrypoint (root directory)
├── README.md               # Project documentation
└── ARCHITECTURE.md         # Detailed architecture documentation
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.12+
- Git
- OpenAI API Key

### Step 1 — Clone the Repository

```bash
git clone https://github.com/Vaibhavjain22/REASEARCH_CORPUS_AGENT.git
cd REASEARCH_CORPUS_AGENT
```

### Step 2 — Create Virtual Environment

```bash
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Mac/Linux
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Set Up Environment Variables

```bash
# Copy the example file
copy .env.example .env

# Open .env and add your API key
OPENAI_API_KEY=your_openai_api_key_here
```

### Step 5 — Download Dataset

1. Go to [ArXiv Dataset on Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv)
2. Download and place the CSV file at `data/arxiv_paper.csv`

### Step 6 — Run Data Ingestion

```bash
# This runs ONCE to populate ChromaDB
python src/ingestion.py
```

Expected output:
```
Loading dataset...
Loaded 136238 documents
Cleaning documents...
Chunking documents...
Total chunks created: 20000
Loading embedding model...
Creating ChromaDB vector store...
Adding 20000 documents in batches of 500...
100%|████████████████| 40/40 [15:00<00:00]
All documents added successfully!
```

---

## 🚀 Usage

### Option 1: Run the Interactive Web UI (Recommended)

Start the FastAPI application:

```bash
python app.py
```

The server will start on port `8080`. Open your browser and navigate to:
```
http://127.0.0.1:8080/
```

- Use the **Search** tab to run research queries.
- Use the **Ingestion Hub** tab to add new papers to the database.
- Use the **Evaluation** tab to see quantitative performance graphs.

### Option 2: Run via CLI

You can run the multi-agent pipeline directly from the command line:

```bash
python src/crew.py
```

### Run Evaluation Scripts

To rerun the retrieval and response evaluation logic:

```bash
python src/evaluate.py
```

---

## 🐳 Docker & Cloud Deployment

You can containerize the application for local testing or cloud hosting.

### Build and Run Locally with Docker

1. **Build the Docker Image**:
   ```bash
   docker build -t research-corpus-agent .
   ```

2. **Run the Docker Container**:
   Pass your OpenAI API Key as an environment variable:
   ```bash
   docker run -d -p 8080:8080 -e OPENAI_API_KEY="your_api_key_here" research-corpus-agent
   ```
   Once launched, navigate to `http://localhost:8080` in your web browser.

### Cloud Deployment Guide

#### 1. Hugging Face Spaces (Docker SDK) — Recommended Free Tier
Hugging Face Spaces offers a free tier with 16GB RAM, which is ideal for hosting local embedding models:
- Create a new Space on [Hugging Face Spaces](https://huggingface.co/new-space).
- Select **Docker** as the SDK (with the Blank template).
- Under the Space's **Settings** tab, add `OPENAI_API_KEY` as a secret environment variable.
- Clone the Space's repository, paste the project files (including `Dockerfile` and `.dockerignore`), commit, and push.

#### 2. Railway.app or Render.com
- Link your GitHub repository.
- Configure environment variables: `OPENAI_API_KEY` and set `PORT` (usually mapped automatically).
- **Tip**: To persist ingested papers from the Paper Ingestion Hub, attach a **Persistent Volume/Disk** (minimum 500MB) and mount it to `/app/chroma_db` (or `/app/data` to persist search history too).

---

## 📈 Evaluation Results

| Metric | Score |
|---|---|
| **Retrieval Precision@5** | 0.9375 (93.75%) |
| **Retrieval Recall@5** | 0.9375 (93.75%) |
| **Answer Keyword Coverage** | 0.9167 (91.67%) |
| **Answer Avg Word Count** | 746.7 words |
| **Answer Success Rate** | 3/3 (100%) |
| **Failure Cases Handled** | 4/5 |

### Query Type Performance

| Query Type | Queries Tested | Performance |
|---|---|---|
| Simple | 5 | Excellent |
| Multi-hop | 5 | Good |
| Comparison | 5 | Good |
| Aggregation | 5 | Good |
| Failure Cases | 5 | Partial |

Full evaluation report available in [evaluation/results.json](evaluation/results.json)

---

## 💬 Example Queries

### Simple Query
```
Input:  "What is dynamic backtracking in search algorithms?"
Output: "Dynamic backtracking is a search technique that allows
         backtrack points to be moved deeper in the search space,
         avoiding the erasure of meaningful progress..."
```

### Comparison Query
```
Input:  "Compare total-order and partial-order planning in AI"
Output: "Total-order planning executes actions in a strict linear
         sequence while partial-order planning allows flexible
         ordering of actions..."
```

### Multi-hop Query
```
Input:  "How does reinforcement learning apply to game tree search
         and what role does temporal difference learning play?"
Output: "Reinforcement learning combined with game tree search,
         as demonstrated in TDLeaf(λ), uses temporal difference
         learning to update value estimates..."
```

---

## ⚠️ Known Limitations

1. **Out-of-domain queries** — System returns unrelated papers for non-research queries like food or sports
2. **API Rate Limits** — OpenAI free/tier 1 rate limits (RPM/TPM) may cause delays
3. **Dataset Coverage** — Only covers AI/ML papers — queries about other domains may return poor results
4. **Chunk Size** — Fixed chunk size of 1500 may cut off important context in some papers
5. **No Hybrid Search** — Currently uses only vector search; BM25 hybrid search not implemented

---

## 🔮 Future Improvements

- Implement BM25 hybrid search for better keyword matching
- Add re-ranking step to improve retrieval quality
- Expand dataset to full 1.7M ArXiv papers
- Add LangSmith tracing for better observability
- Implement streaming responses for faster output

---

## 👤 Author

**Vaibhav Jain**
GitHub: [@Vaibhavjain22](https://github.com/Vaibhavjain22)

---

