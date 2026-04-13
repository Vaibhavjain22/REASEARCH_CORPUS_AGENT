# 🔬 Research Corpus Agent

An intelligent multi-agent AI system that answers complex queries over a large corpus of ArXiv research papers using **CrewAI**, **ChromaDB**, and **Google Gemini**.

---

## 📌 Table of Contents

- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation Results](#evaluation-results)
- [Example Queries](#example-queries)
- [Known Limitations](#known-limitations)

---

## 📖 Project Overview

The **Research Corpus Agent** is an AI-powered system designed to answer complex research questions over a large corpus of 136,000+ ArXiv scientific papers. The system uses a **multi-agent architecture** built with CrewAI where specialized agents collaborate to plan, retrieve, analyze, and validate answers.

### Key Capabilities

- ✅ **Summarization** — Summarize key contributions of research papers
- ✅ **Cross-document Reasoning** — Connect insights across multiple papers
- ✅ **Multi-hop Queries** — Answer questions requiring multiple retrieval steps
- ✅ **Comparisons** — Compare methodologies, models, and approaches
- ✅ **Aggregations** — Identify trends and patterns across papers

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
4. **Embedding** — Generated embeddings using `sentence-transformers/all-MiniLM-L6-v2`
5. **Storage** — Stored 20,000 chunks in ChromaDB with batch size of 500

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA PLATFORM LAYER                     │
│                                                             │
│   ArXiv CSV (170MB)                                         │
│        ↓                                                    │
│   CSVLoader → Text Cleaning → RecursiveCharacterSplitter    │
│        ↓                                                    │
│   HuggingFace Embeddings (all-MiniLM-L6-v2)                 │
│        ↓                                                    │
│   ChromaDB Vector Store (20,000 chunks)                     │
└─────────────────────────────────────────────────────────────┘
                          ↕ similarity search
┌─────────────────────────────────────────────────────────────┐
│                     RETRIEVAL LAYER                         │
│                                                             │
│   Query → Embedding → Vector Similarity Search → Top-K Docs │
└─────────────────────────────────────────────────────────────┘
                          ↕ retrieved docs
┌─────────────────────────────────────────────────────────────┐
│                  AGENTIC LAYER (CrewAI)                     │
│                                                             │
│   🧠 Planner    → Breaks query into search sub-questions    │
│        ↓                                                    │
│   🔍 Retriever  → Searches ChromaDB using vector_search     │
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

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Language** | Python 3.12 |
| **Agent Framework** | CrewAI |
| **LLM** | Google Gemini 2.0 Flash |
| **Vector Database** | ChromaDB |
| **Embeddings** | sentence-transformers/all-MiniLM-L6-v2 |
| **Data Loading** | LangChain CSVLoader |
| **Text Splitting** | LangChain RecursiveCharacterTextSplitter |
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
│   └── test_queries.py     # 25 test queries for evaluation
├── data/
│   └── arxiv_paper.csv     # ArXiv dataset (not tracked by git)
├── evaluation/
│   └── results.json        # Evaluation results
├── chroma_db/              # ChromaDB vector store (not tracked by git)
├── logs/
│   └── crew_runs.json      # Agent run logs for observability
├── .env.example            # Environment variables template
├── .gitignore              # Git ignore rules
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── ARCHITECTURE.md         # Detailed architecture documentation
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.12+
- Git
- Google Gemini API Key (free at [aistudio.google.com](https://aistudio.google.com))

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
GOOGLE_API_KEY=your_gemini_api_key_here
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

### Run the Research Agent

```bash
python src/crew.py
```

### Ask Custom Queries

Open `src/crew.py` and change the query on line 16:

```python
query = "Compare how BERT and GPT approach language understanding"
result = run_research_agent(query)
print(result)
```

### Run Evaluation

```bash
python src/evaluate.py
```

---

## 📈 Evaluation Results

| Metric | Score |
|---|---|
| **Retrieval Precision@5** | 0.9125 (91.25%) |
| **Retrieval Recall@5** | 0.9125 (91.25%) |
| **Answer Keyword Coverage** | 0.9167 (91.67%) |
| **Answer Avg Word Count** | 1525.3 words |
| **Answer Success Rate** | 3/3 (100%) |
| **Failure Cases Handled** | 1/5 |

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
2. **API Rate Limits** — Gemini free tier has quota limits which may cause delays
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

## 📄 License

This project is created for the Uptiq.ai Internship Assignment.
