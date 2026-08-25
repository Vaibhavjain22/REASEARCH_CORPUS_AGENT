"""
Research Corpus Agent — FastAPI Backend
Serves the interactive UI and provides API endpoints for
multi-agent search, paper ingestion, history, and evaluation.
"""

import os
import sys
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

import asyncio
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

# ── Ensure src/ is importable ──────────────────────────────────
SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))

# ── Project paths ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
HISTORY_FILE = DATA_DIR / "history.json"
EVAL_FILE = PROJECT_ROOT / "evaluation" / "results.json"
STATIC_DIR = SRC_DIR / "static"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"


# ── Helpers ────────────────────────────────────────────────────

def _load_history() -> list:
    """Load conversation history from disk."""
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    return []


def _save_history(history: list):
    """Persist conversation history to disk."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


# ── Lifespan ───────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("[+] Research Corpus Agent server starting...")
    yield
    print("[-] Server shutting down.")


# ── App ────────────────────────────────────────────────────────

app = FastAPI(
    title="Research Corpus Agent",
    description="Multi-agent AI research assistant powered by CrewAI",
    version="1.0.0",
    lifespan=lifespan,
)


# ── API Routes ─────────────────────────────────────────────────

@app.post("/api/search")
async def api_search(request: Request):
    """
    Run the full CrewAI multi-agent pipeline on the user query.
    Streams progress messages live, then the final answer word-by-word.
    """
    body = await request.json()
    query = body.get("query", "").strip()

    if not query:
        return JSONResponse(
            status_code=400,
            content={"error": "Query cannot be empty."},
        )

    from src.crew import run_research_agent

    async def generate():
        try:
            # 1. Stream live progress messages immediately
            yield "[STEP] 🧠 Planner is decomposing your query...\n"
            yield "[STEP] 🔍 Retriever is searching 20,000 papers...\n"
            yield "[STEP] 📊 Analyst is synthesizing findings...\n"
            yield "[STEP] ✅ Critic is validating the answer...\n"

            # 2. Run the agent in a background thread (non-blocking)
            start = time.time()
            result = await asyncio.to_thread(run_research_agent, query)
            elapsed = round(time.time() - start, 2)
            answer_text = str(result)

            # 3. Persist to history
            record = {
                "id": str(uuid.uuid4()),
                "query": query,
                "answer": answer_text,
                "time_seconds": elapsed,
                "timestamp": datetime.now().isoformat(),
            }
            history = _load_history()
            history.insert(0, record)
            _save_history(history)

            # 4. Signal start of answer with a separator the JS can detect
            yield f"[ANSWER_START] {record['id']} {elapsed}\n"

            # 5. Stream the answer word-by-word for a typing effect
            for word in answer_text.split(" "):
                yield word + " "

        except Exception as exc:
            yield f"[ERROR] Agent execution failed: {str(exc)}\n"

    return StreamingResponse(generate(), media_type="text/plain")



@app.post("/api/ingest")
async def api_ingest(request: Request):
    """
    Ingest a new paper into the ChromaDB vector store.
    Expects JSON body with: title, authors, category, published_date, abstract.
    """
    body = await request.json()

    title = body.get("title", "").strip()
    authors = body.get("authors", "").strip()
    category = body.get("category", "").strip()
    published_date = body.get("published_date", "").strip()
    abstract = body.get("abstract", "").strip()

    if not title or not abstract:
        return JSONResponse(
            status_code=400,
            content={"error": "Title and Abstract are required fields."},
        )

    try:
        from langchain_core.documents import Document
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_openai import OpenAIEmbeddings
        from langchain_chroma import Chroma

        # Build document text in the same format as the CSV ingestion
        page_content = (
            f"title: {title}\n"
            f"authors: {authors}\n"
            f"category: {category}\n"
            f"published_date: {published_date}\n"
            f"summary: {abstract}"
        ).replace("\n", " ")

        doc = Document(page_content=page_content)

        # Chunk using same settings as ingestion.py
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=50
        )
        chunks = splitter.split_documents([doc])

        # Embed and store
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY")
        )

        vector_store = Chroma(
            collection_name="example_collection",
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR),
        )

        vector_store.add_documents(chunks)
        new_count = vector_store._collection.count()

        return {
            "status": "success",
            "message": f"Paper '{title}' ingested successfully.",
            "chunks_added": len(chunks),
            "total_documents": new_count,
        }

    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"error": f"Ingestion failed: {str(exc)}"},
        )


@app.get("/api/history")
async def api_history():
    """Return the saved conversation history."""
    return _load_history()


@app.get("/api/evaluation")
async def api_evaluation():
    """Return evaluation results from results.json."""
    if not EVAL_FILE.exists():
        return JSONResponse(
            status_code=404,
            content={"error": "Evaluation results file not found."},
        )
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Serve Frontend ─────────────────────────────────────────────

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def serve_index():
    """Serve the main SPA page."""
    return FileResponse(str(STATIC_DIR / "index.html"))


# ── Entrypoint ─────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8080, reload=True)
