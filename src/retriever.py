import asyncio
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

persistence_directory = "./chroma_db"

# ── lazy globals to avoid loading ChromaDB until first search
_db = None

def get_db():
    """Load ChromaDB only when first needed"""
    global _db
    if _db is None:
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )
        _db = Chroma(
            persist_directory=persistence_directory,
            embedding_function=embeddings,
            collection_name="example_collection"
        )
    return _db

def vector_search(query: str, top_k: int = 3, min_score: float = 0.15) -> list:
    """Search ChromaDB using vector similarity with relevance score filtering"""
    db = get_db()
    docs_and_scores = db.similarity_search_with_relevance_scores(query, k=top_k)
    return [doc for doc, score in docs_and_scores if score >= min_score]

async def vector_search_async(query: str, top_k: int = 3, min_score: float = 0.15) -> list:
    """Search ChromaDB using vector similarity asynchronously with relevance score filtering"""
    db = get_db()
    docs_and_scores = await db.asimilarity_search_with_relevance_scores(query, k=top_k)
    return [doc for doc, score in docs_and_scores if score >= min_score]

def format_results(docs: list) -> str:
    """Format retrieved docs into readable text for agents"""
    if not docs:
        return "No relevant papers found."
    output = ""
    for i, doc in enumerate(docs, 1):
        output += f"\n--- Result {i} ---\n"
        output += f"{doc.page_content}\n"
    return output




