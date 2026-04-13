from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

persistence_directory = "./chroma_db"

# ── lazy globals to avoid loading ChromaDB until first search
_db = None

def get_db():
    """Load ChromaDB only when first needed"""
    global _db
    if _db is None:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        _db = Chroma(
            persist_directory=persistence_directory,
            embedding_function=embeddings,
            collection_name="example_collection"
        )
    return _db

def vector_search(query: str, top_k: int = 3) -> list:
    """Search ChromaDB using vector similarity"""
    db = get_db()
    results = db.similarity_search(query, k=top_k)
    return results

def format_results(docs: list) -> str:
    """Format retrieved docs into readable text for agents"""
    if not docs:
        return "No relevant papers found."
    output = ""
    for i, doc in enumerate(docs, 1):
        output += f"\n--- Result {i} ---\n"
        output += f"{doc.page_content}\n"
    return output




