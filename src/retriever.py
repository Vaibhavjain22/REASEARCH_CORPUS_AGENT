import sys
import asyncio
import pydantic.v1 as pydantic_v1
sys.modules['langchain_core.pydantic_v1'] = pydantic_v1

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

persistence_directory = "./chroma_db"

# ── lazy globals to avoid loading ChromaDB/BM25 until first search
_db = None
_bm25_retriever = None

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

def get_bm25_retriever(top_k: int = 5):
    """
    Lazy Singleton: Extracts document chunks from ChromaDB and builds an in-memory BM25 index ONCE.
    Avoids re-building term frequencies on every query.
    """
    global _bm25_retriever
    if _bm25_retriever is None:
        db = get_db()
        data = db.get(include=["documents", "metadatas"])
        documents = data.get("documents", [])
        metadatas = data.get("metadatas", [])
        
        docs = [
            Document(page_content=text, metadata=meta or {})
            for text, meta in zip(documents, metadatas)
        ]
        
        if docs:
            _bm25_retriever = BM25Retriever.from_documents(docs)
            _bm25_retriever.k = top_k
        else:
            _bm25_retriever = None
    return _bm25_retriever

def reciprocal_rank_fusion(results_list: list[list[Document]], weights: list[float] = None, c: int = 60) -> list[Document]:
    """
    Reciprocal Rank Fusion (RRF) algorithm to combine multiple ranked lists.
    RRF Score = sum(weight / (c + rank))
    """
    if weights is None:
        weights = [1.0] * len(results_list)
        
    doc_scores = {}
    doc_map = {}
    
    for results, weight in zip(results_list, weights):
        for rank, doc in enumerate(results, start=1):
            key = doc.page_content.strip()
            doc_map[key] = doc
            rrf_score = weight * (1.0 / (c + rank))
            doc_scores[key] = doc_scores.get(key, 0.0) + rrf_score
            
    sorted_keys = sorted(doc_scores.keys(), key=lambda k: doc_scores[k], reverse=True)
    return [doc_map[k] for k in sorted_keys]



def vector_search(query: str, top_k: int = 3, min_score: float = 0.15) -> list:
    """Hybrid Search combining BM25 Sparse Search and Vector Dense Search via Reciprocal Rank Fusion (RRF)"""
    db = get_db()
    bm25 = get_bm25_retriever(top_k=top_k)
    
    # 1. Sparse BM25 Keyword Search
    bm25_docs = bm25.invoke(query) if bm25 else []
    
    # 2. Dense Vector Search with score thresholding
    vector_docs_and_scores = db.similarity_search_with_relevance_scores(query, k=top_k)
    vector_docs = [doc for doc, score in vector_docs_and_scores if score >= min_score]
    
    # If no vector docs passed threshold (out-of-domain query), return empty list
    if not vector_docs:
        return []
        
    # 3. Fuse results using Reciprocal Rank Fusion (RRF) (30% BM25, 70% Vector)
    hybrid_results = reciprocal_rank_fusion([bm25_docs, vector_docs], weights=[0.3, 0.7])
    return hybrid_results[:top_k]



def format_results(docs: list) -> str:
    """Format retrieved docs into readable text for agents"""
    if not docs:
        return "No relevant papers found."
    output = ""
    for i, doc in enumerate(docs, 1):
        output += f"\n--- Result {i} ---\n"
        output += f"{doc.page_content}\n"
    return output
