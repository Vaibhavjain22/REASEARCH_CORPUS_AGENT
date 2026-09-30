import os
import sys
import asyncio
import pydantic.v1 as pydantic_v1
sys.modules['langchain_core.pydantic_v1'] = pydantic_v1

import cohere
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

_current_dir = os.path.dirname(os.path.abspath(__file__))
persistence_directory = os.path.abspath(os.path.join(_current_dir, "..", "chroma_db"))
if not os.path.exists(persistence_directory):
    persistence_directory = "./chroma_db"

# ── lazy globals to avoid loading ChromaDB/BM25/Cohere until first search
_db = None
_bm25_retriever = None
_cohere_client = None

def get_cohere_client():
    """Load Cohere client lazily from environment variable"""
    global _cohere_client
    if _cohere_client is None:
        api_key = os.getenv("COHERE_API_KEY")
        if api_key:
            _cohere_client = cohere.ClientV2(api_key=api_key)
    return _cohere_client

def rerank_with_cohere(query: str, documents: list[Document], top_n: int = 3) -> list[Document]:
    """
    Stage-2 Reranking: Uses Cohere rerank-v3.5 cross-encoder model to score and re-order candidate documents.
    Falls back gracefully to original candidates if Cohere API is unavailable.
    """
    if not documents:
        return []
    
    client = get_cohere_client()
    if not client:
        return documents[:top_n]
    
    try:
        doc_texts = [doc.page_content for doc in documents]
        response = client.rerank(
            model="rerank-v3.5",
            query=query,
            documents=doc_texts,
            top_n=min(top_n, len(documents))
        )
        reranked_docs = [documents[item.index] for item in response.results]
        return reranked_docs
    except Exception as e:
        print(f"[Warning] Cohere rerank failed: {e}. Falling back to RRF candidates.")
        return documents[:top_n]

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



def vector_search(query: str, top_k: int = 3, min_score: float = 0.15, hyde_doc: str = None) -> list:
    """
    Two-Stage Hybrid Search with HyDE and Cohere Reranking:
    - Stage 1: Candidate Retrieval (BM25 sparse on keywords + Dense vector on HyDE document, fused via RRF [0.2, 0.8])
    - Stage 2: Cross-Encoder Reranking using Cohere rerank-v3.5 down to top_k
    """
    db = get_db()
    candidate_k = max(15, top_k * 3)
    bm25 = get_bm25_retriever(top_k=candidate_k)
    
    # 1. Sparse BM25 Keyword Search
    if bm25:
        bm25.k = candidate_k
        bm25_docs = bm25.invoke(query)
    else:
        bm25_docs = []
    
    # 2. Dense Vector Search (embed HyDE document if available, else original query)
    dense_query = f"{query}\n\n{hyde_doc}" if (hyde_doc and hyde_doc.strip()) else query
    vector_docs_and_scores = db.similarity_search_with_relevance_scores(dense_query, k=candidate_k)
    vector_docs = [doc for doc, score in vector_docs_and_scores if score >= min_score]
    
    # If no vector docs passed threshold (out-of-domain query), return empty list
    if not vector_docs:
        return []
        
    # 3. Fuse Stage 1 candidates using Reciprocal Rank Fusion (20% BM25, 80% Vector)
    candidates = reciprocal_rank_fusion([bm25_docs, vector_docs], weights=[0.2, 0.8])
    candidates = candidates[:candidate_k]
    
    # 4. Stage 2: Cohere Cross-Encoder Rerank down to top_k
    final_docs = rerank_with_cohere(query=query, documents=candidates, top_n=top_k)
    return final_docs



def format_results(docs: list) -> str:
    """Format retrieved docs into readable text for agents"""
    if not docs:
        return "No relevant papers found."
    output = ""
    for i, doc in enumerate(docs, 1):
        output += f"\n--- Result {i} ---\n"
        output += f"{doc.page_content}\n"
    return output
