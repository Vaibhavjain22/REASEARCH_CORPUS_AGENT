"""
eval_retriever.py
-----------------
DeepEval-based Independent Retriever Evaluation Suite for Research Corpus Agent.
Evaluates 2-Stage Hybrid Retrieval (BM25 + ChromaDB + Cohere Reranker) across 20 Golden Queries.

Metrics:
  - Contextual Recall: Validates if retrieved chunks contain all facts needed by the golden answer.
  - Contextual Precision: Validates if the most relevant chunks are ranked at the top (ranks 1-2).

Judge:
  - GPT-4o (configurable via EVAL_JUDGE_MODEL)
"""

import sys
import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv

# Ensure UTF-8 console output for Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

# Setup paths relative to evaluation_scripts/
SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPTS_DIR.parent
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from src.retriever import vector_search
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase

DATASET_PATH = ROOT_DIR / "data" / "golden_eval_dataset.json"
RESULTS_PATH = ROOT_DIR / "evaluation_results" / "retriever_deepeval_results.json"
JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "gpt-4o")

def run_retriever_evaluation():
    print("=" * 70)
    print("   DEEPEVAL — RETRIEVER INDEPENDENT EVALUATION")
    print(f"   Judge Model : {JUDGE_MODEL}")
    print(f"   Dataset     : {DATASET_PATH}")
    print(f"   Output      : {RESULTS_PATH}")
    print("=" * 70)
    
    if not DATASET_PATH.exists():
        print(f"[Error] Golden dataset not found at {DATASET_PATH}")
        sys.exit(1)
        
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    print(f"[+] Loaded {len(dataset)} golden benchmark queries.\n")
    
    # Initialize DeepEval Metrics with GPT-4o Judge
    recall_metric = ContextualRecallMetric(
        threshold=0.7,
        model=JUDGE_MODEL,
        include_reason=True
    )
    
    precision_metric = ContextualPrecisionMetric(
        threshold=0.7,
        model=JUDGE_MODEL,
        include_reason=True
    )
    
    results = []
    
    for i, item in enumerate(dataset, 1):
        q_id = item["id"]
        q_type = item["type"]
        query = item["query"]
        expected_output = item["expected_output"]
        
        print(f"[{i:02d}/{len(dataset)}] [{q_type.upper()}] Query: {query[:65]}...")
        
        # 1. Execute 2-Stage Retrieval (BM25 + Dense + Cohere Rerank)
        start_time = time.time()
        retrieved_docs = vector_search(query, top_k=5)
        retrieval_time = time.time() - start_time
        
        retrieval_context = [doc.page_content.strip() for doc in retrieved_docs]
        
        if not retrieval_context:
            print("    [!] Warning: No documents retrieved for query.")
            retrieval_context = ["No relevant scientific research papers were found."]
            
        # 2. Build DeepEval Test Case
        test_case = LLMTestCase(
            input=query,
            actual_output=retrieval_context[0] if retrieval_context else "",
            expected_output=expected_output,
            retrieval_context=retrieval_context
        )
        
        # 3. Measure Contextual Recall & Precision using GPT-4o Judge
        try:
            recall_metric.measure(test_case)
            recall_score = round(recall_metric.score, 4)
            recall_reason = recall_metric.reason
        except Exception as e:
            print(f"    [Error] Recall eval error: {e}")
            recall_score = 0.0
            recall_reason = str(e)
            
        try:
            precision_metric.measure(test_case)
            precision_score = round(precision_metric.score, 4)
            precision_reason = precision_metric.reason
        except Exception as e:
            print(f"    [Error] Precision eval error: {e}")
            precision_score = 0.0
            precision_reason = str(e)
            
        print(f"    --> Recall: {recall_score:.2f} | Precision: {precision_score:.2f} | Retrieval Latency: {retrieval_time:.2f}s")
        
        results.append({
            "id": q_id,
            "type": q_type,
            "query": query,
            "expected_output": expected_output,
            "retrieved_chunks_count": len(retrieved_docs),
            "retrieval_time_seconds": round(retrieval_time, 2),
            "contextual_recall": {
                "score": recall_score,
                "passed": recall_score >= 0.7,
                "reason": recall_reason
            },
            "contextual_precision": {
                "score": precision_score,
                "passed": precision_score >= 0.7,
                "reason": precision_reason
            }
        })
        
        time.sleep(1.0)
        
    # Aggregate Metrics
    total_queries = len(results)
    mean_recall = sum(r["contextual_recall"]["score"] for r in results) / total_queries
    mean_precision = sum(r["contextual_precision"]["score"] for r in results) / total_queries
    
    # Category Breakdown
    categories = sorted(list(set(r["type"] for r in results)))
    category_metrics = {}
    for cat in categories:
        cat_results = [r for r in results if r["type"] == cat]
        category_metrics[cat] = {
            "count": len(cat_results),
            "mean_recall": round(sum(r["contextual_recall"]["score"] for r in cat_results) / len(cat_results), 4),
            "mean_precision": round(sum(r["contextual_precision"]["score"] for r in cat_results) / len(cat_results), 4),
        }
        
    summary = {
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "judge_model": JUDGE_MODEL,
        "total_queries": total_queries,
        "overall_metrics": {
            "mean_contextual_recall": round(mean_recall, 4),
            "mean_contextual_precision": round(mean_precision, 4),
            "recall_pass_rate": f"{sum(1 for r in results if r['contextual_recall']['passed'])}/{total_queries}",
            "precision_pass_rate": f"{sum(1 for r in results if r['contextual_precision']['passed'])}/{total_queries}",
        },
        "category_breakdown": category_metrics,
        "per_query_details": results
    }
    
    # Save Results
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        
    print("\n" + "=" * 70)
    print("   RETRIEVER EVALUATION RESULTS (DEEPEVAL + GPT-4o JUDGE)")
    print("=" * 70)
    print(f"Overall Mean Contextual Recall    : {mean_recall:.4f} ({mean_recall * 100:.2f}%)")
    print(f"Overall Mean Contextual Precision : {mean_precision:.4f} ({mean_precision * 100:.2f}%)")
    print("-" * 70)
    print(f"{'Category':<15} | {'Count':<6} | {'Recall':<10} | {'Precision':<10}")
    print("-" * 70)
    for cat, m in category_metrics.items():
        print(f"{cat:<15} | {m['count']:<6} | {m['mean_recall']:<10.2f} | {m['mean_precision']:<10.2f}")
    print("=" * 70)
    print(f"[✓] Full evaluation report saved to:\n    {RESULTS_PATH}")

if __name__ == "__main__":
    run_retriever_evaluation()
