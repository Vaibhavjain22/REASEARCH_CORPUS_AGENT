"""
eval_rag_triad.py
-----------------
DeepEval-based End-to-End Pipeline Evaluation Suite for Research Corpus Agent.
Evaluates the complete 4-agent CrewAI pipeline (Planner -> Retriever -> Analyst -> Critic)
against the industry-standard RAG Triad using GPT-4o as Judge.

The RAG Triad:
  1. Contextual Relevancy (Query -> Context): Measures whether the retrieved academic chunks
     are strictly relevant to the user query.
  2. Faithfulness (Context -> Answer): Measures whether the generated answer contains only
     claims directly supported by the actual retrieved context.
  3. Answer Relevancy (Query -> Answer): Measures whether the final response directly and
     completely addresses the user query.

Judge:
  - GPT-4o (configured via EVAL_JUDGE_MODEL)
"""

import sys
import os
import json
import time
import re
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

sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC_DIR))

from src.agents import planner, retriever_agent, analyst, critic
from src.tasks import task1, task2, task3, task4
from crewai import Crew, Process
from deepeval.metrics import (
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    AnswerRelevancyMetric,
)
from deepeval.test_case import LLMTestCase

DATASET_PATH = ROOT_DIR / "data" / "golden_eval_dataset.json"
RESULTS_PATH = ROOT_DIR / "evaluation_results" / "rag_triad_deepeval_results.json"
JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "gpt-4o")
THRESHOLD = float(os.getenv("RAG_TRIAD_THRESHOLD", "0.70"))


def parse_retrieval_chunks(raw_retriever_output: str) -> list:
    """
    Splits the raw retriever agent text into individual document chunks.
    Falls back to paragraph or non-empty lines if format tags are missing.
    """
    if not raw_retriever_output or "no relevant" in raw_retriever_output.lower():
        return ["No relevant scientific research papers were found in the dataset."]

    # Check for '--- Result X ---' delimiter produced by format_results
    chunks = re.split(r"-{2,}\s*Result\s*\d+\s*-{2,}", raw_retriever_output, flags=re.IGNORECASE)
    cleaned_chunks = [c.strip() for c in chunks if c.strip()]
    if cleaned_chunks:
        return cleaned_chunks

    # Fallback to double newline split
    paragraphs = [p.strip() for p in raw_retriever_output.split("\n\n") if len(p.strip()) > 40]
    return paragraphs if paragraphs else [raw_retriever_output.strip()]


def run_pipeline(query: str) -> tuple:
    """
    Executes the full 4-agent CrewAI pipeline end-to-end.
    Returns: (final_answer, retrieved_chunks, latency_seconds)
    """
    crew = Crew(
        agents=[planner, retriever_agent, analyst, critic],
        tasks=[task1, task2, task3, task4],
        process=Process.sequential,
        verbose=False
    )

    start_time = time.time()
    crew_output = crew.kickoff(inputs={"query": query})
    elapsed = time.time() - start_time

    final_answer = str(crew_output.raw if hasattr(crew_output, "raw") else crew_output)

    # Extract Task 2 (Retriever) output
    retrieval_raw = ""
    if hasattr(crew_output, "tasks_output") and len(crew_output.tasks_output) > 1:
        retrieval_raw = str(crew_output.tasks_output[1].raw)

    retrieved_chunks = parse_retrieval_chunks(retrieval_raw)

    return final_answer, retrieved_chunks, elapsed


def run_rag_triad_evaluation(sample_limit: int = None):
    print("=" * 70)
    print("   DEEPEVAL — END-TO-END RAG TRIAD EVALUATION PIPELINE")
    print(f"   Judge Model : {JUDGE_MODEL}")
    print(f"   Threshold   : {THRESHOLD}")
    print(f"   Dataset     : {DATASET_PATH}")
    print(f"   Output      : {RESULTS_PATH}")
    print("=" * 70)

    if not DATASET_PATH.exists():
        print(f"[Error] Golden dataset not found at {DATASET_PATH}")
        sys.exit(1)

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if sample_limit and sample_limit > 0:
        dataset = dataset[:sample_limit]
        print(f"[!] Running on first {len(dataset)} queries as specified.")
    else:
        print(f"[+] Loaded all {len(dataset)} golden benchmark queries.\n")

    # Initialize all 3 RAG Triad metrics with GPT-4o Judge
    contextual_relevancy_metric = ContextualRelevancyMetric(
        threshold=THRESHOLD,
        model=JUDGE_MODEL,
        include_reason=True
    )
    faithfulness_metric = FaithfulnessMetric(
        threshold=THRESHOLD,
        model=JUDGE_MODEL,
        include_reason=True
    )
    answer_relevancy_metric = AnswerRelevancyMetric(
        threshold=THRESHOLD,
        model=JUDGE_MODEL,
        include_reason=True
    )

    results = []

    for i, item in enumerate(dataset, 1):
        q_id = item["id"]
        q_type = item["type"]
        query = item["query"]
        expected_output = item.get("expected_output", "")

        print(f"[{i:02d}/{len(dataset)}] [{q_type.upper()}] Query: {query[:65]}...")

        # 1. Execute live full-pipeline CrewAI
        try:
            final_answer, retrieved_chunks, latency = run_pipeline(query)
            word_count = len(final_answer.split())
        except Exception as e:
            print(f"    [Error] Pipeline execution failed: {e}")
            final_answer = f"Pipeline execution failed: {str(e)}"
            retrieved_chunks = ["Pipeline execution failed."]
            latency = 0.0
            word_count = 0

        # 2. Build DeepEval Test Case
        test_case = LLMTestCase(
            input=query,
            actual_output=final_answer,
            retrieval_context=retrieved_chunks,
            expected_output=expected_output
        )

        # 3. Evaluate Metric 1: Contextual Relevancy (Query -> Context)
        try:
            contextual_relevancy_metric.measure(test_case)
            cr_score = round(float(contextual_relevancy_metric.score), 4)
            cr_passed = bool(contextual_relevancy_metric.is_successful())
            cr_reason = contextual_relevancy_metric.reason or ""
        except Exception as e:
            print(f"    [Error] Contextual Relevancy error: {e}")
            cr_score, cr_passed, cr_reason = 0.0, False, str(e)

        # 4. Evaluate Metric 2: Faithfulness (Context -> Answer)
        try:
            faithfulness_metric.measure(test_case)
            f_score = round(float(faithfulness_metric.score), 4)
            f_passed = bool(faithfulness_metric.is_successful())
            f_reason = faithfulness_metric.reason or ""
        except Exception as e:
            print(f"    [Error] Faithfulness error: {e}")
            f_score, f_passed, f_reason = 0.0, False, str(e)

        # 5. Evaluate Metric 3: Answer Relevancy (Query -> Answer)
        try:
            answer_relevancy_metric.measure(test_case)
            ar_score = round(float(answer_relevancy_metric.score), 4)
            ar_passed = bool(answer_relevancy_metric.is_successful())
            ar_reason = answer_relevancy_metric.reason or ""
        except Exception as e:
            print(f"    [Error] Answer Relevancy error: {e}")
            ar_score, ar_passed, ar_reason = 0.0, False, str(e)

        triad_composite = round((cr_score + f_score + ar_score) / 3.0, 4)
        all_passed = cr_passed and f_passed and ar_passed

        print(
            f"    --> Triad Avg: {triad_composite:.2f} | "
            f"ContextRel: {cr_score:.2f} | Faithfulness: {f_score:.2f} | AnswerRel: {ar_score:.2f} | "
            f"Latency: {latency:.1f}s"
        )

        results.append({
            "id": q_id,
            "type": q_type,
            "query": query,
            "latency_seconds": round(latency, 2),
            "word_count": word_count,
            "retrieved_chunks_count": len(retrieved_chunks),
            "triad_composite_score": triad_composite,
            "all_triad_passed": all_passed,
            "contextual_relevancy": {
                "score": cr_score,
                "passed": cr_passed,
                "threshold": THRESHOLD,
                "reason": cr_reason
            },
            "faithfulness": {
                "score": f_score,
                "passed": f_passed,
                "threshold": THRESHOLD,
                "reason": f_reason
            },
            "answer_relevancy": {
                "score": ar_score,
                "passed": ar_passed,
                "threshold": THRESHOLD,
                "reason": ar_reason
            },
            "final_answer_preview": final_answer[:300] + "..." if len(final_answer) > 300 else final_answer
        })

        time.sleep(2.0)

    # Aggregate Overall Statistics
    total_q = len(results)
    mean_cr = sum(r["contextual_relevancy"]["score"] for r in results) / total_q
    mean_f = sum(r["faithfulness"]["score"] for r in results) / total_q
    mean_ar = sum(r["answer_relevancy"]["score"] for r in results) / total_q
    overall_composite = (mean_cr + mean_f + mean_ar) / 3.0
    avg_latency = sum(r["latency_seconds"] for r in results) / total_q

    # Category Breakdown
    categories = sorted(list(set(r["type"] for r in results)))
    category_metrics = {}
    for cat in categories:
        cat_items = [r for r in results if r["type"] == cat]
        category_metrics[cat] = {
            "count": len(cat_items),
            "mean_contextual_relevancy": round(sum(r["contextual_relevancy"]["score"] for r in cat_items) / len(cat_items), 4),
            "mean_faithfulness": round(sum(r["faithfulness"]["score"] for r in cat_items) / len(cat_items), 4),
            "mean_answer_relevancy": round(sum(r["answer_relevancy"]["score"] for r in cat_items) / len(cat_items), 4),
            "mean_triad_composite": round(sum(r["triad_composite_score"] for r in cat_items) / len(cat_items), 4),
        }

    summary = {
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "evaluation_tier": "End-to-End Pipeline Evaluation (RAG Triad)",
        "judge_model": JUDGE_MODEL,
        "total_queries": total_q,
        "overall_triad_scores": {
            "mean_contextual_relevancy": round(mean_cr, 4),
            "mean_faithfulness": round(mean_f, 4),
            "mean_answer_relevancy": round(mean_ar, 4),
            "rag_triad_composite_score": round(overall_composite, 4),
            "contextual_relevancy_pass_rate": f"{sum(1 for r in results if r['contextual_relevancy']['passed'])}/{total_q}",
            "faithfulness_pass_rate": f"{sum(1 for r in results if r['faithfulness']['passed'])}/{total_q}",
            "answer_relevancy_pass_rate": f"{sum(1 for r in results if r['answer_relevancy']['passed'])}/{total_q}",
            "all_triad_passed_rate": f"{sum(1 for r in results if r['all_triad_passed'])}/{total_q}",
            "avg_pipeline_latency_seconds": round(avg_latency, 2)
        },
        "category_breakdown": category_metrics,
        "per_query_details": results
    }

    # Save to evaluation_results
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("   END-TO-END RAG TRIAD EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Overall RAG Triad Composite : {overall_composite * 100:.2f}%")
    print(f"  1. Contextual Relevancy   : {mean_cr * 100:.2f}% (Pass: {summary['overall_triad_scores']['contextual_relevancy_pass_rate']})")
    print(f"  2. Faithfulness           : {mean_f * 100:.2f}% (Pass: {summary['overall_triad_scores']['faithfulness_pass_rate']})")
    print(f"  3. Answer Relevancy       : {mean_ar * 100:.2f}% (Pass: {summary['overall_triad_scores']['answer_relevancy_pass_rate']})")
    print(f"Avg Pipeline Latency        : {avg_latency:.2f}s")
    print("-" * 70)
    print("Category Breakdown (Composite Scores):")
    for cat, stats in category_metrics.items():
        print(f"  - {cat.upper():<12} : Triad Avg: {stats['mean_triad_composite'] * 100:.1f}% (CR: {stats['mean_contextual_relevancy']*100:.0f}%, F: {stats['mean_faithfulness']*100:.0f}%, AR: {stats['mean_answer_relevancy']*100:.0f}%)")
    print("=" * 70)
    print(f"[+] Results saved to: {RESULTS_PATH}\n")


if __name__ == "__main__":
    run_rag_triad_evaluation()
