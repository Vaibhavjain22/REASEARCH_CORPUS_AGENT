"""
eval_generator.py
-----------------
DeepEval-based Independent Generator Evaluation Suite for Research Corpus Agent.
Evaluates generation fidelity (Analyst + Critic agents) in isolation from retrieval noise.

Workflow:
  1. Ground-truth context ('ideal_context') is fed directly to Analyst and Critic agents,
     bypassing the Retriever.
  2. The generated output is evaluated using DeepEval's FaithfulnessMetric.
  3. Judge model: GPT-4o (configured via EVAL_JUDGE_MODEL).

Metrics:
  - Faithfulness: Validates that all factual claims made in the generated answer
    are strictly grounded in the provided research context without hallucinations.
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
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC_DIR))

from src.agents import analyst, critic
from crewai import Crew, Task, Process
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase

DATASET_PATH = ROOT_DIR / "data" / "golden_eval_dataset.json"
RESULTS_PATH = ROOT_DIR / "evaluation_results" / "generator_deepeval_results.json"
JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "gpt-4o")
THRESHOLD = float(os.getenv("FAITHFULNESS_THRESHOLD", "0.70"))


def generate_answer_from_ideal_context(query: str, ideal_context: list) -> tuple:
    """
    Runs Analyst and Critic agents sequentially using the provided ideal_context,
    completely bypassing retrieval.
    Returns (generated_answer, elapsed_time).
    """
    formatted_context = "\n\n---\n\n".join(ideal_context)

    analyst_task = Task(
        description=f"""
        Read the following research papers carefully:
        ---
        {formatted_context}
        ---

        Synthesize a comprehensive, authoritative answer to the user query: {query}

        Structure your answer into two distinct, well-organized sections:

        ### 1. Conceptual Framework & Practical Workflow
        - Clearly define what the algorithm, methodology, or concept is.
        - Detail step-by-step HOW it works and HOW it is used in practice.

        ### 2. Research Insights & Corpus Advancements
        - Detail the specific findings, algorithmic optimizations, and experimental benchmarks discovered in the provided research papers.
        - Provide comparative analysis across different approaches, noting trade-offs and contributions.

        CRITICAL RULE:
        Base your research insights strictly on the provided research papers above.
        Do NOT invent citations, authors, or claims not found in the provided papers.
        """,
        expected_output="A detailed well-structured answer covering conceptual workflow and corpus research insights.",
        agent=analyst
    )

    critic_task = Task(
        description=f"""
        Review the analyst answer carefully for accuracy, clarity, and completeness against the provided papers:
        ---
        {formatted_context}
        ---

        Verify that:
        1. Both 'Conceptual Framework & Practical Workflow' and 'Research Insights & Corpus Advancements' sections are thoroughly articulated.
        2. All citations in the Research Insights section correspond ONLY to real papers present in the provided context.
        3. No fake citations or hallucinations exist.
        4. Tone is objective, rigorous, and scholarly.

        Produce the final verified answer.
        """,
        expected_output="Final polished, verified research summary with strictly validated citations and no hallucinations.",
        agent=critic
    )

    generator_crew = Crew(
        agents=[analyst, critic],
        tasks=[analyst_task, critic_task],
        process=Process.sequential,
        verbose=False
    )

    start_time = time.time()
    crew_output = generator_crew.kickoff()
    elapsed = time.time() - start_time

    return str(crew_output), elapsed


def run_generator_evaluation():
    print("=" * 70)
    print("   DEEPEVAL — GENERATOR INDEPENDENT EVALUATION (FAITHFULNESS)")
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

    print(f"[+] Loaded {len(dataset)} golden benchmark queries.\n")

    # Initialize DeepEval Faithfulness Metric with GPT-4o
    faithfulness_metric = FaithfulnessMetric(
        threshold=THRESHOLD,
        model=JUDGE_MODEL,
        include_reason=True
    )

    results = []

    for i, item in enumerate(dataset, 1):
        q_id = item["id"]
        q_type = item["type"]
        query = item["query"]
        ideal_context = item["ideal_context"]
        expected_output = item.get("expected_output", "")

        print(f"[{i:02d}/{len(dataset)}] [{q_type.upper()}] Query: {query[:65]}...")

        # 1. Run Generator (Analyst + Critic) with isolated ideal_context
        try:
            generated_answer, gen_time = generate_answer_from_ideal_context(query, ideal_context)
            word_count = len(generated_answer.split())
        except Exception as e:
            print(f"    [Error] Generation failed: {e}")
            generated_answer = f"Error during generation: {str(e)}"
            gen_time = 0.0
            word_count = 0

        # 2. Build DeepEval Test Case
        test_case = LLMTestCase(
            input=query,
            actual_output=generated_answer,
            expected_output=expected_output,
            retrieval_context=ideal_context
        )

        # 3. Measure Faithfulness using GPT-4o Judge
        try:
            faithfulness_metric.measure(test_case)
            f_score = round(float(faithfulness_metric.score), 4)
            f_passed = bool(faithfulness_metric.is_successful())
            f_reason = faithfulness_metric.reason or "No reason provided"
        except Exception as e:
            print(f"    [Error] Faithfulness eval error: {e}")
            f_score = 0.0
            f_passed = False
            f_reason = str(e)

        status_str = "PASSED" if f_passed else "FAILED"
        print(f"    --> [{status_str}] Faithfulness: {f_score:.2f} | Words: {word_count} | Gen Latency: {gen_time:.2f}s")

        results.append({
            "id": q_id,
            "type": q_type,
            "query": query,
            "word_count": word_count,
            "generation_time_seconds": round(gen_time, 2),
            "actual_output_preview": generated_answer[:300] + "..." if len(generated_answer) > 300 else generated_answer,
            "faithfulness": {
                "score": f_score,
                "passed": f_passed,
                "threshold": THRESHOLD,
                "reason": f_reason
            }
        })

        time.sleep(1.0)

    # 4. Aggregate Metrics
    total_queries = len(results)
    mean_faithfulness = sum(r["faithfulness"]["score"] for r in results) / total_queries
    passed_count = sum(1 for r in results if r["faithfulness"]["passed"])
    avg_gen_time = sum(r["generation_time_seconds"] for r in results) / total_queries
    avg_words = sum(r["word_count"] for r in results) / total_queries

    # Category Breakdown
    categories = sorted(list(set(r["type"] for r in results)))
    category_metrics = {}
    for cat in categories:
        cat_results = [r for r in results if r["type"] == cat]
        category_metrics[cat] = {
            "count": len(cat_results),
            "mean_faithfulness": round(sum(r["faithfulness"]["score"] for r in cat_results) / len(cat_results), 4),
            "pass_rate": f"{sum(1 for r in cat_results if r['faithfulness']['passed'])}/{len(cat_results)}"
        }

    summary = {
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "evaluation_tier": "Generator Independent Evaluation",
        "judge_model": JUDGE_MODEL,
        "total_queries": total_queries,
        "overall_metrics": {
            "mean_faithfulness": round(mean_faithfulness, 4),
            "pass_rate": f"{passed_count}/{total_queries}",
            "pass_percentage": f"{round(passed_count / total_queries * 100, 1)}%",
            "avg_generation_time_seconds": round(avg_gen_time, 2),
            "avg_word_count": round(avg_words, 1)
        },
        "category_breakdown": category_metrics,
        "per_query_details": results
    }

    # Save to evaluation_results
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("   GENERATOR EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Mean Faithfulness Score : {mean_faithfulness * 100:.2f}%")
    print(f"Pass Rate               : {passed_count}/{total_queries} ({round(passed_count / total_queries * 100, 1)}%)")
    print(f"Avg Generation Latency  : {avg_gen_time:.2f}s")
    print(f"Avg Answer Word Count   : {avg_words:.1f} words")
    print("-" * 70)
    print("Category Breakdown:")
    for cat, stats in category_metrics.items():
        print(f"  - {cat.upper():<12} : Faithfulness: {stats['mean_faithfulness'] * 100:.1f}% | Pass Rate: {stats['pass_rate']}")
    print("=" * 70)
    print(f"[+] Results saved to: {RESULTS_PATH}\n")


if __name__ == "__main__":
    run_generator_evaluation()
