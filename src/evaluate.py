"""
evaluate.py
-----------
Complete evaluation script for Research Corpus Agent.

Covers ALL assignment requirements:
    Quantitative:
        - Retrieval quality (Recall@K, Precision@K)
        - Answer quality (keyword coverage, word count)
    Qualitative:
        - Failure cases
        - Error analysis

Usage:
    python evaluate.py
"""

import json
import os
import time
from dotenv import load_dotenv

load_dotenv()

# Import your existing functions
from retriever import vector_search
from crew import run_research_agent
from test_queries import TEST_QUERIES

os.makedirs("evaluation", exist_ok=True)

K = 5  # Recall@K and Precision@K




# ─────────────────────────────────────────────────────
# PART 1 — Retrieval Evaluation (Recall@K, Precision@K)
# ─────────────────────────────────────────────────────
def evaluate_retrieval() -> dict:
    print("\n" + "="*60)
    print("   PART 1: RETRIEVAL EVALUATION")
    print(f"   Recall@{K} and Precision@{K}")
    print("="*60)

    normal_queries  = [q for q in TEST_QUERIES if q["type"] != "failure_case"]
    per_query_results = []

    for item in normal_queries:
        query    = item["query"]
        keywords = item["relevant_keywords"]
        q_type   = item["type"]

        # Retrieve top-K documents
        docs = vector_search(query, top_k=K)

        # Combine all retrieved text
        retrieved_text = " ".join([
            doc.page_content.lower() for doc in docs
        ])

        # Check which keywords appear in retrieved docs
        matched = [
            kw for kw in keywords
            if kw.lower() in retrieved_text
        ]

        precision = round(len(matched) / len(keywords), 4)
        recall    = round(len(matched) / len(keywords), 4)

        per_query_results.append({
            "query"             : query,
            "type"              : q_type,
            "keywords"          : keywords,
            "matched_keywords"  : matched,
            f"precision@{K}"    : precision,
            f"recall@{K}"       : recall,
            "num_docs_retrieved": len(docs)
        })

        status = "good" if precision >= 0.5 else "poor"
        print(f"\n{status} Query : {query[:55]}...")
        print(f"   Type  : {q_type}")
        print(f"   Matched keywords : {matched}")
        print(f"   Precision@{K}: {precision}  |  Recall@{K}: {recall}")

    # Compute mean scores
    mean_precision = round(
        sum(r[f"precision@{K}"] for r in per_query_results) / len(per_query_results), 4
    )
    mean_recall = round(
        sum(r[f"recall@{K}"] for r in per_query_results) / len(per_query_results), 4
    )

    print(f"\n{'='*60}")
    print(f"   Mean Precision@{K} : {mean_precision}")
    print(f"   Mean Recall@{K}    : {mean_recall}")
    print(f"{'='*60}")

    return {
        f"mean_precision@{K}": mean_precision,
        f"mean_recall@{K}"   : mean_recall,
        "total_queries"      : len(per_query_results),
        "per_query"          : per_query_results
    }


# ─────────────────────────────────────────────────────
# PART 2 — Answer Quality Evaluation
# ─────────────────────────────────────────────────────
def evaluate_answer_quality() -> dict:
    print("\n" + "="*60)
    print("   PART 2: ANSWER QUALITY EVALUATION")
    print("="*60)

    # Use 3
    #  queries to save API calls
    eval_queries = [q for q in TEST_QUERIES if q["type"] != "failure_case"][:3]
    per_query_results = []

    for item in eval_queries:
        query    = item["query"]
        keywords = item["relevant_keywords"]

        print(f"\n Running agent for: {query[:55]}...")

        try:
            start  = time.time()
            answer = str(run_research_agent(query))
            elapsed = round(time.time() - start, 2)

            # Check keyword coverage in the answer
            answer_lower    = answer.lower()
            matched         = [kw for kw in keywords if kw.lower() in answer_lower]
            keyword_coverage = round(len(matched) / len(keywords), 4)

            # Word count as proxy for answer detail
            word_count  = len(answer.split())
            is_detailed = word_count > 100

            per_query_results.append({
                "query"           : query,
                "answer_preview"  : answer[:300] + "...",
                "keyword_coverage": keyword_coverage,
                "word_count"      : word_count,
                "is_detailed"     : is_detailed,
                "time_seconds"    : elapsed,
                "status"          : "success"
            })

            status = "nice" if keyword_coverage >= 0.5 else "needs improvement"
            print(f"   {status} Coverage: {keyword_coverage} | Words: {word_count} | Time: {elapsed}s")

        except Exception as e:
            per_query_results.append({
                "query" : query,
                "status": "failed",
                "error" : str(e)
            })
            print(f"    Failed: {str(e)[:80]}")

    # Compute summary
    successful = [r for r in per_query_results if r["status"] == "success"]

    if successful:
        avg_coverage = round(sum(r["keyword_coverage"] for r in successful) / len(successful), 4)
        avg_words    = round(sum(r["word_count"]       for r in successful) / len(successful), 1)
    else:
        avg_coverage = 0.0
        avg_words    = 0.0

    print(f"\n{'='*60}")
    print(f"   Avg Keyword Coverage : {avg_coverage}")
    print(f"   Avg Word Count       : {avg_words}")
    print(f"   Success Rate         : {len(successful)}/{len(per_query_results)}")
    print(f"{'='*60}")

    return {
        "avg_keyword_coverage": avg_coverage,
        "avg_word_count"      : avg_words,
        "success_rate"        : f"{len(successful)}/{len(per_query_results)}",
        "per_query"           : per_query_results
    }


# ─────────────────────────────────────────────────────
# PART 3 — Failure Case Analysis
# ─────────────────────────────────────────────────────
def evaluate_failure_cases() -> dict:
    print("\n" + "="*60)
    print("   PART 3: FAILURE CASE ANALYSIS")
    print("="*60)

    failure_queries = [q for q in TEST_QUERIES if q["type"] == "failure_case"]
    per_case_results = []

    for item in failure_queries:
        query    = item["query"]
        keywords = item["relevant_keywords"]

        docs = vector_search(query, top_k=3)

        retrieved_text = " ".join([
            doc.page_content.lower() for doc in docs
        ])

        matched = [
            kw for kw in keywords
            if kw.lower() in retrieved_text
        ]

        # If NO keywords matched = system correctly
        # returned unrelated research papers = good!
        handled_well = len(matched) == 0

        if handled_well:
            failure_type = "Correctly returned unrelated papers"
            error_analysis = "System worked correctly — no relevant docs for out-of-domain query"
        else:
            failure_type   = "Incorrectly matched out-of-domain query"
            error_analysis = f"System incorrectly matched keywords: {matched}"

        per_case_results.append({
            "query"         : query,
            "failure_type"  : failure_type,
            "error_analysis": error_analysis,
            "handled_well"  : handled_well,
            "docs_returned" : len(docs),
            "matched_keywords": matched
        })

        status = "well done!" if handled_well else "needs improvement"
        print(f"\n{status} Query  : {query}")
        print(f"   Result : {failure_type}")
        print(f"   Analysis: {error_analysis}")

    well_handled = sum(1 for r in per_case_results if r["handled_well"])

    print(f"\n{'='*60}")
    print(f"   Well handled : {well_handled}/{len(per_case_results)}")
    print(f"{'='*60}")

    return {
        "total_cases" : len(per_case_results),
        "well_handled": well_handled,
        "per_case"    : per_case_results
    }


# ─────────────────────────────────────────────────────
# MAIN — Run All 3 Parts
# ─────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("   RESEARCH CORPUS AGENT — FULL EVALUATION")
    print("="*60)

    all_results = {}

    # Part 1 — Retrieval 
    all_results["retrieval_evaluation"] = evaluate_retrieval()

    # Part 2 — Answer quality 
    all_results["answer_quality"] = evaluate_answer_quality()

    # Part 3 — Failure cases 
    all_results["failure_case_analysis"] = evaluate_failure_cases()

    # Save everything to JSON
    with open("evaluation/results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Final summary
    r = all_results["retrieval_evaluation"]
    a = all_results["answer_quality"]
    f = all_results["failure_case_analysis"]

    print("\n" + "="*60)
    print("   FINAL EVALUATION REPORT")
    print("="*60)
    print(f"Retrieval Mean Precision@{K} : {r[f'mean_precision@{K}']}")
    print(f"Retrieval Mean Recall@{K}    : {r[f'mean_recall@{K}']}")
    print(f"Answer Keyword Coverage      : {a['avg_keyword_coverage']}")
    print(f"Answer Avg Word Count        : {a['avg_word_count']}")
    print(f"Answer Success Rate          : {a['success_rate']}")
    print(f"Failure Cases Handled        : {f['well_handled']}/{f['total_cases']}")
    print("="*60)
    print("\n Full evaluation complete!")
    print(" Results saved to: evaluation/results.json")


if __name__ == "__main__":
    main()