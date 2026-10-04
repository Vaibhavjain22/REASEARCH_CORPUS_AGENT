from crewai import Task
from agents import planner, retriever_agent, analyst, critic

task1 = Task(
    description="""
        Analyze this research query: {query}
        
        Provide two clear outputs:
        1. SEARCH TERMS: Break the query into 2-3 specific academic search keywords for keyword retrieval.
        2. HYDE ABSTRACT: Write a concise hypothetical scientific paper abstract (approx. 80-120 words) that would directly answer this query. Write it in academic style as if it were an abstract in the corpus.
        
        Output a clear, structured retrieval plan containing both the search terms and the hypothetical abstract.
    """,
    expected_output="A structured retrieval plan containing 2-3 search terms and a hypothetical paper abstract (HyDE).",
    agent=planner
)

task2 = Task(
    description="""
        Using the retrieval plan from the planner:
        1. Identify the search terms and the hypothetical paper abstract (HyDE).
        2. Search the corpus using ONLY the vector_search_tool.
           Pass the search terms in 'query' and the hypothetical abstract in 'hyde_doc'.
        3. Do NOT use brave_search or any external tool.
        4. Return the top papers found along with their key content and findings.
    """,
    expected_output="A list of relevant papers retrieved from the corpus with their key content.",
    agent=retriever_agent
)

task3 = Task(
    description="""
        Read all the retrieved papers carefully.
        Synthesize a comprehensive, authoritative answer to the user query: {query}

        Structure your answer into two distinct, well-organized sections:

        ### 1. Conceptual Framework & Practical Workflow
        - Clearly define what the algorithm, methodology, or concept is.
        - Detail step-by-step HOW it works and HOW it is used in practice (e.g., mathematical intuition/formulation, data pipeline, training/optimization procedure, decision boundaries/thresholds, and practical deployment).

        ### 2. Research Insights & Corpus Advancements
        - Detail the specific findings, algorithmic optimizations, and experimental benchmarks discovered in the retrieved research papers.
        - Provide comparative analysis across different approaches, noting trade-offs, computational complexity, and unique contributions from the authors.

        CRITICAL RULE:
        Base your research insights strictly on the retrieved papers. If the retriever returned 'No relevant papers found' or if no relevant context exists, DO NOT use external knowledge or fabricate an answer. Simply respond: "No relevant scientific research papers were found in the dataset for this query."
    """,
    expected_output="A detailed well-structured answer covering both conceptual workflow and corpus research insights, OR an explicit refusal if no relevant papers were found.",
    agent=analyst
)

task4 = Task(
    description="""
        Review the analyst answer carefully for accuracy, clarity, and completeness against the retrieved papers.

        Verify that:
        1. Both 'Conceptual Framework & Practical Workflow' and 'Research Insights & Corpus Advancements' sections are thoroughly articulated.
        2. All citations in the Research Insights section correspond ONLY to real papers present in the retrieved context.
        3. No fake citations or hallucinations exist.

        FORMATTING RULE: Format your response cleanly using structured Markdown. Always put double newlines (\\n\\n) before headers (### Section Title) and before numbered or bullet items. Never place headers or list points inline on the same line as preceding sentences.

        CRITICAL RULE: If the analyst output or retrieved papers state 'No relevant papers found', or if no relevant papers exist, DO NOT invent an answer, DO NOT use external knowledge, and DO NOT fabricate fake citations. Return ONLY: "No relevant scientific research papers were found in the dataset for this query."

        Do NOT include any planning steps, review logs, self-corrections, or intermediate thought processes in your final output.
    """,
    expected_output="A clean, authoritative Markdown response with clear headers on separate lines and inline citations, OR an explicit refusal if no relevant papers were found.",
    agent=critic
)