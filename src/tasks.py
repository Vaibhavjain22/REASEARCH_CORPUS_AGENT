from crewai import Task
from agents import planner, retriever_agent, analyst, critic

task1 = Task(
    description="""
        Analyze this query: {query}
        Break it into 2-3 specific search terms
        that would help find relevant papers.
        Output a clear retrieval plan.
    """,
    expected_output="A list of specific search terms "
                    "and a retrieval plan.",
    agent=planner
)

task2 = Task(
    description="""
        Using the retrieval plan from the planner,
        search for relevant research papers using
        ONLY the vector_search_tool.
        Do NOT use brave_search or any other tool.
        Only use vector_search_tool.
        Return the top papers found with their details.
    """,
    expected_output="A list of relevant papers with "
                    "their key content.",
    agent=retriever_agent
)


task3 = Task(
    description="""
        Read all the retrieved papers carefully.
        Synthesize a comprehensive answer to the original query: {query}
        Include comparisons, key findings and trends.

        CRITICAL RULE: Base your answer STRICTLY AND ONLY on the retrieved papers provided by the retriever. If the retriever returned 'No relevant papers found' or if no relevant context exists, DO NOT use your own general knowledge or make up an answer. Simply respond: "No relevant scientific research papers were found in the dataset for this query."
    """,
    expected_output="A detailed well-structured answer grounded STRICTLY in retrieved papers, OR an explicit refusal if no relevant papers were found.",
    agent=analyst
)

task4 = Task(
    description="""
        Review the analyst answer carefully.
        Check for accuracy and completeness against the retrieved papers.
        Add citations ONLY to real papers present in the retrieved context.

        CRITICAL RULE: If the analyst output or retrieved papers state 'No relevant papers found', or if no relevant papers exist, DO NOT invent an answer, DO NOT use external knowledge, and DO NOT fabricate fake citations. Return ONLY: "No relevant scientific research papers were found in the dataset for this query."

        Do NOT include any planning steps, review logs, self-corrections, or intermediate thought processes in your final output.
    """,
    expected_output="A clean, formatted final validated answer with inline citations, OR an explicit refusal if no relevant papers were found. No fake citations allowed.",
    agent=critic
)