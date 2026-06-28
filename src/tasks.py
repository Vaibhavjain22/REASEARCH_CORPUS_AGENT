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
        Synthesize a comprehensive answer to
        the original query: {query}
        Include comparisons, key findings and trends.
    """,
    expected_output="A detailed well-structured answer "
                    "with insights from multiple papers.",
    agent=analyst
)

task4 = Task(
    description="""
        Review the analyst answer carefully.
        Check for accuracy and completeness against the retrieved papers.
        Add citations to source papers.
        Return the final validated answer.

        CRITICAL: Do NOT include any planning steps, review logs, self-corrections, or intermediate thought processes in your final output. Your output must contain ONLY the clean, ready-to-read, final validated answer with citations.
    """,
    expected_output="A clean, formatted final validated answer containing ONLY the research summary with inline citations. No meta-commentary, planning, or thought logs allowed.",
    agent=critic
)