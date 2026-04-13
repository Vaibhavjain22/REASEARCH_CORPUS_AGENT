from crewai.tools import tool
from retriever import vector_search, format_results

@tool("Vector Search Tool")
def vector_search_tool(query: str) -> str:
    """
    Search the research paper database using vector
    similarity. Use this to find papers relevant to
    a topic or concept.
    """
    results = vector_search(query, top_k=5)
    return format_results(results)


