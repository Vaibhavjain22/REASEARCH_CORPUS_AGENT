from concurrent.futures import ThreadPoolExecutor
from crewai.tools import tool
from retriever import vector_search, format_results

@tool("Vector Search Tool")
def vector_search_tool(query: str) -> str:
    """
    Search the research paper database. You can pass a single query, 
    or multiple queries separated by commas to search them in parallel 
    (e.g., 'machine learning, deep learning, neural networks').
    Use this to find papers relevant to your research topics.
    """
    queries = [q.strip() for q in query.split(",") if q.strip()]
    if not queries:
        return "No valid search queries provided."
    
    # Execute all searches concurrently using a ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=len(queries)) as executor:
        results_list = list(executor.map(lambda q: vector_search(q, top_k=5), queries))
    
    # Merge and deduplicate results based on page_content
    seen_contents = set()
    merged_results = []
    for results in results_list:
        for doc in results:
            if doc.page_content not in seen_contents:
                seen_contents.add(doc.page_content)
                merged_results.append(doc)
                
    return format_results(merged_results)


