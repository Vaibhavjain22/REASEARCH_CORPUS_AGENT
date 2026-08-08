import asyncio
from crewai.tools import tool
from retriever import vector_search_async, format_results

@tool("Vector Search Tool")
async def vector_search_tool(query: str) -> str:
    """
    Search the research paper database. You can pass a single query, 
    or multiple queries separated by commas to search them in parallel 
    (e.g., 'machine learning, deep learning, neural networks').
    Use this to find papers relevant to your research topics.
    """
    queries = [q.strip() for q in query.split(",") if q.strip()]
    if not queries:
        return "No valid search queries provided."
    
    # Execute all searches concurrently
    tasks = [vector_search_async(q, top_k=5) for q in queries]
    results_list = await asyncio.gather(*tasks)
    
    # Merge and deduplicate results based on page_content
    seen_contents = set()
    merged_results = []
    for results in results_list:
        for doc in results:
            if doc.page_content not in seen_contents:
                seen_contents.add(doc.page_content)
                merged_results.append(doc)
                
    return format_results(merged_results)


