from crewai import Crew, Process
from agents import planner, retriever_agent, analyst, critic
from tasks import task1, task2, task3, task4

def run_research_agent(query: str) -> str:
    crew = Crew(
        agents=[planner, retriever_agent, analyst, critic],
        tasks=[task1, task2, task3, task4],
        process=Process.sequential,
        verbose=False
    )
    result = crew.kickoff(inputs={"query": query})
    return result

if __name__ == "__main__":
    query = '''What is the difference between ML and DL'''
    
    print("\n" + "="*60)
    print(f"Query: {query}")
    print("="*60 + "\n")
    
    result = run_research_agent(query)
    
    print("\n" + "="*60)
    print("FINAL ANSWER:")
    print("="*60)
    print(result)