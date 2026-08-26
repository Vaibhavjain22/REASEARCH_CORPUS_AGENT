import queue
from crewai import Crew, Process
from agents import planner, retriever_agent, analyst, critic
from tasks import task1, task2, task3, task4

_AGENT_LABELS = {
    "Research Planner": "Planner",
    "Research Retriever": "Retriever",
    "Research Analyst": "Analyst",
    "Research Critic": "Critic",
}

def run_research_agent(query: str, event_queue: queue.Queue = None) -> str:
    """
    Executes CrewAI multi-agent pipeline.
    If event_queue is provided, pushes live agent task completion events.
    """
    def on_task_end(task_output):
        if event_queue is not None:
            agent_raw = str(getattr(task_output, "agent", ""))
            agent_name = _AGENT_LABELS.get(agent_raw, agent_raw)
            event_queue.put({"type": "step", "agent": agent_name})

    crew = Crew(
        agents=[planner, retriever_agent, analyst, critic],
        tasks=[task1, task2, task3, task4],
        process=Process.sequential,
        verbose=False,
        task_callback=on_task_end if event_queue is not None else None
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