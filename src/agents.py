import os
from crewai import Agent, LLM
from dotenv import load_dotenv
from tools import vector_search_tool

load_dotenv()



llm = LLM(
    model="gpt-4o-mini", 
    api_key=os.getenv("OPENAI_API_KEY")
)


planner = Agent(
    role="Research Query Planner",
    goal="Analyze the user query and break it into "
         "specific search sub-questions",
    backstory="You are an expert at decomposing complex "
              "research questions into searchable components.",
    llm=llm,
    tools=[],
    verbose=True
)

retriever_agent = Agent(
    role="Research Paper Retriever",
    goal="Fetch the most relevant papers using "
         "the search tools available",
    backstory="You are an expert at finding relevant "
              "academic papers from large databases.",
    tools=[vector_search_tool],
    llm=llm,
    verbose=True
)

analyst = Agent(
    role="Research Analyst",
    goal="Read retrieved papers and synthesize a "
         "detailed insightful answer",
    backstory="You are an expert at reading multiple "
              "papers and extracting key insights "
              "comparisons and trends.",
    llm=llm,
    tools=[],
    verbose=True
)

critic = Agent(
    role="Answer Critic",
    goal="Validate the answer for accuracy and completeness and cite source papers",
    backstory="You are a strict fact-checker for research summaries. "
              "You ONLY allow claims grounded in real retrieved papers. "
              "NEVER invent citations, fake papers, or general knowledge claims that do not exist in the retrieved context.",
    llm=llm,
    tools=[],
    verbose=True
)


