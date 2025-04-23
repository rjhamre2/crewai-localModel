from crewai import Agent, Task, Crew
from langchain.llms.base import LLM
from typing import Optional, List
import requests
from langchain_core.language_models import BaseLLM
from langchain_openai import ChatOpenAI
import os


os.environ["OPENAI_API_KEY"] = "DUMMY_KEY"

llm = ChatOpenAI(model = "llama2:13b",
                 base_url="http://localhost:11434/v1",
                 )


agent = Agent(
    role="Research Assistant",
    goal="Find the latest AI news",
    backstory="An expert in tech news",
    verbose=True,
    llm=llm,
)

task = Task(
    description=(
        "Identify the next big trend in AI news."
        "Focus on identifying pros and cons and the overall narrative."
        "Your final report should clearly articulate the key points,"
        "its market opportunities, and potential risks."
    ),
    expected_output="A comprehensive 3 paragraphs long report on the latest AI trends.",
    agent=agent
)
crew = Crew(
    agents=[agent],
    tasks=[task],
    verbose=True
)

crew.kickoff()