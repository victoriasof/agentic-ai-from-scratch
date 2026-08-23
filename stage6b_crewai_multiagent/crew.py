import os
import glob
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool

load_dotenv()

## CrewAI reads the API key from this environment variable name by default
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")


## ============================================================
## A simple tool, shared by the crew — reuses your docs/ folder
## ============================================================
@tool("search_blorbex_docs")
def search_blorbex_docs(query: str) -> str:
    """Searches internal documents about the company Blorbex and returns
    all document content relevant to the query."""
    all_text = []
    for filepath in glob.glob("docs/*.txt"):
        with open(filepath, "r", encoding="utf-8") as f:
            all_text.append(f.read())
    return "\n\n".join(all_text)


## ============================================================
## Step 1: Define the agents — each one is a role, not a task
## ============================================================
planner = Agent(
    role="Content Planner",
    goal="Decide what a short paragraph about Blorbex should focus on",
    backstory="You're skilled at identifying the most interesting angle for a short piece of company content.",
    llm="anthropic/claude-sonnet-4-6",
)

writer = Agent(
    role="Writer",
    goal="Write a short, engaging paragraph about Blorbex based on the plan",
    backstory="You write clear, concise company content, always grounded in facts you're given.",
    llm="anthropic/claude-sonnet-4-6",
    tools=[search_blorbex_docs],
)

reviewer = Agent(
    role="Reviewer",
    goal="Check the paragraph is accurate, well-written, and free of invented facts",
    backstory="You're a careful editor who flags anything that isn't clearly supported by the source material.",
    llm="anthropic/claude-sonnet-4-6",
    tools=[search_blorbex_docs],
)

## ============================================================
## Step 2: Define the tasks — what each agent needs to produce,
## and which agent is responsible for it
## ============================================================
plan_task = Task(
    description="Decide on the most interesting angle for a short paragraph about Blorbex.",
    expected_output="One or two sentences describing the chosen angle.",
    agent=planner,
)

write_task = Task(
    description="Using the search_blorbex_docs tool to find real facts, write a short paragraph "
                "(3-4 sentences) about Blorbex based on the planner's chosen angle.",
    expected_output="A short paragraph about Blorbex.",
    agent=writer,
    context=[plan_task],  # this task can see the planner's output
)

review_task = Task(
    description="Review the paragraph. Use the search_blorbex_docs tool to independently verify "
                "each factual claim against Blorbex's actual documents. "
                "If anything looks invented or unsupported, point it out. Otherwise, approve it.",
    expected_output="Either an approval, or a list of issues to fix.",
    agent=reviewer,
    context=[write_task],
)

## ============================================================
## Step 3: Assemble the crew and run it
## ============================================================
crew = Crew(
    agents=[planner, writer, reviewer],
    tasks=[plan_task, write_task, review_task],
    process=Process.sequential,  # run tasks in order, each seeing prior results
    verbose=True,  # print what each agent is doing, as it happens
)

result = crew.kickoff()
print("\n=== FINAL RESULT ===")
print(result)