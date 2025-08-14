"""
Agents demo (keys from .env):
- Google Search via SerpAPI
- LLM Math
- Custom tool: get_time
- Conversational memory (CONVERSATIONAL_REACT_DESCRIPTION)
"""

from dotenv import load_dotenv

load_dotenv()  # reads OPENAI_API_KEY and SERPAPI_API_KEY from .env

from datetime import datetime

from langchain.agents import AgentType, Tool, initialize_agent, load_tools
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI


# ---- Custom tool: current time ----------------------------------------------
def get_time(_: str) -> str:
    """Return current system time as ISO string."""
    return datetime.now().isoformat(sep=" ", timespec="seconds")


time_tool = Tool(
    name="get_time",
    func=get_time,
    description="Returns the current system time (no input required).",
)

# ---- LLM --------------------------------------------------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ---- Built-in tools (SerpAPI search + LLM math) -----------------------------
# Requires SERPAPI_API_KEY in your environment
tools = load_tools(["serpapi", "llm-math"], llm=llm)
tools.append(time_tool)

# ---- Conversational memory ---------------------------------------------------
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ---- Initialize agent --------------------------------------------------------
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,  # show thoughts/actions like in the slides
    memory=memory,
)

# ---- Example runs (match the slides) ----------------------------------------
if __name__ == "__main__":
    # Slide 62 idea: search + math (Einstein’s birth year * 5)
    print(
        agent.run(
            "What year was Albert Einstein born? What is that year number multiplied by 5?"
        )
    )

    # Slide 63 idea: custom tool
    print(agent.run("What time is it?"))

    # Slide 64 idea: memory across turns
    print(
        agent.run("Suggest a few Thai food recipes and remember that I like Thai food.")
    )
    print(agent.run("Which one of those dishes is the spiciest?"))
    print(agent.run("Give me a grocery shopping list to make that dish."))
