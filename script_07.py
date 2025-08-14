"""
Tavily Search: Direct & Agentic Demos
-------------------------------------

Part A  (Slide 85/86): Direct TavilyClient with include_answer=True
Part B  (Slide 87):     Read first result's JSON content (max_results=1)
Part C  (Slide 81/82):  Agentic search via a ReAct agent using a Tavily tool

The agent asks multi-part questions and the LLM decides when to call Tavily.
Keys are read from .env. LLM selection:
  - If OPENAI_API_KEY -> ChatOpenAI('gpt-4o-mini')
  - elif GROQ_API_KEY  -> ChatGroq('gemma2-9b-it')
  - else               -> raises a helpful error
"""

from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

# --- A/B: Tavily direct client -----------------------------------------------
from tavily import TavilyClient

# --- C: Agentic search (ReAct) -----------------------------------------------
from langchain_core.tools import Tool
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent


# ---------- Helpers -----------------------------------------------------------

def get_tavily_api_key() -> str:
    # Accept both correct and commonly misspelled env var names.
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        raise RuntimeError(
            "Tavily API key not found. Set TAVILY_API_KEY in your .env."
        )
    return key


def get_llm():
    """Pick an LLM based on available env vars."""
    if os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)
    if os.getenv("GROQ_API_KEY"):
        return ChatGroq(model="gemma2-9b-it", temperature=0)
    raise RuntimeError(
        "No LLM key found. Set either OPENAI_API_KEY or GROQ_API_KEY in your .env."
    )


# ---------- Part A: Direct Q&A (include_answer) -------------------------------

def demo_direct_answer():
    """
    Mirrors the slide where we call TavilyClient.search with include_answer=True.
    """
    print("\n=== Part A: Direct Tavily answer ===")
    client = TavilyClient(api_key=get_tavily_api_key())
    question = "What is in Nvidia's new Blackwell GPU?"
    result = client.search(question, include_answer=True)
    print("Q:", question)
    print("A:", result.get("answer", "(no answer returned)"))


# ---------- Part B: First result content (structured) -------------------------

def demo_first_result_json():
    """
    Mirrors the slide that inspects the first result's 'content' payload.
    Example query: current weather in Singapore (max_results=1).
    """
    print("\n=== Part B: First result JSON content ===")
    client = TavilyClient(api_key=get_tavily_api_key())
    query = "current weather in Singapore"
    result = client.search(query, max_results=1)
    results = result.get("results", [])
    if not results:
        print("No results.")
        return
    content = results[0].get("content")
    print(f"Query: {query}")
    print("First result content (truncated preview):")
    print(content if content else "(no content field)")


# ---------- Part C: Agentic ReAct with Tavily tool ---------------------------

def demo_agentic_search():
    """
    Recreates the multi-question Tavily Search Agent Demo:
    - The tool: TavilySearchResults (returns ranked web results as JSON)
    - The agent: ReAct graph that decides when to call the tool
    - Prompt: Olympics + country + GDP (multi-step)
    """
    print("\n=== Part C: ReAct Agent with Tavily tool ===")

    # Tool configured to return JSON-like results (list of dicts)
    tavily_tool = TavilySearch(
        max_results=5,
        include_answer=False,   # Let the agent synthesize from results
        api_key=get_tavily_api_key(),
    )
    tool = Tool.from_function(
        name="tavily_search_results_json",
        func=tavily_tool.run,
        description=(
            "Use this to perform a web search. "
            "Input should be a short search query. Returns JSON-like results."
        ),
    )

    llm = get_llm()
    graph = create_react_agent(llm, tools=[tool])

    # Multi-part query adapted from the slide for reliable answers.
    query = (
        "Who won the women's tennis singles at the 2024 Olympics? "
        "In what country is the champion located? "
        "What is the GDP of that country in 2023? "
        "Answer each question clearly."
    )
    print("User:", query)

    final = None
    # Stream so you could see tool calls if you choose to print events.
    for ev in graph.stream({"messages": [{"role": "user", "content": query}]}):
        final = ev

    # Extract and print the agent's final message
    if final:
        for node_state in final.values():
            msgs = node_state.get("messages", [])
            if msgs:
                last = msgs[-1]
                content = getattr(last, "content", None) or last.get("content", "")
                print("\nAssistant:\n", content)
                break


# ---------- Main --------------------------------------------------------------

if __name__ == "__main__":
    # Run all three mini-demos (like the slides, but in one file).
    demo_direct_answer()
    demo_first_result_json()
    demo_agentic_search()
