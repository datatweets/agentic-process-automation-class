"""
LangGraph Chatbot (Groq) + Prebuilt ReAct Agent demo
====================================================

This script mirrors slides 77–79:

A) Chatbot Agent (Open Source LLM via Groq)
   - Define LLM (ChatGroq, model="gemma2-9b-it")
   - Define StateGraph with `messages: Annotated[list, add_messages]`
   - Add chatbot node
   - Connect START -> chatbot -> END
   - Compile graph and interact in a simple REPL using stream()

B) Pre-built Agent (ReAct) with a tiny custom tool `get_weather`
   - Demonstrates `langgraph.prebuilt.create_react_agent(model, tools=[...])`

Keys:
- GROQ_API_KEY is loaded from `.env` (python-dotenv)
"""

from __future__ import annotations

import os
import sys
from typing import Annotated, Literal, TypedDict

# 0) Load environment (reads GROQ_API_KEY)
from dotenv import load_dotenv

# 2) Imports for Path B (prebuilt ReAct agent)
from langchain_core.tools import tool

# 1) Imports for Path A (hand-built StateGraph chatbot)
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent

# Load and validate environment
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("❌ Error: GROQ_API_KEY not found in .env file")
    sys.exit(1)


# =========================
# A) CHATBOT: StateGraph
# =========================


# -- Define the State (slide 77/78: messages: Annotated[list, add_messages])
class ChatState(TypedDict):
    # `add_messages` appends new messages rather than overwriting the list
    messages: Annotated[list, add_messages]


def build_chatbot_graph():
    """
    Build a minimal 1-node graph:
      START -> chatbot -> END
    The node reads ChatState['messages'], calls the LLM, and
    returns {"messages": [assistant_message]} which `add_messages` appends.
    """
    # LLM: open-source path via Groq (Gemma2-9b-it)
    llm = ChatGroq(model="gemma2-9b-it", temperature=0.5)

    # Define the node function
    def chatbot_node(state: ChatState) -> ChatState:
        """
        Node receives the entire state. We expect state['messages'] to be a list
        of dicts or LangChain messages. For quick demos, we pass the raw list
        to `llm.invoke` (the Groq wrapper accepts LC message format and plain strings).
        """
        # Let the LLM produce one assistant turn from the conversation so far
        ai_msg = llm.invoke(state["messages"])
        # Return as a dict with a *list* under "messages" so add_messages can append
        return {"messages": [ai_msg]}

    # Create the graph
    builder = StateGraph(ChatState)
    builder.add_node("chatbot", chatbot_node)
    # Either add explicit edges:
    builder.add_edge(START, "chatbot")
    builder.add_edge("chatbot", END)

    # Or equivalently:
    # builder.set_entry_point("chatbot")
    # builder.set_finish_point("chatbot")

    graph = builder.compile()
    return graph


def run_chat_repl(graph):
    """
    Simple console REPL:
    - User types -> appended to messages
    - Stream events from the graph and print assistant output
    Type 'q' to quit.
    """
    print("\n=== Chatbot (Groq / Gemma2-9b-it) ===")
    print("Type 'q' to quit.\n")

    # Start with empty conversation state
    state: ChatState = {"messages": []}

    while True:
        user_text = input("User: ").strip()
        if user_text.lower() in {"q", "quit", "exit"}:
            print("Goodbye.")
            break

        # Append user turn to state (LangChain-style role/content dict)
        state["messages"].append({"role": "user", "content": user_text})

        # Stream the graph execution so you can see incremental events if desired
        # For a single node this will be brief, but it's a good habit.
        events = graph.stream(state)

        # Collect the final updated state
        final_state = None
        for ev in events:
            # ev is a dict of node_name -> node_output
            # For debugging, you could: print("EVENT:", ev)
            # Keep the latest piece as the final state
            for node_name, node_state in ev.items():
                final_state = node_state

        # Print last assistant message (if present)
        if final_state and final_state.get("messages"):
            last = final_state["messages"][-1]
            # `last` is a LC message or dict; handle both
            content = getattr(last, "content", None) or last.get("content", "")
            print(f"Assistant: {content}\n")

        # Update state with the final_state for next turn
        if final_state:
            state = final_state


# =========================
# B) PREBUILT ReAct AGENT
# =========================


@tool
def get_weather(city: Literal["nyc", "sf"]) -> str:
    """Toy weather tool: returns predictable text for 'nyc' or 'sf'."""
    if city == "nyc":
        return "It might be cloudy in nyc."
    elif city == "sf":
        return "It's always sunny in sf."
    else:
        raise AssertionError("Unknown city")


def run_prebuilt_agent_demo():
    """
    Prebuilt ReAct agent graph:
      START -> agent --(tool calls / continue)--> tools or END
    Uses the same Groq LLM for consistency with the slides.
    """
    print("\n=== Prebuilt ReAct Agent (Groq) ===")
    print("Try: 'What is the weather in nyc?' or 'in sf?'\n")

    llm = ChatGroq(model="gemma2-9b-it", temperature=0.0)
    graph = create_react_agent(llm, tools=[get_weather])

    while True:
        user_text = input("User: ").strip()
        if user_text.lower() in {"q", "quit", "exit"}:
            print("Goodbye.")
            break

        # The prebuilt graph expects a dict with 'messages' containing a list where
        # each item is a role/content pair. We can pass a simplified message list:
        # We'll just stream and print assistant outputs as they arrive.
        result = None
        for ev in graph.stream({"messages": [{"role": "user", "content": user_text}]}):
            # Each ev is a dict of node_name -> state_delta
            # You can uncomment the next line to see the flow:
            # print("EVENT:", ev)
            result = ev

        # Extract final assistant message if present
        if result:
            # The last event normally contains the updated "messages"
            for node_state in result.values():
                msgs = node_state.get("messages", [])
                if msgs:
                    last = msgs[-1]
                    content = getattr(last, "content", None) or last.get("content", "")
                    print(f"Assistant: {content}\n")
                    break


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    # Choose which demo to run from CLI: "chat" or "react"
    # Default: run both sequentially.
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "both"

    if mode in {"chat", "both"}:
        g = build_chatbot_graph()
        run_chat_repl(g)

    if mode in {"react", "both"}:
        run_prebuilt_agent_demo()
