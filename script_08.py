"""
Persistent Chatbot with LangGraph + MemorySaver + Threads
---------------------------------------------------------
- State: messages: Annotated[list, add_messages]
- Node: chatbot(state) -> appends assistant reply
- Persistence: MemorySaver() passed to compile(checkpointer=...)
- Threads: config={"configurable": {"thread_id": "<id>"}}
  Same thread_id => remembers prior messages; different thread_id => fresh convo
"""

from __future__ import annotations

# 0) Imports & setup
from typing import TypedDict, Annotated
from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

import os


# 1) LLM selector: uses OpenAI GPT-4o-mini
def pick_llm():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OpenAI API key not found. Set OPENAI_API_KEY in your .env."
        )
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


# 2) Define LangGraph State
class State(TypedDict):
    # Annotated + add_messages tells LangGraph to APPEND new messages,
    # not overwrite the list, when nodes return {"messages": [ ... ]}.
    messages: Annotated[list, add_messages]


# 3) Build the graph (START -> chatbot -> END)
def build_graph():
    llm = pick_llm()

    def chatbot(state: State) -> State:
        """
        Reads conversation from state["messages"], asks the LLM for the next turn,
        and returns the assistant reply to be appended to the message list.
        """
        ai_msg = llm.invoke(state["messages"])
        return {"messages": [ai_msg]}

    builder = StateGraph(State)
    builder.add_node("chatbot", chatbot)
    builder.add_edge(START, "chatbot")
    builder.add_edge("chatbot", END)

    # Persistence layer: in-memory key-value store that survives across calls
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    return graph


# 4) Helper to run a single turn, streaming values like the slide
def run_turn(graph, text: str, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    input_message = HumanMessage(content=text)

    # We stream in "values" mode so each event is the latest state snapshot.
    last_state = None
    for event in graph.stream({"messages": [input_message]}, config=config, stream_mode="values"):
        last_state = event  # event is a dict with the current state (e.g., {"messages":[...]]})

    # Pretty print like the slide
    print("=" * 30, "Human Message", "=" * 30)
    print(text)
    print("=" * 30, "Ai Message", "=" * 30)
    if last_state and "messages" in last_state and last_state["messages"]:
        # last item is the assistant reply we just appended
        reply = last_state["messages"][-1]
        content = getattr(reply, "content", None) or reply.get("content", "")
        print(content)
    print()  # spacer


if __name__ == "__main__":
    app = build_graph()

    # Optional: seed a system message to guide style (kept in thread state)
    sys_prompt = SystemMessage(content="You are a helpful, concise assistant.")
    # Initialize both threads with a system message so they behave consistently
    for tid in ("2", "99"):
        for _ in app.stream({"messages": [sys_prompt]}, config={"configurable": {"thread_id": tid}}, stream_mode="values"):
            pass

    # ---- Thread "2": shows persistence (remembers name) ----
    run_turn(app, "hi! I'm bob", thread_id="2")
    run_turn(app, "what is my name?", thread_id="2")

    # ---- Thread "99": independent conversation (no memory of Bob) ----
    run_turn(app, "what is my name?", thread_id="99")

    # Tip:
    # - Re-run this script or call run_turn() again with thread_id="2" and the
    #   bot should still remember "Bob" because MemorySaver kept the checkpoints
    #   for that thread in the compiled app.

