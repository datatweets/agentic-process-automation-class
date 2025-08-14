# LangGraph Tutorial: Building a Persistent Chatbot with Memory

This tutorial explains how to build a chatbot that remembers conversations using LangGraph's persistence features. Unlike simple chatbots that forget everything after each interaction, this chatbot maintains memory across multiple conversations using different "threads."

## What You'll Learn

- How to create a persistent chatbot with memory
- Understanding conversation threads and thread isolation
- Using LangGraph's `MemorySaver` for state persistence
- Managing multiple independent conversations

## Prerequisites

Before starting, make sure you have:
- Python installed on your system
- An OpenAI API key (sign up at [platform.openai.com](https://platform.openai.com))
- Required packages: `langchain-openai`, `langgraph`, `python-dotenv`

## Setup

1. Create a `.env` file in your project directory:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

2. Install required packages:
```bash
pip install langchain-openai langgraph python-dotenv
```

## Core Concepts

### What is Persistence in Chatbots?

**Without Persistence**: Each conversation starts fresh - the bot has no memory of previous messages.
```
User: "Hi, I'm Alice"
Bot: "Hello! How can I help?"
User: "What's my name?"
Bot: "I don't know your name."  ← Forgot Alice!
```

**With Persistence**: The bot remembers the entire conversation history.
```
User: "Hi, I'm Alice"
Bot: "Hello Alice! How can I help?"
User: "What's my name?"
Bot: "Your name is Alice."  ← Remembers!
```

### What are Conversation Threads?

Think of threads like separate chat rooms:
- **Thread A**: Alice's conversation with the bot
- **Thread B**: Bob's conversation with the bot
- **Thread C**: Carol's conversation with the bot

Each thread maintains its own memory independently.

## Step-by-Step Implementation

### Step 1: Define the State Structure

The state holds all the messages in a conversation:

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]
```

**Key concept**: `add_messages` tells LangGraph to append new messages instead of replacing the entire list.

### Step 2: Create the LLM Connection

```python
import os
from langchain_openai import ChatOpenAI

def pick_llm():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OpenAI API key not found. Set OPENAI_API_KEY in your .env."
        )
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)
```

**What this does**:
- Checks for OpenAI API key in environment variables
- Creates a connection to GPT-4o-mini with zero temperature (deterministic responses)

### Step 3: Build the Chatbot Node

A "node" is a function that processes the conversation state:

```python
def chatbot(state: State) -> State:
    """
    Takes the current conversation state, sends it to the AI,
    and returns the AI's response to be added to the conversation.
    """
    ai_msg = llm.invoke(state["messages"])
    return {"messages": [ai_msg]}
```

**What happens here**:
1. Receives the current state (all conversation messages)
2. Sends the conversation history to OpenAI
3. Gets the AI's response
4. Returns the response in a format that gets appended to the message list

### Step 4: Create the Graph with Persistence

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

def build_graph():
    llm = pick_llm()
    
    # Create the chatbot node function
    def chatbot(state: State) -> State:
        ai_msg = llm.invoke(state["messages"])
        return {"messages": [ai_msg]}
    
    # Build the graph structure
    builder = StateGraph(State)
    builder.add_node("chatbot", chatbot)
    builder.add_edge(START, "chatbot")
    builder.add_edge("chatbot", END)
    
    # THE MAGIC: Add persistence with MemorySaver
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    return graph
```

**Key components**:
- **Graph flow**: START → chatbot → END
- **MemorySaver**: Stores conversation state between interactions
- **checkpointer=memory**: Enables the graph to save and load states

### Step 5: Implement Thread-Based Conversations

```python
from langchain_core.messages import HumanMessage

def run_turn(graph, text: str, thread_id: str):
    # Thread configuration - this is where the magic happens!
    config = {"configurable": {"thread_id": thread_id}}
    input_message = HumanMessage(content=text)
    
    # Stream the conversation through the graph
    last_state = None
    for event in graph.stream(
        {"messages": [input_message]}, 
        config=config, 
        stream_mode="values"
    ):
        last_state = event
    
    # Extract and display the AI's response
    if last_state and "messages" in last_state:
        reply = last_state["messages"][-1]
        content = reply.content
        print(f"Assistant: {content}")
```

**Understanding thread_id**:
- **Same thread_id**: Continues the same conversation with full memory
- **Different thread_id**: Starts a fresh conversation with no shared memory

## Complete Example Walkthrough

Let's trace through the script's execution:

### 1. Initialize System Message (Optional)
```python
sys_prompt = SystemMessage(content="You are a helpful, concise assistant.")

# Set up both threads with the same system prompt
for tid in ("2", "99"):
    for _ in app.stream(
        {"messages": [sys_prompt]}, 
        config={"configurable": {"thread_id": tid}}, 
        stream_mode="values"
    ):
        pass
```
This seeds both conversation threads with a system message to ensure consistent behavior.

### 2. Thread "2" - First Conversation
```python
run_turn(app, "hi! I'm bob", thread_id="2")
# Result: "Hi Bob! How can I help you today?"

run_turn(app, "what is my name?", thread_id="2")  
# Result: "Your name is Bob. How can I assist you further?"
```

**What's happening**:
- Thread "2" stores: [SystemMessage, "hi! I'm bob", "Hi Bob!", "what is my name?", "Your name is Bob..."]
- The AI remembers Bob's name because it's in the same thread

### 3. Thread "99" - Separate Conversation  
```python
run_turn(app, "what is my name?", thread_id="99")
# Result: "I don't have access to personal information about you, including your name."
```

**What's happening**:
- Thread "99" only stores: [SystemMessage, "what is my name?", "I don't have access..."]
- The AI doesn't know about Bob because it's a different thread

## Memory Persistence Features

### How Memory Works
1. **MemorySaver** stores conversation states in memory
2. Each **thread_id** creates an isolated conversation space
3. **Checkpointer** saves state after each interaction
4. **State recovery** happens automatically when using the same thread_id

### Memory Lifetime
- **During script execution**: Memory persists across all interactions
- **After script ends**: Memory is lost (it's in-memory only)
- **For persistent storage**: You can use database checkpointers instead of MemorySaver

## Real-World Applications

### Customer Support Bot
```python
# Each customer gets their own thread
run_turn(app, "I have an issue with my order #1234", thread_id="customer_alice")
run_turn(app, "What was my order number again?", thread_id="customer_alice")  
# Bot remembers: "Your order number is #1234"
```

### Multi-User Chat System
```python
# Different users in different threads
run_turn(app, "Set my language to Spanish", thread_id="user_123")
run_turn(app, "What's my language preference?", thread_id="user_123")
# Bot remembers: "Your language is set to Spanish"

run_turn(app, "What's my language preference?", thread_id="user_456")  
# Bot doesn't know - different thread: "I don't have your language preference"
```

## Advanced Features

### Custom Stream Processing
```python
def run_turn_with_events(graph, text: str, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    input_message = HumanMessage(content=text)
    
    print("Processing...")
    for event in graph.stream({"messages": [input_message]}, config=config):
        # You can see each step of the graph execution
        print(f"Event: {event}")
    
    print("Done!")
```

### Error Handling
```python
def safe_run_turn(graph, text: str, thread_id: str):
    try:
        config = {"configurable": {"thread_id": thread_id}}
        input_message = HumanMessage(content=text)
        
        for event in graph.stream({"messages": [input_message]}, config=config, stream_mode="values"):
            last_state = event
            
        return last_state["messages"][-1].content
    except Exception as e:
        return f"Error: {str(e)}"
```

## Testing the Implementation

### Basic Memory Test
```python
# Test 1: Memory within same thread
run_turn(app, "Remember: my favorite color is blue", thread_id="test1")
run_turn(app, "What's my favorite color?", thread_id="test1")
# Expected: "Your favorite color is blue"

# Test 2: No memory across different threads  
run_turn(app, "What's my favorite color?", thread_id="test2")
# Expected: "I don't know your favorite color"
```

### Multi-Step Conversation Test
```python
# Build context over multiple turns
run_turn(app, "I'm planning a trip to Japan", thread_id="travel")
run_turn(app, "I want to visit temples", thread_id="travel") 
run_turn(app, "What should I pack for my temple visits in Japan?", thread_id="travel")
# Bot remembers both the destination and the activity
```

## Common Patterns and Best Practices

### 1. Thread ID Strategies
```python
# User-based threads
thread_id = f"user_{user_id}"

# Session-based threads  
thread_id = f"session_{session_id}"

# Topic-based threads
thread_id = f"topic_travel_planning"
```

### 2. System Message Setup
Always initialize threads with clear instructions:
```python
system_msg = SystemMessage(content="""
You are a helpful travel assistant. 
Remember user preferences and provide personalized recommendations.
Always be concise but informative.
""")
```

### 3. Memory Management
For production systems, consider:
- Implementing thread cleanup after inactivity
- Using database-backed checkpointers for persistence
- Setting message limits to prevent excessive memory usage

## Troubleshooting

### Common Issues

1. **"No memory between calls"**
   - Check that you're using the same `thread_id`
   - Verify `MemorySaver` is properly configured

2. **"API key errors"**
   - Ensure `OPENAI_API_KEY` is in your `.env` file
   - Check that the key is valid and has credits

3. **"Empty responses"**
   - Verify your input messages are properly formatted
   - Check that the graph is compiled with the checkpointer

## Next Steps

Try extending this example:
1. **Add multiple users**: Create a web interface with user-specific threads
2. **Implement conversation summaries**: Summarize old messages to manage context length
3. **Add conversation export**: Let users download their conversation history
4. **Database persistence**: Replace MemorySaver with SQLite or PostgreSQL checkpointer
5. **Add conversation metadata**: Store timestamps, user preferences, etc.

## Running the Code

```bash
# Make sure your .env file has OPENAI_API_KEY
python script_08.py
```

**Expected Output**:
```
============================== Human Message ==============================
hi! I'm bob
============================== Ai Message ==============================
Hi Bob! How can I help you today?

============================== Human Message ==============================
what is my name?
============================== Ai Message ==============================
Your name is Bob. How can I assist you further?

============================== Human Message ==============================  
what is my name?
============================== Ai Message ==============================
I'm sorry, but I don't have access to personal information about you, including your name. How can I help you otherwise?
```

The last response shows the AI forgot Bob's name because it's running in a different thread (thread "99" vs thread "2").

## Summary

This tutorial demonstrated:
- ✅ Building persistent chatbots with memory
- ✅ Using thread-based conversation isolation  
- ✅ Implementing state management with MemorySaver
- ✅ Creating multi-user conversation systems
- ✅ Understanding LangGraph's checkpointing system

The key insight is that **persistence + threads = powerful conversation management**. Each thread maintains its own memory, allowing you to build sophisticated multi-user applications while keeping conversations isolated and secure.