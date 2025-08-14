# LangGraph Tutorial: Building Chatbots and AI Agents

This tutorial explains how to build two types of AI systems using LangGraph and Groq:
1. A simple chatbot using StateGraph
2. A ReAct (Reasoning + Acting) agent with tools

## Prerequisites

Before starting, make sure you have:
- Python installed on your system
- A Groq API key (sign up at [console.groq.com](https://console.groq.com))
- Required packages: `langchain-groq`, `langgraph`, `python-dotenv`

## Setup

1. Create a `.env` file in your project directory:
```bash
GROQ_API_KEY=your_groq_api_key_here
```

2. Install required packages:
```bash
pip install langchain-groq langgraph python-dotenv
```

## Part A: Building a Simple Chatbot

### Understanding the State

In LangGraph, we define a "state" that represents the current conversation. Our chatbot uses a simple state with just messages:

```python
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
```

**Key concept**: The `add_messages` annotation tells LangGraph to append new messages to the list instead of replacing it entirely.

### Creating the Chatbot Node

A "node" in LangGraph is a function that processes the state and returns updates:

```python
def chatbot_node(state: ChatState) -> ChatState:
    # Get the conversation history
    llm = ChatGroq(model="gemma2-9b-it", temperature=0.5)
    
    # Generate AI response based on all messages so far
    ai_msg = llm.invoke(state["messages"])
    
    # Return the new message to be added to the conversation
    return {"messages": [ai_msg]}
```

**What happens here**:
1. The function receives the current state (all conversation messages)
2. The LLM generates a response based on the conversation history
3. We return the AI's response, which gets appended to the message list

### Building the Graph

```python
from langgraph.graph import START, END, StateGraph

def build_chatbot_graph():
    # Create the graph structure
    builder = StateGraph(ChatState)
    builder.add_node("chatbot", chatbot_node)
    
    # Define the flow: START -> chatbot -> END
    builder.add_edge(START, "chatbot")
    builder.add_edge("chatbot", END)
    
    return builder.compile()
```

**Graph flow**:
- START: Entry point
- chatbot: Our AI response node
- END: Exit point

### Running the Chatbot

The REPL (Read-Eval-Print Loop) lets users chat interactively:

```python
def run_chat_repl(graph):
    state: ChatState = {"messages": []}  # Start with empty conversation
    
    while True:
        user_input = input("User: ")
        if user_input.lower() == 'q':
            break
            
        # Add user message to state
        state["messages"].append({"role": "user", "content": user_input})
        
        # Process through the graph
        for event in graph.stream(state):
            state = event.get("chatbot", state)  # Update state
            
        # Print AI response
        if state["messages"]:
            last_message = state["messages"][-1]
            print(f"Assistant: {last_message.content}")
```

## Part B: Building a ReAct Agent with Tools

### What is a ReAct Agent?

ReAct stands for "Reasoning + Acting". This agent can:
- **Reason**: Think about what it needs to do
- **Act**: Use tools to gather information or perform actions
- **Observe**: Look at tool results and decide next steps

### Creating a Tool

Tools are functions the agent can call. We use the `@tool` decorator:

```python
from langchain_core.tools import tool
from typing import Literal

@tool
def get_weather(city: Literal["nyc", "sf"]) -> str:
    """Get weather for a specific city. Only supports NYC and SF."""
    if city == "nyc":
        return "It might be cloudy in NYC."
    elif city == "sf":
        return "It's always sunny in SF."
    else:
        return "Weather data not available for this city."
```

**Important**:
- The docstring explains what the tool does (the agent reads this!)
- Type hints help the agent understand what inputs are valid
- Return a string that the agent can understand

### Using the Prebuilt ReAct Agent

LangGraph provides a ready-to-use ReAct agent:

```python
from langgraph.prebuilt import create_react_agent

def create_weather_agent():
    llm = ChatGroq(model="gemma2-9b-it", temperature=0.0)
    
    # Create agent with our weather tool
    agent = create_react_agent(llm, tools=[get_weather])
    return agent
```

### How the ReAct Agent Works

When you ask "What's the weather in NYC?", the agent:

1. **Thinks**: "I need weather information for NYC"
2. **Acts**: Calls `get_weather("nyc")`
3. **Observes**: Receives "It might be cloudy in NYC"
4. **Responds**: "Based on the weather tool, it might be cloudy in NYC."

### Running the Agent

```python
def run_agent_demo():
    agent = create_weather_agent()
    
    while True:
        user_input = input("User: ")
        if user_input.lower() == 'q':
            break
            
        # Process user input through the agent
        for event in agent.stream({"messages": [{"role": "user", "content": user_input}]}):
            # Extract and print the final response
            for node_state in event.values():
                messages = node_state.get("messages", [])
                if messages:
                    last_msg = messages[-1]
                    print(f"Assistant: {last_msg.content}")
```

## Key Differences Between Approaches

| Feature | Simple Chatbot | ReAct Agent |
|---------|----------------|-------------|
| **Capabilities** | Just conversation | Conversation + tool usage |
| **Complexity** | Manual graph building | Prebuilt solution |
| **Use Cases** | Chat, Q&A | Research, actions, complex tasks |
| **State Management** | Manual | Handled automatically |

## Common Patterns

### Message Structure
Both approaches use messages with this format:
```python
{"role": "user", "content": "Hello"}
{"role": "assistant", "content": "Hi there!"}
```

### Streaming Events
Both use `.stream()` to process requests:
- Returns events as the graph executes
- Useful for real-time updates
- Final event contains the complete result

### Error Handling
Always check for:
- Missing API keys
- Empty responses
- Tool execution errors

## Next Steps

Try extending these examples:
1. **Add more tools**: Create tools for calculations, web searches, or database queries
2. **Improve state**: Add memory, user preferences, or context
3. **Combine approaches**: Use custom nodes within a ReAct agent
4. **Add persistence**: Save conversations to a database

## Running the Code

```bash
# Run both demos
python script_06.py

# Run just the chatbot
python script_06.py chat

# Run just the ReAct agent
python script_06.py react
```

The script will start an interactive session where you can chat with the AI or test the weather tool functionality.