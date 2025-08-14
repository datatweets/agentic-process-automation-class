# Tutorial: Building an AI Agent with Custom Tools and Memory

This tutorial explains how to create an AI agent that can perform web searches, mathematical calculations, and remember conversations using LangChain and OpenAI's GPT model.

## Prerequisites

```bash
pip install langchain langchain-openai python-dotenv google-search-results
```

You'll need two API keys in your .env file:
```plaintext
OPENAI_API_KEY=your-openai-key
SERPAPI_API_KEY=your-serpapi-key
```

## 1. Setting Up the Environment

```python
from dotenv import load_dotenv
load_dotenv()
```
**What:** Loads environment variables from .env file.  
**Why:** Securely store API keys outside of code.  
**How:** Creates a .env file in your project root and loads it at runtime.

## 2. Creating a Custom Time Tool

```python
from datetime import datetime

def get_time(_: str) -> str:
    """Return current system time as ISO string."""
    return datetime.now().isoformat(sep=" ", timespec="seconds")

time_tool = Tool(
    name="get_time",
    func=get_time,
    description="Returns the current system time (no input required)."
)
```
**What:** A custom tool that returns the current time.  
**Why:** Demonstrates how to create custom tools for the agent.  
**How:** 
- Creates a function that returns the current time
- Wraps it in a LangChain Tool object with a name and description

## 3. Setting Up the Language Model

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
```
**What:** Initializes the GPT model.  
**Why:** This is the brain of our agent.  
**How:** Uses OpenAI's API with temperature=0 for consistent outputs.

## 4. Loading Built-in Tools

```python
from langchain.agents import load_tools

tools = load_tools(["serpapi", "llm-math"], llm=llm)
tools.append(time_tool)
```
**What:** Loads pre-built tools and adds our custom tool.  
**Why:** Gives the agent capabilities for:
- Web searches (serpapi)
- Mathematical calculations (llm-math)
- Getting current time (our custom tool)  
**How:** Uses LangChain's tool loader and appends our custom tool.

## 5. Setting Up Conversation Memory

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history", 
    return_messages=True
)
```
**What:** Creates a memory system for the agent.  
**Why:** Allows the agent to remember previous conversations.  
**How:** Uses a buffer to store conversation history.

## 6. Initializing the Agent

```python
from langchain.agents import AgentType, initialize_agent

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory
)
```
**What:** Creates an agent with all our components.  
**Why:** Combines tools, LLM, and memory into a functional agent.  
**How:**
- Passes all tools to the agent
- Sets the agent type to conversational
- Enables verbose mode for seeing agent's thoughts
- Adds memory for conversation history

## 7. Using the Agent

```python
# Example 1: Search and Math
response = agent.run(
    "What year was Einstein born? What is that year number multiplied by 5?"
)

# Example 2: Custom Tool
response = agent.run("What time is it?")

# Example 3: Memory Usage
response = agent.run(
    "Suggest a few Thai food recipes and remember that I like Thai food."
)
response = agent.run("Which one of those dishes is the spiciest?")
```

**What:** Different examples of agent usage.  
**Why:** Demonstrates various capabilities:
- Web search + math calculations
- Custom tool usage
- Memory across conversations  
**How:** Use the `run()` method with your queries.

## Running the Script

```bash
python script_05.py
```

## Expected Output
The agent will:
1. Search for Einstein's birth year and multiply it
2. Show the current time
3. Remember your Thai food preference across multiple questions

## Common Issues and Solutions

1. **Missing API Keys**
   ```plaintext
   Error: Missing API key
   Solution: Check your .env file has both OPENAI_API_KEY and SERPAPI_API_KEY
   ```

2. **SerpAPI Rate Limits**
   ```plaintext
   Error: Quota exceeded
   Solution: Check your SerpAPI usage or upgrade your plan
   ```

3. **Memory Issues**
   - Memory only persists during the script's runtime
   - For permanent storage, you'll need to implement a database solution

## Next Steps

1. Try creating your own custom tools
2. Experiment with different agent types
3. Implement persistent memory storage
4. Add error handling for API calls

This tutorial covered the basics of creating an AI agent with custom tools and memory. The code is modular and can be extended with additional tools and capabilities as needed.