# Tavily Search Tutorial: Web Search with AI Agents

This tutorial demonstrates three different approaches to web search using Tavily, a search API optimized for AI applications. You'll learn how to perform direct searches, extract structured data, and build intelligent agents that can search the web.

## Prerequisites

Before starting, make sure you have:
- Python installed on your system
- A Tavily API key (sign up at [tavily.com](https://tavily.com))
- Either an OpenAI API key OR a Groq API key for the AI agent
- Required packages: `tavily-python`, `langchain-tavily`, `langchain-openai`, `langchain-groq`, `langgraph`, `python-dotenv`

## Setup

1. Create a `.env` file in your project directory:
```bash
TAVILY_API_KEY=your_tavily_api_key_here
OPENAI_API_KEY=your_openai_key_here  # OR
GROQ_API_KEY=your_groq_key_here      # Use either OpenAI or Groq
```

2. Install required packages:
```bash
pip install tavily-python langchain-tavily langchain-openai langchain-groq langgraph python-dotenv
```

## What is Tavily?

Tavily is a search API designed specifically for AI applications. Unlike traditional search APIs, Tavily:
- Provides AI-optimized search results
- Can return direct answers to questions
- Offers structured data extraction
- Works well with AI agents and chatbots

## Part A: Direct Question Answering

### The Simplest Approach

The most straightforward way to use Tavily is to ask it a question and get a direct answer:

```python
from tavily import TavilyClient

def demo_direct_answer():
    client = TavilyClient(api_key="your_tavily_key")
    question = "What is in Nvidia's new Blackwell GPU?"
    
    # include_answer=True tells Tavily to provide a direct answer
    result = client.search(question, include_answer=True)
    
    print("Q:", question)
    print("A:", result.get("answer"))
```

**How it works**:
1. Tavily searches the web for relevant information
2. Uses AI to synthesize an answer from multiple sources
3. Returns a concise, factual response

**When to use this**:
- Simple question-answering
- Quick fact lookup
- When you need a direct answer without sources

## Part B: Extracting Structured Search Results

### Getting Raw Search Data

Sometimes you need the actual search results, not just an answer. This approach gives you structured data:

```python
def demo_first_result_json():
    client = TavilyClient(api_key="your_tavily_key")
    query = "current weather in Singapore"
    
    # max_results=1 limits to just the first result
    result = client.search(query, max_results=1)
    
    # Extract the first result's content
    results = result.get("results", [])
    if results:
        content = results[0].get("content")
        print("First result content:")
        print(content)
```

**What you get**:
- URL of the source
- Title of the page
- Extracted text content
- Relevance score
- Publication date (if available)

**When to use this**:
- When you need source attribution
- Building search interfaces
- Processing multiple results
- Extracting specific data formats (like weather data)

## Part C: Intelligent Search Agents

### Building a ReAct Agent with Search

The most powerful approach combines Tavily with an AI agent that can reason about when and how to search:

```python
from langchain_tavily import TavilySearch
from langchain_core.tools import Tool
from langgraph.prebuilt import create_react_agent

def create_search_agent():
    # Create the search tool
    tavily_tool = TavilySearch(
        max_results=5,
        include_answer=False,  # Let the agent synthesize answers
        api_key="your_tavily_key"
    )
    
    # Wrap it as a LangChain tool
    search_tool = Tool.from_function(
        name="web_search",
        func=tavily_tool.run,
        description="Search the web for current information. Use for facts, news, or recent data."
    )
    
    # Choose LLM (OpenAI or Groq)
    if os.getenv("OPENAI_API_KEY"):
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    else:
        llm = ChatGroq(model="gemma2-9b-it", temperature=0)
    
    # Create the ReAct agent
    agent = create_react_agent(llm, tools=[search_tool])
    return agent
```

### How the Agent Works

When you ask a complex question like:
> "Who won the women's tennis singles at the 2024 Olympics? In what country is the champion located? What is the GDP of that country in 2023?"

The agent follows this process:

1. **Analyze**: Breaks down the multi-part question
2. **Search**: "2024 Olympics women's tennis singles winner"
3. **Extract**: Finds that Zheng Qinwen from China won
4. **Search Again**: "China GDP 2023"
5. **Synthesize**: Combines all information into a comprehensive answer

### Running the Agent

```python
def run_search_agent():
    agent = create_search_agent()
    
    # Complex multi-part question
    query = """
    Who won the women's tennis singles at the 2024 Olympics? 
    In what country is the champion located? 
    What is the GDP of that country in 2023? 
    Answer each question clearly.
    """
    
    # Stream the agent's work
    for event in agent.stream({"messages": [{"role": "user", "content": query}]}):
        # Extract the final response
        for node_state in event.values():
            messages = node_state.get("messages", [])
            if messages:
                last_msg = messages[-1]
                print("Assistant:", last_msg.content)
```

## Comparison of Approaches

| Approach | Best For | Pros | Cons |
|----------|----------|------|------|
| **Direct Answer** | Simple Q&A | Fast, concise | No sources, limited complexity |
| **Structured Results** | Data extraction | Full control, sources included | Requires processing |
| **AI Agent** | Complex queries | Multi-step reasoning, flexible | More complex, uses more tokens |

## Advanced Features

### Search Parameters

You can customize Tavily searches with various parameters:

```python
result = client.search(
    query="artificial intelligence trends 2024",
    max_results=10,           # Number of results
    include_answer=True,      # Get direct answer
    include_raw_content=True, # Include full page content
    include_domains=["techcrunch.com", "wired.com"],  # Specific domains
    exclude_domains=["spam-site.com"]  # Exclude domains
)
```

### Error Handling

Always include proper error handling:

```python
def safe_search(query):
    try:
        client = TavilyClient(api_key=get_tavily_api_key())
        result = client.search(query, include_answer=True)
        return result.get("answer", "No answer found")
    except Exception as e:
        return f"Search error: {str(e)}"
```

## Real-World Use Cases

### News Monitoring Agent
```python
# Agent that searches for latest news on specific topics
query = "Latest developments in renewable energy storage technology"
```

### Research Assistant
```python
# Multi-step research with source verification
query = "Compare the latest electric vehicle sales in Europe vs USA in 2024"
```

### Fact Checker
```python
# Verify claims against current information
query = "Is it true that cryptocurrency adoption increased in developing countries in 2024?"
```

## Best Practices

### 1. Choose the Right Approach
- **Direct answers**: Simple facts or definitions
- **Structured results**: When you need sources or multiple perspectives
- **AI agents**: Complex, multi-step queries

### 2. Optimize Query Design
```python
# Good: Specific and clear
"GDP of Japan in 2023"

# Better: More context for complex topics
"Japan's nominal GDP in 2023 compared to previous year"

# Best: Multi-part for comprehensive answers
"What was Japan's GDP in 2023? How did it compare to 2022? What were the main economic factors?"
```

### 3. Handle Rate Limits
```python
import time

def search_with_backoff(query, retries=3):
    for attempt in range(retries):
        try:
            return client.search(query)
        except Exception as e:
            if "rate limit" in str(e).lower():
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise e
```

### 4. Validate Results
Always check that search results are relevant and recent:

```python
def validate_search_result(result, query):
    # Check if result exists
    if not result.get("results"):
        return False
    
    # Check relevance (basic keyword matching)
    content = result["results"][0].get("content", "").lower()
    query_terms = query.lower().split()
    
    return any(term in content for term in query_terms)
```

## Running the Complete Example

The script demonstrates all three approaches:

```bash
# Run all demos
python script_07.py
```

**What you'll see**:
1. Direct answer about Nvidia's Blackwell GPU
2. Structured weather data from Singapore
3. Complex multi-step search about Olympics, countries, and GDP

## Next Steps

Try extending these examples:
1. **Build a news summarizer**: Search for recent news and create summaries
2. **Create a research tool**: Multi-step fact gathering with source citations
3. **Make a comparison agent**: Search and compare different topics side-by-side
4. **Add memory**: Let the agent remember previous searches in a conversation

## Troubleshooting

### Common Issues

1. **"No API key found"**: Check your `.env` file has `TAVILY_API_KEY`
2. **Rate limit errors**: Add delays between requests
3. **No results**: Try broader search terms or check internet connection
4. **Agent not searching**: Ensure the tool description is clear and specific

### API Key Priority

The script automatically chooses your LLM provider:
1. If `OPENAI_API_KEY` exists → Uses OpenAI GPT-4o-mini
2. Else if `GROQ_API_KEY` exists → Uses Groq Gemma2-9b-it
3. Else → Raises error

This flexibility lets you use either provider based on your preference and API availability.