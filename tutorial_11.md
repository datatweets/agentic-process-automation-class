# AutoGen Multi-Agent Stock Analysis Tutorial: Complete Beginner's Guide

## Table of Contents
1. [What We're Building](#what-were-building)
2. [Understanding the Architecture](#understanding-the-architecture)
3. [Prerequisites and Setup](#prerequisites-and-setup)
4. [Code Walkthrough](#code-walkthrough)
5. [How the Agents Work Together](#how-the-agents-work-together)
6. [Running the System](#running-the-system)
7. [Understanding the Output](#understanding-the-output)
8. [Troubleshooting](#troubleshooting)

---

## What We're Building

We're creating a **Multi-Agent AI System** that analyzes stocks using two specialized AI agents:

- 🧑‍💼 **User Proxy Agent**: Acts as your assistant, executes code, and talks to you
- 🤖 **Assistant Agent**: A Python expert that writes financial analysis code

**Think of it like this**: You ask a question about stocks → Assistant writes Python code → User Proxy runs it → You get charts and analysis!

---

## Understanding the Architecture

### The Team Structure
```
You (Human) ↔ User Proxy Agent ↔ Assistant Agent
     ↑              ↑                    ↑
   Asks         Executes             Writes
  Questions      Code               Python Code
```

### Why Two Agents?
- **Separation of Skills**: One agent writes code, another runs it
- **Safety**: Human oversight on what code gets executed
- **Error Handling**: If code fails, agents can work together to fix it
- **Collaboration**: Agents can discuss and improve solutions

---

## Prerequisites and Setup

### Step 1: Install Required Packages
```bash
pip install autogen-agentchat yfinance matplotlib pandas python-dotenv
```

**What each package does:**
- `autogen-agentchat`: The multi-agent framework
- `yfinance`: Downloads stock data from Yahoo Finance
- `matplotlib`: Creates charts and graphs
- `pandas`: Handles data manipulation
- `python-dotenv`: Reads environment variables from .env files

### Step 2: Get an OpenAI API Key
1. Go to [OpenAI's website](https://platform.openai.com/)
2. Sign up/login and create an API key
3. Copy your key (it starts with "sk-...")

### Step 3: Create Environment File
Create a file named `.env` in your project folder:
```
OPENAI_API_KEY=sk-your-actual-api-key-here
```

**Important**: Replace `sk-your-actual-api-key-here` with your real API key!

---

## Code Walkthrough

Let's go through the code step by step, explaining what each part does.

### Part 1: Imports and Setup

```python
import os
from dotenv import load_dotenv
from autogen import ConversableAgent, UserProxyAgent

# Load environment variables from .env file
load_dotenv()
```

**What this does:**
- `os`: Helps us read environment variables
- `load_dotenv()`: Reads our .env file to get the API key
- `ConversableAgent, UserProxyAgent`: The building blocks for our AI agents

### Part 2: The Main Class Structure

```python
class StockAnalysisSystem:
    """Multi-Agent Stock Analysis System with User Proxy and Assistant agents."""
    
    def __init__(self):
        """Initialize the multi-agent system."""
        self.validate_api_key()
        self.setup_config()
        self.create_agents()
```

**What this does:**
- Creates a class to organize our system
- `__init__` runs when we create the system
- Validates API key → Sets up configuration → Creates agents

### Part 3: API Key Validation

```python
def validate_api_key(self):
    """Check if OpenAI API key is available."""
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OpenAI API key not found!")
        print("Please create a .env file with:")
        print("OPENAI_API_KEY=your_api_key_here")
        exit(1)
```

**What this does:**
- Checks if the API key exists in environment variables
- If not found, shows clear error message and stops the program
- Prevents confusing errors later

### Part 4: LLM Configuration

```python
def setup_config(self):
    """Configure the LLM settings for AutoGen."""
    self.config_list = [{
        "model": "gpt-4o-mini",
        "api_key": os.getenv("OPENAI_API_KEY"),
    }]
    
    self.llm_config = {
        "config_list": self.config_list,
        "temperature": 0.1,
        "timeout": 60,
    }
```

**What this does:**
- **Model**: Uses GPT-4o-mini (cost-effective and capable)
- **Temperature**: 0.1 = more predictable, consistent responses
- **Timeout**: 60 seconds maximum wait time for responses

### Part 5: Creating the User Proxy Agent

```python
self.user_proxy = UserProxyAgent(
    name="UserProxy",
    system_message="""You are a User Proxy Agent for stock analysis.
    - Execute Python code provided by the Assistant Agent
    - Handle package installation if needed
    - Provide feedback on outputs
    - Ask for human input when needed""",
    
    human_input_mode="ALWAYS",
    max_consecutive_auto_reply=3,
    is_termination_msg=lambda x: "TERMINATE" in x.get("content", ""),
    
    code_execution_config={
        "work_dir": "stock_analysis",
        "use_docker": False,
    },
    llm_config=self.llm_config
)
```

**Breaking this down:**

**System Message**: Tells the agent its role and responsibilities

**Key Settings:**
- `human_input_mode="ALWAYS"`: Always asks human before proceeding
- `max_consecutive_auto_reply=3`: Limits automatic responses
- `is_termination_msg`: Knows when to stop (when sees "TERMINATE")

**Code Execution:**
- `work_dir="stock_analysis"`: Creates folder for code execution
- `use_docker=False`: Runs code directly (not in Docker container)

### Part 6: Creating the Assistant Agent

```python
self.assistant = ConversableAgent(
    name="AssistantAgent", 
    system_message="""You are an expert Python programmer for financial data analysis.
    
    Your tasks:
    1. Write Python code for stock analysis and visualization
    2. Use yfinance for data fetching
    3. Create matplotlib visualizations
    4. Handle errors gracefully
    5. Install required packages when needed
    
    Code requirements:
    - Always import required libraries
    - Include error handling
    - Add helpful comments
    - Create clear visualizations""",
    
    llm_config=self.llm_config,
    human_input_mode="NEVER",
)
```

**What this does:**
- Creates a specialized Python coding agent
- `human_input_mode="NEVER"`: This agent doesn't need human input
- Detailed instructions on how to write good financial analysis code

### Part 7: Starting the Analysis

```python
def start_analysis(self, request=None):
    """Start the multi-agent stock analysis conversation."""
    if not request:
        try:
            request = input("Enter your stock analysis request: ").strip()
            if not request:
                print("❌ No request provided.")
                return None
        except (EOFError, KeyboardInterrupt):
            print("❌ Input not available. Use start_analysis('your request') directly.")
            return None
    
    print("🚀 Starting Multi-Agent Stock Analysis")
    print("=" * 50)
    print(f"📋 Request: {request}")
    print("=" * 50)
    
    # Start the conversation between agents
    try:
        chat_result = self.user_proxy.initiate_chat(
            self.assistant,
            message=request,
            max_turns=10
        )
        return chat_result
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return None
```

**What this does:**
- Gets your request (either from parameter or user input)
- Handles input errors gracefully
- Starts conversation between User Proxy and Assistant
- `max_turns=10`: Limits conversation to prevent infinite loops

---

## How the Agents Work Together

### The Conversation Flow

Here's exactly what happens when you run the system:

#### Step 1: You Make a Request
```
You: "Plot META and TESLA stock price YTD"
```

#### Step 2: User Proxy Forwards to Assistant
```
User Proxy → Assistant: "The user wants to plot META and TESLA stock price YTD"
```

#### Step 3: Assistant Writes Code
```python
Assistant generates:
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

# Get YTD data for META and TESLA
start_date = f"{datetime.now().year}-01-01"
meta = yf.download("META", start=start_date)
tesla = yf.download("TSLA", start=start_date)

# Create visualization
plt.figure(figsize=(12, 6))
plt.plot(meta.index, meta['Close'], label='META')
plt.plot(tesla.index, tesla['Close'], label='TESLA')
plt.title('META vs TESLA YTD Stock Price')
plt.legend()
plt.show()
```

#### Step 4: User Proxy Executes Code
- User Proxy asks for your permission
- You approve → Code runs
- Charts appear on your screen

#### Step 5: Feedback Loop
If there are errors or you want changes:
```
You: "Show percentage change instead of absolute prices"
User Proxy → Assistant: "User wants percentage change"
Assistant: Writes new code with percentage calculations
```

### Error Handling Example

**What happens if packages are missing:**

1. **Assistant** writes code using `yfinance`
2. **User Proxy** tries to run it
3. **Error**: "ModuleNotFoundError: No module named 'yfinance'"
4. **User Proxy** reports error to Assistant
5. **Assistant** responds: "Let's install yfinance first"
6. **User Proxy** runs: `pip install yfinance`
7. **User Proxy** re-runs the original code
8. **Success**: Charts appear!

---

## Running the System

### Method 1: Using the Main Function
```bash
python stock_analysis.py
```

**What you'll see:**
```
🎯 Multi-Agent Stock Analysis System
========================================

Example requests:
- 'Plot META and TESLA stock price YTD'
- 'Compare Apple and Microsoft performance'
- 'Show Amazon stock with moving averages'

🧪 Running test with: Plot META and TESLA stock price change YTD
```

### Method 2: Interactive Mode
```python
# In Python console or script
system = StockAnalysisSystem()
system.start_analysis("Compare Apple and Google stock performance over 6 months")
```

### Method 3: Custom Requests
You can ask for any stock analysis:

```python
system.start_analysis("Show Tesla stock volatility with Bollinger Bands")
system.start_analysis("Create a correlation heatmap for tech stocks")
system.start_analysis("Analyze S&P 500 performance with moving averages")
```

---

## Understanding the Output

### What You'll See in the Console

#### 1. System Initialization
```
🚀 Starting Multi-Agent Stock Analysis
==================================================
📋 Request: Plot META and TESLA stock price YTD
==================================================
```

#### 2. Agent Conversation
```
UserProxy: I'll help you analyze META and TESLA stocks. Let me work with the Assistant to create this visualization.

AssistantAgent: I'll write Python code to fetch and plot the YTD stock prices for META and TESLA.

[Code block appears here]

UserProxy: I'll execute this code for you. 

[Charts and analysis appear]
```

#### 3. Human Interaction Points
```
Do you want to proceed with executing this code? (y/n): y
```

### Types of Visualizations You'll Get

#### 1. **Stock Price Charts**
- Line charts showing price over time
- Multiple stocks on same chart
- Professional formatting with legends

#### 2. **Percentage Change Analysis**
- Shows relative performance
- Better for comparing different-priced stocks
- Normalized to starting point

#### 3. **Technical Analysis**
- Moving averages
- Bollinger Bands
- RSI indicators
- Volume analysis

---

## Example Conversations

### Example 1: Basic Stock Comparison
```
You: "Compare Apple and Microsoft stock this year"