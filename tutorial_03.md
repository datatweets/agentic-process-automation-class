# Building Your First AI Agent: A ReAct Tutorial for Beginners

## Introduction: What Are We Building?

Imagine teaching a smart assistant to solve problems step-by-step, just like a human would. That's exactly what we're going to build today! We'll create a **ReAct Agent** - a simple AI system that can:
- **Think** about problems
- **Use tools** to get information
- **Reason** through solutions
- **Give you answers**

Think of it like having a helpful assistant who can look things up, do calculations, and explain their thinking along the way.

## Part 1: Understanding the ReAct Pattern

### The Magic Loop

Our AI agent follows a simple pattern called **ReAct** (Reason + Act):

```
1. Thought → "I need to find out X"
2. Action → "Let me use this tool"
3. PAUSE → (We run the tool)
4. Observation → "The tool gave me this result"
5. (Repeat if needed)
6. Answer → "Here's your final answer!"
```

### Real-World Example

Let's say you ask: "How much does a pen cost?"

The agent thinks like this:
1. **Thought:** "I should look up the pen cost"
2. **Action:** "Use the price-checking tool for 'pen'"
3. **PAUSE** (waits for us to run the tool)
4. **Observation:** "A pen costs $5"
5. **Answer:** "A pen costs $5"

## Part 2: Setting Up Your Environment

### What You'll Need

1. **Python installed** on your computer
2. **An OpenAI API key** (get one from platform.openai.com)
3. **Three Python packages** to install

### Step-by-Step Setup

1. **Install the required packages:**
   ```bash
   pip install openai python-dotenv httpx
   ```

2. **Create a file called `.env`** in your project folder:
   ```
   OPENAI_API_KEY=sk-your-actual-key-here
   ```
   ⚠️ **Important:** Never share this key or commit it to GitHub!

3. **Create your Python file** called `my_first_agent.py`

## Part 3: Understanding the Code Components

### The Three Tools

Our agent has three "tools" it can use - think of these as special abilities:

```python
def calculate(what):
    """Does math calculations"""
    return eval(what)  # Note: Only safe for classroom use!

def get_cost(thing):
    """Looks up prices from our toy database"""
    if "pen" in thing:
        return "A pen costs $5"
    elif "book" in thing:
        return "A book costs $20"
    else:
        return "Price not found"

def wikipedia(q):
    """Searches Wikipedia for information"""
    # This connects to Wikipedia's API
    # Returns the first search result
```

### The Agent Brain

The `Agent` class is the "brain" of our system:

```python
class Agent:
    def __init__(self, system=""):
        # Stores the instructions (system prompt)
        # Keeps track of the conversation
        
    def __call__(self, message):
        # Receives your message
        # Adds it to the conversation history
        # Gets a response from the AI
        
    def execute(self):
        # Actually calls OpenAI's API
        # Returns the AI's response
```

### The Instructions (System Prompt)

The most important part! This teaches the AI how to behave:

```python
prompt = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer.

Your available actions are:
- calculate: Does math (e.g., calculate: 4 * 7)
- get_cost: Gets prices (e.g., get_cost: pen)
- wikipedia: Searches Wikipedia (e.g., wikipedia: Albert Einstein)

[Examples showing the pattern...]
"""
```

## Part 4: How It All Works Together

### The Flow

```
You → "What's 1879 × 5?"
    ↓
Agent → "Thought: I need to calculate this"
Agent → "Action: calculate: 1879 * 5"
Agent → "PAUSE"
    ↓
Your Code → Runs calculate(1879 * 5) = 9395
Your Code → "Observation: 9395"
    ↓
Agent → "Answer: 1879 × 5 = 9395"
```

### Running the Demos

When you run the code, you'll see two demonstrations:

**Demo 1: Simple Price Lookup**
- Question: "How much does a pen cost?"
- Agent uses the `get_cost` tool
- Gets the price ($5)
- Gives you the answer

**Demo 2: Multi-Step Problem**
- Question: "What year was Einstein born? Multiply by 5"
- Agent first uses `wikipedia` to find the year (1879)
- Then uses `calculate` to multiply (1879 × 5 = 9395)
- Combines both results in the final answer

## Part 5: Understanding the Output

### What You'll See

```
--- Demo 1 ---
Thought: I should look up the pen cost using get_cost.
Action: get_cost: pen
PAUSE
Observation: A pen costs $5
Answer: A pen costs $5
```

### Why Sometimes You See Extra Lines?

If the agent "guesses" before getting the tool result, you might see:
- First answer (the guess): "A pen costs $3"
- Then the observation: "A pen costs $5"
- Updated answer (the correction): "A pen costs $5"

**This is good!** It shows the agent learns from the tools and corrects itself.

## Part 6: Try It Yourself!

### Experiment Ideas

1. **Add a new price** to the `get_cost` function:
   ```python
   elif "laptop" in thing:
       return "A laptop costs $1000"
   ```

2. **Ask different questions:**
   ```python
   my_agent("Question: How much would 3 books cost?")
   # The agent should use get_cost then calculate!
   ```

3. **Create a new tool:**
   ```python
   def get_weather(city):
       return f"The weather in {city} is sunny and 72°F"
   ```

### Common Issues and Solutions

| Problem | Solution |
|---------|----------|
| "API key not found" | Check your `.env` file is in the right folder |
| Agent gives wrong answer | Make sure temperature=0 for consistency |
| Tool not recognized | Add it to the `known_actions` dictionary |
| Agent keeps talking after Answer | In production, stop after the first "Answer:" |

## Part 7: Key Concepts to Remember

### The Mental Model

Think of your ReAct agent like a smart assistant with:
- 🧠 **Brain** (LLM) - Plans and decides
- 🙌 **Hands** (Tools) - Do the actual work
- 👀 **Eyes** (Observations) - See the results
- ✅ **Answer** - The final conclusion

### Why This Matters

1. **Transparency**: You can see the agent's thinking process
2. **Accuracy**: Tools provide real data, not guesses
3. **Control**: You decide what tools are available
4. **Learning**: Perfect for understanding how AI agents work!

## Part 8: Next Steps

Once you're comfortable with this basic agent:

1. **Add more tools** (weather API, calculator with safety checks, database lookups)
2. **Improve the prompt** to handle edge cases
3. **Build a web interface** so others can use your agent
4. **Create specialized agents** for specific tasks (research assistant, math tutor, etc.)

## Summary

Congratulations! You've built your first AI agent that can:
- Think through problems step-by-step
- Use tools to get real information
- Combine multiple steps to solve complex problems
- Explain its reasoning clearly

This ReAct pattern is the foundation of many modern AI applications. You're now ready to explore more advanced agent architectures!

---

## Quick Reference Card

```python
# The ReAct Loop
Thought → Action → PAUSE → Observation → Answer

# Key Commands
"Thought: [reasoning]"      # Agent explains its thinking
"Action: [tool]: [input]"   # Agent requests a tool
"PAUSE"                     # Agent waits for tool result
"Observation: [result]"     # We provide tool output
"Answer: [final answer]"    # Agent gives final response

# Available Tools
calculate: Does math
get_cost: Looks up prices
wikipedia: Searches for information
```

Remember: The beauty of this system is its simplicity. Each piece has one job, and together they create intelligent behavior. Happy coding! 🚀