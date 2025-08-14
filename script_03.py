"""
Create Simple ReAct Agent from Scratch
=====================================================

This script mirrors the slide snippets as closely as possible:

- Agent class with __init__, __call__, and execute() that hits OpenAI Chat Completions.
- A long 'prompt' string describing the loop: Thought -> Action -> PAUSE -> Observation -> Answer.
- Three tools: calculate (eval-based like the slide), get_cost (toy DB), wikipedia (httpx to Wikipedia API).
- A small driver that:
   1) asks "How much does a pen cost?" → PAUSE → sends Observation with get_cost('pen') → Answer
   2) asks "What year was Albert Einstein born? Multiply that year by 5."
      → PAUSE → sends Observation with wikipedia('Albert Einstein birth year')
      → PAUSE → sends Observation with calculate('1879 * 5') → Answer

Setup
-----
1) pip install -U openai python-dotenv httpx
2) Create .env in the same folder:
     OPENAI_API_KEY=sk-your-real-key
3) python react_agent_slides_exact.py
"""

import os

import httpx  # per the slide's wikipedia example
from dotenv import load_dotenv
from openai import OpenAI

# ------------- Load API key from .env (added for safe classroom use) -------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise SystemExit(
        "Missing OPENAI_API_KEY. Create a .env file with OPENAI_API_KEY=..."
    )

client = OpenAI(api_key=api_key)


# ------------- Tools (exactly as on slides) -------------
def calculate(what):
    """Slide-style calculate: uses Python's eval() like the slide snippet."""
    return eval(what)


def get_cost(thing):
    """Slide-style toy cost lookup."""
    if "pen" in thing:
        return "A pen costs $5"
    elif "book" in thing:
        return "A book costs $20"
    elif "stapler" in thing:
        return "A stapler costs $10"
    else:
        return "A random thing for writing costs $12."


def wikipedia(q):
    """Slide-style wikipedia search using httpx.get."""
    resp = httpx.get(
        "https://en.wikipedia.org/w/api.php",
        params={"action": "query", "list": "search", "srsearch": q, "format": "json"},
        timeout=10.0,
    )
    data = resp.json()
    results = data.get("query", {}).get("search", [])
    if not results:
        return None
    return results[0].get("snippet", None)


# ------------- Agent class (same shape as slide 39) -------------
class Agent:
    def __init__(self, system=""):
        self.system = system
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": self.system})

    def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self):
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",  # same model line as on the slide
            temperature=0,
            messages=self.messages,
        )
        return completion.choices[0].message.content


# ------------- Prompt text (modeled after slide 40) -------------
prompt = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer.

Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you – then return PAUSE.
Observation will be the result of running those actions.

Your available actions are:

calculate:
e.g., calculate: 4 * 7 / 3
Runs a calculation and returns the number — uses Python so be sure to use floating point syntax if necessary.

get_cost:
e.g., get_cost: book
returns the cost of a book

wikipedia:
e.g., wikipedia: Albert Einstein birth year
Returns a summary from searching Wikipedia

Always look things up on Wikipedia if you have the opportunity to do so.

Example session #1:
Question: How much does a pen cost?
Thought: I should look up the pen cost using get_cost
Action: get_cost: pen
PAUSE
Observation: A pen costs $5
You then output:
Answer: A pen costs $5

Example session #2:
Question: What year was Albert Einstein born? What is that year number multiplied by 5?
Thought: I should find the year using wikipedia
Action: wikipedia: Albert Einstein birth year
PAUSE
Observation: March 14, 1879
Thought: Now I can use the calculator to multiply 1879 by 5.
Action: calculate: 1879 * 5
PAUSE
Observation: 9395
You then output:
Answer: The year Albert Einstein was born is 1879. When multiplied by 5, the result is 9395.
"""

# ------------- Demo driver (matches slide 42 flow: manual Observation steps) -------------
if __name__ == "__main__":
    # Map function names to functions, like the slide shows
    known_actions = {
        "calculate": calculate,
        "get_cost": get_cost,
        "wikipedia": wikipedia,
    }

    # Create the agent with the slide-style prompt
    my_agent = Agent(prompt)

    # ---- Example / Demo 1: "How much does a pen cost?" ----
    print("\n--- Demo 1 ---")
    result = my_agent("Question: How much does a pen cost?")
    print(result)  # expect Thought + Action + PAUSE

    # The slide then feeds back an Observation produced by calling the tool directly
    obs1 = f"Observation: {get_cost('pen')}"
    print(obs1)
    result = my_agent(obs1)
    print(result)  # expect final Answer line

    # ---- Example / Demo 2: Einstein birth year * 5 ----
    print("\n--- Demo 2 ---")
    result = my_agent(
        "Question: What year was Albert Einstein born? What is that year number multiplied by 5?"
    )
    print(result)  # expect Thought + Action wikipedia + PAUSE

    # First observation from wikipedia
    wiki_obs = wikipedia("Albert Einstein birth year") or "No result"
    obs2 = f"Observation: {wiki_obs if wiki_obs else 'No result'}"
    print(obs2)
    result = my_agent(obs2)
    print(result)  # expect Thought + Action calculate + PAUSE

    # For classroom simplicity (as per slide text), we multiply 1879 * 5
    calc_obs = f"Observation: {calculate('1879 * 5')}"
    print(calc_obs)
    result = my_agent(calc_obs)
    print(result)  # expect final Answer
