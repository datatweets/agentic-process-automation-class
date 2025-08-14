"""
Robust DuckDuckGo search with LangChain + ddgs (no deprecation warnings).
- Reads OPENAI_API_KEY from .env
- LLM compresses user request -> search query
- ddgs performs the search with a retry strategy

Requirements:
pip install langchain langchain-openai python-dotenv ddgs
"""

import os
from typing import Dict, List, Optional

from ddgs import DDGS  # duckduckgo_search has been renamed to ddgs!
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Load environment variables from .env
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    print("❌ No OPENAI_API_KEY found in .env file.")
    exit(1)


def ddg_search(
    query: str, max_results: int = 5, timelimit: Optional[str] = "d"
) -> List[Dict]:
    """
    Perform a DuckDuckGo search using ddgs directly.
    timelimit: None | 'd' (day) | 'w' | 'm' | 'y'
    Returns: list of {title, href, body}
    """
    results: List[Dict] = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(
                query=query,  # Changed from keywords to query
                region="wt-wt",
                safesearch="moderate",
                timelimit=timelimit,
                max_results=max_results,
            ):
                results.append(r)
    except Exception as e:
        print(f"Error during DuckDuckGo search: {e}")
        return results
    return results


def search_with_retries(query: str) -> List[Dict]:
    """
    Try strict → relaxed:
    1) timelimit='d' (fresh news today)
    2) timelimit='w' (past week)
    3) broaden query by removing quotes and generic words, timelimit=None
    """
    # 1) Today
    items = ddg_search(query, max_results=6, timelimit="d")
    if items:
        return items

    # 2) Past week
    items = ddg_search(query, max_results=6, timelimit="w")
    if items:
        return items

    # 3) Broaden query
    broadened = (
        query.replace('"', "").replace(" tonight", "").replace(" schedule", "").strip()
    )
    if broadened == query:
        broadened = f"{query} fixtures live"
    return ddg_search(broadened, max_results=6, timelimit=None)


if __name__ == "__main__":
    # --- LLM: compress user request into a short, keyword-rich query ----------
    prompt = ChatPromptTemplate.from_template(
        "Rewrite the user's request as a concise web search query (few keywords, no punctuation):\n\nUser: {input}"
    )
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    to_query = prompt | model | StrOutputParser()

    user_input = input("Enter your search request: ").strip()
    if not user_input:
        print("No input provided.")
        exit(0)

    query = to_query.invoke({"input": user_input})

    results = search_with_retries(query)

    print("\n--- User input ---")
    print(user_input)
    print("\n--- Generated query ---")
    print(query)

    print("\n--- Top results ---")
    if not results:
        print("(no results found; try a different phrasing or remove e.g. 'tonight')")
    else:
        for i, r in enumerate(results, 1):
            title = r.get("title", "").strip()
            href = r.get("href", "").strip()
            body = r.get("body", "").strip()
            print(f"{i}. {title}\n   {href}\n   {body}\n")
