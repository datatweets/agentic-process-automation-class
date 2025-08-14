import os

from dotenv import load_dotenv
from openai import OpenAI

# Load API key from .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("❌ No API key found in .env file.")
    exit(1)

client = OpenAI(api_key=api_key)

try:
    # Try a very small API call
    response = client.models.list()
    print("✅ API key is valid. Models available:", len(response.data))
except Exception as e:
    print("❌ Invalid API key or connection issue:", e)
