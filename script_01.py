import datetime
import json
import os

import pytz
from dotenv import load_dotenv
from openai import OpenAI

# Load API key from .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Create OpenAI client
client = OpenAI(api_key=api_key)

# Function definition (your JSON schema)
functions = [
    {
        "name": "get_current_time",
        "description": "Gets the current time for a location",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    }
]


# Function implementation
def get_current_time(location: str):
    try:
        tz = pytz.timezone(location)
        return {"time": datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")}
    except Exception as e:
        return {"error": str(e)}


# User's question
user_message = "What's the current time in Asia/Kuala_Lumpur?"

# Step 1: Ask the model and provide the tool definition
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": user_message}],
    tools=[{"type": "function", "function": functions[0]}],
)

# Step 2: Get tool call
tool_call = response.choices[0].message.tool_calls[0]
args = json.loads(tool_call.function.arguments)  # safer parsing

# Step 3: Run the function locally
function_result = get_current_time(**args)
print(function_result)
# Step 4: Send result back to model for natural language reply
final_response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": user_message},
        {"role": "assistant", "tool_calls": response.choices[0].message.tool_calls},
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(function_result),
        },
    ],
)

print(final_response.choices[0].message.content)
