# Beginner-Friendly Tutorial: How This Python AI Assistant Script Works

This tutorial will walk you through the code in script_02.py line by line and function by function. The script is an interactive Python assistant that can answer questions about the weather or Wikipedia topics using OpenAI's GPT model and external APIs.

---

## 1. **Importing Required Libraries**

```python
import json
import os

import requests
from dotenv import load_dotenv
from openai import OpenAI
```
- `json` and `os`: Standard Python libraries for handling JSON data and environment variables.
- `requests`: Lets you make HTTP requests to APIs.
- `dotenv`: Loads environment variables from a .env file (like your API key).
- `openai`: The official OpenAI Python library for accessing GPT models.

---

## 2. **Loading the OpenAI API Key**

```python
# Load API key from .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ No API key found in .env file.")
    exit(1)

client = OpenAI(api_key=api_key)
```
- `load_dotenv()`: Loads variables from a .env file into your environment.
- `os.getenv("OPENAI_API_KEY")`: Gets your OpenAI API key.
- If the key is missing, the script prints an error and stops.
- `client = OpenAI(api_key=api_key)`: Creates an OpenAI client for making requests.

---

## 3. **Weather Function**

```python
def get_weather(city_name):
    try:
        result_city = requests.get(
            url="https://geocoding-api.open-meteo.com/v1/search?name=" + city_name
        )
        result_city.raise_for_status()
        location = result_city.json()
        if "results" not in location or not location["results"]:
            return f"Could not find location for {city_name}."
        longitude = str(location["results"][0]["longitude"])
        latitude = str(location["results"][0]["latitude"])
        BASE_URL = "https://api.open-meteo.com/v1/forecast"
        complete_url = f"{BASE_URL}?latitude={latitude}&longitude={longitude}&hourly=temperature_2m,weathercode"
        response = requests.get(complete_url)
        response.raise_for_status()
        data = response.json()
        current_weather = data["hourly"]["temperature_2m"][0]
        weather_sentence = (
            f"The current temperature in {city_name} is {current_weather}°C."
        )
        return weather_sentence
    except Exception as e:
        return f"Error fetching weather: {e}"
```
- **Purpose:** Gets the current temperature for a city.
- **How it works:**
  - Calls a geocoding API to get the city's latitude and longitude.
  - Uses those coordinates to call a weather API.
  - Returns the current temperature.
- **Error Handling:** If anything goes wrong, it returns an error message.

---

## 4. **Wikipedia Function**

```python
def get_wikipedia_info(topic):
    try:
        apiUrl = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "titles": topic,
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
        }
        response = requests.get(apiUrl, params=params)
        response.raise_for_status()
        data = response.json()
        page = next(iter(data["query"]["pages"].values()))
        formattedWikiInfo = page.get("extract", "No information found.")
        return formattedWikiInfo
    except Exception as e:
        return f"Error fetching Wikipedia info: {e}"
```
- **Purpose:** Gets a summary about a topic from Wikipedia.
- **How it works:**
  - Calls Wikipedia's API with the topic.
  - Extracts and returns the summary text.
- **Error Handling:** Returns an error message if something fails.

---

## 5. **Defining the Tools for the AI**

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Fetch weather data for a specified city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city_name": {
                        "type": "string",
                        "description": "The name of the city to fetch weather data for",
                    },
                },
                "required": ["city_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_wikipedia_info",
            "description": "Fetch information from Wikipedia for a specified topic",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The topic to fetch information about from Wikipedia",
                    },
                },
                "required": ["topic"],
            },
        },
    },
]
```
- **Purpose:** Tells the OpenAI model which functions it can call and what parameters they need.
- **How it works:** Each function is described with its name, what it does, and what arguments it takes.

---

## 6. **Getting User Input**

```python
user_query = input("Please enter your query: ")
```
- **Purpose:** Asks the user for a question or command.

---

## 7. **Preparing the Conversation for the AI**

```python
messages = [
    {"role": "system", "content": "I am a helpful assistant."},
    {"role": "user", "content": user_query},
]
```
- **Purpose:** Sets up the conversation history for the AI.
- **How it works:** The system message sets the assistant's behavior, and the user message is your question.

---

## 8. **Calling the OpenAI Chat Model**

```python
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages,
    tools=tools,
    tool_choice="auto",
)
tool_calls = response.choices[0].message.tool_calls
```
- **Purpose:** Asks the AI to answer the user's question, possibly by calling one of the defined functions.
- **How it works:** The AI decides if it needs to call a function and with what arguments.

---

## 9. **Mapping Function Names to Python Functions**

```python
available_functions = {
    "get_weather": get_weather,
    "get_wikipedia_info": get_wikipedia_info,
}
```
- **Purpose:** Lets the script call the correct Python function based on the AI's response.

---

## 10. **Handling the AI's Function Calls**

```python
if tool_calls:
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_to_call = available_functions.get(function_name)
        if function_to_call:
            print(f"Calling function: {function_name}")
            arguments = json.loads(tool_call.function.arguments)
            result = function_to_call(**arguments)
            print(result)
        else:
            print(f"Function {function_name} not found.")
else:
    print("No function call detected.")
```
- **Purpose:** Checks if the AI wants to call a function.
- **How it works:**
  - For each function call the AI suggests:
    - Looks up the function.
    - Parses the arguments.
    - Calls the function and prints the result.
  - If no function is called, it prints a message.

---

## **Summary**

- The script loads your OpenAI API key and sets up two tools: weather and Wikipedia lookup.
- It asks you for a question.
- It sends your question to OpenAI's GPT model, which decides if it needs to call a function.
- If so, it calls the function and prints the answer.

---

## **How to Use**

1. **Install dependencies:**  
   ```
   pip install openai python-dotenv requests
   ```
2. **Create a .env file** in the same folder with this line:  
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```
3. **Run the script:**  
   ```
   python script_02.py
   ```
4. **Type your question!**  
   - Example: `What's the weather in Paris?`
   - Example: `Tell me about Python programming language.`

---

**Congratulations!**  
You now understand how this AI assistant script works, step by step.