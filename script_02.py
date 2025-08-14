import json
import os

import requests
from dotenv import load_dotenv
from openai import OpenAI

# Load API key from .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ No API key found in .env file.")
    exit(1)

client = OpenAI(api_key=api_key)


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


# Define the functions with their descriptions and parameters
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

user_query = input("Please enter your query: ")

# Prepare the conversation message
messages = [
    {"role": "system", "content": "I am a helpful assistant."},
    {"role": "user", "content": user_query},
]

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages,
    tools=tools,
    tool_choice="auto",
)
tool_calls = response.choices[0].message.tool_calls

# Map functions to their names
available_functions = {
    "get_weather": get_weather,
    "get_wikipedia_info": get_wikipedia_info,
}

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
