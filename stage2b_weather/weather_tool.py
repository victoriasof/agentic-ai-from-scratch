import os
import requests
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


## --- Step 1: The real function — this one calls an external API ---
def get_weather(latitude, longitude):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,wind_speed_10m"
    }
    response = requests.get(url, params=params)
    data = response.json()
    temp = data["current"]["temperature_2m"]
    wind = data["current"]["wind_speed_10m"]
    return f"{temp}°C, wind speed {wind} km/h"


## --- Step 2: Describe it to Claude ---
tools = [
    {
        "name": "get_weather",
        "description": "Gets the current temperature and wind speed for a location, given its latitude and longitude.",
        "input_schema": {
            "type": "object",
            "properties": {
                "latitude": {"type": "number", "description": "Latitude of the location"},
                "longitude": {"type": "number", "description": "Longitude of the location"}
            },
            "required": ["latitude", "longitude"]
        }
    }
]

## --- Step 3: This is the "memory" — a list that grows as the conversation continues ---
messages = []


def ask(question):
    messages.append({"role": "user", "content": question})

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        tools=tools,
        messages=messages
    )

    if response.stop_reason == "tool_use":
        tool_use_block = next(b for b in response.content if b.type == "tool_use")
        print(f"Claude wants: {tool_use_block.name} with {tool_use_block.input}")

        result = get_weather(**tool_use_block.input)
        print(f"Tool result: {result}")

        messages.append({"role": "assistant", "content": response.content})
        messages.append({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_use_block.id,
                "content": result
            }]
        })

        final = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            tools=tools,
            messages=messages
        )
        answer = final.content[0].text
        messages.append({"role": "assistant", "content": answer})
        print("Claude:", answer)
    else:
        answer = response.content[0].text
        messages.append({"role": "assistant", "content": answer})
        print("Claude:", answer)


## --- Try some questions in a row, in the same conversation ---
ask("What's the weather like in Antwerp right now? Its coordinates are 51.2194, 4.4025")
ask("Is that warmer or colder than 15 degrees?") 

ask("should I wear a jacket?")

#make it interactive:

#while True:
#    user_question = input("You: ")
#    if user_question.lower() in ("quit", "exit"):
#        break
#    ask(user_question)