import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

user_input = input("You: ")

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1000,
    messages=[
        {"role": "user", "content": user_input}
    ]
)

print("Claude:", response.content[0].text)