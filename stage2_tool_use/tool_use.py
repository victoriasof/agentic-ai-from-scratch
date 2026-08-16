import os
import json
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


## --- Step 1: The actual function Claude will be able to call ---
def calculator(a, b, operation):
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        return a / b
    else:
        return "Unknown operation"


## --- Step 2: Describe that function to Claude, in a format it understands ---
tools = [
    {
        "name": "calculator",
        "description": "Performs basic arithmetic: add, subtract, multiply, or divide two numbers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "The first number"},
                "b": {"type": "number", "description": "The second number"},
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "Which operation to perform"
                }
            },
            "required": ["a", "b", "operation"]
        }
    }
]

## --- Step 3: Send the user's question, along with the tool description ---
user_question = input("You: ")

messages = [{"role": "user", "content": user_question}]

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1000,
    tools=tools,
    messages=messages
)

print("--- First response from Claude ---")
print(response.stop_reason)  # will show "tool_use" if Claude wants to call the tool

## --- Step 4: Check if Claude wants to use the tool, and if so, run it ---
if response.stop_reason == "tool_use":
    # Find the tool_use block in Claude's response
    tool_use_block = next(
        block for block in response.content if block.type == "tool_use"
    )

    tool_name = tool_use_block.name
    tool_input = tool_use_block.input
    print(f"Claude wants to call: {tool_name} with {tool_input}")

    # Actually run the function, using the inputs Claude chose
    if tool_name == "calculator":
        result = calculator(**tool_input)

    print(f"Tool result: {result}")

    # --- Step 5: Send the result back to Claude so it can give a final answer ---
    messages.append({"role": "assistant", "content": response.content})
    messages.append({
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_use_block.id,
                "content": str(result)
            }
        ]
    })

    final_response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        tools=tools,
        messages=messages
    )

    print("--- Final answer ---")
    print(final_response.content[0].text)
else:
    # Claude answered directly without needing the tool
    print(response.content[0].text)