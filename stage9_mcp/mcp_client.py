import os
import asyncio
from dotenv import load_dotenv
from anthropic import Anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Tells the client how to start the server process
server_params = StdioServerParameters(
    command="python",
    args=["mcp_server.py"],
)


async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # --- Step 1: Discover what tools the server offers ---
            tools_response = await session.list_tools()
            print("Discovered tools:", [t.name for t in tools_response.tools])

            # Convert the MCP tool descriptions into the format Claude's API expects
            claude_tools = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.inputSchema,
                }
                for t in tools_response.tools
            ]

            # --- Step 2: Ask Claude a question, offering the MCP-discovered tools ---
            messages = [{"role": "user", "content": "What is 156 divided by 12?"}]

            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1000,
                tools=claude_tools,
                messages=messages,
            )

            if response.stop_reason == "tool_use":
                tool_use_block = next(b for b in response.content if b.type == "tool_use")
                print(f"Claude wants: {tool_use_block.name} with {tool_use_block.input}")

                # --- Step 3: Call the tool on the MCP server, not a local function ---
                result = await session.call_tool(tool_use_block.name, tool_use_block.input)
                result_text = result.content[0].text
                print(f"MCP server returned: {result_text}")

                # --- Step 4: Send the result back to Claude, same as every prior stage ---
                messages.append({"role": "assistant", "content": response.content})
                messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_use_block.id,
                        "content": result_text,
                    }],
                })

                final = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=1000,
                    tools=claude_tools,
                    messages=messages,
                )
                print("Claude:", final.content[0].text)


asyncio.run(main())