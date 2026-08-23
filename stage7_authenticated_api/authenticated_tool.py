import os
import requests
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")


## ============================================================
## The tool — this is where authentication actually happens
## ============================================================
def get_repo_info(owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}"

    # This is the authentication step: the token is attached
    # to every request via a header, proving who's asking
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }

    response = requests.get(url, headers=headers)

    # Real APIs fail sometimes — check before assuming success
    if response.status_code == 401:
        return "Authentication failed — check your GITHUB_TOKEN in .env"
    elif response.status_code == 404:
        return f"Repository {owner}/{repo} not found"
    elif response.status_code != 200:
        return f"Request failed with status {response.status_code}"

    data = response.json()
    return (
        f"{data['full_name']}: {data['description']}. "
        f"{data['stargazers_count']} stars. "
        f"Last updated {data['updated_at']}."
    )


## ============================================================
## Describe it to Claude, same pattern as every prior stage
## ============================================================
tools = [
    {
        "name": "get_repo_info",
        "description": "Gets information about a public GitHub repository, including its description, star count, and last update date.",
        "input_schema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string", "description": "The GitHub username or organization that owns the repo"},
                "repo": {"type": "string", "description": "The repository name"}
            },
            "required": ["owner", "repo"]
        }
    }
]

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
        print(f"  [calling get_repo_info with {tool_use_block.input}]")

        result = get_repo_info(**tool_use_block.input)
        print(f"  [result: {result}]")

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


ask("Tell me about the anthropics/anthropic-sdk-python repository on GitHub.")