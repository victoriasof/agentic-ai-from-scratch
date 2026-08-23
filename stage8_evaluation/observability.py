import time
import json
from datetime import datetime


def log_interaction(question, tool_calls, answer, duration_seconds, input_tokens, output_tokens):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "tool_calls": tool_calls,
        "answer": answer,
        "duration_seconds": round(duration_seconds, 2),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
    with open("agent_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")