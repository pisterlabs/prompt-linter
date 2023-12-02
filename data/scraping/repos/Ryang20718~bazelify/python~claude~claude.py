import anthropic
import os
from dotenv import load_dotenv

load_dotenv()

def claude_api(prompt: str) -> str:
    api_key=os.getenv("ANTHROPIC_API_KEY")
    client = anthropic.Client(api_key)
    response = client.completion(
        prompt=f"{anthropic.HUMAN_PROMPT} {prompt}?{anthropic.AI_PROMPT}",
        stop_sequences = [anthropic.HUMAN_PROMPT],
        model="claude-v1-100k",
        temperature=0,
        max_tokens_to_sample=70000)

    if response["exception"] is None:
        # print(response["completion"])
        return response["completion"]
    return ""