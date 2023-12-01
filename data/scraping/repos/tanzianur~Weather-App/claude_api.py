import sys
import os
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from dotenv import load_dotenv
load_dotenv()

anthropic = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

def generate_response(prompt):
    completion = anthropic.completions.create(
        model="claude-instant-1",
        max_tokens_to_sample=300,
        prompt=f"{HUMAN_PROMPT} {prompt} {AI_PROMPT}",
    )
    return completion.completion


arg1 = sys.argv[1]

print(
    generate_response(
        arg1,
    )
)
