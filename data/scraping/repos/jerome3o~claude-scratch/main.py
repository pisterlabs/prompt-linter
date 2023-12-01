import os
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT


anthropic = Anthropic(
    api_key=os.environ.get("API_KEY")
)

completion = anthropic.completions.create(
    model="claude-2",
    max_tokens_to_sample=300,
    prompt=f"{HUMAN_PROMPT} how does a court case get to the Supreme Court?{AI_PROMPT}",
)

print(completion.completion)
