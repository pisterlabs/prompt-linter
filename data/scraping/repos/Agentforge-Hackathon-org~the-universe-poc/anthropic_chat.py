import os
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from dotenv import load_dotenv

load_dotenv()

class AnthropicChatBot:
    def __init__(self):
        self.anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def generate_prompt(self,message):
        completion = self.anthropic.completions.create(
            model="claude-1-100k",
            max_tokens_to_sample=300,
            prompt=f"{HUMAN_PROMPT} {message} {AI_PROMPT}",
        )
        with open("docs/out/output.txt", "w") as f:
            f.write(completion.completion)
        return completion.completion
