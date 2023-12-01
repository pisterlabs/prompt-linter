from dotenv import load_dotenv

load_dotenv(".env")

from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

anthropic = Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
)

completion = anthropic.completions.create(
    model="claude-2",
    max_tokens_to_sample=300,
    prompt=f"{HUMAN_PROMPT} how does a court case get to the Supreme Court? {AI_PROMPT}",
)
print(completion.completion)
