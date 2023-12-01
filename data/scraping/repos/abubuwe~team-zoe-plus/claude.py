import os

from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

anthropic = Anthropic(
    api_key = ANTHROPIC_API_KEY
)

def complete(human_prompt):
  try:
    completion = anthropic.completions.create(
        model="claude-2",
        max_tokens_to_sample=3000,
        prompt=f"{HUMAN_PROMPT} {human_prompt}{AI_PROMPT}",
    )
  except anthropic.APIConnectionError as e:
      print("The server could not be reached")
      print(e.__cause__)  # an underlying Exception, likely raised within httpx.
  except anthropic.RateLimitError as e:
      print("A 429 status code was received; we should back off a bit.")
  except anthropic.APIStatusError as e:
      print("Another non-200-range status code was received")
      print(e.status_code)
      print(e.response)
  
  return completion.completion.strip()