from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from dotenv import load_dotenv
import os
import json

load_dotenv()

anthropic = Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

# print("human_prompt",HUMAN_PROMPT) # just says "Human: "
# print("ai_prompt",AI_PROMPT) # just says "Assistant: "

user_json = json.load(open('jess.json'))
print(user_json)

# first decide what the main concerns are
completion = anthropic.completions.create(
    model="claude-2",
    max_tokens_to_sample=50,
    prompt=f"{HUMAN_PROMPT} Our brand account tweeted the following, and received the comments below. Summarize the 3 top concerns. {AI_PROMPT}",
)
print(completion.completion)

# for each concern, pull up the tweets that are related to that concern
completion = anthropic.completions.create(
    model="claude-2",
    max_tokens_to_sample=50,
    prompt=f"{HUMAN_PROMPT} Our brand account tweeted the following, and received the comments below. Summarize the 3 top concerns. {AI_PROMPT}",
)
print(completion.completion)