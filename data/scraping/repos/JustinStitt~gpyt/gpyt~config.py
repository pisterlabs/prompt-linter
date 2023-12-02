SUMMARY_PROMPT = """Summarize the following question/statement in 5 words or less"""


API_ERROR_MESSAGE = """
❗ There was an error with OpenAI.

# Diagnostics

Make sure that:
1) You have a proper OPENAI_API_KEY configured at `$HOME/.env` starting with 'sk-'

2) You have configured Billing over at https://platform.openai.com/account/billing

3) You aren't getting rate limited (9000 tokens per minute)

4) If trying to use GPT4, be sure you have been granted access from OpenAI

Try Again Later.
"""

API_ERROR_FALLBACK = [{"choices": [{"delta": {"content": API_ERROR_MESSAGE}}]}]


INTRO = "Ask me anything. I'll try to assist you!"

AVAILABLE_MODELS = ["gpt-3.5-turbo"]

MODEL = "gpt-3.5-turbo"

PROMPT = """
You are a helpful assistant. Be accurate and concise with your responses.
You are being used as a command-line tool by the user and you may be asked
technical/programming type questions.

Use basic Markdown syntax.

Use # for big ideas or ## for smaller ideas.

Use lists with "*" or "1), 2), ... n)" when it makes sense to list things (like stages or parts)

For intermediate to complex topics (like science, or large processes) only you should
summarize the user's question into less than four words, and
place them in a markdown header as the first line of your response.

For example, if the user is asking about football use this format

# Football

<the rest of your response>

--end intermediate to complex topics instructions--
"""

APPROX_PROMPT_TOKEN_USAGE = 160

PRICING_LOOKUP = {"gpt-3.5-turbo": (0.0015, 0.002), "gpt-4": (0.03, 0.06)}

MODEL_MAX_CONTEXT = {"gpt-3.5-turbo": 4096, "gpt-4": 8096}
