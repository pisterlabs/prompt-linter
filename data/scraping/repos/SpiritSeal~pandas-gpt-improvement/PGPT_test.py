from . import pandasGPT as pgpt
import openai

# OpenAI API key
openai_key = None
with open('openai_api.key', 'r') as f:
    openai_key = f.read().strip()
openai.api_key = openai_key