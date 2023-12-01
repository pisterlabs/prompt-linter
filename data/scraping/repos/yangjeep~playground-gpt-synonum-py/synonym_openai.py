import openai
from dotenv import load_dotenv
import os
import re

# Load API key from .env file
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

# Send API request
word = "levis"
context = "fashion"
prompt = f"Provide a list of synonyms for the word '{word}' in the context of '{context}', including at least 5 different synonyms."


seed = 0

response = openai.Completion.create(
    engine="davinci",
    prompt=prompt,
    max_tokens=50,
    n=3,
    stop=None,
    temperature=0.5,
    frequency_penalty=0,
    presence_penalty=0,
    seed=seed
)

print(response.choices[0].text)

# Select completion that matches desired format
pattern = r"^[\w, ]+$"
for choice in response.choices:
    if re.match(pattern, choice.text):
        synonyms_str = choice.text
        break

# Extract list of synonyms from response
synonyms_list = synonyms_str.split(", ")

# Print list of synonyms
print(synonyms_list)
