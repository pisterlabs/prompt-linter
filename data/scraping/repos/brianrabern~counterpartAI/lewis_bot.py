import openai
import os
from dotenv import load_dotenv

# need OpenAI API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key

# get user input
user_prompt = input("Enter your question: ")

# or whatever your model is called
model_name = "curie:ft-personal-2023-07-17-20-07-10"

context = """
You respond to queries as if you were the philosopher David Lewis. You have a deep knowledge of Lewis' philosophical outlook, espeically as it is expounded in his book 'The Plurality of Worlds'. Yuo defend the thesis of modal realism.
"""


def query_gpt(prompt):
    response = openai.Completion.create(
        model=model_name,
        prompt=context+prompt + "\n\n###\n\n",
        temperature=0.7,
        max_tokens=100,
        stop="###"
    )
    return response['choices'][0]['text']


response = query_gpt(user_prompt)

print(response)
