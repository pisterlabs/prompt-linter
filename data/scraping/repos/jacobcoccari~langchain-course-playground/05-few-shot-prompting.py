from dotenv import load_dotenv
import openai
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def chat(message, model="gpt-3.5-turbo"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": message}],
    )
    return response.choices[0].message["content"]


dialogue = f"""
<child>: Teach me about resilience.

<grandparent>: Resilience is like a tree in a storm. It may sway and bend, but it remains firmly rooted and steadfast through even the toughest of winds. It is the ability to bounce back from adversity and grow stronger, much like how the tree heals and thrives after the storm has passed.

<child>: Teach me about patience.

"""

prompt = f"""Please complete this dialouge while \
maintining the same tone of the grandparent.

dialogue: ```{dialogue}```
"""

response = chat(prompt)
print(response)
