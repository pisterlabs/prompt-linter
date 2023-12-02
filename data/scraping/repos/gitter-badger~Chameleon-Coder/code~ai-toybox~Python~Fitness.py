import os

import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
    model="text-davinci-002",
    prompt="Brainstorm some ideas combining VR and fitness:",
    temperature=0.6,
    max_tokens=150,
    top_p=1.0,
    frequency_penalty=1,
    presence_penalty=1,
)
