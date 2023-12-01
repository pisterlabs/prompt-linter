from dotenv import load_dotenv
load_dotenv()

import os
import openai

#model = "gpt-35-turbo"
model = "gpt-4"

prompt: str = "Write an introductory paragraph to explain Generative AI to the reader of this content." 
system_prompt: str = "Explain in detail to help student understand the concept.", 
assistant_prompt: str = None,

messages = [
    {"role": "user", "content": f"{prompt}"},
    {"role": "system", "content": f"{system_prompt}"},
    {"role": "assistant", "content": f"{assistant_prompt}"}
]

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_version = '2020-11-07'

completion = openai.ChatCompletion.create(
    model = model,
    messages = messages,
    temperature = 0.7
)

print(completion)
response = completion["choices"][0]["message"].content
print(response)
