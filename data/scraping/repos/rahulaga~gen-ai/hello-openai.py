import os
import openai

openai.api_key  = os.environ['OPENAI_API_KEY']

#chat completion example, multiple messages maybe sent
messages = [{"role": "user", "content": "What other plants can grow well where grapes thrive? Think step by step"}]
response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                        messages=messages, temperature=0)
print(response)