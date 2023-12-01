import os
import openai
from zapsalvo import msg

openai.api_key = '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'

response = openai.Completion.create(
  model="text-davinci-003",
  prompt="qual o peso da terra?",
  temperature=1,
  max_tokens=256,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)

resposta = response['choices'][0]['text']
print(resposta)
