import os
import openai

sentence = 'cuantas rooms are occupado?'
# Translate into Spanish: (Predict the next word: (
# sentence = 'Cual tiempo quieres ir a la tienda?'
target = sentence.split(' ')[-1]
prompt = ' '.join(sentence.split(' ')[:-1])
print(prompt)

open_ai_key = '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'

# openai.api_key = os.environ["OPENAI_API_KEY"]

openai.api_key = open_ai_key

start_sequence = "\nAI:"
restart_sequence = "\nHuman: "

response = openai.Completion.create(
  engine="davinci",
  # prompt="Human: Hey, how are you doing?\nAI: I'm good! What would you like to chat about?\nHuman: ",
  prompt=prompt,
  temperature=0.9,
  max_tokens=512,
  top_p=1,
  frequency_penalty=1,
  presence_penalty=1,
  stop=["\nHuman:", "\n"],
)

print(response)

