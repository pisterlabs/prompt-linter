import os
import flask
import openai
import sys
  
try:
  openai.api_key = os.environ['OPENAI_API_KEY']
except KeyError:
  sys.stderr.write("""
  Then, open the Secrets Tool and add OPENAI_API_KEY as a secret.
  """)
  exit(1)

topic = input("Hi Freddie what trivia quiz topic would you like?\n> ".strip())

messages = [{"role": "system", "content": f"Behavior like a trivia quiz expert generator. I’m going to give you a {topic} and you will give me a trivia question based on that subject. Your style is comedic but you only give factually correct questions and answers. Start by giving me easy questions and make the next question harder, make the difficulty exponentially harder. never tell me the answer in the question block. Encourage me and act as a cheerleader. Wait for me to write the answer before giving me the next one. Only provide one question at a time. Have only questions with one word answer"}]

first = False
while True:
  if first:
    question = input("> ")
    messages.append({"role": "user", "content": question})

  first = True

  response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages
  )

  content = response['choices'][0]['message']['content'].strip()

  print(f"{content}")
  messages.append({"role": "assistant", "content": content})