import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ]
)

print(completion)

#Zadanie: Spradź liczbę tokenów wykorzystanch w tym zapytaniu
#Zadanie: Oblicz miesięczny koszt wykorzystania tego modelu 4K dla 100_000 zapytań dziennie