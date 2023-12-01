import os
import openai

# Configuration of the API key
with open("fine_tuning/key.txt", "r") as file:
    api_key = file.read().strip()
    
openai.api_key = api_key

response = openai.ChatCompletion.create(
  model="ft:gpt-3.5-turbo-0613:personal::8FgZJBt6",
  messages=[
        {"role": "system", "content": "You are a sadness rating assistant."},
        {"role": "user", "content": "In the following tweet, what is the rate of sadness over 10? Here is the tweet: I feel like singing the song human sadness by Julian Cassablancas and The Voidz or some other sad music. #depression #sad #sadness"}
    ]
)

print(response.choices[0].message['content'])
