import os, sys
import openai

openai.api_key = os.getenv('OPENAI_API_KEY')

while True:
    user_input = input("Enter a phrase to calculate sentiment: ")
    if user_input == 'exit' or user_input == 'quit':
        break

    response = openai.ChatCompletion.create(
        #model = 'gpt-4',
        model = 'gpt-3.5-turbo',
        messages = [
            {"role":"system", "content":"You are a sentiment classification bot, respond with the sentiment of the user"},
            {"role":"user", "content":user_input}
        ],
        temperature=0.7,
        max_tokens=150
    )
    
    response_message = response["choices"][0]["message"]
    print(response_message)
    