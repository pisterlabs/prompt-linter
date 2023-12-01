from api_key import API_KEY



import os
import openai

openai.api_key = API_KEY 

start_sequence = "\nAI:"
restart_sequence = "\nHuman: "


while True:
    ask = input("Enter a question:  ")
   
    if ask == "break":
        print("thank you")
        break
    else:

    
        response = openai.Completion.create(
        model="text-davinci-003",
        prompt=ask,
        temperature=0.9,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.6,
        stop=[" Human:", " AI:"]
        )


        print(response["choices"][0]["text"])