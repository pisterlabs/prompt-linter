import openai
import os

openai.api_key = "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"

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user" , "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    print(str(response.choices[0].message))
    return response.choices[0].message["content"]



# user_input = " "

# while user_input != stop:
#     user_input = input("Enter your text: ")
#     print(get_completion(user_input))


while True:
    user_input = input("Enter your text (type 'stop' to exit): ")
    if user_input == "stop":
        break

    prompt_input = input("Enter your prompt: ")

    prompt =  prompt_input + f""" {user_input} """ 

    print(get_completion(prompt))