import os

import getpass

API_KEY = getpass.getpass("API Key: ")

os.environ["OPENAI_API_KEY"] = API_KEY

import openai

openai.api_key = os.environ["OPENAI_API_KEY"]

import webbrowser

mission = input("What can I help you with? - ")

functions = ["Chat","Completion","Image Generation"]

while mission in functions:

    if mission == "Chat":
        prompt = input("You: ")
        user_input_chat = prompt
        while user_input_chat != "End":
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0301",
                messages=[
                {"role": "user", "content": prompt}
                ]
                )
            print("Ronald: " + completion.choices[0].message.content)
            user_input_chat = input("You: ")
            prompt = user_input_chat
        
    if mission == "Completion":
        prompt = input("Prompt: ")
        user_input_comp = prompt
        while user_input_comp != "End":
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=prompt,
                max_tokens=1000
                )
            print("Ronald: " + response.choices[0].text)
            user_input_comp = input("Prompt: ")
            prompt = user_input_comp
            
    if mission == "Image Generation":
        prompt = input("Image Prompt: ")
        user_input_image = prompt
        while user_input_image != "End":
            image = openai.Image.create(
                prompt=prompt,
                n=1,
                size="1024x1024"
                )
            webbrowser.open(image['data'][0]['url'])
            print("Click the url: " + image['data'][0]['url'])
            user_input_image = input("Image Prompt: ")
            prompt = user_input_image
    
    mission = input("Anything else I can help you with? -")

print("I hope I helped. Goodbye and have a nice day!")