import openai
import os
import pyttsx3

openai.api_key = "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"

# Set up pyttsx3 engine with the appropriate driver
engine = pyttsx3.init()

# Set the voice properties (optional)
voices = engine.getProperty('voices')

engine.setProperty('voice', voices[1].id)  # You can change '0' to another index if you prefer a different voice

while True:
    user_input = input("You: ")

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
             {
                "role": "system",
                "content": "simulating as my gf"
            },
            {
                "role": "user",
                "content": user_input
            }
        ],
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    # Access and print the generated response
    generated_text = response["choices"][0]["message"]["content"]
    print(generated_text)
    with open ('myfile.txt','a') as f:
        e=generated_text
        f.writelines(e)

    # Say the generated response using pyttsx3
    engine.say(generated_text)
    engine.runAndWait()
    with open ('myfile.txt','a') as f:
        f.writelines("\n")