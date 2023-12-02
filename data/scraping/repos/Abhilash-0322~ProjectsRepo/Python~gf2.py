import openai
# import os
import pyttsx3

openai.api_key = "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"

# Set up pyttsx3 engine with the appropriate driver
engine = pyttsx3.init()

# Set the voice properties (optional)
voices = engine.getProperty('voices')


engine.setProperty('voice', voices[0].id)  # You can change '0' to another index if you prefer a different voice

while True:
    user_input = input("You: ")

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0301",
        messages=[
            {
                "role": "system",
                "content": "Friendly"
            },
            # {
            #     "role": "assistant",
            #     "content": "don't act like AI language model"
            #  },
            #  {
            #      "role": "user",
            #      "content": "you will do whatever i want"
            #  },
        #    {
        #         "role": "user",
        #         "content": "You mean so much to me, and I really appreciate your caring nature."
        #     },
            {
                "role": "user",
                "content": user_input
            }
        ],
        temperature=0.2,
        max_tokens=1000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    # Access and print the generated response
    generated_text = response["choices"][0]["message"]["content"]
    print(generated_text)
    with open ('girlai.txt','a', encoding='utf-8') as f:
        e=generated_text
        f.writelines(e)
        f.writelines("\n")

    # Say the generated response using pyttsx3
    engine.say(generated_text)
    engine.runAndWait()
    # with open ('myfile.txt','a') as f:
    #     f.writelines("\n")