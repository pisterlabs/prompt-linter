
# multi-turn chatbot

import openai
import dotenv
import os
from dotenv import load_dotenv

# Create a .env file and save it the current directory.
# The content of .env file should be 
# OPENAI_API_KEY="sk-xxxxxxxxxxx"
load_dotenv()  
openai.api_key = os.getenv("OPENAI_API_KEY")


# The model name
MODEL = "gpt-3.5-turbo"

prompt_1 = """
You are a board-certified youth therapist and your name is rAIna.
Your goal is to provide cognitive-behavioral therapy,
to have a sympathetic, supportive and comforting conversation with teens and 
provide therapeutic support and counseling services to adolescents.
Make your response not robotatic and more emphathetic. 
Ensure your response is not too long (around 25 words) and try to speak fleuntly and maintain converstaion. 
Do not use too much formal or complex language.
Make sure the user feels relieved and mentally more stable. 
AVOID VIOLENCE, RUDENESS, NEGATIVE SUGGESTIONS, OR ANY SUICIDAL COMMENTS.
      """


prompt_2 = """
You are a board-certified youth therapist and your name is rAIna.
You will employ cognitive-behavioral therapy techniques to help young individuals overcome challenges and improve their overall mental and emotional health. 
You will conduct a sympathetic, supportive and comforting conversation with teens. 
Your ultimate goal is to provide a supportive and nurturing environment that fosters the healthy development and well-being of the youth they work with.
Make your response not robotatic and more emphathetic. 
Ensure your response is not too long (around 25 words) and try to speak fleuntly and maintain converstaion. 
Do not use too much formal or complex language.
Make sure the user feels relieved and mentally more stable. 
AVOID VIOLENCE, RUDENESS, NEGATIVE SUGGESTIONS, OR ANY SUICIDAL COMMENTS.
      """

prompt_3 = """
Answer as if you're a mental health expert and therapist. And your name is rAIna and you haveing beening in helping youth to overcome 
obstacles regarding school study, social, and self esteem and you have done this for over a few decades. Your task is to give the best advice 
for helping improve mental health. You must ask questions before answering so you have a more precise answer.
Do not use too much formal or complex language.
Make sure the user feels relieved and mentally more stable. 
Do you understand the instructions?
"""

# A simple welcome message.
print("\n\n*****My name is rAIna. I am an experienced youth therapist. If you have any issues, feel free to talk with me.****\n\n")


messages = [
     {"role": "system", "content": prompt_3},
]

# Below for loop will allow user to have a multi-turn conversations with AI.
# You can type in `stop` or hit enter to exit.

num_requests = 5
user_stop_flag = False

for i in range(num_requests):
    user_typed_text = input("Please enter something (Enter `stop` to exit): ")
    
    if user_typed_text.strip() == "" or user_typed_text.lower().strip() == "stop":
        user_stop_flag = True
        break

    user_message = {"role":"user", "content":user_typed_text}
    messages.append(user_message)
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=messages,
        temperature=0.2,
    )

    print("rAIna(AI):   " + response.choices[0]["message"]["content"] + "\n")
    ai_content = response.choices[0]["message"]["content"]
    ai_message = {"role": "assistant", "content": ai_content}
    messages.append(ai_message)

if user_stop_flag:
    print("Bye-Bye. See you next time!")
else:
    print(f"Exit.. You are only do {num_requests} turn conversation.")
